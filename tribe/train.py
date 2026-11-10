"""M1 training: one learner, one scripted adult, PPO + next-frame prediction.

    python -m tribe.train --preset cpu      # smoke test
    python -m tribe.train --preset gpu      # the real M1 run

Everything inside one jitted update: rollout (lax.scan over T steps across N
environments, auto-reset), GAE, then PPO epochs over minibatches that rebuild
each sample's K-frame window from the stored per-env frame sequence.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import pickle
import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import schema as S
from . import jax_world as J
from .model import Learner, XLLearner, Predictor, ModelConfig, prediction_losses


# ------------------------------------------------------------- config ---
@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = ModelConfig()
    n_envs: int = 64
    T: int = 32                    # rollout length
    epochs: int = 2
    minibatches: int = 4
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    gamma: float = 0.98
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_action: float = 0.02
    ent_point: float = 0.01
    ent_say: float = 0.0           # words are not needed in M1; keep the head quiet
    w_world: float = 1.0
    w_heard: float = 1.0
    w_pi: float = 1.0
    updates: int = 200
    seed: int = 0
    log_every: int = 1
    ckpt_every: int = 50
    out_dir: str = "runs/m1"


PRESETS = {
    "cpu": TrainConfig(model=ModelConfig(d=64, n_layers=2, n_heads=4, K=8), n_envs=32, T=32,
                       epochs=2, minibatches=4, updates=60, out_dir="runs/m1_cpu"),
    # fits a 16 GB T4 (Colab free): ~1.6M params, 2k-sample minibatches of 16-frame windows
    "colab": TrainConfig(model=ModelConfig(d=192, n_layers=4, n_heads=6, K=16), n_envs=512, T=64,
                         epochs=2, minibatches=16, updates=2000, lr=3e-4, ckpt_every=20,
                         out_dir="runs/m1_colab"),
    # memory scales with (n_envs * T / minibatches) * K: 128 minibatches -> 1024 windows of 32 frames
    "gpu": TrainConfig(model=ModelConfig(d=256, n_layers=6, n_heads=8, K=32), n_envs=2048, T=64,
                       epochs=3, minibatches=128, updates=3000, lr=2e-4, out_dir="runs/m1_gpu"),
    # XL memory (see make_train_xl): each frame is encoded once per pass instead of K times.
    # minibatches split envs; 32 -> 64 envs x 64 steps = 4096 samples per gradient step.
    "cpu_xl": TrainConfig(model=ModelConfig(d=64, n_layers=2, n_heads=4, K=8, arch="xl", remat=False),
                          n_envs=32, T=32, epochs=2, minibatches=4, updates=60, out_dir="runs/m1_cpu_xl"),
    "gpu_xl": TrainConfig(model=ModelConfig(d=256, n_layers=6, n_heads=8, K=32, arch="xl", remat=False),
                          n_envs=2048, T=64, epochs=3, minibatches=32, updates=3000, lr=2e-4,
                          out_dir="runs/m1_gpu_xl"),
}

# reward shaping
R_ALIVE = 0.02
R_NEED = 2.0
R_DEATH = -1.0


def world_step(env, out, key):
    """Learner output (N, 3) + scripted adult -> (next env with auto-reset, reward, done)."""
    N = out.shape[0]
    aa, ap, asay = jax.vmap(J.adult_act, in_axes=(0, None))(env, 1)
    acts = jnp.stack([out[:, 0], aa], axis=1)
    pts = jnp.stack([out[:, 1], ap.astype(jnp.int32)], axis=1)
    says = jnp.stack([out[:, 2], asay], axis=1)
    before = (env.hunger[:, 0], env.thirst[:, 0], env.warmth[:, 0])
    env2 = jax.vmap(J.step)(env, acts, pts, says)
    gains = sum(jnp.maximum(a2 - a1, 0.0) for a1, a2 in zip(before, (env2.hunger[:, 0], env2.thirst[:, 0], env2.warmth[:, 0])))
    died = env2.died_this_step[:, 0]
    reward = R_ALIVE * env2.alive[:, 0] + R_NEED * gains + R_DEATH * died
    done = died | (env2.t >= S.EPISODE_STEPS)
    fresh = J.reset_batch(jax.random.split(key, N), 1, 1)
    env3 = jax.tree.map(lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), env2, fresh)
    return env3, reward, done


def logp_of(logits, out):
    """logits leaves (M, n); out (M, 3) -> (M,)"""
    la = jax.nn.log_softmax(logits["action"])[jnp.arange(out.shape[0]), out[:, 0]]
    lp = jax.nn.log_softmax(logits["pointing"])[jnp.arange(out.shape[0]), out[:, 1]]
    ls = jax.nn.log_softmax(logits["saying"])[jnp.arange(out.shape[0]), out[:, 2]]
    return la + lp + ls


def entropy_of(logits):
    def H(l):
        p = jax.nn.softmax(l)
        return -(p * jax.nn.log_softmax(l)).sum(-1)
    return H(logits["action"]), H(logits["pointing"]), H(logits["saying"])


def sample_out(key, logits):
    k1, k2, k3 = jax.random.split(key, 3)
    a = jax.random.categorical(k1, logits["action"])
    p = jax.random.categorical(k2, logits["pointing"])
    s = jax.random.categorical(k3, logits["saying"])
    out = jnp.stack([a, p, s], axis=-1).astype(jnp.int32)
    return out, logp_of(logits, out)


def ppo_loss(cfg, logits, predict_fn, out, old_logp, adv, ret, next_frame, has_next):
    """Shared PPO + prediction loss over flat (M, ...) samples. Prediction losses reach the
    trunk only through backprop; they never enter the advantage."""
    logp = logp_of(logits, out)
    ratio = jnp.exp(logp - old_logp)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_n).mean()
    vf = 0.5 * ((logits["value"] - ret) ** 2).mean()
    Ha, Hp, Hs = entropy_of(logits)
    ent = cfg.ent_action * Ha.mean() + cfg.ent_point * Hp.mean() + cfg.ent_say * Hs.mean()
    sl, ll = predict_fn(logits["h"], out)
    world, heard = prediction_losses(sl, ll, next_frame)
    m = has_next.astype(jnp.float32)
    world = (world * m).sum() / jnp.maximum(m.sum(), 1.0)
    heard = (heard * m).sum() / jnp.maximum(m.sum(), 1.0)
    total = cfg.w_pi * (pg + cfg.vf_coef * vf - ent) + cfg.w_world * world + cfg.w_heard * heard
    approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
    return total, {"pg": pg, "vf": vf, "ent_action": Ha.mean(), "world": world, "heard": heard,
                   "approx_kl": approx_kl, "clipfrac": (jnp.abs(ratio - 1) > cfg.clip).mean()}


def gae(cfg, trs, last_value):
    """trs.reward/value/done: (T, N). Returns (advantages, returns), both (T, N)."""
    def body(carry, x):
        adv_next, v_next = carry
        r, v, d = x
        nonterm = 1.0 - d.astype(jnp.float32)
        delta = r + cfg.gamma * v_next * nonterm - v
        adv = delta + cfg.gamma * cfg.lam * nonterm * adv_next
        return (adv, v), adv
    _, advs = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value),
                           (trs.reward, trs.value, trs.done), reverse=True)
    return advs, advs + trs.value


# ------------------------------------------------------------ runner ---
class RunnerState(NamedTuple):
    env: J.EnvState                # batched
    buf_frames: jnp.ndarray        # (N, K-1, F)  frames before the current one
    buf_out: jnp.ndarray           # (N, K-1, 3)
    buf_ep: jnp.ndarray            # (N, K-1)     episode ids
    ep_id: jnp.ndarray             # (N,)
    key: jnp.ndarray


class Transition(NamedTuple):
    frame: jnp.ndarray             # (N, F)   frame the decision was made on
    out: jnp.ndarray               # (N, 3)   (action, pointing, saying) taken
    ep: jnp.ndarray                # (N,)
    logp: jnp.ndarray              # (N,)
    value: jnp.ndarray             # (N,)
    reward: jnp.ndarray            # (N,)
    done: jnp.ndarray              # (N,)     episode ended after this step


def make_train(cfg: TrainConfig):
    mcfg = cfg.model
    K, N, T, F = mcfg.K, cfg.n_envs, cfg.T, S.FRAME_LEN
    model = Learner(mcfg)
    predictor = Predictor(mcfg)

    def init_params(key):
        k1, k2 = jax.random.split(key)
        f = jnp.zeros((1, K, F), jnp.int32)
        o = jnp.zeros((1, K, 3), jnp.int32)
        v = jnp.ones((1, K), bool)
        lp = model.init(k1, f, o, v)
        pp = predictor.init(k2, jnp.zeros((1, mcfg.d)), jnp.zeros((1, 3), jnp.int32))
        return {"learner": lp, "predictor": pp}

    def forward(params, frames, prev_out, valid):
        return model.apply(params["learner"], frames, prev_out, valid)

    def predict(params, h, out):
        return predictor.apply(params["predictor"], h, out)

    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))

    def reset_runner(key):
        k1, k2 = jax.random.split(key)
        env = J.reset_batch(jax.random.split(k1, N), 1, 1)
        return RunnerState(
            env=env,
            buf_frames=jnp.zeros((N, K - 1, F), jnp.int32),
            buf_out=jnp.zeros((N, K - 1, 3), jnp.int32),
            buf_ep=jnp.full((N, K - 1), -1, jnp.int32),
            ep_id=jnp.zeros((N,), jnp.int32),
            key=k2,
        )

    def window(buf_frames, buf_out, buf_ep, frame, ep):
        """Assemble the model input for the current step."""
        frames = jnp.concatenate([buf_frames, frame[:, None]], axis=1)           # (N, K, F)
        prev_out = jnp.concatenate([buf_out, jnp.zeros((N, 1, 3), jnp.int32)], axis=1)
        eps = jnp.concatenate([buf_ep, ep[:, None]], axis=1)
        valid = eps == ep[:, None]
        return frames, prev_out, valid

    def env_step(rs: RunnerState, params):
        env = rs.env
        frame = jax.vmap(J.observe, in_axes=(0, None))(env, 0)                     # learner frames (N, F)
        frames, prev_out, valid = window(rs.buf_frames, rs.buf_out, rs.buf_ep, frame, rs.ep_id)
        logits = forward(params, frames, prev_out, valid)
        key, k_s, k_r = jax.random.split(rs.key, 3)
        out, logp = sample_out(k_s, logits)
        env3, reward, done = world_step(env, out, k_r)
        # buffers: push this frame/out, then advance episode id where done
        buf_frames = jnp.concatenate([rs.buf_frames[:, 1:], frame[:, None]], axis=1)
        buf_out = jnp.concatenate([rs.buf_out[:, 1:], out[:, None]], axis=1)
        buf_ep = jnp.concatenate([rs.buf_ep[:, 1:], rs.ep_id[:, None]], axis=1)
        ep_id = rs.ep_id + done.astype(jnp.int32)
        tr = Transition(frame=frame, out=out, ep=rs.ep_id, logp=logp, value=logits["value"],
                        reward=reward, done=done)
        return RunnerState(env3, buf_frames, buf_out, buf_ep, ep_id, key), tr

    def rollout(rs, params):
        init_buf = (rs.buf_frames, rs.buf_out, rs.buf_ep)
        rs, trs = jax.lax.scan(lambda r, _: env_step(r, params), rs, None, length=T)
        # bootstrap value and the frame after the last step (prediction target)
        env = rs.env
        last_frame = jax.vmap(J.observe, in_axes=(0, None))(env, 0)
        frames, prev_out, valid = window(rs.buf_frames, rs.buf_out, rs.buf_ep, last_frame, rs.ep_id)
        last_value = forward(params, frames, prev_out, valid)["value"]
        return rs, trs, init_buf, last_frame, last_value

    def build_windows(seq_frames, seq_out, seq_ep, n_idx, t_idx):
        """seq_*: (N, K-1+T+1, ...) ; sample (n, t) -> window ending at rollout step t."""
        ar = jnp.arange(K)
        ti = t_idx[:, None] + ar[None, :]                       # (M, K) into seq (offset K-1 already folded)
        frames = seq_frames[n_idx[:, None], ti]                 # (M, K, F)
        out = seq_out[n_idx[:, None], ti]                       # (M, K, 3)
        eps = seq_ep[n_idx[:, None], ti]
        valid = eps == eps[:, -1:]
        prev_out = out.at[:, -1].set(0)
        return frames, prev_out, valid

    def loss_fn(params, batch):
        frames, prev_out, valid, out, old_logp, adv, ret, next_frame, has_next = batch
        logits = forward(params, frames, prev_out, valid)
        return ppo_loss(cfg, logits, lambda h, o: predict(params, h, o), out, old_logp, adv, ret,
                        next_frame, has_next)

    @jax.jit
    def update(params, opt_state, rs):
        rs, trs, init_buf, last_frame, last_value = rollout(rs, params)
        adv, ret = gae(cfg, trs, last_value)
        # per-env sequences: [buffer (K-1)] + [rollout T] + [frame after]
        seq_frames = jnp.concatenate([init_buf[0], jnp.swapaxes(trs.frame, 0, 1), last_frame[:, None]], axis=1)
        seq_out = jnp.concatenate([init_buf[1], jnp.swapaxes(trs.out, 0, 1), jnp.zeros((N, 1, 3), jnp.int32)], axis=1)
        seq_ep = jnp.concatenate([init_buf[2], jnp.swapaxes(trs.ep, 0, 1), rs.ep_id[:, None]], axis=1)
        # flat sample table
        t_grid, n_grid = jnp.meshgrid(jnp.arange(T), jnp.arange(N), indexing="ij")
        t_flat, n_flat = t_grid.reshape(-1), n_grid.reshape(-1)
        out_flat = trs.out.reshape(-1, 3)
        logp_flat, adv_flat, ret_flat = trs.logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        has_next = (~trs.done).reshape(-1)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, T * N)
            mb = perm.reshape(cfg.minibatches, -1)

            def mini(carry, idx):
                params, opt_state = carry
                n_idx, t_idx = n_flat[idx], t_flat[idx]
                frames, prev_out, valid = build_windows(seq_frames, seq_out, seq_ep, n_idx, t_idx)
                next_frame = seq_frames[n_idx, t_idx + K]        # frame after step t
                batch = (frames, prev_out, valid, out_flat[idx], logp_flat[idx], adv_flat[idx],
                         ret_flat[idx], next_frame, has_next[idx])
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
                upd, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, upd)
                aux["loss"] = loss
                return (params, opt_state), aux
            (params, opt_state), auxs = jax.lax.scan(mini, (params, opt_state), mb)
            return (params, opt_state, key), jax.tree.map(jnp.mean, auxs)

        (params, opt_state, key), auxs = jax.lax.scan(epoch, (params, opt_state, rs.key), None, length=cfg.epochs)
        rs = rs._replace(key=key)
        metrics = jax.tree.map(jnp.mean, auxs)
        metrics["reward_per_step"] = trs.reward.mean()
        metrics["episodes_ended"] = trs.done.sum()
        metrics["deaths"] = (trs.reward <= R_DEATH + 1e-6).sum()
        return params, opt_state, rs, trs, metrics

    return init_params, tx, reset_runner, update


# ---------------------------------------------------------- XL runner ---
class XLRunnerState(NamedTuple):
    env: J.EnvState                # batched
    mem: jnp.ndarray               # (N, n_layers, K-1, d) layer inputs at the previous K-1 positions
    mem_ep: jnp.ndarray            # (N, K-1) their episode ids (-1 = empty)
    last_out: jnp.ndarray          # (N, 3) output taken at the previous position
    ep_id: jnp.ndarray             # (N,)
    key: jnp.ndarray


class XLTransition(NamedTuple):
    frame: jnp.ndarray             # (N, F)
    out: jnp.ndarray               # (N, 3)
    prev_out: jnp.ndarray          # (N, 3)   model input: output at the previous position
    prev_same: jnp.ndarray         # (N,)     previous position is in this episode
    ep: jnp.ndarray                # (N,)
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def make_train_xl(cfg: TrainConfig):
    """Same PPO + prediction objective as make_train, with XLLearner: the rollout encodes one
    frame per step against cached per-layer memory, and training runs each env's whole
    segment once (minibatches are groups of envs) instead of rebuilding a K-frame window
    per sample. The training pass reproduces the rollout's computation exactly, so the
    first minibatch has ratio 1."""
    mcfg = cfg.model
    K, N, T, F, L, d = mcfg.K, cfg.n_envs, cfg.T, S.FRAME_LEN, mcfg.n_layers, mcfg.d
    assert N % cfg.minibatches == 0, "xl minibatches split envs: n_envs must divide evenly"
    model = XLLearner(mcfg)
    predictor = Predictor(mcfg)

    def init_params(key):
        k1, k2 = jax.random.split(key)
        lp = model.init(k1, jnp.zeros((1, 2, F), jnp.int32), jnp.zeros((1, 2, 3), jnp.int32),
                        jnp.ones((1, 2), bool), jnp.zeros((1, L, K - 1, d)),
                        jnp.full((1, K - 1), -1, jnp.int32), jnp.zeros((1, 2), jnp.int32))
        pp = predictor.init(k2, jnp.zeros((1, d)), jnp.zeros((1, 3), jnp.int32))
        return {"learner": lp, "predictor": pp}

    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))

    def reset_runner(key):
        k1, k2 = jax.random.split(key)
        return XLRunnerState(
            env=J.reset_batch(jax.random.split(k1, N), 1, 1),
            mem=jnp.zeros((N, L, K - 1, d), jnp.float32),
            mem_ep=jnp.full((N, K - 1), -1, jnp.int32),
            last_out=jnp.zeros((N, 3), jnp.int32),
            ep_id=jnp.zeros((N,), jnp.int32),
            key=k2,
        )

    def policy_step(params, rs, frame):
        prev_same = rs.mem_ep[:, -1] == rs.ep_id
        logits, new_mem = model.apply(params["learner"], frame, rs.last_out, prev_same, rs.mem,
                                      rs.mem_ep, rs.ep_id, method=XLLearner.step)
        return logits, new_mem, prev_same

    def env_step(rs: XLRunnerState, params):
        frame = jax.vmap(J.observe, in_axes=(0, None))(rs.env, 0)
        logits, new_mem, prev_same = policy_step(params, rs, frame)
        key, k_s, k_r = jax.random.split(rs.key, 3)
        out, logp = sample_out(k_s, logits)
        env3, reward, done = world_step(rs.env, out, k_r)
        mem_ep = jnp.concatenate([rs.mem_ep[:, 1:], rs.ep_id[:, None]], axis=1)
        tr = XLTransition(frame=frame, out=out, prev_out=rs.last_out, prev_same=prev_same, ep=rs.ep_id,
                          logp=logp, value=logits["value"], reward=reward, done=done)
        return XLRunnerState(env3, new_mem, mem_ep, out, rs.ep_id + done.astype(jnp.int32), key), tr

    def loss_fn(params, batch):
        frames, prev_out, prev_same, mem, mem_ep, eps, out, old_logp, adv, ret, next_frame, has_next = batch
        logits = model.apply(params["learner"], frames, prev_out, prev_same, mem, mem_ep, eps,
                             method=XLLearner.sequence)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        logits = jax.tree.map(flat, logits)
        return ppo_loss(cfg, logits, lambda h, o: predictor.apply(params["predictor"], h, o),
                        flat(out), flat(old_logp), flat(adv), flat(ret), flat(next_frame), flat(has_next))

    @jax.jit
    def update(params, opt_state, rs):
        mem0, mem_ep0 = rs.mem, rs.mem_ep
        rs, trs = jax.lax.scan(lambda r, _: env_step(r, params), rs, None, length=T)
        last_frame = jax.vmap(J.observe, in_axes=(0, None))(rs.env, 0)
        last_value = policy_step(params, rs, last_frame)[0]["value"]
        adv, ret = gae(cfg, trs, last_value)
        # per-env sequences (N, T, ...)
        sw = lambda x: jnp.swapaxes(x, 0, 1)
        frames, out, eps = sw(trs.frame), sw(trs.out), sw(trs.ep)
        next_frame = jnp.concatenate([frames[:, 1:], last_frame[:, None]], axis=1)
        data = (frames, sw(trs.prev_out), sw(trs.prev_same), mem0, mem_ep0, eps, out, sw(trs.logp),
                sw(adv), sw(ret), next_frame, sw(~trs.done))

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k = jax.random.split(key)
            mb = jax.random.permutation(k, N).reshape(cfg.minibatches, -1)

            def mini(carry, idx):
                params, opt_state = carry
                batch = jax.tree.map(lambda x: x[idx], data)
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
                upd, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, upd)
                aux["loss"] = loss
                return (params, opt_state), aux
            (params, opt_state), auxs = jax.lax.scan(mini, (params, opt_state), mb)
            return (params, opt_state, key), jax.tree.map(jnp.mean, auxs)

        (params, opt_state, key), auxs = jax.lax.scan(epoch, (params, opt_state, rs.key), None, length=cfg.epochs)
        rs = rs._replace(key=key)
        metrics = jax.tree.map(jnp.mean, auxs)
        metrics["reward_per_step"] = trs.reward.mean()
        metrics["episodes_ended"] = trs.done.sum()
        metrics["deaths"] = (trs.reward <= R_DEATH + 1e-6).sum()
        return params, opt_state, rs, trs, metrics

    return init_params, tx, reset_runner, update


# ---------------------------------------------------------- analysis ---
def gaze_metrics(frames: np.ndarray, actions: np.ndarray) -> dict:
    """Does the learner turn the way a visible, pointing adult faces?
    frames: (M, F); actions: (M,). Returns rates and a chance baseline."""
    slots = frames[:, S.SELF_LEN:].reshape(len(frames), S.N_SLOTS, S.SLOT_LEN)
    is_adult = (slots[..., 0] > 0) & (slots[..., 1] == S.ADULT) & (slots[..., 5] == 1)
    has = is_adult.any(1)
    first = is_adult.argmax(1)
    rf = slots[np.arange(len(frames)), first, 4]
    right = has & (rf == 1)
    left = has & (rf == 3)
    n_cue = int(right.sum() + left.sum())
    if n_cue == 0:
        return {"gaze_follow_rate": float("nan"), "gaze_chance": float("nan"), "gaze_cues": 0}
    follow = (actions[right] == S.TURN_RIGHT).sum() + (actions[left] == S.TURN_LEFT).sum()
    # chance: how often the learner turns right / left in general
    pr, pl = (actions == S.TURN_RIGHT).mean(), (actions == S.TURN_LEFT).mean()
    chance = (right.sum() * pr + left.sum() * pl) / n_cue
    return {"gaze_follow_rate": float(follow / n_cue), "gaze_chance": float(chance), "gaze_cues": n_cue}


# -------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="cpu")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pkl if present")
    ap.add_argument("--minibatches", type=int, default=None, help="more = smaller minibatches = less GPU memory")
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--remat", action="store_true", help="gradient checkpointing (xl presets default off)")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    if args.updates:
        cfg = dataclasses.replace(cfg, updates=args.updates)
    if args.out:
        cfg = dataclasses.replace(cfg, out_dir=args.out)
    if args.minibatches:
        cfg = dataclasses.replace(cfg, minibatches=args.minibatches)
    if args.envs:
        cfg = dataclasses.replace(cfg, n_envs=args.envs)
    if args.remat:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, remat=True))
    cfg = dataclasses.replace(cfg, seed=args.seed)
    mb = cfg.n_envs * cfg.T // cfg.minibatches
    if cfg.model.arch == "xl":
        print(f"minibatch = {cfg.n_envs // cfg.minibatches} envs x {cfg.T} steps = {mb} samples (xl memory)", flush=True)
    else:
        print(f"minibatch = {mb} windows x {cfg.model.K} frames (memory scales with this product)", flush=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    init_params, tx, reset_runner, update = (make_train_xl if cfg.model.arch == "xl" else make_train)(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    k_p, k_r = jax.random.split(key)
    params = init_params(k_p)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
    opt_state = tx.init(params)
    rs = reset_runner(k_r)
    start = 1
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pkl")
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        params, opt_state, start = ck["params"], ck["opt_state"], ck["update"] + 1
        rs = reset_runner(jax.random.fold_in(k_r, start))
        print(f"resumed from update {ck['update']}", flush=True)
    print(f"preset={args.preset} params={n_params:,} envs={cfg.n_envs} T={cfg.T} K={cfg.model.K} "
          f"device={jax.devices()[0].platform} out={cfg.out_dir}", flush=True)

    log_path = os.path.join(cfg.out_dir, "log.csv")
    writer = None
    t0 = time.time()
    for u in range(start, cfg.updates + 1):
        params, opt_state, rs, trs, metrics = update(params, opt_state, rs)
        m = {k: float(v) for k, v in metrics.items()}
        fr = np.asarray(trs.frame).reshape(-1, S.FRAME_LEN)
        ac = np.asarray(trs.out)[..., 0].reshape(-1)
        m.update(gaze_metrics(fr, ac))
        m["update"] = u
        m["steps"] = u * cfg.n_envs * cfg.T
        m["elapsed_s"] = time.time() - t0
        if writer is None:
            append = start > 1 and os.path.exists(log_path)
            fh = open(log_path, "a" if append else "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=list(m.keys()))
            if not append:
                writer.writeheader()
        writer.writerow(m); fh.flush()
        if u % cfg.log_every == 0:
            print(f"u{u:5d} steps {m['steps']:>9,} | R/step {m['reward_per_step']:+.4f} deaths {int(m['deaths']):4d} "
                  f"| world {m['world']:.3f} heard {m['heard']:.3f} pg {m['pg']:+.3f} vf {m['vf']:.3f} H {m['ent_action']:.2f} "
                  f"| gaze {m['gaze_follow_rate']:.2f} vs chance {m['gaze_chance']:.2f} (n={m['gaze_cues']}) "
                  f"| {m['elapsed_s']:.0f}s", flush=True)
        if u % cfg.ckpt_every == 0 or u == cfg.updates:
            with open(os.path.join(cfg.out_dir, "params.pkl"), "wb") as f:
                pickle.dump({"cfg": cfg, "params": jax.device_get(params)}, f)
            tmp = ckpt_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"cfg": cfg, "update": u, "params": jax.device_get(params),
                             "opt_state": jax.device_get(opt_state)}, f)
            os.replace(tmp, ckpt_path)


if __name__ == "__main__":
    main()
