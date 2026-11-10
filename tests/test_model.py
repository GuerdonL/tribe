import jax
import jax.numpy as jnp
import numpy as np

from tribe import schema as S
from tribe.model import Learner, Predictor, ModelConfig, prediction_losses
from tribe.train import gaze_metrics


def test_forward_and_predict_shapes():
    cfg = ModelConfig(d=32, n_layers=1, n_heads=2, K=4)
    m, p = Learner(cfg), Predictor(cfg)
    B, K = 3, cfg.K
    frames = jnp.zeros((B, K, S.FRAME_LEN), jnp.int32)
    prev = jnp.zeros((B, K, 3), jnp.int32)
    valid = jnp.ones((B, K), bool)
    lp = m.init(jax.random.PRNGKey(0), frames, prev, valid)
    out = m.apply(lp, frames, prev, valid)
    assert out["action"].shape == (B, S.N_ACTIONS)
    assert out["saying"].shape == (B, S.N_WORDS)
    assert out["value"].shape == (B,)
    pp = p.init(jax.random.PRNGKey(1), out["h"], jnp.zeros((B, 3), jnp.int32))
    sl, ll = p.apply(pp, out["h"], jnp.zeros((B, 3), jnp.int32))
    assert len(sl) == S.SELF_LEN and len(ll) == S.SLOT_LEN
    assert ll[1].shape == (B, S.N_SLOTS, S.N_TYPES)
    world, heard, heard_spoken, spoken = prediction_losses(sl, ll, frames[:, -1])
    assert heard_spoken.shape == (B,) and spoken.shape == (B,)
    assert world.shape == (B,) and heard.shape == (B,)
    assert bool(jnp.isfinite(world).all()) and bool(jnp.isfinite(heard).all())


def test_padding_mask_blocks_past_episodes():
    """With the earlier frames marked invalid, output must not depend on them."""
    cfg = ModelConfig(d=32, n_layers=1, n_heads=2, K=4)
    m = Learner(cfg)
    B, K = 2, cfg.K
    key = jax.random.PRNGKey(0)
    frames = jax.random.randint(key, (B, K, S.FRAME_LEN), 0, 2).astype(jnp.int32)
    prev = jnp.zeros((B, K, 3), jnp.int32)
    valid = jnp.array([[False, False, False, True]] * B)
    lp = m.init(key, frames, prev, valid)
    a = m.apply(lp, frames, prev, valid)["action"]
    frames2 = frames.at[:, :3].set(1 - frames[:, :3])     # perturb only the masked frames
    b = m.apply(lp, frames2, prev, valid)["action"]
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_gaze_metric():
    F = S.FRAME_LEN
    frames = np.zeros((4, F), np.int64)
    acts = np.array([S.TURN_RIGHT, S.TURN_LEFT, S.FORWARD, S.TURN_RIGHT])
    # frame 0: adult pointing, rel_facing 1 -> correct is TURN_RIGHT
    base = S.SELF_LEN
    frames[0, base:base + 7] = [1, S.ADULT, 5 + S.REL_X_OFFSET, 2, 1, 1, 0]
    # frame 1: adult pointing, rel_facing 3 -> correct is TURN_LEFT
    frames[1, base:base + 7] = [1, S.ADULT, 0 + S.REL_X_OFFSET, 2, 3, 1, 0]
    # frame 2: adult not pointing -> no cue
    frames[2, base:base + 7] = [1, S.ADULT, 0 + S.REL_X_OFFSET, 2, 3, 0, 0]
    m = gaze_metrics(frames, acts)
    assert m["gaze_cues"] == 2 and m["gaze_follow_rate"] == 1.0


def _xl_rollout(m, lp, frames, outs, eps, mem, mem_ep, last_out):
    """Run XLLearner.step over T positions; return stacked logits and the per-step inputs."""
    from tribe.model import XLLearner
    T = frames.shape[1]
    acts, prev_outs, prev_sames = [], [], []
    for t in range(T):
        prev_same = mem_ep[:, -1] == eps[:, t]
        o, mem = m.apply(lp, frames[:, t], last_out, prev_same, mem, mem_ep, eps[:, t], method=XLLearner.step)
        acts.append(o["action"]); prev_outs.append(last_out); prev_sames.append(prev_same)
        mem_ep = jnp.concatenate([mem_ep[:, 1:], eps[:, t:t + 1]], axis=1)
        last_out = outs[:, t]
    return jnp.stack(acts, 1), jnp.stack(prev_outs, 1), jnp.stack(prev_sames, 1), mem


def test_xl_step_matches_sequence():
    """The rollout (cached, one step at a time) and training (whole segment) passes agree,
    including across an episode boundary and from a non-empty starting memory."""
    from tribe.model import XLLearner
    cfg = ModelConfig(d=32, n_layers=2, n_heads=2, K=4, arch="xl")
    m = XLLearner(cfg)
    B, T, L, K = 3, 9, cfg.n_layers, cfg.K
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    frames = jax.random.randint(k1, (B, T, S.FRAME_LEN), 0, 2).astype(jnp.int32)
    outs = jax.random.randint(k2, (B, T, 3), 0, 2).astype(jnp.int32)
    eps = jnp.array([[0] * 9, [0] * 5 + [1] * 4, [0] * 2 + [1] * 7])
    mem = 0.5 * jax.random.normal(k3, (B, L, K - 1, cfg.d))
    mem_ep = jnp.array([[-1, 0, 0], [0, 0, 0], [-1, -1, 0]])
    last_out = jnp.ones((B, 3), jnp.int32)
    lp = m.init(k4, frames, outs, jnp.ones((B, T), bool), mem, mem_ep, eps)
    a_step, prev_outs, prev_sames, _ = _xl_rollout(m, lp, frames, outs, eps, mem, mem_ep, last_out)
    seq = m.apply(lp, frames, prev_outs, prev_sames, mem, mem_ep, eps, method=XLLearner.sequence)
    assert np.allclose(np.asarray(a_step), np.asarray(seq["action"]), atol=1e-4)


def test_xl_memory_from_other_episode_is_ignored():
    from tribe.model import XLLearner
    cfg = ModelConfig(d=32, n_layers=2, n_heads=2, K=4, arch="xl")
    m = XLLearner(cfg)
    B, T = 2, 3
    key = jax.random.PRNGKey(1)
    frames = jax.random.randint(key, (B, T, S.FRAME_LEN), 0, 2).astype(jnp.int32)
    outs = jnp.zeros((B, T, 3), jnp.int32)
    same = jnp.array([[False, True, True]] * B)
    eps = jnp.ones((B, T), jnp.int32)
    mem_ep = jnp.zeros((B, cfg.K - 1), jnp.int32)          # all from episode 0
    mem_a = jax.random.normal(key, (B, cfg.n_layers, cfg.K - 1, cfg.d))
    lp = m.init(key, frames, outs, same, mem_a, mem_ep, eps)
    a = m.apply(lp, frames, outs, same, mem_a, mem_ep, eps)["action"]
    b = m.apply(lp, frames, outs, same, 3.0 * mem_a + 1.0, mem_ep, eps)["action"]
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
