"""Gaze-following confound check for a trained XL learner.
usage: python scripts/gaze_eval.py runs/m1_gpu_xl/params.pkl [n_envs] [steps]"""
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

from tribe import schema as S
from tribe import jax_world as J
from tribe.model import XLLearner
from tribe.train import world_step, sample_out, TrainConfig
import __main__; __main__.TrainConfig = TrainConfig

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 512
ck = pickle.load(open(path, "rb"))
cfg, params = ck["cfg"], ck["params"]
mc = cfg.model
K, L, d = mc.K, mc.n_layers, mc.d
model = XLLearner(mc)
lp = params["learner"]

SO = S.SELF_LEN
F_VALID, F_TYPE, F_X, F_FACE, F_POINT = 0, 1, 2, 4, 5


def adult_cue(frame):
    """first visible pointing adult slot index and its fields; frame (N, F)"""
    slots = frame[:, SO:].reshape(frame.shape[0], S.N_SLOTS, S.SLOT_LEN)
    is_a = (slots[..., F_VALID] > 0) & (slots[..., F_TYPE] == S.ADULT) & (slots[..., F_POINT] == 1)
    first = jnp.argmax(is_a, 1)
    s = slots[jnp.arange(frame.shape[0]), first]
    return is_a.any(1), first, s


def flip_facing(frame, first):
    """swap rel_facing 1 <-> 3 (right <-> left) on the cue slot only"""
    idx = SO + first * S.SLOT_LEN + F_FACE
    cur = frame[jnp.arange(frame.shape[0]), idx]
    new = jnp.where(cur == 1, 3, jnp.where(cur == 3, 1, cur))
    return frame.at[jnp.arange(frame.shape[0]), idx].set(new)


@jax.jit
def step(carry, _):
    env, mem, mem_ep, last_out, ep, key = carry
    frame = jax.vmap(J.observe, in_axes=(0, None))(env, 0)
    prev_same = mem_ep[:, -1] == ep
    logits, new_mem = model.apply(lp, frame, last_out, prev_same, mem, mem_ep, ep, method=XLLearner.step)
    has, first, s = adult_cue(frame)
    lflip, _ = model.apply(lp, flip_facing(frame, first), last_out, prev_same, mem, mem_ep, ep, method=XLLearner.step)
    p = jax.nn.softmax(logits["action"])
    pf = jax.nn.softmax(lflip["action"])
    key, k_s, k_r = jax.random.split(key, 3)
    out, _ = sample_out(k_s, logits)
    env2, reward, done = world_step(env, out, k_r)
    mem_ep2 = jnp.concatenate([mem_ep[:, 1:], ep[:, None]], 1)
    rec = dict(has=has, face=s[:, F_FACE], x=s[:, F_X] - S.REL_X_OFFSET, act=out[:, 0],
               pr=p[:, S.TURN_RIGHT], pl=p[:, S.TURN_LEFT], fr=pf[:, S.TURN_RIGHT], fl=pf[:, S.TURN_LEFT])
    return (env2, new_mem, mem_ep2, out, ep + done.astype(jnp.int32), key), rec


key = jax.random.PRNGKey(123)
k1, k2 = jax.random.split(key)
carry = (J.reset_batch(jax.random.split(k1, N), 1, 1), jnp.zeros((N, L, K - 1, d)),
         jnp.full((N, K - 1), -1, jnp.int32), jnp.zeros((N, 3), jnp.int32), jnp.zeros((N,), jnp.int32), k2)
_, rec = jax.lax.scan(step, carry, None, length=STEPS)
rec = {k: np.asarray(v).reshape(-1) for k, v in rec.items()}
cue = rec["has"] & np.isin(rec["face"], [1, 3])
print(f"{cue.sum():,} cue steps out of {rec['has'].size:,}")

R, Lf = rec["face"] == 1, rec["face"] == 3
a = rec["act"]
print("\n1) Stratified by adult position (rel_x <0 left, 0 ahead, >0 right of learner)")
print(f"{'adult pos':>10} {'n(face R)':>9} {'n(face L)':>9} {'P(turnR|faceR)':>15} {'P(turnR|faceL)':>15} {'P(turnL|faceL)':>15} {'P(turnL|faceR)':>15}")
for name, m in [("left", rec["x"] < 0), ("ahead", rec["x"] == 0), ("right", rec["x"] > 0)]:
    mr, ml = cue & m & R, cue & m & Lf
    if mr.sum() < 30 or ml.sum() < 30:
        print(f"{name:>10} {mr.sum():9d} {ml.sum():9d}   (too few)")
        continue
    print(f"{name:>10} {mr.sum():9d} {ml.sum():9d} {np.mean(a[mr]==S.TURN_RIGHT):15.3f} {np.mean(a[ml]==S.TURN_RIGHT):15.3f} "
          f"{np.mean(a[ml]==S.TURN_LEFT):15.3f} {np.mean(a[mr]==S.TURN_LEFT):15.3f}")

print("\n2) Intervention: flip only the adult's observed facing, same memory")
for name, m, match, other, fmatch, fother in [("faces right", cue & R, "pr", "pl", "fr", "fl"),
                                              ("faces left", cue & Lf, "pl", "pr", "fl", "fr")]:
    real = rec[match][m] - rec[other][m]
    flip = rec[fmatch][m] - rec[fother][m]
    diff = real - flip
    se = diff.std() / np.sqrt(len(diff))
    print(f"  adult {name:11s} n={m.sum():7,}  P(match turn)-P(other turn): real {real.mean():+.3f}, flipped {flip.mean():+.3f}, "
          f"effect {diff.mean():+.3f} (95% CI {diff.mean()-1.96*se:+.3f} to {diff.mean()+1.96*se:+.3f})")
    print(f"  {'':17s} mean P(match turn): real {rec[match][m].mean():.3f} -> flipped {rec[fmatch][m].mean():.3f}")
