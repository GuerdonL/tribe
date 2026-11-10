"""How does the policy respond to each direction the adult can face? At every step where the adult is
visible, set its observed facing to each of the four values (same memory) and average the action probabilities.
usage: python scripts/facing_eval.py <params.pkl> [n_envs]"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import evallib as E
from tribe import schema as S

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
cfg, params = E.load(path)


def extra(apply, frame, obs, logits, ctx):
    me, sl = E.split(frame)
    am = E.adult_mask(sl)
    first = jnp.argmax(am, 1)
    s = sl[jnp.arange(frame.shape[0]), first]
    idx = E.SO + first * S.SLOT_LEN + E.F_FACE
    ps = jnp.stack([jax.nn.softmax(apply(frame.at[jnp.arange(frame.shape[0]), idx].set(f))["action"]) for f in range(4)], 1)
    return dict(has=am.any(1), face=s[:, E.F_FACE], point=s[:, E.F_POINT], x=s[:, E.F_X] - S.REL_X_OFFSET, y=s[:, E.F_Y],
                word=me[:, E.I_HW], ps=ps.astype(jnp.float16))


rec = E.make_rollout(cfg, extra=extra)(params, jax.random.PRNGKey(7), N, S.EPISODE_STEPS, env_seed=11)
names = ["same way as me", "to my right", "toward me/behind", "to my left"]
target = np.isin(rec["word"], [S.WORD_ID["food"], S.WORD_ID["water"]])
print(f"{path}: adult visible on {rec['has'].mean()*100:.1f}% of steps")
for label, m in (("adult visible, hearing food/water", rec["has"] & target), ("adult visible, not hearing food/water", rec["has"] & ~target)):
    print(f"\n{label}: n={m.sum():,}; real facing split " + ", ".join(f"{names[f]} {np.mean(rec['face'][m]==f)*100:.0f}%" for f in range(4)))
    print(f"{'adult set to face':>20} " + " ".join(f"{a:>10}" for a in S.ACTION_NAMES))
    ps = rec["ps"][m].astype(np.float64)
    for f in range(4):
        print(f"{names[f]:>20} " + " ".join(f"{v:10.3f}" for v in ps[:, f].mean(0)))
