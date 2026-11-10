"""Q2: does the trained learner still need the adult? Blank parts of what it observes of the
adult and compare survival over one full episode, on the same worlds.
usage: python scripts/ablate_eval.py <params.pkl> [n_envs] [conditions,comma,separated]"""
import sys

import jax
import numpy as np

import evallib as E
from tribe import schema as S

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
conds = sys.argv[3].split(",") if len(sys.argv) > 3 else list(E.TRANSFORMS)
cfg, params = E.load(path)

res = {}
for c in conds:
    run = E.make_rollout(cfg, E.get_transform(c))
    res[c] = E.first_episode(run(params, jax.random.PRNGKey(7), N, S.EPISODE_STEPS, env_seed=11))

print(f"{path}: {N} worlds, one {S.EPISODE_STEPS}-step episode each, same worlds in every condition")
print(f"{'condition':>28} {'died %':>7} {'steps':>7} {'return':>7} {'hunger':>7} {'thirst':>7} {'warmth':>7} {'night in':>9} | "
      + " ".join(f"{c:>8}" for c in E.CAUSES[1:]))
for c in conds:
    r = res[c]
    causes = " ".join(f"{100*np.mean(r['cause']==k):8.2f}" for k in range(1, len(E.CAUSES)))
    print(f"{c:>28} {100*r['died'].mean():7.2f} {r['steps'].mean():7.1f} {r['ret'].mean():7.2f} {r['hunger'].mean():7.3f} "
          f"{r['thirst'].mean():7.3f} {r['warmth'].mean():7.3f} {r['night_sheltered'].mean():9.3f} | {causes}")
if "normal" in res:
    print("\npaired difference vs normal (95% bootstrap CI over worlds)")
    for c in conds:
        if c == "normal":
            continue
        out = []
        for k, scale in (("died", 100), ("steps", 1), ("ret", 1)):
            m, lo, hi = E.boot_ci((res[c][k].astype(float) - res["normal"][k].astype(float)) * scale)
            out.append(f"{k} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"{c:>28}  " + "   ".join(out))
