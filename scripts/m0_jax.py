"""M0 in the JAX world: same comparison as m0.py, batched environments.

    python scripts/m0_jax.py --envs 200
    python scripts/m0_jax.py --bench          # env+adult throughput with a random learner
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tribe import schema as S
from tribe import jax_world as J
from tribe.policies import GazeIgnorer, GazeFollower


def run(policy_cls, n_envs, steps, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    st = J.reset_batch(keys, 1, 1)
    pols = [policy_cls(seed=1000 + e) for e in range(n_envs)]
    survived = np.full(n_envs, steps)
    done = np.zeros(n_envs, bool)
    for t in range(steps):
        frames = np.asarray(J.observe_all_batch(st))
        acts = np.zeros((n_envs, 2), np.int32)
        pts = np.zeros((n_envs, 2), np.int32)
        says = np.zeros((n_envs, 2), np.int32)
        for e in range(n_envs):
            if not done[e]:
                acts[e, 0], pts[e, 0], says[e, 0] = pols[e].act(frames[e, 0])
        aa, ap, asay = J.adult_act_batch(st, 1)
        acts[:, 1], pts[:, 1], says[:, 1] = np.asarray(aa), np.asarray(ap), np.asarray(asay)
        st = J.step_batch(st, jnp.asarray(acts), jnp.asarray(pts), jnp.asarray(says))
        alive = np.asarray(st.alive[:, 0])
        newly = ~alive & ~done
        survived[newly] = t + 1
        done |= ~alive
    return survived


@jax.jit
def _bench_step(st, key):
    n = st.pos.shape[0]
    aa, ap, asay = jax.vmap(J.adult_act, in_axes=(0, None))(st, 1)
    la = jax.random.randint(key, (n,), 0, S.N_ACTIONS)
    acts = jnp.stack([la, aa], axis=1)
    pts = jnp.stack([jnp.zeros(n, jnp.int32), ap.astype(jnp.int32)], axis=1)
    says = jnp.stack([jnp.zeros(n, jnp.int32), asay], axis=1)
    st = jax.vmap(J.step)(st, acts, pts, says)
    frames = jax.vmap(J.observe_all)(st)
    return st, frames


def bench(n_envs, steps):
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    st = J.reset_batch(keys, 1, 1)
    key = jax.random.PRNGKey(1)
    st, fr = _bench_step(st, key)
    fr.block_until_ready()
    t0 = time.time()
    for _ in range(steps):
        key, k = jax.random.split(key)
        st, fr = _bench_step(st, k)
    fr.block_until_ready()
    dt = time.time() - t0
    print(f"{n_envs} envs x {steps} steps (env + adult + frames): {n_envs*steps/dt:,.0f} agent-steps/s "
          f"on {jax.devices()[0].platform}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=S.EPISODE_STEPS)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.bench:
        for n in (64, 512, 2048):
            bench(n, 50)
        return
    a = run(GazeIgnorer, args.envs, args.steps, 0)
    b = run(GazeFollower, args.envs, args.steps, 0)
    for name, s in (("ignorer", a), ("follower", b)):
        print(f"{name:>9}: mean steps {s.mean():6.1f}  survived whole episode {np.mean(s == args.steps)*100:5.1f}%")
    diff = b - a
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"paired mean gain {diff.mean():+.1f} steps, 95% CI [{lo:+.1f}, {hi:+.1f}]  "
          f"-> M0 (JAX) {'PASS' if lo > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
