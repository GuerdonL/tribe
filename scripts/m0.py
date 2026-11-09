"""M0 go/no-go: does a learner that follows adult gaze outlive one that ignores it?

    python scripts/m0.py --episodes 200

Same seeds for both policies, so the worlds are identical and only the
exploration rule differs.
"""

from __future__ import annotations

import argparse
import sys
import os
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tribe import schema as S
from tribe.episode import run_episode
from tribe.policies import GazeIgnorer, GazeFollower


def summarize(name, results):
    steps = np.array([r["steps_survived"] for r in results])
    full = np.mean([r["steps_survived"] == S.EPISODE_STEPS for r in results])
    causes = Counter(r["cause"] for r in results)
    print(f"{name:>9}: mean steps {steps.mean():6.1f}  median {np.median(steps):5.0f}  "
          f"survived whole episode {full*100:5.1f}%  causes {dict(causes)}")
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=S.EPISODE_STEPS)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    seeds = range(args.seed0, args.seed0 + args.episodes)
    ign = [run_episode(GazeIgnorer, s, steps=args.steps) for s in seeds]
    fol = [run_episode(GazeFollower, s, steps=args.steps) for s in seeds]

    print(f"M0  episodes={args.episodes} steps={args.steps} grid={S.GRID} "
          f"view={S.VIEW_DEPTH} day={S.DAY_LENGTH}/{S.NIGHT_START}")
    a = summarize("ignorer", ign)
    b = summarize("follower", fol)

    diff = b - a
    wins = np.mean(diff > 0); losses = np.mean(diff < 0)
    # paired bootstrap on mean difference
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\npaired: follower longer in {wins*100:.0f}% of worlds, shorter in {losses*100:.0f}%; "
          f"mean gain {diff.mean():+.1f} steps, 95% CI [{lo:+.1f}, {hi:+.1f}]")
    verdict = "PASS" if lo > 0 else "FAIL"
    print(f"M0 {verdict}: {'gaze-following helps' if verdict == 'PASS' else 'no reliable advantage'}")


if __name__ == "__main__":
    main()
