"""Watch one episode in the terminal.

    python scripts/replay.py --seed 7 --policy follower --every 2
    python scripts/replay.py --seed 7 --policy follower --frames   # print decoded learner frames too

Uppercase / comma cells are what learner 0 can see. A = adult, ^>v< = adult pointing
in that direction, P = predator (night only), % = incapacitated agent.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tribe.episode import run_episode
from tribe.frame import decode
from tribe.policies import GazeIgnorer, GazeFollower

POLICIES = {"ignorer": GazeIgnorer, "follower": GazeFollower}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=POLICIES, default="follower")
    ap.add_argument("--every", type=int, default=1, help="show every Nth step")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between frames")
    ap.add_argument("--frames", action="store_true", help="also print the decoded learner frame")
    args = ap.parse_args()

    r = run_episode(POLICIES[args.policy], args.seed, record=True)
    for t, txt in enumerate(r["renders"]):
        if t % args.every:
            continue
        print(txt)
        if args.frames:
            d = decode(r["frames"][t])
            print("  self:", d["self"])
            for s in d["slots"]:
                print("  slot:", {k: v for k, v in s.items() if k not in ("valid",)})
        print()
        if args.delay:
            time.sleep(args.delay)
    print(f"seed {r['seed']} {args.policy}: survived {r['steps_survived']} steps, cause: {r['cause']}")


if __name__ == "__main__":
    main()
