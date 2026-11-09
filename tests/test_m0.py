"""M0 smoke test: the harness runs and the follower is not worse on a small sample.
The real go/no-go is scripts/m0.py with 300 episodes."""
import numpy as np

from tribe.episode import run_episode
from tribe.policies import GazeIgnorer, GazeFollower


def test_episodes_run_and_follower_not_worse():
    seeds = range(12)
    a = np.array([run_episode(GazeIgnorer, s)["steps_survived"] for s in seeds])
    b = np.array([run_episode(GazeFollower, s)["steps_survived"] for s in seeds])
    assert a.min() > 0 and b.min() > 0
    assert b.mean() >= a.mean() - 10
