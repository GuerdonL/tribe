"""Run one episode with a learner policy class and scripted adults."""

from __future__ import annotations

import numpy as np

from . import schema as S
from . import world as W
from .adult import adult_act
from .frame import observe


def run_episode(policy_cls, seed: int, n_learners: int = 1, n_adults: int = 1,
                steps: int = S.EPISODE_STEPS, record: bool = False) -> dict:
    st = W.reset(seed=seed, n_learners=n_learners, n_adults=n_adults)
    pols = [policy_cls(seed=seed * 101 + i) for i in range(n_learners)]
    frames_log, states_log = [], []
    survived = steps
    cause = "survived"
    for t in range(steps):
        A = st.n_agents
        acts = np.zeros(A, dtype=np.int64)
        pts = np.zeros(A, dtype=np.int64)
        says = np.zeros(A, dtype=np.int64)
        for i in range(A):
            if not st.alive[i]:
                continue
            if st.is_adult[i]:
                acts[i], pts[i], says[i] = adult_act(st, i)
            else:
                acts[i], pts[i], says[i] = pols[i].act(observe(st, i))
        if record:
            states_log.append(W.render(st, viewer=0))
            frames_log.append(observe(st, 0).copy())
        W.step(st, acts, pts, says)
        for e in st.events:
            if "agent 0" in e and "incapacitated" in e or "agent 0" in e and ("taken" in e or "hazard" in e):
                cause = e.split(": ")[-1] if "incapacitated" in e else ("predator" if "taken" in e else "hazard")
        if not st.alive[st.learners()].any():
            survived = t + 1
            break
    return {
        "seed": seed,
        "steps_survived": survived,
        "alive_at_end": int(st.alive[st.learners()].sum()),
        "cause": cause,
        "final_hunger": float(st.hunger[0]),
        "final_thirst": float(st.thirst[0]),
        "final_warmth": float(st.warmth[0]),
        "renders": states_log,
        "frames": frames_log,
    }
