"""Frame builder: State -> fixed-size integer observation for one agent.

`observe(state, i)` returns a numpy int64 vector of length FRAME_LEN laid out
exactly as schema.py describes. `decode(frame)` turns it back into a dict for
tests and the replay viewer; it is the round-trip check that the layout is
what the schema says it is.
"""

from __future__ import annotations

import numpy as np

from . import schema as S
from .world import State, view_cells, heard


def _slot_entities(state: State, i: int) -> list[tuple]:
    """Visible entities for agent i as (rel_x, rel_y, type, rel_facing, pointing, saying),
    nearest first, excluding the agent itself and empty cells."""
    my_f = int(state.facing[i])
    out = []
    for rx, ry, x, y in view_cells(state, i):
        if x < 0:                                   # map edge
            out.append((rx, ry, S.ROCK, 0, 0, 0))
            continue
        # another agent on the cell?
        j = state.agent_at(x, y)
        if j != -1 and j != i:
            kind = S.ADULT if state.is_adult[j] else S.LEARNER
            rel_f = (int(state.facing[j]) - my_f) % 4
            out.append((rx, ry, kind, rel_f, int(state.pointing[j]), int(state.saying[j])))
            continue
        if state.predator_active and (x, y) == (int(state.predator_pos[0]), int(state.predator_pos[1])):
            out.append((rx, ry, S.PREDATOR, 0, 0, 0))
            continue
        terr = int(state.terrain[y, x])
        if terr != S.EMPTY and (rx, ry) != (0, 0):
            out.append((rx, ry, terr, 0, 0, 0))
        elif terr != S.EMPTY and (rx, ry) == (0, 0):
            # the tile we stand on: worth knowing (food/water/shelter underfoot)
            out.append((rx, ry, terr, 0, 0, 0))
    return out


def observe(state: State, i: int) -> np.ndarray:
    f = np.zeros(S.FRAME_LEN, dtype=np.int64)
    word, wdir = heard(state, i)
    f[S.SELF_OFF:S.SELF_OFF + S.SELF_LEN] = [
        S.quantize(float(state.hunger[i])),
        S.quantize(float(state.thirst[i])),
        S.quantize(float(state.warmth[i])),
        int(state.facing[i]),
        state.phase(),
        word,
        wdir,
    ]
    ents = _slot_entities(state, i)[: S.N_SLOTS]
    for k, (rx, ry, kind, rel_f, pt, say) in enumerate(ents):
        base = S.SLOTS_OFF + k * S.SLOT_LEN
        f[base:base + S.SLOT_LEN] = [1, kind, rx + S.REL_X_OFFSET, ry, rel_f, pt, say]
    return f


def observe_all(state: State) -> np.ndarray:
    """(A, FRAME_LEN) frames for every agent."""
    return np.stack([observe(state, i) for i in range(state.n_agents)])


def decode(frame: np.ndarray) -> dict:
    """Inverse view of a frame for humans and tests."""
    f = frame
    me = {name: int(f[S.SELF_OFF + k]) for k, name in enumerate(S.SELF_FIELDS)}
    me["heard_word_str"] = S.WORDS[me["heard_word"]]
    slots = []
    for k in range(S.N_SLOTS):
        base = S.SLOTS_OFF + k * S.SLOT_LEN
        if f[base] == 0:
            continue
        s = {name: int(f[base + j]) for j, name in enumerate(S.SLOT_FIELDS)}
        s["rel_x"] -= S.REL_X_OFFSET
        s["type_str"] = S.TYPE_NAMES[s["type"]]
        s["saying_str"] = S.WORDS[s["saying"]]
        slots.append(s)
    return {"self": me, "slots": slots}


def validate(frame: np.ndarray) -> None:
    """Assert every field is within its schema cardinality."""
    assert frame.shape == (S.FRAME_LEN,), frame.shape
    for k, card in enumerate(S.SELF_CARD):
        v = int(frame[S.SELF_OFF + k])
        assert 0 <= v < card, (S.SELF_FIELDS[k], v, card)
    for s in range(S.N_SLOTS):
        base = S.SLOTS_OFF + s * S.SLOT_LEN
        for k, card in enumerate(S.SLOT_CARD):
            v = int(frame[base + k])
            assert 0 <= v < card, (s, S.SLOT_FIELDS[k], v, card)


def find(frame: np.ndarray, kind: int) -> dict | None:
    """Nearest visible slot of a given entity type, decoded, or None."""
    for s in decode(frame)["slots"]:
        if s["type"] == kind:
            return s
    return None
