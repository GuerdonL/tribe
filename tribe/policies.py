"""Hand-written learner policies for the M0 go/no-go.

Both policies share every local rule (eat when standing on food, flee the
predator, enter shelter at night, walk toward a visible resource you need).
They differ in exactly one thing: what to do when nothing useful is in view.

  GazeIgnorer : random walk.
  GazeFollower: if an adult is visible and pointing, turn to face the way it
                faces and walk that way for a while; if a word is heard from
                an unseen adult, turn toward the voice.

Both read only the frame (the learner's observation), never the State.
That is the point: the follower has no extra information, only a social cue.
"""

from __future__ import annotations

import numpy as np

from . import schema as S
from .frame import decode

LOW = 0.55          # go for a visible resource below this (as a meter fraction)
FULL = 0.9
COMMIT_STEPS = 16   # how long to keep walking along a pointed direction
SEEK_STEPS = 4      # how long to keep walking toward a voice
ATTEND_STEPS = 3    # how long to keep watching a silent adult after it spoke
TARGET_TTL = 12     # steps a seen-but-now-hidden target is remembered


def _meter(v: int) -> float:
    return (v + 0.5) / S.N_METER_BINS


def _nearest(slots, kind):
    for s in slots:
        if s["type"] == kind:
            return s
    return None


def _ahead(slots):
    """Entity type directly ahead, or EMPTY."""
    for s in slots:
        if (s["rel_x"], s["rel_y"]) == (0, 1):
            return s["type"]
    return S.EMPTY


def _ahead_blocked(slots) -> bool:
    return _ahead(slots) in S.BLOCKS_MOVE or _ahead(slots) in (S.LEARNER, S.ADULT)


def _ahead_is_agent(slots) -> bool:
    return _ahead(slots) in (S.LEARNER, S.ADULT)


def _underfoot(slots):
    for s in slots:
        if (s["rel_x"], s["rel_y"]) == (0, 0):
            return s["type"]
    return S.EMPTY


def _go_to(s, slots, rng) -> int:
    """One step toward an egocentric slot."""
    rx, ry = s["rel_x"], s["rel_y"]
    if rx > 0:
        return S.TURN_RIGHT
    if rx < 0:
        return S.TURN_LEFT
    if ry > 0:
        if _ahead_blocked(slots) and s["rel_y"] != 1:
            return S.TURN_RIGHT if rng.random() < 0.5 else S.TURN_LEFT
        return S.FORWARD
    return S.NOOP


def _shift(rel, action):
    """Dead-reckon a remembered egocentric offset through our own action."""
    rx, ry = rel
    if action == S.FORWARD:
        return (rx, ry - 1)
    if action == S.TURN_RIGHT:
        return (-ry, rx)
    if action == S.TURN_LEFT:
        return (ry, -rx)
    return (rx, ry)


def _go_to_rel(rx: int, ry: int, slots, rng) -> int:
    """One step toward an egocentric offset, visible or remembered.
    Close the forward gap first (keeps the target in the cone), turn only when abreast."""
    if ry > 0 and ry >= abs(rx):
        if _ahead_is_agent(slots):
            return S.NOOP                      # agents move; wait a tick
        if _ahead_blocked(slots) or _hazard_ahead(slots):
            return S.TURN_RIGHT if rng.random() < 0.5 else S.TURN_LEFT
        return S.FORWARD
    if rx > 0:
        return S.TURN_RIGHT
    if rx < 0:
        return S.TURN_LEFT
    if ry < 0:
        return S.TURN_RIGHT
    return S.NOOP


def _hazard_ahead(slots) -> bool:
    return any(s["type"] == S.HAZARD and (s["rel_x"], s["rel_y"]) == (0, 1) for s in slots)


class Learner:
    """Base: shared local rules. Subclasses implement `explore`."""

    name = "base"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.mem: dict = {"commit": 0, "sheltering": False, "plan": [], "target": None}

    def reset(self):
        self.mem = {"commit": 0, "sheltering": False, "plan": [], "target": None}

    def act(self, frame: np.ndarray) -> tuple[int, int, int]:
        a = self._decide(frame)
        # dead-reckon the remembered target through the action we are about to take
        tgt = self.mem.get("target")
        if tgt is not None:
            kind, rel, ttl = tgt
            rel = _shift(rel, a)
            ttl -= 1
            self.mem["target"] = (kind, rel, ttl) if ttl > 0 and rel != (0, 0) else None
        return a, 0, S.NO_WORD

    def _seek(self, slots, kind: int):
        """Step toward a resource of this kind: visible if we can see it, remembered if not."""
        s = _nearest(slots, kind)
        if s is not None:
            self.mem["target"] = (kind, (s["rel_x"], s["rel_y"]), TARGET_TTL)
            return _go_to_rel(s["rel_x"], s["rel_y"], slots, self.rng)
        tgt = self.mem.get("target")
        if tgt is not None and tgt[0] == kind:
            return _go_to_rel(tgt[1][0], tgt[1][1], slots, self.rng)
        return None

    def _decide(self, frame: np.ndarray) -> int:
        d = decode(frame)
        me, slots = d["self"], d["slots"]
        hunger, thirst = _meter(me["hunger"]), _meter(me["thirst"])
        night = me["phase"] == S.NIGHT
        dusk_or_night = me["phase"] != S.DAY
        under = _underfoot(slots)

        # --- shelter: once inside, stay put until day
        if self.mem["sheltering"]:
            if dusk_or_night:
                return S.NOOP
            self.mem["sheltering"] = False

        # --- predator in view: shelter if we know one, else turn tail
        pred = _nearest(slots, S.PREDATOR)
        if pred is not None:
            if under == S.SHELTER:
                self.mem["sheltering"] = True
                return S.ENTER
            a = self._seek(slots, S.SHELTER)
            if a is not None:
                return a
            if pred["rel_y"] >= 0 and abs(pred["rel_x"]) <= pred["rel_y"]:
                return S.TURN_RIGHT
            return S.TURN_RIGHT if _ahead_blocked(slots) else S.FORWARD

        # --- consume what is underfoot
        if under == S.FOOD and hunger < FULL:
            self.mem["target"] = None
            return S.EAT
        if under == S.WATER and thirst < FULL:
            self.mem["target"] = None
            return S.DRINK
        if under == S.SHELTER and dusk_or_night:
            self.mem["sheltering"] = True
            self.mem["target"] = None
            return S.ENTER

        # --- go to a resource you need, seen or remembered
        if dusk_or_night:
            a = self._seek(slots, S.SHELTER)
            if a is not None:
                return a
        if hunger < LOW:
            a = self._seek(slots, S.FOOD)
            if a is not None:
                return a
        if thirst < LOW:
            a = self._seek(slots, S.WATER)
            if a is not None:
                return a

        # --- nothing useful in view or memory
        a = self.explore(me, slots)
        if a == S.FORWARD and _ahead_is_agent(slots):
            return S.NOOP
        if a == S.FORWARD and (_ahead_blocked(slots) or _hazard_ahead(slots)):
            turn = S.TURN_RIGHT if self.rng.random() < 0.5 else S.TURN_LEFT
            if self.mem["commit"] > 0:
                # sidestep: turn, one step, turn back, keep going the pointed way
                back = S.TURN_LEFT if turn == S.TURN_RIGHT else S.TURN_RIGHT
                self.mem["plan"] = [S.FORWARD, back]
            a = turn
        return a

    def explore(self, me, slots) -> int:
        raise NotImplementedError


class GazeIgnorer(Learner):
    name = "ignorer"

    def explore(self, me, slots) -> int:
        r = self.rng.random()
        if r < 0.7:
            return S.FORWARD
        return S.TURN_RIGHT if r < 0.85 else S.TURN_LEFT


class GazeFollower(Learner):
    name = "follower"

    def explore(self, me, slots) -> int:
        adult = _nearest(slots, S.ADULT)
        word = me["heard_word"]
        # an adult is pointing: read its facing relative to ours, plan the turns, then walk
        if adult is not None and adult["pointing"]:
            self.mem["seek"] = 0
            rf = adult["rel_facing"]
            turns = {0: [], 1: [S.TURN_RIGHT], 2: [S.TURN_RIGHT, S.TURN_RIGHT], 3: [S.TURN_LEFT]}[rf]
            self.mem["plan"] = turns
            # "danger" means look, not go: turn to see it and let the flee rule take over
            warned = word == S.WORD_ID["danger"] or adult["saying"] == S.WORD_ID["danger"]
            self.mem["commit"] = 0 if warned else COMMIT_STEPS
            self.mem["commit_word"] = word
            if self.mem["plan"]:
                return self.mem["plan"].pop(0)
            return S.NOOP if warned else S.FORWARD
        # a plan already in motion (turns read off the adult's gaze, then a walk)
        if self.mem.get("plan"):
            return self.mem["plan"].pop(0)
        # a new word interrupts a committed walk
        if word != S.NO_WORD and word != self.mem.get("commit_word", S.NO_WORD):
            self.mem["commit"] = 0
        if self.mem["commit"] > 0:
            self.mem["commit"] -= 1
            return S.FORWARD
        # adult in view and talking: hold still and watch where it looks
        if adult is not None and me["heard_word"] != S.NO_WORD:
            self.mem["attend"] = ATTEND_STEPS
            return S.NOOP
        if adult is not None and self.mem.get("attend", 0) > 0:
            self.mem["attend"] -= 1
            return S.NOOP
        # a voice from somewhere: turn toward it and look for the speaker
        if me["heard_word"] != S.NO_WORD and me["heard_dir"] != S.NO_DIR and adult is None:
            self.mem["seek"] = SEEK_STEPS
            hd = me["heard_dir"]
            if hd == 1:
                return S.TURN_RIGHT
            if hd == 3:
                return S.TURN_LEFT
            if hd == 2:
                return S.TURN_RIGHT
            return S.FORWARD     # facing the voice: walk toward it (blocked -> base turns us)
        if self.mem.get("seek", 0) > 0:
            self.mem["seek"] -= 1
            return S.FORWARD
        # otherwise the same random walk as the ignorer
        r = self.rng.random()
        if r < 0.7:
            return S.FORWARD
        return S.TURN_RIGHT if r < 0.85 else S.TURN_LEFT
