"""Scripted adult — stage one caretaker.

Omniscient, needless, attentive. Each tick, for the nearest living learner:
  1. predator awake and closing        -> face it, point, say "danger"
  2. night near and learner unsheltered -> face shelter, point, say "home"
  3. learner hungry                     -> face nearest stocked food, point, say "food"
  4. learner thirsty                    -> face nearest water, point, say "water"
  5. otherwise                          -> face the learner, say "look" now and then
Stays within a few cells of the learner so it is usually in view and earshot.
The adult never judges; it only looks, points and names.
"""

from __future__ import annotations

import numpy as np

from . import schema as S
from .world import State, to_egocentric, from_egocentric, line_of_sight

FOLLOW_DIST = 3
PREDATOR_WARN_DIST = 10
NIGHT_WARN_LEAD = 12
HUNGRY = 0.5
THIRSTY = 0.5
CRITICAL = 0.4


def _nearest(state: State, kind: int, x: int, y: int, need_stock: bool = False):
    ys, xs = np.where(state.terrain == kind)
    if need_stock:
        keep = state.food_stock[ys, xs] >= 0.34
        ys, xs = ys[keep], xs[keep]
    if len(xs) == 0:
        return None
    d = np.abs(xs - x) + np.abs(ys - y)
    k = int(np.argmin(d))
    return int(xs[k]), int(ys[k])


def _facing_toward(dx: int, dy: int) -> int:
    if abs(dx) >= abs(dy):
        return S.EAST if dx > 0 else S.WEST
    return S.SOUTH if dy > 0 else S.NORTH


def _turn_toward(cur: int, want: int) -> int:
    if cur == want:
        return S.NOOP
    return S.TURN_RIGHT if (want - cur) % 4 == 1 else S.TURN_LEFT


def adult_act(state: State, i: int) -> tuple[int, int, int]:
    """(action, pointing, saying) for adult i."""
    ax, ay = int(state.pos[i, 0]), int(state.pos[i, 1])
    learners = [j for j in state.learners() if state.alive[j]]
    if not learners:
        return S.NOOP, 0, S.NO_WORD
    d = [abs(state.pos[j, 0] - ax) + abs(state.pos[j, 1] - ay) for j in learners]
    L = learners[int(np.argmin(d))]
    lx, ly = int(state.pos[L, 0]), int(state.pos[L, 1])

    # --- pick a target and a word
    target, word = None, S.NO_WORD
    tod = state.t % S.DAY_LENGTH
    night_near = state.phase() != S.DAY
    if state.predator_active:
        px, py = int(state.predator_pos[0]), int(state.predator_pos[1])
        if abs(px - lx) + abs(py - ly) <= PREDATOR_WARN_DIST and not state.sheltered[L]:
            target, word = (px, py), S.WORD_ID["danger"]
    needs = []
    if state.hunger[L] < HUNGRY:
        needs.append((float(state.hunger[L]), S.FOOD, "food"))
    if state.thirst[L] < THIRSTY:
        needs.append((float(state.thirst[L]), S.WATER, "water"))
    needs.sort()

    def need_target():
        for _, kind, w in needs:            # most urgent first
            cell = _nearest(state, kind, lx, ly, need_stock=(kind == S.FOOD))
            if cell is not None:
                return cell, S.WORD_ID[w]
        return None, S.NO_WORD

    # a critical need outranks the nightly call home
    if target is None and needs and needs[0][0] < CRITICAL:
        target, word = need_target()
    if target is None and night_near and not state.sheltered[L]:
        sh = _nearest(state, S.SHELTER, lx, ly)
        if sh is not None:
            target, word = sh, S.WORD_ID["home"]
    if target is None and needs:
        target, word = need_target()

    # --- stay close and visible: stand in the learner's view, with line of sight
    dist = abs(lx - ax) + abs(ly - ay)
    lf = int(state.facing[L])
    in_learner_view = _visible_from(state, lx, ly, lf, ax, ay)
    if dist > FOLLOW_DIST or (target is not None and not in_learner_view):
        goal = _best_viewpoint(state, L, i)
        if goal is not None and (ax, ay) != goal:
            a = _step_toward(state, i, goal)
            # keep naming the thing while repositioning so the learner can turn toward the voice
            return a, 0, (word if target is not None else S.NO_WORD)

    # --- face the target and point; speak while pointing
    if target is not None:
        tx, ty = target
        if (tx, ty) == (ax, ay):
            # standing on it: face the learner instead so it comes here
            want = _facing_toward(lx - ax, ly - ay)
            a = _turn_toward(int(state.facing[i]), want)
            return a, 1, S.WORD_ID["here"] if a == S.NOOP else S.NO_WORD
        want = _facing_toward(tx - ax, ty - ay)
        a = _turn_toward(int(state.facing[i]), want)
        pointing = 1 if a == S.NOOP else 0
        return a, pointing, word          # name it while turning too, so the learner keeps watching

    # --- idle: face the learner, say "look" every few steps
    want = _facing_toward(lx - ax, ly - ay) if (lx, ly) != (ax, ay) else int(state.facing[i])
    a = _turn_toward(int(state.facing[i]), want)
    say = S.WORD_ID["look"] if (a == S.NOOP and state.t % 10 == 0) else S.NO_WORD
    return a, 0, say


def _visible_from(state: State, lx: int, ly: int, lf: int, x: int, y: int) -> bool:
    rx, ry = to_egocentric(lf, x - lx, y - ly)
    return 0 <= ry <= S.VIEW_DEPTH and abs(rx) <= min(ry, S.VIEW_HALF) \
        and line_of_sight(state, lx, ly, x, y)


def _best_viewpoint(state: State, L: int, i: int):
    """Free cell inside learner L's cone, with line of sight, nearest to adult i."""
    lx, ly, lf = int(state.pos[L, 0]), int(state.pos[L, 1]), int(state.facing[L])
    ax, ay = int(state.pos[i, 0]), int(state.pos[i, 1])
    best, bd = None, 10 ** 9
    for ry in range(2, 5):
        for rx in sorted(range(-ry, ry + 1), key=lambda v: (v == 0, abs(v))):
            if rx == 0 and ry < 4:
                continue                       # not directly in the learner's path
            dx, dy = from_egocentric(lf, rx, ry)
            x, y = lx + dx, ly + dy
            if not state.in_bounds(x, y) or state.terrain[y, x] in S.BLOCKS_MOVE:
                continue
            occ = state.agent_at(x, y)
            if occ not in (-1, i):
                continue
            if not line_of_sight(state, lx, ly, x, y):
                continue
            d = abs(x - ax) + abs(y - ay) + (2 if rx == 0 else 0)
            if d < bd:
                best, bd = (x, y), d
    return best


def _step_toward(state: State, i: int, goal) -> int:
    """BFS over walkable cells; returns the action that makes progress (turn or forward)."""
    from collections import deque
    ax, ay = int(state.pos[i, 0]), int(state.pos[i, 1])
    G = S.GRID
    prev = {(ax, ay): None}
    q = deque([(ax, ay)])
    found = False
    while q:
        x, y = q.popleft()
        if (x, y) == tuple(goal):
            found = True
            break
        for f in range(4):
            dx, dy = S.FACING_DXY[f]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < G and 0 <= ny < G) or (nx, ny) in prev:
                continue
            if state.terrain[ny, nx] in S.BLOCKS_MOVE:
                continue
            if state.agent_at(nx, ny) not in (-1, i):
                continue
            prev[(nx, ny)] = (x, y)
            q.append((nx, ny))
    if not found:
        return S.NOOP
    # walk back to the first step
    cur = tuple(goal)
    while prev[cur] != (ax, ay):
        cur = prev[cur]
    want = _facing_toward(cur[0] - ax, cur[1] - ay)
    a = _turn_toward(int(state.facing[i]), want)
    return S.FORWARD if a == S.NOOP else a
