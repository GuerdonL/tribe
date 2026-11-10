"""Tribe gridworld in JAX — batched, jit-able, same semantics as world.py.

One `EnvState` is a pytree of fixed-shape arrays for ONE environment; use
`jax.vmap` over `reset`/`step`/`observe` for a batch. The scripted adult is
`adult_act` below and runs inside the same jit.

Differences from the NumPy world, all deliberate:
  * agents act simultaneously, not in index order (collisions resolve to "stay");
  * the adult navigates by a bounded distance-field relaxation instead of BFS;
  * resource placement is by masked random choice rather than rejection loops.
The frame layout is identical (schema.py) and the NumPy M0 policies run on
its frames unchanged, which is how it is tested.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import schema as S

G = S.GRID
INF = 10_000

# ------------------------------------------------------------ constants ---
HUNGER_DRAIN = 1.0 / 240
THIRST_DRAIN = 1.0 / 200
WARMTH_DRAIN_NIGHT = 1.0 / 35
WARMTH_RECOVER = 1.0 / 30
EAT_GAIN = 0.5
DRINK_GAIN = 0.6
FOOD_REGEN = 1.0 / 120
FOOD_BITE = 0.34
PREDATOR_MOVE_EVERY = 2
N_FOOD_TILES = 6
N_WATER_TILES = 5
N_HAZARDS = 4
TREE_DENSITY = 0.08
ROCK_DENSITY = 0.03
RELAX_ITERS = 28           # distance-field relaxation steps for adult pathing

# adult
# NAME_IDLE: when the adult has nothing the learner needs, it points at the nearest landmark
# within NAME_RADIUS and names it (tree, rock, food, water, shelter, hazard -> danger). This
# decouples word identity from the learner's meters: in M1 the adult only ever said "water"
# when the learner was thirsty, so a word was predictable from self-state alone. Off by
# default so M1 stays reproducible; set TRIBE_NAME_IDLE=1 to turn it on.
import os as _os
NAME_IDLE = _os.environ.get("TRIBE_NAME_IDLE", "0") == "1"
NAME_RADIUS = 6
FOLLOW_DIST = 3
PREDATOR_WARN_DIST = 10
HUNGRY = 0.5
THIRSTY = 0.5
CRITICAL = 0.4

FACING_DXY = jnp.array(S.FACING_DXY, dtype=jnp.int32)          # (4, 2)


# ------------------------------------------------- static geometry tables ---
def _bresenham(x1: int, y1: int):
    """Intermediate cells strictly between (0,0) and (x1,y1)."""
    x0 = y0 = 0
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    out = []
    while True:
        if (x, y) != (x0, y0) and (x, y) != (x1, y1):
            out.append((x, y))
        if (x, y) == (x1, y1):
            return out
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _cone_cells():
    cells = []
    for ry in range(0, S.VIEW_DEPTH + 1):
        half = min(ry, S.VIEW_HALF)
        for rx in range(-half, half + 1):
            cells.append((rx, ry))
    cells.sort(key=lambda c: (abs(c[0]) + c[1], c[1], c[0]))
    return cells


CONE = _cone_cells()                                   # list of (rx, ry), nearest first
N_CONE = len(CONE)
CONE_RX = jnp.array([c[0] for c in CONE], dtype=jnp.int32)
CONE_RY = jnp.array([c[1] for c in CONE], dtype=jnp.int32)

# line-of-sight intermediates in EGOCENTRIC coords (facing north frame: dx=rx, dy=-ry)
_MAX_MID = max(len(_bresenham(rx, -ry)) for rx, ry in CONE) if CONE else 0
_mid = np.zeros((N_CONE, max(_MAX_MID, 1), 2), dtype=np.int32)
_mid_mask = np.zeros((N_CONE, max(_MAX_MID, 1)), dtype=bool)
for k, (rx, ry) in enumerate(CONE):
    for m, (mx, my) in enumerate(_bresenham(rx, -ry)):
        _mid[k, m] = (mx, -my)        # back to (rel_x, rel_y)
        _mid_mask[k, m] = True
MID_REL = jnp.array(_mid)             # (N_CONE, M, 2) egocentric offsets of intermediates
MID_MASK = jnp.array(_mid_mask)       # (N_CONE, M)

# adult viewpoint candidates: (rx, ry) with ry in 2..4, |rx| <= ry, not directly ahead unless ry == 4
VIEWPOINTS = [(rx, ry) for ry in range(2, 5) for rx in range(-ry, ry + 1) if not (rx == 0 and ry < 4)]
VP_RX = jnp.array([c[0] for c in VIEWPOINTS], dtype=jnp.int32)
VP_RY = jnp.array([c[1] for c in VIEWPOINTS], dtype=jnp.int32)
VP_CONE_IDX = jnp.array([CONE.index(c) for c in VIEWPOINTS], dtype=jnp.int32)

YY, XX = jnp.meshgrid(jnp.arange(G), jnp.arange(G), indexing="ij")     # (G, G) each


# --------------------------------------------------------------- state ---
class EnvState(NamedTuple):
    t: jnp.ndarray                 # ()
    terrain: jnp.ndarray           # (G, G) int32
    food_stock: jnp.ndarray        # (G, G) f32
    is_adult: jnp.ndarray          # (A,) bool
    pos: jnp.ndarray               # (A, 2) int32 [x, y]
    facing: jnp.ndarray            # (A,) int32
    hunger: jnp.ndarray            # (A,) f32
    thirst: jnp.ndarray            # (A,) f32
    warmth: jnp.ndarray            # (A,) f32
    alive: jnp.ndarray             # (A,) bool
    sheltered: jnp.ndarray         # (A,) bool
    pointing: jnp.ndarray          # (A,) bool
    saying: jnp.ndarray            # (A,) int32
    predator_pos: jnp.ndarray      # (2,) int32
    predator_lair: jnp.ndarray     # (2,) int32
    predator_active: jnp.ndarray   # () bool
    died_this_step: jnp.ndarray    # (A,) bool   for reward shaping


def phase(t):
    tod = t % S.DAY_LENGTH
    return jnp.where(tod >= S.NIGHT_START, S.NIGHT, jnp.where(tod >= S.DUSK_START, S.DUSK, S.DAY))


# ---------------------------------------------------------- geometry ---
def to_ego(facing, dx, dy):
    """World offset -> (rel_x, rel_y). facing scalar or broadcastable."""
    rx = jnp.select([facing == S.NORTH, facing == S.EAST, facing == S.SOUTH], [dx, dy, -dx], -dy)
    ry = jnp.select([facing == S.NORTH, facing == S.EAST, facing == S.SOUTH], [-dy, dx, dy], -dx)
    return rx, ry


def from_ego(facing, rx, ry):
    dx = jnp.select([facing == S.NORTH, facing == S.EAST, facing == S.SOUTH], [rx, ry, -rx], -ry)
    dy = jnp.select([facing == S.NORTH, facing == S.EAST, facing == S.SOUTH], [-ry, rx, ry], -rx)
    return dx, dy


def in_bounds(x, y):
    return (x >= 0) & (x < G) & (y >= 0) & (y < G)


def terrain_at(terrain, x, y):
    """Terrain with out-of-bounds reading as ROCK."""
    xc = jnp.clip(x, 0, G - 1)
    yc = jnp.clip(y, 0, G - 1)
    return jnp.where(in_bounds(x, y), terrain[yc, xc], S.ROCK)


def cone_world_cells(terrain, x0, y0, facing):
    """For the viewer at (x0, y0, facing): world coords of every cone cell, plus
    visible mask (in bounds and line of sight) and edge mask (out of bounds, ry <= 1)."""
    dx, dy = from_ego(facing, CONE_RX, CONE_RY)                 # (N_CONE,)
    xs, ys = x0 + dx, y0 + dy
    inb = in_bounds(xs, ys)
    # intermediates
    mdx, mdy = from_ego(facing, MID_REL[..., 0], MID_REL[..., 1])  # (N_CONE, M)
    mt = terrain_at(terrain, x0 + mdx, y0 + mdy)
    blocked = ((mt == S.TREE) | (mt == S.ROCK)) & MID_MASK
    los = ~blocked.any(axis=1)
    visible = inb & los
    edge = (~inb) & (CONE_RY <= 1)
    return xs, ys, visible, edge


def direction_to(facing, dx, dy):
    rx, ry = to_ego(facing, dx, dy)
    ahead_or_behind = jnp.abs(ry) >= jnp.abs(rx)
    return jnp.where(ahead_or_behind, jnp.where(ry >= 0, 0, 2), jnp.where(rx > 0, 1, 3))


# ------------------------------------------------------------- reset ---
def _masked_choice(key, mask):
    """Index (flat) of a uniformly random True cell; falls back to argmax if none."""
    logits = jnp.where(mask.reshape(-1), 0.0, -1e9)
    return jax.random.categorical(key, logits)


def reset(key, n_learners: int = 1, n_adults: int = 1) -> EnvState:
    A = n_learners + n_adults
    k = jax.random.split(key, 12)
    noise = jax.random.uniform(k[0], (G, G))
    terrain = jnp.where(noise < TREE_DENSITY, S.TREE,
                        jnp.where(noise < TREE_DENSITY + ROCK_DENSITY, S.ROCK, S.EMPTY)).astype(jnp.int32)
    cx = cy = G // 2
    center_clear = (jnp.abs(XX - cx) <= 2) & (jnp.abs(YY - cy) <= 2)
    terrain = jnp.where(center_clear, S.EMPTY, terrain)
    cdist = jnp.abs(XX - cx) + jnp.abs(YY - cy)

    def place_cluster(terrain, key, kind, n, dmin, dmax):
        ks = jax.random.split(key, n + 1)
        free = (terrain == S.EMPTY) & (cdist >= dmin) & (cdist <= dmax)
        a = _masked_choice(ks[0], free)
        ax, ay = a % G, a // G
        terrain = terrain.at[ay, ax].set(kind)
        for j in range(1, n):
            near = (jnp.abs(XX - ax) <= 1) & (jnp.abs(YY - ay) <= 1) & (terrain == S.EMPTY)
            b = _masked_choice(ks[j], near)
            bx, by = b % G, b // G
            ok = near[by, bx]
            terrain = jnp.where(ok, terrain.at[by, bx].set(kind), terrain)
        return terrain

    def place_scatter(terrain, key, kind, n, dmin, dmax):
        ks = jax.random.split(key, n)
        for j in range(n):
            free = (terrain == S.EMPTY) & (cdist >= dmin) & (cdist <= dmax)
            a = _masked_choice(ks[j], free)
            ax, ay = a % G, a // G
            terrain = jnp.where(free[ay, ax], terrain.at[ay, ax].set(kind), terrain)
        return terrain

    terrain = place_cluster(terrain, k[1], S.FOOD, N_FOOD_TILES // 2, 8, 14)
    terrain = place_cluster(terrain, k[2], S.FOOD, N_FOOD_TILES - N_FOOD_TILES // 2, 8, 14)
    terrain = place_cluster(terrain, k[3], S.WATER, N_WATER_TILES, 8, 14)
    terrain = place_scatter(terrain, k[4], S.SHELTER, 1, 6, 12)
    terrain = place_scatter(terrain, k[5], S.HAZARD, N_HAZARDS, 4, 10 ** 6)
    food_stock = jnp.where(terrain == S.FOOD, 1.0, 0.0).astype(jnp.float32)

    # agents on distinct empty cells near the center
    spawn = center_clear & (terrain == S.EMPTY)
    ks = jax.random.split(k[6], A)
    pos = []
    taken = jnp.zeros((G, G), dtype=bool)
    for i in range(A):
        a = _masked_choice(ks[i], spawn & ~taken)
        ax, ay = a % G, a // G
        taken = taken.at[ay, ax].set(True)
        pos.append(jnp.stack([ax, ay]))
    pos = jnp.stack(pos).astype(jnp.int32)
    facing = jax.random.randint(k[7], (A,), 0, 4).astype(jnp.int32)
    corners = jnp.array([(2, 2), (G - 3, 2), (2, G - 3), (G - 3, G - 3)], dtype=jnp.int32)
    lair = corners[jax.random.randint(k[8], (), 0, 4)]
    is_adult = jnp.array([False] * n_learners + [True] * n_adults)

    return EnvState(
        t=jnp.int32(0), terrain=terrain, food_stock=food_stock, is_adult=is_adult,
        pos=pos, facing=facing,
        hunger=jnp.full((A,), 0.8, jnp.float32), thirst=jnp.full((A,), 0.8, jnp.float32),
        warmth=jnp.ones((A,), jnp.float32),
        alive=jnp.ones((A,), bool), sheltered=jnp.zeros((A,), bool),
        pointing=jnp.zeros((A,), bool), saying=jnp.zeros((A,), jnp.int32),
        predator_pos=lair, predator_lair=lair, predator_active=jnp.bool_(False),
        died_this_step=jnp.zeros((A,), bool),
    )


# -------------------------------------------------------------- step ---
def step(st: EnvState, actions, pointing, saying) -> EnvState:
    A = st.pos.shape[0]
    actions = actions.astype(jnp.int32)
    alive_before = st.alive

    # --- turns
    facing = jnp.where(actions == S.TURN_LEFT, (st.facing - 1) % 4,
                       jnp.where(actions == S.TURN_RIGHT, (st.facing + 1) % 4, st.facing))
    # --- proposed moves
    d = FACING_DXY[facing]                                     # (A, 2)
    want = st.pos + d
    tx, ty = want[:, 0], want[:, 1]
    tt = terrain_at(st.terrain, tx, ty)
    walkable = in_bounds(tx, ty) & (tt != S.TREE) & (tt != S.ROCK)
    moving = (actions == S.FORWARD) & st.alive & walkable
    # occupied by a living agent that is not itself moving away? simultaneous rule:
    # a cell is blocked if any living agent currently stands there (conservative)
    same = (want[:, None, 0] == st.pos[None, :, 0]) & (want[:, None, 1] == st.pos[None, :, 1])
    occupied = (same & st.alive[None, :] & ~jnp.eye(A, dtype=bool)).any(axis=1)
    moving = moving & ~occupied
    # two agents moving into the same cell: lower index wins
    clash = (want[:, None, 0] == want[None, :, 0]) & (want[:, None, 1] == want[None, :, 1]) \
        & moving[:, None] & moving[None, :] & (jnp.arange(A)[None, :] < jnp.arange(A)[:, None])
    moving = moving & ~clash.any(axis=1)
    pos = jnp.where(moving[:, None], want, st.pos)
    sheltered = st.sheltered & ~moving
    fell = moving & (tt == S.HAZARD) & ~st.is_adult

    # --- consumption on the (new) tile
    px, py = pos[:, 0], pos[:, 1]
    under = st.terrain[py, px]
    stock = st.food_stock[py, px]
    eats = (actions == S.EAT) & (under == S.FOOD) & (stock >= FOOD_BITE) & st.alive
    drinks = (actions == S.DRINK) & (under == S.WATER) & st.alive
    enters = (actions == S.ENTER) & (under == S.SHELTER) & st.alive
    food_stock = st.food_stock
    for i in range(A):   # A is small and static
        food_stock = food_stock.at[py[i], px[i]].add(jnp.where(eats[i], -FOOD_BITE, 0.0))
    hunger = jnp.minimum(1.0, st.hunger + jnp.where(eats, EAT_GAIN, 0.0))
    thirst = jnp.minimum(1.0, st.thirst + jnp.where(drinks, DRINK_GAIN, 0.0))
    sheltered = sheltered | enters

    # --- needs (learners only)
    night = phase(st.t) == S.NIGHT
    learner = ~st.is_adult
    hunger = jnp.where(learner, hunger - HUNGER_DRAIN, hunger)
    thirst = jnp.where(learner, thirst - THIRST_DRAIN, thirst)
    warm_ok = sheltered | ~night
    warmth = jnp.where(learner, jnp.where(warm_ok, jnp.minimum(1.0, st.warmth + WARMTH_RECOVER),
                                          st.warmth - WARMTH_DRAIN_NIGHT), st.warmth)
    starved = learner & ((hunger <= 0) | (thirst <= 0) | (warmth <= 0))
    hunger, thirst, warmth = jnp.maximum(hunger, 0.0), jnp.maximum(thirst, 0.0), jnp.maximum(warmth, 0.0)

    # --- food regrowth
    food_stock = jnp.where(st.terrain == S.FOOD, jnp.minimum(1.0, food_stock + FOOD_REGEN), food_stock)

    # --- predator
    wake = night & ~st.predator_active
    ppos = jnp.where(wake, st.predator_lair, st.predator_pos)
    active = night
    alive_now = st.alive & ~fell & ~starved
    targetable = alive_now & learner & ~sheltered
    dist = jnp.abs(pos[:, 0] - ppos[0]) + jnp.abs(pos[:, 1] - ppos[1])
    dist = jnp.where(targetable, dist, INF)
    tgt = jnp.argmin(dist)
    has_t = dist[tgt] < INF
    ddx = pos[tgt, 0] - ppos[0]
    ddy = pos[tgt, 1] - ppos[1]
    step_x = jnp.where(jnp.abs(ddx) >= jnp.abs(ddy), jnp.sign(ddx), 0)
    step_y = jnp.where(jnp.abs(ddx) >= jnp.abs(ddy), 0, jnp.sign(ddy))
    moves = night & has_t & ((st.t % PREDATOR_MOVE_EVERY) == 0)
    ppos = jnp.where(moves, ppos + jnp.stack([step_x, step_y]).astype(jnp.int32), ppos)
    ppos = jnp.where(night, ppos, st.predator_lair)
    pd = jnp.abs(pos[:, 0] - ppos[0]) + jnp.abs(pos[:, 1] - ppos[1])
    taken = night & targetable & (pd <= 1)

    alive = alive_now & ~taken
    died = alive_before & ~alive

    return EnvState(
        t=st.t + 1, terrain=st.terrain, food_stock=food_stock, is_adult=st.is_adult,
        pos=pos, facing=facing, hunger=hunger, thirst=thirst, warmth=warmth,
        alive=alive, sheltered=sheltered & alive,
        pointing=pointing.astype(bool), saying=saying.astype(jnp.int32),
        predator_pos=ppos, predator_lair=st.predator_lair, predator_active=active,
        died_this_step=died,
    )


# ------------------------------------------------------------ observe ---
def heard(st: EnvState, i):
    A = st.pos.shape[0]
    dx = st.pos[:, 0] - st.pos[i, 0]
    dy = st.pos[:, 1] - st.pos[i, 1]
    d = jnp.abs(dx) + jnp.abs(dy)
    speaking = st.alive & (st.saying != S.NO_WORD) & (jnp.arange(A) != i) & (d <= S.HEAR_RADIUS)
    d = jnp.where(speaking, d, INF)
    j = jnp.argmin(d)
    ok = d[j] < INF
    word = jnp.where(ok, st.saying[j], S.NO_WORD)
    wdir = jnp.where(ok, direction_to(st.facing[i], dx[j], dy[j]), S.NO_DIR)
    return word, wdir


def quantize(x):
    return jnp.clip(jnp.floor(x * S.N_METER_BINS), 0, S.N_METER_BINS - 1).astype(jnp.int32)


def observe(st: EnvState, i) -> jnp.ndarray:
    """Frame (FRAME_LEN,) int32 for agent i. Same layout as frame.observe."""
    A = st.pos.shape[0]
    x0, y0, f = st.pos[i, 0], st.pos[i, 1], st.facing[i]
    xs, ys, visible, edge = cone_world_cells(st.terrain, x0, y0, f)        # (N_CONE,)
    xc, yc = jnp.clip(xs, 0, G - 1), jnp.clip(ys, 0, G - 1)
    terr = st.terrain[yc, xc]
    # agents on cells
    same = (xs[:, None] == st.pos[None, :, 0]) & (ys[:, None] == st.pos[None, :, 1]) \
        & st.alive[None, :] & (jnp.arange(A)[None, :] != i)                 # (N_CONE, A)
    has_agent = same.any(axis=1)
    j = jnp.argmax(same, axis=1)
    agent_kind = jnp.where(st.is_adult[j], S.ADULT, S.LEARNER)
    rel_f = (st.facing[j] - f) % 4
    a_point = st.pointing[j].astype(jnp.int32)
    a_say = st.saying[j]
    is_pred = st.predator_active & (xs == st.predator_pos[0]) & (ys == st.predator_pos[1])
    own = (CONE_RX == 0) & (CONE_RY == 0)
    kind = jnp.where(has_agent, agent_kind,
                     jnp.where(is_pred, S.PREDATOR, terr))
    kind = jnp.where(edge, S.ROCK, kind)
    present = (visible & (kind != S.EMPTY)) | edge
    # own cell: only if terrain is non-empty (never an agent: that's us)
    present = jnp.where(own, visible & (terr != S.EMPTY), present)
    kind = jnp.where(own, terr, kind)
    rel_f = jnp.where(has_agent & ~own, rel_f, 0)
    a_point = jnp.where(has_agent & ~own, a_point, 0)
    a_say = jnp.where(has_agent & ~own, a_say, 0)

    # compact present entries into the first N_SLOTS slots, in cone order (nearest first)
    order = jnp.cumsum(present.astype(jnp.int32)) - 1                   # slot index per cone cell
    take = present & (order < S.N_SLOTS)
    slots = jnp.zeros((S.N_SLOTS, S.SLOT_LEN), dtype=jnp.int32)
    vals = jnp.stack([jnp.ones_like(kind), kind, CONE_RX + S.REL_X_OFFSET, CONE_RY,
                      rel_f, a_point, a_say], axis=1)                     # (N_CONE, 7)
    idx = jnp.where(take, order, S.N_SLOTS)                               # overflow row is dropped
    slots = jnp.zeros((S.N_SLOTS + 1, S.SLOT_LEN), dtype=jnp.int32).at[idx].set(vals)[: S.N_SLOTS]
    # .at[].set with duplicates at the overflow row is fine: that row is discarded

    word, wdir = heard(st, i)
    me = jnp.stack([quantize(st.hunger[i]), quantize(st.thirst[i]), quantize(st.warmth[i]),
                    f, phase(st.t), word, wdir]).astype(jnp.int32)
    return jnp.concatenate([me, slots.reshape(-1)])


def observe_all(st: EnvState) -> jnp.ndarray:
    A = st.pos.shape[0]
    return jnp.stack([observe(st, i) for i in range(A)])


# ------------------------------------------------------- scripted adult ---
def _nearest_cell(mask, lx, ly):
    """(x, y, found) of the nearest True cell to (lx, ly) by Manhattan distance."""
    d = jnp.abs(XX - lx) + jnp.abs(YY - ly)
    d = jnp.where(mask, d, INF)
    a = jnp.argmin(d)
    return a % G, a // G, d.reshape(-1)[a] < INF


def _facing_toward(dx, dy):
    return jnp.where(jnp.abs(dx) >= jnp.abs(dy),
                     jnp.where(dx > 0, S.EAST, S.WEST),
                     jnp.where(dy > 0, S.SOUTH, S.NORTH))


def _turn_toward(cur, want):
    return jnp.where(cur == want, S.NOOP,
                     jnp.where((want - cur) % 4 == 1, S.TURN_RIGHT, S.TURN_LEFT))


def _distance_field(passable, gx, gy):
    """Bounded min-plus relaxation from goal cell. Unreached cells stay INF."""
    d0 = jnp.where((XX == gx) & (YY == gy), 0, INF)

    def body(_, d):
        up = jnp.concatenate([jnp.full((1, G), INF), d[:-1]], axis=0)
        dn = jnp.concatenate([d[1:], jnp.full((1, G), INF)], axis=0)
        lf = jnp.concatenate([jnp.full((G, 1), INF), d[:, :-1]], axis=1)
        rt = jnp.concatenate([d[:, 1:], jnp.full((G, 1), INF)], axis=1)
        nb = jnp.minimum(jnp.minimum(up, dn), jnp.minimum(lf, rt)) + 1
        return jnp.where(passable, jnp.minimum(d, nb), jnp.where(d == 0, 0, INF))

    return jax.lax.fori_loop(0, RELAX_ITERS, body, d0)


def _step_toward(st: EnvState, i, gx, gy):
    """Action that makes progress toward (gx, gy): turn or forward, by distance descent."""
    A = st.pos.shape[0]
    ax, ay = st.pos[i, 0], st.pos[i, 1]
    passable = (st.terrain != S.TREE) & (st.terrain != S.ROCK)
    occ = jnp.zeros((G, G), bool)
    for j in range(A):
        occ = occ.at[st.pos[j, 1], st.pos[j, 0]].set(occ[st.pos[j, 1], st.pos[j, 0]] | (st.alive[j] & (j != i)))
    passable = passable & ~occ
    dist = _distance_field(passable, gx, gy)
    # neighbor with smallest distance
    nx = ax + FACING_DXY[:, 0]
    ny = ay + FACING_DXY[:, 1]
    nd = jnp.where(in_bounds(nx, ny), dist[jnp.clip(ny, 0, G - 1), jnp.clip(nx, 0, G - 1)], INF)
    best = jnp.argmin(nd)
    reachable = nd[best] < INF
    # fallback when unreachable within RELAX_ITERS: greedy
    want = jnp.where(reachable, best, _facing_toward(gx - ax, gy - ay))
    a = _turn_toward(st.facing[i], want)
    return jnp.where(a == S.NOOP, S.FORWARD, a)


def adult_act(st: EnvState, i):
    """(action, pointing, saying) for adult i. Mirrors adult.py."""
    A = st.pos.shape[0]
    ax, ay = st.pos[i, 0], st.pos[i, 1]
    learner = ~st.is_adult & st.alive
    d = jnp.abs(st.pos[:, 0] - ax) + jnp.abs(st.pos[:, 1] - ay)
    d = jnp.where(learner, d, INF)
    L = jnp.argmin(d)
    any_learner = d[L] < INF
    lx, ly, lf = st.pos[L, 0], st.pos[L, 1], st.facing[L]

    # --- target and word, by priority
    tgt_x, tgt_y, has, word = jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.int32(S.NO_WORD)

    def choose(cond, x, y, w, cur):
        cx, cy, ch, cw = cur
        take = cond & ~ch
        return (jnp.where(take, x, cx), jnp.where(take, y, cy), ch | take, jnp.where(take, w, cw))

    cur = (tgt_x, tgt_y, has, word)
    px, py = st.predator_pos
    pred_close = st.predator_active & (jnp.abs(px - lx) + jnp.abs(py - ly) <= PREDATOR_WARN_DIST) & ~st.sheltered[L]
    cur = choose(pred_close, px, py, S.WORD_ID["danger"], cur)

    hungry = st.hunger[L] < HUNGRY
    thirsty = st.thirst[L] < THIRSTY
    water_first = st.thirst[L] <= st.hunger[L]
    fx, fy, ffound = _nearest_cell((st.terrain == S.FOOD) & (st.food_stock >= FOOD_BITE), lx, ly)
    wx, wy, wfound = _nearest_cell(st.terrain == S.WATER, lx, ly)
    sx, sy, sfound = _nearest_cell(st.terrain == S.SHELTER, lx, ly)
    critical = jnp.minimum(st.hunger[L], st.thirst[L]) < CRITICAL

    def need(cur, only_if):
        c1 = choose(only_if & water_first & thirsty & wfound, wx, wy, S.WORD_ID["water"], cur)
        c1 = choose(only_if & hungry & ffound, fx, fy, S.WORD_ID["food"], c1)
        c1 = choose(only_if & thirsty & wfound, wx, wy, S.WORD_ID["water"], c1)
        return c1

    cur = need(cur, critical)
    night_near = phase(st.t) != S.DAY
    cur = choose(night_near & ~st.sheltered[L] & sfound, sx, sy, S.WORD_ID["home"], cur)
    cur = need(cur, jnp.bool_(True))
    tgt_x, tgt_y, has, word = cur

    # --- am I in the learner's view?
    rx, ry = to_ego(lf, ax - lx, ay - ly)
    xs, ys, visible, _ = cone_world_cells(st.terrain, lx, ly, lf)
    in_view = ((xs == ax) & (ys == ay) & visible).any()
    dist_l = jnp.abs(lx - ax) + jnp.abs(ly - ay)
    need_move = (dist_l > FOLLOW_DIST) | (has & ~in_view)

    # --- best viewpoint: free, visible cone cell, off-axis, nearest to me
    vx, vy = xs[VP_CONE_IDX], ys[VP_CONE_IDX]
    vvis = visible[VP_CONE_IDX]
    vt = terrain_at(st.terrain, vx, vy)
    free = (vt != S.TREE) & (vt != S.ROCK)
    occ = ((vx[:, None] == st.pos[None, :, 0]) & (vy[:, None] == st.pos[None, :, 1])
           & st.alive[None, :] & (jnp.arange(A)[None, :] != i)).any(axis=1)
    ok = vvis & free & ~occ
    cost = jnp.abs(vx - ax) + jnp.abs(vy - ay) + jnp.where(VP_RX == 0, 2, 0)
    cost = jnp.where(ok, cost, INF)
    b = jnp.argmin(cost)
    vp_found = cost[b] < INF
    gx = jnp.where(vp_found, vx[b], lx)
    gy = jnp.where(vp_found, vy[b], ly)
    at_goal = (gx == ax) & (gy == ay)
    need_move = need_move & ~at_goal            # standing at the viewpoint: fall through to pointing
    move_a = _step_toward(st, i, gx, gy)

    # --- face the target and point
    on_target = has & (tgt_x == ax) & (tgt_y == ay)
    want_face = jnp.where(on_target, _facing_toward(lx - ax, ly - ay), _facing_toward(tgt_x - ax, tgt_y - ay))
    turn = _turn_toward(st.facing[i], want_face)
    pointing = has & (turn == S.NOOP)
    say_target = jnp.where(on_target, jnp.where(pointing, S.WORD_ID["here"], S.NO_WORD), word)

    # --- idle: face learner, say "look" every 10 steps
    idle_face = _facing_toward(lx - ax, ly - ay)
    idle_turn = _turn_toward(st.facing[i], idle_face)
    idle_say = jnp.where((idle_turn == S.NOOP) & (st.t % 10 == 0), S.WORD_ID["look"], S.NO_WORD)
    idle_point = jnp.bool_(False)
    if NAME_IDLE:
        # point at the nearest landmark to the adult and name it, whatever the learner needs
        land = (st.terrain != S.EMPTY) & ((jnp.abs(XX - ax) + jnp.abs(YY - ay)) <= NAME_RADIUS) \
            & ~((XX == ax) & (YY == ay))
        nx_, ny_, nfound = _nearest_cell(land, ax, ay)
        ntype = st.terrain[ny_, nx_]
        name_of = jnp.array([S.NO_WORD, S.WORD_ID["food"], S.WORD_ID["water"], S.WORD_ID["tree"],
                             S.WORD_ID["rock"], S.WORD_ID["shelter"], S.NO_WORD, S.WORD_ID["danger"],
                             S.NO_WORD, S.NO_WORD], dtype=jnp.int32)
        name_face = _facing_toward(nx_ - ax, ny_ - ay)
        name_turn = _turn_toward(st.facing[i], name_face)
        # alternate: name the landmark for 10 steps, then look at the learner for 10
        naming = nfound & ((st.t // 10) % 2 == 0)
        idle_turn = jnp.where(naming, name_turn, idle_turn)
        idle_point = naming & (name_turn == S.NOOP)
        idle_say = jnp.where(naming, jnp.where(idle_point, name_of[ntype], S.NO_WORD), idle_say)

    action = jnp.where(need_move, move_a, jnp.where(has, turn, idle_turn))
    point_out = jnp.where(need_move, False, jnp.where(has, pointing, idle_point))
    say_out = jnp.where(need_move, jnp.where(has, word, S.NO_WORD),
                        jnp.where(has, say_target, idle_say))
    action = jnp.where(any_learner, action, S.NOOP)
    point_out = point_out & any_learner
    say_out = jnp.where(any_learner, say_out, S.NO_WORD)
    return action.astype(jnp.int32), point_out, say_out.astype(jnp.int32)


# ------------------------------------------------------------ batching ---
reset_batch = jax.jit(jax.vmap(reset, in_axes=(0, None, None)), static_argnums=(1, 2))
step_batch = jax.jit(jax.vmap(step))
observe_all_batch = jax.jit(jax.vmap(observe_all))
adult_act_batch = jax.jit(jax.vmap(adult_act, in_axes=(0, None)))
