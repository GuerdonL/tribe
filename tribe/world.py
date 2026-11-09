"""Tribe gridworld — stage one.

Pure-ish functions over fixed-shape NumPy arrays so the whole thing ports to
JAX mechanically later. No per-entity Python objects.

A `State` holds the terrain grid, the agents (learners and adults share one
table; adults have no needs), and the predator. `reset` builds one; `step`
advances it one tick given every agent's (action, pointing, saying).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import schema as S


@dataclass
class State:
    rng: np.random.Generator
    t: int
    terrain: np.ndarray          # (GRID, GRID) int: EMPTY/FOOD/WATER/TREE/ROCK/SHELTER/HAZARD
    food_stock: np.ndarray       # (GRID, GRID) float in [0,1]; only meaningful on FOOD tiles
    # agents
    n_agents: int
    is_adult: np.ndarray         # (A,) bool
    pos: np.ndarray              # (A, 2) int  [x, y]
    facing: np.ndarray           # (A,) int
    hunger: np.ndarray           # (A,) float [0,1]  1 = full
    thirst: np.ndarray           # (A,) float
    warmth: np.ndarray           # (A,) float
    alive: np.ndarray            # (A,) bool   False = incapacitated for the episode
    sheltered: np.ndarray        # (A,) bool
    pointing: np.ndarray         # (A,) bool   what each agent is doing this tick
    saying: np.ndarray           # (A,) int    word id, 0 = silent
    # predator
    predator_pos: np.ndarray     # (2,) int
    predator_lair: np.ndarray    # (2,) int
    predator_active: bool
    # bookkeeping
    events: list = field(default_factory=list)   # human-readable log of this tick

    # ------------------------------------------------------------ helpers
    def phase(self) -> int:
        tod = self.t % S.DAY_LENGTH
        if tod >= S.NIGHT_START:
            return S.NIGHT
        return S.DUSK if tod >= S.DUSK_START else S.DAY

    def learners(self) -> np.ndarray:
        return np.where(~self.is_adult)[0]

    def adults(self) -> np.ndarray:
        return np.where(self.is_adult)[0]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < S.GRID and 0 <= y < S.GRID

    def agent_at(self, x: int, y: int) -> int:
        """Index of a living agent on (x, y), or -1."""
        hits = np.where((self.pos[:, 0] == x) & (self.pos[:, 1] == y) & self.alive)[0]
        return int(hits[0]) if len(hits) else -1


# ---------------------------------------------------------------- tuning ---
HUNGER_DRAIN = 1.0 / 240
THIRST_DRAIN = 1.0 / 200
WARMTH_DRAIN_NIGHT = 1.0 / 35
WARMTH_RECOVER = 1.0 / 30
EAT_GAIN = 0.5
DRINK_GAIN = 0.6
FOOD_REGEN = 1.0 / 120
FOOD_BITE = 0.34                  # fraction of a tile's stock one meal uses
PREDATOR_MOVE_EVERY = 2           # steps per predator move at night (half learner speed)

N_FOOD_TILES = 6
N_WATER_TILES = 5
N_HAZARDS = 4
TREE_DENSITY = 0.08
ROCK_DENSITY = 0.03


# ----------------------------------------------------------------- reset ---
def reset(seed: int | None = None, n_learners: int = 1, n_adults: int = 1,
          rng: np.random.Generator | None = None) -> State:
    rng = rng if rng is not None else np.random.default_rng(seed)
    G = S.GRID
    terrain = np.full((G, G), S.EMPTY, dtype=np.int64)

    # scatter trees and rocks, keep a clear ring at the center for spawning
    noise = rng.random((G, G))
    terrain[noise < TREE_DENSITY] = S.TREE
    terrain[(noise >= TREE_DENSITY) & (noise < TREE_DENSITY + ROCK_DENSITY)] = S.ROCK
    cx = cy = G // 2
    terrain[cy - 2:cy + 3, cx - 2:cx + 3] = S.EMPTY

    def place(kind: int, n: int, min_center_dist: int = 0, cluster: bool = False,
              max_center_dist: int = 10 ** 6):
        placed = 0
        anchor = None
        tries = 0
        while placed < n and tries < 5000:
            tries += 1
            if cluster and anchor is not None:
                x = int(np.clip(anchor[0] + rng.integers(-1, 2), 0, G - 1))
                y = int(np.clip(anchor[1] + rng.integers(-1, 2), 0, G - 1))
            else:
                x, y = int(rng.integers(0, G)), int(rng.integers(0, G))
            if terrain[y, x] != S.EMPTY:
                continue
            if not (min_center_dist <= abs(x - cx) + abs(y - cy) <= max_center_dist):
                continue
            terrain[y, x] = kind
            placed += 1
            if anchor is None:
                anchor = (x, y)

    # food: two small patches, far enough that undirected search is slow
    place(S.FOOD, N_FOOD_TILES // 2, min_center_dist=8, max_center_dist=14, cluster=True)
    place(S.FOOD, N_FOOD_TILES - N_FOOD_TILES // 2, min_center_dist=8, max_center_dist=14, cluster=True)
    place(S.WATER, N_WATER_TILES, min_center_dist=8, max_center_dist=14, cluster=True)
    place(S.SHELTER, 1, min_center_dist=6, max_center_dist=12)
    place(S.HAZARD, N_HAZARDS, min_center_dist=4)

    food_stock = np.where(terrain == S.FOOD, 1.0, 0.0)

    # agents around the center
    A = n_learners + n_adults
    is_adult = np.array([False] * n_learners + [True] * n_adults)
    pos = np.zeros((A, 2), dtype=np.int64)
    taken = set()
    for i in range(A):
        while True:
            x = cx + int(rng.integers(-2, 3))
            y = cy + int(rng.integers(-2, 3))
            if (x, y) not in taken and terrain[y, x] == S.EMPTY:
                taken.add((x, y))
                pos[i] = (x, y)
                break
    facing = rng.integers(0, 4, size=A)

    # predator lair in a corner region, far from center
    corners = [(2, 2), (G - 3, 2), (2, G - 3), (G - 3, G - 3)]
    lair = np.array(corners[int(rng.integers(0, 4))], dtype=np.int64)

    return State(
        rng=rng, t=0, terrain=terrain, food_stock=food_stock,
        n_agents=A, is_adult=is_adult, pos=pos, facing=facing,
        hunger=np.full(A, 0.8), thirst=np.full(A, 0.8), warmth=np.ones(A),
        alive=np.ones(A, dtype=bool), sheltered=np.zeros(A, dtype=bool),
        pointing=np.zeros(A, dtype=bool), saying=np.zeros(A, dtype=np.int64),
        predator_pos=lair.copy(), predator_lair=lair, predator_active=False,
    )


# -------------------------------------------------------------- geometry ---
def line_of_sight(state: State, x0: int, y0: int, x1: int, y1: int) -> bool:
    """True if no sight-blocking terrain lies strictly between the two cells."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if (x, y) != (x0, y0) and (x, y) != (x1, y1):
            if state.terrain[y, x] in S.BLOCKS_SIGHT:
                return False
        if (x, y) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def to_egocentric(facing: int, dx: int, dy: int) -> tuple[int, int]:
    """World offset (dx, dy) -> (rel_x right+, rel_y forward+) for an agent facing `facing`."""
    if facing == S.NORTH:
        return dx, -dy
    if facing == S.EAST:
        return dy, dx
    if facing == S.SOUTH:
        return -dx, dy
    return -dy, -dx  # WEST


def from_egocentric(facing: int, rel_x: int, rel_y: int) -> tuple[int, int]:
    """Inverse of to_egocentric."""
    if facing == S.NORTH:
        return rel_x, -rel_y
    if facing == S.EAST:
        return rel_y, rel_x
    if facing == S.SOUTH:
        return -rel_x, rel_y
    return -rel_y, -rel_x


def view_cells(state: State, i: int) -> list[tuple[int, int, int, int]]:
    """Cells agent i can see: (rel_x, rel_y, abs_x, abs_y), nearest first.
    Includes the agent's own cell at (0, 0)."""
    x0, y0 = int(state.pos[i, 0]), int(state.pos[i, 1])
    f = int(state.facing[i])
    out = []
    for ry in range(0, S.VIEW_DEPTH + 1):
        half = min(ry, S.VIEW_HALF)
        for rx in range(-half, half + 1):
            dx, dy = from_egocentric(f, rx, ry)
            x, y = x0 + dx, y0 + dy
            if not state.in_bounds(x, y):
                if ry <= 1:
                    out.append((rx, ry, -1, -1))      # edge marker; frame shows it as rock
                continue
            if (rx, ry) != (0, 0) and not line_of_sight(state, x0, y0, x, y):
                continue
            out.append((rx, ry, x, y))
    out.sort(key=lambda c: abs(c[0]) + c[1])
    return out


def direction_to(facing: int, dx: int, dy: int) -> int:
    """Relative direction (0 ahead, 1 right, 2 behind, 3 left) of a world offset."""
    rx, ry = to_egocentric(facing, dx, dy)
    if abs(ry) >= abs(rx):
        return 0 if ry >= 0 else 2
    return 1 if rx > 0 else 3


def heard(state: State, i: int) -> tuple[int, int]:
    """(word, rel_dir) of the nearest speaker within HEAR_RADIUS, or (0, NO_DIR)."""
    best = None
    for j in range(state.n_agents):
        if j == i or not state.alive[j] or state.saying[j] == S.NO_WORD:
            continue
        dx = int(state.pos[j, 0] - state.pos[i, 0])
        dy = int(state.pos[j, 1] - state.pos[i, 1])
        d = abs(dx) + abs(dy)
        if d <= S.HEAR_RADIUS and (best is None or d < best[0]):
            best = (d, int(state.saying[j]), direction_to(int(state.facing[i]), dx, dy))
    if best is None:
        return S.NO_WORD, S.NO_DIR
    return best[1], best[2]


# ------------------------------------------------------------------ step ---
def step(state: State, actions: np.ndarray, pointing: np.ndarray, saying: np.ndarray) -> State:
    """Advance one tick. Mutates and returns `state`.

    actions, pointing, saying: (A,) arrays for every agent, adults included.
    """
    st = state
    st.events = []
    st.pointing[:] = pointing.astype(bool)
    st.saying[:] = saying.astype(np.int64)

    # --- movement and consumption, agents in index order
    for i in range(st.n_agents):
        if not st.alive[i]:
            continue
        a = int(actions[i])
        x, y = int(st.pos[i, 0]), int(st.pos[i, 1])
        if a == S.TURN_LEFT:
            st.facing[i] = (st.facing[i] - 1) % 4
        elif a == S.TURN_RIGHT:
            st.facing[i] = (st.facing[i] + 1) % 4
        elif a == S.FORWARD:
            dx, dy = S.FACING_DXY[int(st.facing[i])]
            nx, ny = x + dx, y + dy
            if st.in_bounds(nx, ny) and st.terrain[ny, nx] not in S.BLOCKS_MOVE \
                    and st.agent_at(nx, ny) == -1:
                st.pos[i] = (nx, ny)
                st.sheltered[i] = False
                if st.terrain[ny, nx] == S.HAZARD and not st.is_adult[i]:
                    st.alive[i] = False
                    st.events.append(f"agent {i} fell into hazard at {(nx, ny)}")
        elif a == S.EAT:
            if st.terrain[y, x] == S.FOOD and st.food_stock[y, x] >= FOOD_BITE:
                st.food_stock[y, x] -= FOOD_BITE
                st.hunger[i] = min(1.0, st.hunger[i] + EAT_GAIN)
        elif a == S.DRINK:
            if st.terrain[y, x] == S.WATER:
                st.thirst[i] = min(1.0, st.thirst[i] + DRINK_GAIN)
        elif a == S.ENTER:
            if st.terrain[y, x] == S.SHELTER:
                st.sheltered[i] = True

    # --- needs (learners only)
    night = st.phase() == S.NIGHT
    for i in st.learners():
        if not st.alive[i]:
            continue
        st.hunger[i] -= HUNGER_DRAIN
        st.thirst[i] -= THIRST_DRAIN
        if st.sheltered[i] or not night:
            st.warmth[i] = min(1.0, st.warmth[i] + WARMTH_RECOVER)
        else:
            st.warmth[i] -= WARMTH_DRAIN_NIGHT
        for name, arr in (("hunger", st.hunger), ("thirst", st.thirst), ("warmth", st.warmth)):
            if arr[i] <= 0.0:
                arr[i] = 0.0
                st.alive[i] = False
                st.events.append(f"agent {i} incapacitated: {name}")

    # --- food regrowth
    mask = st.terrain == S.FOOD
    st.food_stock[mask] = np.minimum(1.0, st.food_stock[mask] + FOOD_REGEN)

    # --- predator
    if night:
        if not st.predator_active:
            st.predator_active = True
            st.predator_pos = st.predator_lair.copy()
            st.events.append("predator wakes")
        if st.t % PREDATOR_MOVE_EVERY == 0:
            targets = [i for i in st.learners() if st.alive[i] and not st.sheltered[i]]
            if targets:
                d = [abs(st.pos[i, 0] - st.predator_pos[0]) + abs(st.pos[i, 1] - st.predator_pos[1])
                     for i in targets]
                tgt = targets[int(np.argmin(d))]
                dx = int(np.sign(st.pos[tgt, 0] - st.predator_pos[0]))
                dy = int(np.sign(st.pos[tgt, 1] - st.predator_pos[1]))
                # move along the longer axis first; terrain does not stop it
                if abs(st.pos[tgt, 0] - st.predator_pos[0]) >= abs(st.pos[tgt, 1] - st.predator_pos[1]):
                    st.predator_pos[0] += dx
                else:
                    st.predator_pos[1] += dy
        # catch: adjacent or on top of an unsheltered learner
        for i in st.learners():
            if st.alive[i] and not st.sheltered[i]:
                if abs(st.pos[i, 0] - st.predator_pos[0]) + abs(st.pos[i, 1] - st.predator_pos[1]) <= 1:
                    st.alive[i] = False
                    st.events.append(f"agent {i} taken by predator at {tuple(st.pos[i])}")
    else:
        if st.predator_active:
            st.predator_active = False
            st.predator_pos = st.predator_lair.copy()
            st.events.append("predator sleeps")

    st.t += 1
    return st


# ------------------------------------------------------------ rendering ---
def render(state: State, viewer: int | None = None) -> str:
    """ASCII map. With `viewer`, cells that agent can see are uppercased."""
    G = S.GRID
    rows = [[S.TYPE_GLYPH[int(state.terrain[y, x])] for x in range(G)] for y in range(G)]
    if viewer is not None:
        seen = {(c[2], c[3]) for c in view_cells(state, viewer) if c[2] >= 0}
        for (x, y) in seen:
            if rows[y][x] == ".":
                rows[y][x] = ","
            else:
                rows[y][x] = rows[y][x].upper()
    if state.predator_active:
        px, py = int(state.predator_pos[0]), int(state.predator_pos[1])
        rows[py][px] = "P"
    for i in range(state.n_agents):
        x, y = int(state.pos[i, 0]), int(state.pos[i, 1])
        if not state.alive[i]:
            rows[y][x] = "%"
        elif state.is_adult[i]:
            rows[y][x] = S.FACING_GLYPH[int(state.facing[i])] if state.pointing[i] else "A"
        else:
            rows[y][x] = str(i) if i < 10 else "l"
    header = f"t={state.t} {['day', 'dusk', 'NIGHT'][state.phase()]}  " + "  ".join(
        f"[{i}{'A' if state.is_adult[i] else ''} {S.FACING_GLYPH[int(state.facing[i])]} "
        f"h{state.hunger[i]:.2f} t{state.thirst[i]:.2f} w{state.warmth[i]:.2f}"
        f"{' shel' if state.sheltered[i] else ''}{' DOWN' if not state.alive[i] else ''}"
        f"{' says ' + S.WORDS[int(state.saying[i])] if state.saying[i] else ''}]"
        for i in range(state.n_agents))
    return header + "\n" + "\n".join("".join(r) for r in rows)
