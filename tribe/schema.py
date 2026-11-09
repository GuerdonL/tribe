"""Frozen frame schema — version 1.

Everything the learner perceives, hears, does and says is an integer in a
fixed-size frame. One frame per environment step. One word per frame, which
is also the maximum speaking rate.

Layout
------
self  : [hunger, thirst, warmth, facing, phase, heard_word, heard_dir]      (7)
slots : N_SLOTS x [valid, type, rel_x, rel_y, rel_facing, pointing, saying] (N_SLOTS * 7)
out   : [action, pointing, saying]                                          (3)

All coordinates are egocentric: rel_x is right(+)/left(-) of the agent,
rel_y is forward(+) of the agent. rel_facing is the other agent's facing
relative to ours (0 = same way we face, 1 = our right, 2 = behind us, 3 = our left).

Changing anything here is a schema version bump; every downstream file
(model, logger, replay) reads this module and nothing else for layout.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

# ---------------------------------------------------------------- world ---
GRID = 32
VIEW_DEPTH = 5          # cells forward
VIEW_HALF = 5           # |rel_x| <= min(rel_y, VIEW_HALF): a 90 degree cone
HEAR_RADIUS = 6         # words carry this far, through trees
DAY_LENGTH = 80
NIGHT_START = 60        # steps [NIGHT_START, DAY_LENGTH) are night
DUSK_START = 48         # steps [DUSK_START, NIGHT_START) are dusk: head home
EPISODE_STEPS = 320

# ---------------------------------------------------------- entity types ---
EMPTY, FOOD, WATER, TREE, ROCK, SHELTER, PREDATOR, HAZARD, LEARNER, ADULT = range(10)
N_TYPES = 10
TYPE_NAMES = ["empty", "food", "water", "tree", "rock", "shelter",
              "predator", "hazard", "learner", "adult"]
TYPE_GLYPH = [".", "f", "w", "T", "#", "H", "P", "x", "l", "A"]
BLOCKS_SIGHT = {TREE, ROCK}
BLOCKS_MOVE = {TREE, ROCK}          # water is shallow: walk in to drink

# --------------------------------------------------------------- facing ---
NORTH, EAST, SOUTH, WEST = range(4)
FACING_DXY = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # (dx, dy) per facing; y grows south
FACING_GLYPH = ["^", ">", "v", "<"]

# -------------------------------------------------------------- actions ---
NOOP, FORWARD, TURN_LEFT, TURN_RIGHT, EAT, DRINK, ENTER = range(7)
N_ACTIONS = 7
ACTION_NAMES = ["noop", "forward", "turn_left", "turn_right", "eat", "drink", "enter"]

# ---------------------------------------------------------------- words ---
# Stage-one vocabulary, concrete leaves of the dictionary graph first.
# Index 0 is silence. Order is part of the schema.
WORDS = [
    "<none>",
    # rung 1
    "look", "here", "there",
    # rung 1-2
    "food", "water", "tree", "rock", "shelter", "danger",
    # rung 2
    "come", "go", "wait", "get",
    # rung 3
    "night", "home", "now", "together",
    # rung 4
    "behind", "near", "far", "left", "right",
]
WORD_ID = {w: i for i, w in enumerate(WORDS)}
N_WORDS = len(WORDS)
NO_WORD = 0

# ------------------------------------------------------------ quantizing ---
N_METER_BINS = 8        # meters in [0, 1] -> {0..7}
N_PHASES = 3            # 0 day, 1 dusk, 2 night
DAY, DUSK, NIGHT = range(3)
NO_DIR = 4              # heard_dir when nothing heard

# ---------------------------------------------------------------- frame ---
N_SLOTS = 12

SELF_FIELDS = ["hunger", "thirst", "warmth", "facing", "phase", "heard_word", "heard_dir"]
SLOT_FIELDS = ["valid", "type", "rel_x", "rel_y", "rel_facing", "pointing", "saying"]
OUT_FIELDS = ["action", "pointing", "saying"]

SELF_LEN = len(SELF_FIELDS)
SLOT_LEN = len(SLOT_FIELDS)
OUT_LEN = len(OUT_FIELDS)
FRAME_LEN = SELF_LEN + N_SLOTS * SLOT_LEN

SELF_OFF = 0
SLOTS_OFF = SELF_LEN

# Offsets that make every field a non-negative small int, for embedding.
REL_X_OFFSET = VIEW_HALF          # rel_x in [-5, 5] -> [0, 10]
REL_X_RANGE = 2 * VIEW_HALF + 1
REL_Y_RANGE = VIEW_DEPTH + 1      # rel_y in [0, 5]

# Cardinality of each field, so a model can size one embedding table per field.
SELF_CARD = [N_METER_BINS, N_METER_BINS, N_METER_BINS, 4, N_PHASES, N_WORDS, 5]
SLOT_CARD = [2, N_TYPES, REL_X_RANGE, REL_Y_RANGE, 4, 2, N_WORDS]
OUT_CARD = [N_ACTIONS, 2, N_WORDS]


def quantize(x: float, bins: int = N_METER_BINS) -> int:
    """Map a meter in [0, 1] to {0 .. bins-1}."""
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return bins - 1
    return int(x * bins)


def field_index(group: str, name: str, slot: int = 0) -> int:
    """Flat index of a named field in a frame vector."""
    if group == "self":
        return SELF_OFF + SELF_FIELDS.index(name)
    if group == "slot":
        return SLOTS_OFF + slot * SLOT_LEN + SLOT_FIELDS.index(name)
    raise KeyError(group)
