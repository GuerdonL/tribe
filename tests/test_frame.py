import numpy as np

from tribe import schema as S
from tribe import world as W
from tribe.frame import observe, decode, validate, find


def test_schema_lengths():
    assert S.FRAME_LEN == S.SELF_LEN + S.N_SLOTS * S.SLOT_LEN
    assert len(S.SELF_CARD) == S.SELF_LEN
    assert len(S.SLOT_CARD) == S.SLOT_LEN
    assert len(S.OUT_CARD) == S.OUT_LEN
    assert S.WORDS[0] == "<none>"


def test_egocentric_roundtrip():
    for f in range(4):
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                rx, ry = W.to_egocentric(f, dx, dy)
                assert W.from_egocentric(f, rx, ry) == (dx, dy)


def test_egocentric_semantics():
    # facing north, a cell one step north is straight ahead
    assert W.to_egocentric(S.NORTH, 0, -1) == (0, 1)
    # facing north, a cell to the east is on the right
    assert W.to_egocentric(S.NORTH, 1, 0) == (1, 0)
    # facing east, a cell to the south is on the right
    assert W.to_egocentric(S.EAST, 0, 1) == (1, 0)
    # facing south, a cell to the north is behind
    assert W.to_egocentric(S.SOUTH, 0, -1) == (0, -1)


def test_frames_validate_over_random_play():
    rng = np.random.default_rng(0)
    for seed in range(5):
        st = W.reset(seed=seed, n_learners=2, n_adults=1)
        for _ in range(80):
            A = st.n_agents
            acts = rng.integers(0, S.N_ACTIONS, size=A)
            pts = rng.integers(0, 2, size=A)
            says = rng.integers(0, S.N_WORDS, size=A)
            W.step(st, acts, pts, says)
            for i in range(A):
                fr = observe(st, i)
                validate(fr)
                d = decode(fr)
                # decode agrees with the raw vector
                assert d["self"]["facing"] == st.facing[i]
                assert d["self"]["phase"] == st.phase()
                for s in d["slots"]:
                    assert -S.VIEW_HALF <= s["rel_x"] <= S.VIEW_HALF
                    assert 0 <= s["rel_y"] <= S.VIEW_DEPTH
                    assert abs(s["rel_x"]) <= s["rel_y"]  # inside the cone


def test_adult_seen_with_relative_facing_and_word():
    st = W.reset(seed=3, n_learners=1, n_adults=1)
    # put learner at center facing north, adult two cells ahead facing east, saying "look"
    st.pos[0] = (16, 16); st.facing[0] = S.NORTH
    st.pos[1] = (16, 14); st.facing[1] = S.EAST
    st.terrain[14, 16] = S.EMPTY
    st.terrain[15, 16] = S.EMPTY
    st.saying[1] = S.WORD_ID["look"]
    st.pointing[1] = True
    fr = observe(st, 0)
    validate(fr)
    a = find(fr, S.ADULT)
    assert a is not None
    assert (a["rel_x"], a["rel_y"]) == (0, 2)
    assert a["rel_facing"] == 1          # adult faces east = learner's right
    assert a["pointing"] == 1
    assert a["saying_str"] == "look"
    d = decode(fr)
    assert d["self"]["heard_word_str"] == "look"
    assert d["self"]["heard_dir"] == 0   # speaker is ahead


def test_occlusion():
    st = W.reset(seed=4, n_learners=1, n_adults=1)
    st.pos[0] = (16, 16); st.facing[0] = S.NORTH
    st.pos[1] = (16, 13); st.facing[1] = S.NORTH
    for y in (13, 14, 15):
        st.terrain[y, 16] = S.EMPTY
    assert find(observe(st, 0), S.ADULT) is not None
    st.terrain[15, 16] = S.TREE
    assert find(observe(st, 0), S.ADULT) is None
    # but you can still hear through trees
    st.saying[1] = S.WORD_ID["danger"]
    assert decode(observe(st, 0))["self"]["heard_word_str"] == "danger"
