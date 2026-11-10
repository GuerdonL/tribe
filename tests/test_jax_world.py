"""The JAX world must produce schema-valid frames and the M0 effect must survive the port."""
import jax
import jax.numpy as jnp
import numpy as np

from tribe import schema as S
from tribe import jax_world as J
from tribe.frame import validate, decode
from tribe.policies import GazeIgnorer, GazeFollower


def test_reset_and_frames_validate():
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    st = J.reset_batch(keys, 1, 1)
    assert st.pos.shape == (8, 2, 2)
    frames = np.asarray(J.observe_all_batch(st))
    assert frames.shape == (8, 2, S.FRAME_LEN)
    for e in range(8):
        for i in range(2):
            validate(frames[e, i])


def test_ego_roundtrip_matches_numpy():
    from tribe import world as W
    for f in range(4):
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                rx, ry = J.to_ego(jnp.int32(f), jnp.int32(dx), jnp.int32(dy))
                assert (int(rx), int(ry)) == W.to_egocentric(f, dx, dy)
                bx, by = J.from_ego(jnp.int32(f), jnp.int32(rx), jnp.int32(ry))
                assert (int(bx), int(by)) == (dx, dy)


def _run(policy_cls, n_envs=24, steps=S.EPISODE_STEPS, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    st = J.reset_batch(keys, 1, 1)
    pols = [policy_cls(seed=1000 + e) for e in range(n_envs)]
    survived = np.full(n_envs, steps)
    done = np.zeros(n_envs, bool)
    for t in range(steps):
        frames = np.asarray(J.observe_all_batch(st))
        acts = np.zeros((n_envs, 2), np.int32)
        pts = np.zeros((n_envs, 2), np.int32)
        says = np.zeros((n_envs, 2), np.int32)
        for e in range(n_envs):
            validate(frames[e, 0])
            acts[e, 0], pts[e, 0], says[e, 0] = pols[e].act(frames[e, 0])
        aa, ap, asay = J.adult_act_batch(st, 1)
        acts[:, 1], pts[:, 1], says[:, 1] = np.asarray(aa), np.asarray(ap), np.asarray(asay)
        st = J.step_batch(st, jnp.asarray(acts), jnp.asarray(pts), jnp.asarray(says))
        alive = np.asarray(st.alive[:, 0])
        newly = ~alive & ~done
        survived[newly] = t + 1
        done |= ~alive
    return survived


def test_m0_effect_survives_port():
    a = _run(GazeIgnorer)
    b = _run(GazeFollower)
    assert a.min() > 0 and b.min() > 0
    # not a significance test at this size; the real one is scripts/m0_jax.py
    assert b.mean() >= a.mean() - 15, (a.mean(), b.mean())


def test_adult_points_and_speaks_sometimes():
    keys = jax.random.split(jax.random.PRNGKey(3), 8)
    st = J.reset_batch(keys, 1, 1)
    pointed = spoke = 0
    for t in range(120):
        aa, ap, asay = J.adult_act_batch(st, 1)
        pointed += int(np.asarray(ap).sum())
        spoke += int((np.asarray(asay) != 0).sum())
        acts = jnp.stack([jnp.zeros(8, jnp.int32) + S.NOOP, aa], axis=1)
        pts = jnp.stack([jnp.zeros(8, jnp.int32), ap.astype(jnp.int32)], axis=1)
        says = jnp.stack([jnp.zeros(8, jnp.int32), asay], axis=1)
        st = J.step_batch(st, acts, pts, says)
    assert pointed > 0 and spoke > 0


def test_name_idle_names_landmarks(monkeypatch):
    """With NAME_IDLE on, an idle adult points at and names nearby terrain in both worlds."""
    monkeypatch.setattr(J, "NAME_IDLE", True)
    from tribe import adult as A
    monkeypatch.setattr(A, "NAME_IDLE", True)
    keys = jax.random.split(jax.random.PRNGKey(5), 8)
    st = J.reset_batch(keys, 1, 1)
    words = set()
    for t in range(60):
        aa, ap, asay = jax.vmap(J.adult_act, in_axes=(0, None))(st, 1)
        words |= {int(w) for w, p in zip(np.asarray(asay), np.asarray(ap)) if p}
        acts = jnp.stack([jnp.zeros(8, jnp.int32) + S.NOOP, aa], axis=1)
        pts = jnp.stack([jnp.zeros(8, jnp.int32), ap.astype(jnp.int32)], axis=1)
        says = jnp.stack([jnp.zeros(8, jnp.int32), asay], axis=1)
        st = J.step_batch(st, acts, pts, says)
    landmark_words = {S.WORD_ID[w] for w in ("tree", "rock", "food", "water", "shelter", "danger")}
    assert words & landmark_words, words
    # NumPy world agrees in kind
    from tribe import world as W
    from tribe.adult import adult_act
    stn = W.reset(seed=5)
    nwords = set()
    for t in range(60):
        a, p, w = adult_act(stn, 1)
        if p:
            nwords.add(w)
        W.step(stn, np.array([S.NOOP, a]), np.array([0, p]), np.array([0, w]))
    assert nwords & landmark_words, nwords
