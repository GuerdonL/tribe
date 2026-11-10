"""Shared evaluation helpers for trained XL learners: load a checkpoint, roll it out on fresh
episodes with an optional transform of what the learner observes, and record what happened.

The transform changes only the frame fed to the model. The world, and the scripted adult
(which reads the true state), are untouched.
"""
import os
import pickle
from functools import partial
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tribe import schema as S
from tribe import jax_world as J
from tribe.model import XLLearner, Predictor
from tribe.train import world_step_raw, sample_out, TrainConfig
import __main__; __main__.TrainConfig = TrainConfig          # checkpoints pickle cfg as __main__.TrainConfig

SO = S.SELF_LEN
F_VALID, F_TYPE, F_X, F_Y, F_FACE, F_POINT, F_SAY = range(7)
I_HW, I_HD = S.SELF_FIELDS.index("heard_word"), S.SELF_FIELDS.index("heard_dir")


def load(path):
    ck = pickle.load(open(path, "rb"))
    return ck["cfg"], ck["params"]


# ------------------------------------------------------- frame surgery ---
def split(frame):
    return frame[:, :SO], frame[:, SO:].reshape(frame.shape[0], S.N_SLOTS, S.SLOT_LEN)


def join(me, slots):
    return jnp.concatenate([me, slots.reshape(slots.shape[0], -1)], axis=1)


def adult_mask(slots):
    return (slots[..., F_VALID] > 0) & (slots[..., F_TYPE] == S.ADULT)


def t_identity(frame, key):
    return frame


def t_no_point(frame, key):
    me, sl = split(frame)
    return join(me, sl.at[..., F_POINT].set(jnp.where(adult_mask(sl), 0, sl[..., F_POINT])))


def t_rand_face(frame, key):
    me, sl = split(frame)
    r = jax.random.randint(key, sl.shape[:2], 0, 4)
    return join(me, sl.at[..., F_FACE].set(jnp.where(adult_mask(sl), r, sl[..., F_FACE])))


def t_flip_face(frame, key):
    """adult's facing reversed: right <-> left, toward <-> away"""
    me, sl = split(frame)
    return join(me, sl.at[..., F_FACE].set(jnp.where(adult_mask(sl), (sl[..., F_FACE] + 2) % 4, sl[..., F_FACE])))


def t_no_gaze(frame, key):
    return t_rand_face(t_no_point(frame, key), key)


def t_no_words(frame, key):
    me, sl = split(frame)
    me = me.at[:, I_HW].set(S.NO_WORD).at[:, I_HD].set(S.NO_DIR)
    return join(me, sl.at[..., F_SAY].set(0))


def t_no_adult(frame, key):
    """adult removed from view (slots compacted) and nothing heard"""
    me, sl = split(t_no_words(frame, key))
    keep = (sl[..., F_VALID] > 0) & ~adult_mask(sl)
    order = jnp.cumsum(keep.astype(jnp.int32), axis=1) - 1
    idx = jnp.where(keep, order, S.N_SLOTS)
    n = sl.shape[0]
    out = jnp.zeros((n, S.N_SLOTS + 1, S.SLOT_LEN), sl.dtype).at[jnp.arange(n)[:, None], idx].set(sl)
    return join(me, out[:, : S.N_SLOTS])


def _map_words(frame, table):
    me, sl = split(frame)
    table = jnp.asarray(table)
    return join(me.at[:, I_HW].set(table[me[:, I_HW]]), sl.at[..., F_SAY].set(table[sl[..., F_SAY]]))


def t_const_word(frame, key):
    """every word becomes "look": timing and direction of speech kept, identity removed"""
    return _map_words(frame, [0] + [S.WORD_ID["look"]] * (S.N_WORDS - 1))


def t_swap_food_water(frame, key):
    tab = list(range(S.N_WORDS))
    tab[S.WORD_ID["food"]], tab[S.WORD_ID["water"]] = S.WORD_ID["water"], S.WORD_ID["food"]
    return _map_words(frame, tab)


def t_no_heard_dir(frame, key):
    """word kept, direction of the voice removed"""
    me, sl = split(frame)
    return join(me.at[:, I_HD].set(S.NO_DIR), sl)


TRANSFORMS = {"normal": t_identity, "no_point": t_no_point, "rand_face": t_rand_face, "flip_face": t_flip_face,
              "no_gaze": t_no_gaze, "no_words": t_no_words, "no_adult": t_no_adult,
              "const_word": t_const_word, "swap_fw": t_swap_food_water, "no_hdir": t_no_heard_dir}

def word_map_transform(spec):
    """spec like "food=look+water=look": replace heard/said words by name"""
    tab = list(range(S.N_WORDS))
    for pair in spec.split("+"):
        a, b = pair.split("=")
        tab[S.WORD_ID[a]] = S.WORD_ID[b]
    return lambda frame, key: _map_words(frame, tab)


def get_transform(name):
    return word_map_transform(name[4:]) if name.startswith("map:") else TRANSFORMS[name]


CAUSES = ["alive", "hunger", "thirst", "cold", "hazard", "predator"]


def death_cause(env2):
    """0 alive, else index into CAUSES, from the post-step state (before reset)."""
    died = env2.died_this_step[:, 0]
    x, y = env2.pos[:, 0, 0], env2.pos[:, 0, 1]
    under = env2.terrain[jnp.arange(x.shape[0]), y, x]
    c = jnp.where(env2.hunger[:, 0] <= 0, 1, jnp.where(env2.thirst[:, 0] <= 0, 2, jnp.where(
        env2.warmth[:, 0] <= 0, 3, jnp.where(under == S.HAZARD, 4, 5))))
    return jnp.where(died, c, 0)


# -------------------------------------------------------------- rollout ---
def make_rollout(cfg, transform=t_identity, extra=None, reset=True):
    """Returns run(params, key, n_envs, steps) -> dict of (steps, n_envs, ...) numpy arrays. Params are an
    argument of the compiled function, so one compile serves every snapshot of the same model.
    extra(apply, frame, obs, logits, ctx) -> dict of more things to record. apply(frame2) re-runs this
    step's policy on a different frame with the same memory; ctx has env, env2, out, key, params."""
    mc = cfg.model
    K, L, d = mc.K, mc.n_layers, mc.d
    model = XLLearner(mc)

    def step(params, carry, _):
        env, mem, mem_ep, last_out, ep, key = carry
        key, k_t, k_s, k_r = jax.random.split(key, 4)
        frame = jax.vmap(J.observe, in_axes=(0, None))(env, 0)
        obs = transform(frame, k_t)
        prev_same = mem_ep[:, -1] == ep

        def apply(f):
            return model.apply(params["learner"], f, last_out, prev_same, mem, mem_ep, ep, method=XLLearner.step)

        logits, new_mem = apply(obs)
        out, _ = sample_out(k_s, logits)
        env2, reward, done = world_step_raw(env, out)
        rec = dict(ep=ep, act=out[:, 0], reward=reward, done=done, cause=death_cause(env2), value=logits["value"],
                   hunger=env.hunger[:, 0], thirst=env.thirst[:, 0], warmth=env.warmth[:, 0],
                   sheltered=env.sheltered[:, 0], phase=J.phase(env.t), t=env.t)
        if extra is not None:
            rec.update(extra(lambda f: apply(f)[0], frame, obs, logits,
                             dict(env=env, env2=env2, out=out, key=k_t, params=params)))
        if reset:
            fresh = J.reset_batch(jax.random.split(k_r, out.shape[0]), 1, 1)
            env2 = jax.tree.map(lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), env2, fresh)
        mem_ep2 = jnp.concatenate([mem_ep[:, 1:], ep[:, None]], 1)
        return (env2, new_mem, mem_ep2, out, ep + done.astype(jnp.int32), key), rec

    @partial(jax.jit, static_argnums=(3,))
    def run_jit(params, key, env, steps):
        n = env.t.shape[0]
        carry = (env, jnp.zeros((n, L, K - 1, d)), jnp.full((n, K - 1), -1, jnp.int32),
                 jnp.zeros((n, 3), jnp.int32), jnp.zeros((n,), jnp.int32), key)
        return jax.lax.scan(partial(step, params), carry, None, length=steps)[1]

    def run(params, key, n_envs, steps, env_seed=0):
        env = J.reset_batch(jax.random.split(jax.random.PRNGKey(env_seed), n_envs), 1, 1)
        params = jax.tree.map(jnp.asarray, params)
        return {k: np.asarray(v) for k, v in run_jit(params, key, env, steps).items()}

    return run


def first_episode(rec):
    """Per-env summary of episode 0 from a rollout of >= EPISODE_STEPS steps."""
    m = rec["ep"] == 0                                         # (T, N)
    steps = m.sum(0)
    cause = (rec["cause"] * m).max(0)
    f = m.astype(np.float64)
    mean = lambda x: (x * f).sum(0) / np.maximum(f.sum(0), 1)
    night = m & (rec["phase"] == S.NIGHT)
    return dict(steps=steps, died=cause > 0, cause=cause, ret=(rec["reward"] * f).sum(0),
                hunger=mean(rec["hunger"]), thirst=mean(rec["thirst"]), warmth=mean(rec["warmth"]),
                night_sheltered=(rec["sheltered"] & night).sum(0) / np.maximum(night.sum(0), 1))


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])
