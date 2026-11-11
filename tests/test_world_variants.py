"""The M2 world and reward variants: nights that can kill, and a warmth reward that does not pay for getting cold.
The defaults must stay the M1 world."""
import dataclasses
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from tribe import schema as S
from tribe import jax_world as J
from tribe import world as W
from tribe.train import TrainConfig, world_step_raw, R_ALIVE

COLD = 1.0 / 15


def _idle_outside(cfg, steps, n=6):
    """Learner does nothing (never shelters). Returns per-step reward, warmth, alive: each (steps, n)."""
    env = J.reset_batch(jax.random.split(jax.random.PRNGKey(0), n), 1, 1)
    out = jnp.zeros((n, 3), jnp.int32)
    f = jax.jit(lambda e: world_step_raw(e, out, cfg))
    rew, warm, alive = [], [], []
    for _ in range(steps):
        env, r, _ = f(env)
        rew.append(np.asarray(r)); warm.append(np.asarray(env.warmth[:, 0])); alive.append(np.asarray(env.alive[:, 0]))
    return np.array(rew), np.array(warm), np.array(alive)


def test_m1_night_is_survivable_and_cold_night_kills():
    _, warm, alive = _idle_outside(TrainConfig(), S.DAY_LENGTH)
    assert alive[-1].all() and warm[-1].min() > 0.4            # 20 night steps at 1/35 leave 0.43
    _, warm, alive = _idle_outside(TrainConfig(night_drain=COLD), S.DAY_LENGTH)
    assert not alive[-1].any()
    assert alive[S.NIGHT_START + 13].all() and not alive[S.NIGHT_START + 15].any()   # empties on night step 15 or 16 (float32)


def test_recover_pays_for_getting_cold_and_deficit_does_not():
    steps = S.DAY_LENGTH + 10                                   # one night outside, then the morning after
    rec, _, alive = _idle_outside(TrainConfig(), steps)
    dfc, warm, _ = _idle_outside(TrainConfig(warmth_reward="deficit"), steps)
    assert alive.all()
    day, morning = slice(0, S.NIGHT_START), slice(S.DAY_LENGTH, steps)
    np.testing.assert_allclose(rec[day], R_ALIVE, atol=1e-6)    # warm and idle: both pay only for being alive
    np.testing.assert_allclose(dfc[day], R_ALIVE, atol=1e-6)
    assert (rec[morning] > R_ALIVE + 0.05).all()                # M1: paid 2 * 1/30 per step for warming back up
    assert (dfc[morning] < R_ALIVE).all()                       # variant: still cold, so still charged
    np.testing.assert_allclose(dfc[morning], R_ALIVE - 0.1 * (1 - warm[morning]), atol=1e-6)


def test_numpy_world_takes_the_same_night_drain():
    def nights_out(**kw):
        st = W.reset(seed=0)
        z = np.zeros(st.n_agents, np.int64)
        for _ in range(S.DAY_LENGTH):
            W.step(st, z, z, z, **kw)
        return bool(st.alive[0]), float(st.warmth[0])
    alive, warmth = nights_out()
    assert alive and warmth > 0.4
    alive, warmth = nights_out(night_drain=COLD)
    assert not alive and warmth == 0.0


def test_old_checkpoint_config_reads_as_m1():
    """Configs pickled before the variant fields existed must load with the M1 values."""
    cfg = TrainConfig()
    state = {k: v for k, v in cfg.__dict__.items() if k not in ("warmth_reward", "r_cold", "night_drain", "predator_move_every")}
    old = object.__new__(TrainConfig)
    old.__dict__.update(state)
    old = pickle.loads(pickle.dumps(old))
    assert old.warmth_reward == "recover" and old.night_drain == J.WARMTH_DRAIN_NIGHT and old.predator_move_every == 2
