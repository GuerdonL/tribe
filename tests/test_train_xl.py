import dataclasses

import jax
import numpy as np

from tribe.model import ModelConfig
from tribe.train import PRESETS, TrainConfig, make_train_xl


def test_xl_training_pass_reproduces_rollout_logp():
    """With lr=0 the params never move, so every minibatch's ratio must be exactly 1:
    the segment pass recomputes what the cached step-by-step rollout computed."""
    cfg = TrainConfig(model=ModelConfig(d=32, n_layers=2, n_heads=2, K=4, arch="xl", remat=False),
                      n_envs=8, T=12, epochs=2, minibatches=2, lr=0.0)
    init_params, tx, reset_runner, update = make_train_xl(cfg)
    params = init_params(jax.random.PRNGKey(0))
    opt_state = tx.init(params)
    rs = reset_runner(jax.random.PRNGKey(1))
    for _ in range(3):      # later updates start from non-empty memory and span episode resets
        params, opt_state, rs, trs, m = update(params, opt_state, rs)
        assert float(m["approx_kl"]) < 1e-6, m["approx_kl"]
        assert float(m["clipfrac"]) == 0.0
        assert np.isfinite(float(m["loss"]))
