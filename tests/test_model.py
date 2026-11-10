import jax
import jax.numpy as jnp
import numpy as np

from tribe import schema as S
from tribe.model import Learner, Predictor, ModelConfig, prediction_losses
from tribe.train import gaze_metrics


def test_forward_and_predict_shapes():
    cfg = ModelConfig(d=32, n_layers=1, n_heads=2, K=4)
    m, p = Learner(cfg), Predictor(cfg)
    B, K = 3, cfg.K
    frames = jnp.zeros((B, K, S.FRAME_LEN), jnp.int32)
    prev = jnp.zeros((B, K, 3), jnp.int32)
    valid = jnp.ones((B, K), bool)
    lp = m.init(jax.random.PRNGKey(0), frames, prev, valid)
    out = m.apply(lp, frames, prev, valid)
    assert out["action"].shape == (B, S.N_ACTIONS)
    assert out["saying"].shape == (B, S.N_WORDS)
    assert out["value"].shape == (B,)
    pp = p.init(jax.random.PRNGKey(1), out["h"], jnp.zeros((B, 3), jnp.int32))
    sl, ll = p.apply(pp, out["h"], jnp.zeros((B, 3), jnp.int32))
    assert len(sl) == S.SELF_LEN and len(ll) == S.SLOT_LEN
    assert ll[1].shape == (B, S.N_SLOTS, S.N_TYPES)
    world, heard = prediction_losses(sl, ll, frames[:, -1])
    assert world.shape == (B,) and heard.shape == (B,)
    assert bool(jnp.isfinite(world).all()) and bool(jnp.isfinite(heard).all())


def test_padding_mask_blocks_past_episodes():
    """With the earlier frames marked invalid, output must not depend on them."""
    cfg = ModelConfig(d=32, n_layers=1, n_heads=2, K=4)
    m = Learner(cfg)
    B, K = 2, cfg.K
    key = jax.random.PRNGKey(0)
    frames = jax.random.randint(key, (B, K, S.FRAME_LEN), 0, 2).astype(jnp.int32)
    prev = jnp.zeros((B, K, 3), jnp.int32)
    valid = jnp.array([[False, False, False, True]] * B)
    lp = m.init(key, frames, prev, valid)
    a = m.apply(lp, frames, prev, valid)["action"]
    frames2 = frames.at[:, :3].set(1 - frames[:, :3])     # perturb only the masked frames
    b = m.apply(lp, frames2, prev, valid)["action"]
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_gaze_metric():
    F = S.FRAME_LEN
    frames = np.zeros((4, F), np.int64)
    acts = np.array([S.TURN_RIGHT, S.TURN_LEFT, S.FORWARD, S.TURN_RIGHT])
    # frame 0: adult pointing, rel_facing 1 -> correct is TURN_RIGHT
    base = S.SELF_LEN
    frames[0, base:base + 7] = [1, S.ADULT, 5 + S.REL_X_OFFSET, 2, 1, 1, 0]
    # frame 1: adult pointing, rel_facing 3 -> correct is TURN_LEFT
    frames[1, base:base + 7] = [1, S.ADULT, 0 + S.REL_X_OFFSET, 2, 3, 1, 0]
    # frame 2: adult not pointing -> no cue
    frames[2, base:base + 7] = [1, S.ADULT, 0 + S.REL_X_OFFSET, 2, 3, 0, 0]
    m = gaze_metrics(frames, acts)
    assert m["gaze_cues"] == 2 and m["gaze_follow_rate"] == 1.0
