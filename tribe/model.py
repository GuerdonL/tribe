"""The learner — one network, three heads, two losses plus RL.

    frame (91 ints)  --row embed-->  13 rows  --set attention-->  frame vector
    K frame vectors (+ previous action)  --GTrXL decoder-->  h_t
    h_t  -->  policy (action, pointing, saying), value
    [h_t, a_t]  -->  next-frame prediction (every field of every row)

The next-frame head is the dense signal; heard_word / heard_dir are two of
its fields and get their own loss weight so word-prediction can be tracked
on its own. Nothing here knows what a word means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from . import schema as S


@dataclass(frozen=True)
class ModelConfig:
    d: int = 128                 # width
    n_layers: int = 3            # temporal decoder layers
    n_heads: int = 4
    K: int = 16                  # frames of context
    row_layers: int = 1          # set-attention layers over rows within a frame
    mlp_mult: int = 4
    dropout: float = 0.0
    remat: bool = True           # gradient checkpointing: ~3x less activation memory, ~30% slower
    arch: str = "window"         # "window": Learner, re-encodes K frames per sample; "xl": XLLearner, cached memory


SELF_CARD = tuple(S.SELF_CARD)
SLOT_CARD = tuple(S.SLOT_CARD)
OUT_CARD = tuple(S.OUT_CARD)
N_ROWS = 1 + S.N_SLOTS


class FieldEmbed(nn.Module):
    """Sum of per-field embeddings for a row of integer fields."""
    cards: Sequence[int]
    d: int

    @nn.compact
    def __call__(self, x):                       # x: (..., n_fields) ints
        out = 0.0
        for k, card in enumerate(self.cards):
            out = out + nn.Embed(card, self.d, name=f"f{k}")(x[..., k])
        return out


class RowAttention(nn.Module):
    d: int
    n_heads: int
    mlp_mult: int

    @nn.compact
    def __call__(self, rows, mask):              # rows: (..., R, d); mask: (..., R) bool
        attn_mask = mask[..., None, None, :]     # (..., 1, 1, R) keys masked
        h = nn.LayerNorm()(rows)
        a = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=self.d)(h, h, mask=attn_mask)
        rows = rows + a
        h = nn.LayerNorm()(rows)
        m = nn.Dense(self.d * self.mlp_mult)(h)
        m = nn.Dense(self.d)(nn.gelu(m))
        return rows + m


class FrameEncoder(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, frames):                  # (..., FRAME_LEN) ints -> (..., d)
        d = self.cfg.d
        me = frames[..., : S.SELF_LEN]
        slots = frames[..., S.SELF_LEN:].reshape(frames.shape[:-1] + (S.N_SLOTS, S.SLOT_LEN))
        me_e = FieldEmbed(SELF_CARD, d, name="self_embed")(me)[..., None, :]        # (..., 1, d)
        sl_e = FieldEmbed(SLOT_CARD, d, name="slot_embed")(slots)                   # (..., N_SLOTS, d)
        pos = self.param("row_pos", nn.initializers.normal(0.02), (N_ROWS, d))
        rows = jnp.concatenate([me_e, sl_e], axis=-2) + pos
        valid = jnp.concatenate([jnp.ones(frames.shape[:-1] + (1,), bool), slots[..., 0] > 0], axis=-1)
        Row = nn.remat(RowAttention) if self.cfg.remat else RowAttention
        for l in range(self.cfg.row_layers):
            rows = Row(d, self.cfg.n_heads, self.cfg.mlp_mult, name=f"row{l}")(rows, valid)
        # masked mean pool
        w = valid.astype(rows.dtype)[..., None]
        pooled = (rows * w).sum(-2) / jnp.maximum(w.sum(-2), 1.0)
        return nn.LayerNorm()(pooled)


class GRUGate(nn.Module):
    """GTrXL gating on the residual path (Parisotto et al. 2019), bias toward identity."""
    d: int
    bias: float = 2.0

    @nn.compact
    def __call__(self, x, y):
        Wr = nn.Dense(self.d, use_bias=False); Ur = nn.Dense(self.d, use_bias=False)
        Wz = nn.Dense(self.d, use_bias=False); Uz = nn.Dense(self.d, use_bias=False)
        Wg = nn.Dense(self.d, use_bias=False); Ug = nn.Dense(self.d, use_bias=False)
        bg = self.param("bg", nn.initializers.constant(self.bias), (self.d,))
        r = jax.nn.sigmoid(Wr(y) + Ur(x))
        z = jax.nn.sigmoid(Wz(y) + Uz(x) - bg)
        h = jnp.tanh(Wg(y) + Ug(r * x))
        return (1 - z) * x + z * h


class GTrXLBlock(nn.Module):
    d: int
    n_heads: int
    mlp_mult: int

    @nn.compact
    def __call__(self, x, mask):                  # x: (..., K, d); mask: (..., 1, K, K)
        h = nn.LayerNorm()(x)
        a = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=self.d)(h, h, mask=mask)
        x = GRUGate(self.d, name="gate_attn")(x, nn.relu(a))
        h = nn.LayerNorm()(x)
        m = nn.Dense(self.d)(nn.gelu(nn.Dense(self.d * self.mlp_mult)(h)))
        x = GRUGate(self.d, name="gate_mlp")(x, nn.relu(m))
        return x


class Learner(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, frames, prev_out, valid):
        """
        frames  : (B, K, FRAME_LEN) ints, oldest first; frames[:, -1] is the current frame
        prev_out: (B, K, 3) ints — the (action, pointing, saying) taken at each frame's step
                  (for the current frame this is unknown; pass zeros, it is masked below)
        valid   : (B, K) bool — False for padding before an episode's first frame
        returns dict of logits for the LAST position plus features for the prediction head
        """
        cfg = self.cfg
        d = cfg.d
        B, K = frames.shape[:2]
        x = FrameEncoder(cfg, name="frame")(frames)                                     # (B, K, d)
        # previous output embedding shifted so position t sees the output taken at t-1
        po = FieldEmbed(OUT_CARD, d, name="prev_out_embed")(prev_out)                   # (B, K, d)
        po = jnp.concatenate([jnp.zeros_like(po[:, :1]), po[:, :-1]], axis=1)
        tpos = self.param("time_pos", nn.initializers.normal(0.02), (cfg.K, d))
        x = x + po + tpos[-K:]
        causal = jnp.tril(jnp.ones((K, K), bool))
        mask = causal[None, None] & valid[:, None, None, :]                              # (B, 1, K, K)
        Blk = nn.remat(GTrXLBlock) if cfg.remat else GTrXLBlock
        for l in range(cfg.n_layers):
            x = Blk(d, cfg.n_heads, cfg.mlp_mult, name=f"blk{l}")(x, mask)
        h = nn.LayerNorm()(x)                                                           # (B, K, d)
        hl = h[:, -1]
        pi_action = nn.Dense(S.N_ACTIONS, name="pi_action")(hl)
        pi_point = nn.Dense(2, name="pi_point")(hl)
        pi_say = nn.Dense(S.N_WORDS, name="pi_say")(hl)
        value = nn.Dense(1, name="value")(hl)[..., 0]
        return {"action": pi_action, "pointing": pi_point, "saying": pi_say, "value": value, "h": hl, "h_all": h}


# ---------------------------------------------------------------- XL ---
# Transformer-XL style learner. Same frame encoder, gating and heads as Learner, but
# position is a learned relative bias over distance 0..K-1 and context is a per-layer
# memory: layer l at position i attends to layer-l inputs at positions i-K+1..i of the
# same episode. The rollout runs it one step at a time with cached memory (`step`);
# training runs a whole rollout segment at once from the memory cached at its start
# (`sequence`). Both compute the same function, so PPO's old_logp is exact.

class XLBlock(nn.Module):
    d: int
    n_heads: int
    mlp_mult: int
    K: int

    @nn.compact
    def __call__(self, xq, xkv, mask, dist):
        """xq (B,Q,d) queries; xkv (B,S,d) keys/values (memory + xq); mask (B,Q,S) bool;
        dist (Q,S) int in [0, K-1]: query position minus key position."""
        H = self.n_heads
        dh = self.d // H
        ln = nn.LayerNorm(name="ln_attn")
        hq, hk = ln(xq), ln(xkv)
        q = nn.DenseGeneral((H, dh), name="q")(hq)
        k = nn.DenseGeneral((H, dh), name="k")(hk)
        v = nn.DenseGeneral((H, dh), name="v")(hk)
        bias = self.param("rel_bias", nn.initializers.zeros, (H, self.K))
        logits = jnp.einsum("bqhd,bshd->bhqs", q, k) / jnp.sqrt(dh).astype(q.dtype) + bias[:, dist][None]
        logits = jnp.where(mask[:, None], logits, jnp.finfo(logits.dtype).min)
        a = jnp.einsum("bhqs,bshd->bqhd", jax.nn.softmax(logits, -1), v)
        a = nn.DenseGeneral(self.d, axis=(-2, -1), name="o")(a)
        x = GRUGate(self.d, name="gate_attn")(xq, nn.relu(a))
        h = nn.LayerNorm(name="ln_mlp")(x)
        m = nn.Dense(self.d, name="mlp_out")(nn.gelu(nn.Dense(self.d * self.mlp_mult, name="mlp_in")(h)))
        return GRUGate(self.d, name="gate_mlp")(x, nn.relu(m))


class XLLearner(nn.Module):
    """mem: (B, n_layers, K-1, d) layer inputs at the K-1 previous positions, oldest first.
    mem_ep: (B, K-1) their episode ids (-1 = empty). prev_out/prev_same: the output taken at
    the previous position and whether that position is in the same episode."""
    cfg: ModelConfig

    def setup(self):
        c = self.cfg
        self.frame = FrameEncoder(c)
        self.prev_out_embed = FieldEmbed(OUT_CARD, c.d)
        Blk = nn.remat(XLBlock) if c.remat else XLBlock
        self.blocks = [Blk(c.d, c.n_heads, c.mlp_mult, c.K) for _ in range(c.n_layers)]
        self.ln_f = nn.LayerNorm()
        self.pi_action = nn.Dense(S.N_ACTIONS)
        self.pi_point = nn.Dense(2)
        self.pi_say = nn.Dense(S.N_WORDS)
        self.value = nn.Dense(1)

    def _embed(self, frames, prev_out, prev_same):
        po = self.prev_out_embed(prev_out) * prev_same[..., None].astype(jnp.float32)
        return self.frame(frames) + po

    def _heads(self, x):
        h = self.ln_f(x)
        return {"action": self.pi_action(h), "pointing": self.pi_point(h), "saying": self.pi_say(h),
                "value": self.value(h)[..., 0], "h": h}

    def step(self, frame, prev_out, prev_same, mem, mem_ep, ep):
        """One position per env. frame (B,F); returns (outputs for this position, new mem)."""
        K = self.cfg.K
        x = self._embed(frame, prev_out, prev_same)[:, None]                         # (B,1,d)
        mask = jnp.concatenate([mem_ep == ep[:, None], jnp.ones_like(ep[:, None], bool)], 1)[:, None]
        dist = jnp.arange(K - 1, -1, -1)[None]                                       # (1,K)
        new_mem = []
        for l, blk in enumerate(self.blocks):
            new_mem.append(jnp.concatenate([mem[:, l, 1:], x], axis=1))
            x = blk(x, jnp.concatenate([mem[:, l], x], axis=1), mask, dist)
        return self._heads(x[:, 0]), jnp.stack(new_mem, axis=1)

    def sequence(self, frames, prev_out, prev_same, mem, mem_ep, eps):
        """T consecutive positions per env starting after mem. frames (B,T,F), eps (B,T).
        Memory is treated as fixed input (no gradient), as in the rollout that produced it."""
        K = self.cfg.K
        T = frames.shape[1]
        x = self._embed(frames, prev_out, prev_same)                                 # (B,T,d)
        kv_ep = jnp.concatenate([mem_ep, eps], axis=1)                                # (B,K-1+T)
        dist = (K - 1 + jnp.arange(T))[:, None] - jnp.arange(K - 1 + T)[None]        # (T,K-1+T)
        mask = ((dist >= 0) & (dist < K))[None] & (kv_ep[:, None, :] == eps[:, :, None])
        dist = jnp.clip(dist, 0, K - 1)
        mem = jax.lax.stop_gradient(mem)
        for l, blk in enumerate(self.blocks):
            x = blk(x, jnp.concatenate([mem[:, l], x], axis=1), mask, dist)
        return self._heads(x)

    def __call__(self, *args):
        return self.sequence(*args)


class Predictor(nn.Module):
    """Next-frame logits from hidden state h (..., d) and the output taken (..., 3).
    A separate module (Flax allows one @compact method per module); params live beside the Learner's."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, h, out):
        d = self.cfg.d
        a = FieldEmbed(OUT_CARD, d, name="pred_out_embed")(out)
        z = nn.gelu(nn.Dense(d * 2, name="pred_in")(jnp.concatenate([h, a], axis=-1)))
        z = nn.gelu(nn.Dense(d * 2, name="pred_mid")(z))
        self_logits = [nn.Dense(c, name=f"pred_self{k}")(z) for k, c in enumerate(SELF_CARD)]
        # slot logits: one dense per field, shared across the 12 slots, conditioned on slot index
        zs = z[..., None, :] + self.param("pred_slot_pos", nn.initializers.normal(0.02), (S.N_SLOTS, d * 2))
        slot_logits = [nn.Dense(c, name=f"pred_slot{k}")(zs) for k, c in enumerate(SLOT_CARD)]
        return self_logits, slot_logits


def prediction_losses(self_logits, slot_logits, next_frame):
    """Per-example cross-entropies: (world, heard). next_frame: (..., FRAME_LEN) ints."""
    me = next_frame[..., : S.SELF_LEN]
    slots = next_frame[..., S.SELF_LEN:].reshape(next_frame.shape[:-1] + (S.N_SLOTS, S.SLOT_LEN))

    def ce(logits, target):
        return -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), target[..., None], -1)[..., 0]

    heard_idx = {S.SELF_FIELDS.index("heard_word"), S.SELF_FIELDS.index("heard_dir")}
    world = 0.0
    heard = 0.0
    for k in range(S.SELF_LEN):
        l = ce(self_logits[k], me[..., k])
        if k in heard_idx:
            heard = heard + l
        else:
            world = world + l
    # slot fields: validity always; other fields only where the slot is valid in the target
    valid = (slots[..., 0] > 0).astype(jnp.float32)
    for k in range(S.SLOT_LEN):
        l = ce(slot_logits[k], slots[..., k])                   # (..., N_SLOTS)
        if k == 0:
            world = world + l.sum(-1)
        else:
            world = world + (l * valid).sum(-1)
    return world, heard
