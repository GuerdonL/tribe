"""Q4: what does a low `heard` cross-entropy mean? Compare the model's next-word prediction with
count-based baselines that see no world at all (only what the learner hears, does and feels),
and split the loss by what kind of step it is.
usage: python scripts/heard_eval.py <params.pkl> [n_envs] [steps]"""
import sys
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

import evallib as E
from tribe import schema as S
from tribe.model import Predictor

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 640
cfg, params = E.load(path)
pred = Predictor(cfg.model)


def extra(apply, frame, obs, logits, ctx):
    sl, _ = pred.apply(ctx["params"]["predictor"], logits["h"], ctx["out"])
    return dict(me=frame[:, : S.SELF_LEN].astype(jnp.int8),
                lw=jax.nn.log_softmax(sl[E.I_HW]).astype(jnp.float16), ld=jax.nn.log_softmax(sl[E.I_HD]).astype(jnp.float16))


rec = E.make_rollout(cfg, extra=extra)(params, jax.random.PRNGKey(3), N, STEPS, env_seed=5)
me = rec["me"].astype(np.int64)                         # (T, N, 7)
T = STEPS - 1
ok = ~rec["done"][:T]                                    # next frame is the same episode
w, d = me[:T, :, E.I_HW], me[:T, :, E.I_HD]
w2, d2 = me[1:, :, E.I_HW], me[1:, :, E.I_HD]
ce_w = -np.take_along_axis(rec["lw"][:T].astype(np.float64), w2[..., None], -1)[..., 0]
ce_d = -np.take_along_axis(rec["ld"][:T].astype(np.float64), d2[..., None], -1)[..., 0]

# steps since the last word, and what it was (reset at episode start)
since = np.zeros((STEPS, N), np.int64); lastw = np.zeros((STEPS, N), np.int64)
c = np.full(N, 10); lw = np.zeros(N, np.int64)
for t in range(STEPS):
    heard = me[t, :, E.I_HW] != 0
    c = np.where(heard, 0, np.minimum(c + 1, 10)); lw = np.where(heard, me[t, :, E.I_HW], lw)
    since[t], lastw[t] = c, lw
    c = np.where(rec["done"][t], 10, c); lw = np.where(rec["done"][t], 0, lw)

feats = dict(w=w, d=d, a=rec["act"][:T], hu=me[:T, :, 0], th=me[:T, :, 1], ph=me[:T, :, 4], c=since[:T], lw=lastw[:T])
y = w2 * 5 + d2
train = np.zeros((T, N), bool); train[:, : N // 2] = True
tr, te = ok & train, ok & ~train


def table_ce(names, backoff=None, alpha=0.5):
    """held-out CE (nats) of the joint (word, dir) from a count table on the named features"""
    key = np.zeros((T, N), np.int64)
    for n in names:
        key = key * 32 + feats[n]
    cnt = defaultdict(lambda: np.zeros(S.N_WORDS * 5))
    ks, ys = key[tr], y[tr]
    order = np.argsort(ks, kind="stable")
    ks, ys = ks[order], ys[order]
    uniq, start = np.unique(ks, return_index=True)
    for u, a, b in zip(uniq, start, list(start[1:]) + [len(ks)]):
        cnt[u] = np.bincount(ys[a:b], minlength=S.N_WORDS * 5).astype(float)
    prior = np.bincount(y[tr], minlength=S.N_WORDS * 5) + 1e-3
    prior = prior / prior.sum()
    kt, yt = key[te], y[te]
    uq, inv = np.unique(kt, return_inverse=True)
    P = np.stack([(cnt[u] + alpha * prior) / (cnt[u].sum() + alpha) if u in cnt else prior for u in uq])
    return -np.log(P[inv, yt]), P[inv]


print(f"{path}: {ok.sum():,} steps; silent next step in {np.mean(w2[ok]==0)*100:.1f}%")
print(f"\nheld-out cross-entropy of the next (heard_word, heard_dir), nats  [chance = ln 23 + ln 5 = {np.log(23)+np.log(5):.2f}]")
rows = [("nothing (overall frequencies)", []),
        ("current word + direction", ["w", "d"]),
        ("+ own action", ["w", "d", "a"]),
        ("+ hunger, thirst, phase", ["w", "d", "a", "hu", "th", "ph"]),
        ("+ steps since last word, last word", ["w", "d", "a", "hu", "th", "ph", "c", "lw"])]
base = {}
for name, f in rows:
    ce, _ = table_ce(f)
    base[name] = ce
    print(f"  table on {name:38s} {ce.mean():.3f}")
m_te = (ce_w + ce_d)[te]
print(f"  {'model (sees the world, K=32 memory)':47s} {m_te.mean():.3f}   (word {ce_w[te].mean():.3f} + dir {ce_d[te].mean():.3f})")

best = base[rows[-1][0]]
kinds = [("silence -> silence", (w == 0) & (w2 == 0)), ("silence -> word (onset)", (w == 0) & (w2 != 0)),
         ("word -> same word", (w != 0) & (w2 == w)), ("word -> other word", (w != 0) & (w2 != 0) & (w2 != w)),
         ("word -> silence", (w != 0) & (w2 == 0))]
print(f"\nby kind of step (held-out half): share of steps, model CE, best no-world table CE, share of model's total loss")
tot = m_te.sum()
for name, k in kinds:
    k = k[te]
    print(f"  {name:24s} {k.mean()*100:5.1f}%   model {m_te[k].mean():.3f}   table {best[k].mean():.3f}   {m_te[k].sum()/tot*100:5.1f}% of loss")

print("\nword identity only, at steps where a word is heard next (model word-CE vs table), by word")
wt, cw = w2[te], ce_w[te]
_, P = table_ce(rows[-1][1])
Pw = P.reshape(len(P), S.N_WORDS, 5).sum(-1)
tw = -np.log(Pw[np.arange(len(wt)), wt])
for i, name in enumerate(S.WORDS):
    k = wt == i
    if i and k.sum() >= 200:
        print(f"  {name:8s} n={k.sum():8,}  model {cw[k].mean():.3f}   table {tw[k].mean():.3f}")
