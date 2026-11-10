"""Q4b: is "food" tied to food? Swap the words food <-> water in everything the learner hears, for whole
episodes, on the same worlds. The adult still leads a hungry learner to food while (now) saying "water".
If the model's expectations follow the word, it should expect water tiles; if they follow its own need
and the adult's lead, nothing changes.
usage: python scripts/word_ground_eval.py <params.pkl> [n_envs]"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import evallib as E
from tribe import schema as S
from tribe.model import Predictor

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
cfg, params = E.load(path)
pred = Predictor(cfg.model)


def extra(apply, frame, obs, logits, ctx):
    _, sl = pred.apply(ctx["params"]["predictor"], logits["h"], ctx["out"])
    pv = jax.nn.softmax(sl[E.F_VALID])[..., 1]                       # (N, slots)
    pt = jax.nn.softmax(sl[E.F_TYPE])                                # (N, slots, types)
    exp = (pv[..., None] * pt).sum(1)                                # expected count of each type next frame
    p = jax.nn.softmax(logits["action"])
    me, slots = E.split(frame)
    return dict(true_word=me[:, E.I_HW], exp_food=exp[:, S.FOOD], exp_water=exp[:, S.WATER],
                p_eat=p[:, S.EAT], p_drink=p[:, S.DRINK],
                n_food=((slots[..., 0] > 0) & (slots[..., 1] == S.FOOD)).sum(1),
                n_water=((slots[..., 0] > 0) & (slots[..., 1] == S.WATER)).sum(1))


res = {c: E.make_rollout(cfg, E.get_transform(c), extra=extra)(params, jax.random.PRNGKey(7), N, S.EPISODE_STEPS, env_seed=11)
       for c in ("normal", "swap_fw")}
print(f"{path}: {N} worlds x {S.EPISODE_STEPS} steps. Steps grouped by the word the adult REALLY said; no food or water in view")
print(f"{'adult says':>10} {'learner hears':>14} {'n':>9} {'expect food':>12} {'expect water':>13} {'food - water':>13} {'P(eat)':>8} {'P(drink)':>9}")
for word in ("food", "water"):
    for c in ("normal", "swap_fw"):
        r = res[c]
        m = (r["true_word"] == S.WORD_ID[word]) & (r["n_food"] == 0) & (r["n_water"] == 0)
        heard = word if c == "normal" else ("water" if word == "food" else "food")
        d = r["exp_food"][m] - r["exp_water"][m]
        print(f"{word:>10} {heard:>14} {m.sum():9,} {r['exp_food'][m].mean():12.4f} {r['exp_water'][m].mean():13.4f} "
              f"{d.mean():+13.4f} {r['p_eat'][m].mean():8.4f} {r['p_drink'][m].mean():9.4f}")
