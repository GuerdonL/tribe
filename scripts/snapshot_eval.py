"""Q1: how does reliance on the adult change over training? For each snapshot: survival with parts of
the adult hidden from the learner, and the causal effect of the adult's facing on the learner's turn.
usage: python scripts/snapshot_eval.py <run_dir or params.pkl ...> [--envs 2048]
Appends one row per snapshot to <run_dir>/snap_eval.csv (skips snapshots already there)."""
import argparse
import csv
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np

import evallib as E
from tribe import schema as S

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--envs", type=int, default=2048)
args = ap.parse_args()
CONDS = ["normal", "no_gaze", "no_words", "no_adult"]


def extra(apply, frame, obs, logits, ctx):
    me, sl = E.split(frame)
    am = E.adult_mask(sl)
    first = jnp.argmax(am, 1)
    s = sl[jnp.arange(frame.shape[0]), first]
    idx = E.SO + first * S.SLOT_LEN + E.F_FACE
    cur = frame[jnp.arange(frame.shape[0]), idx]
    flipped = frame.at[jnp.arange(frame.shape[0]), idx].set(jnp.where(cur == 1, 3, jnp.where(cur == 3, 1, cur)))
    p, pf = jax.nn.softmax(logits["action"]), jax.nn.softmax(apply(flipped)["action"])
    ent = -(p * jnp.log(p + 1e-9)).sum(-1)
    return dict(word=me[:, E.I_HW], has=am.any(1), face=s[:, E.F_FACE], point=s[:, E.F_POINT], pr=p[:, S.TURN_RIGHT], pl=p[:, S.TURN_LEFT],
                fr=pf[:, S.TURN_RIGHT], fl=pf[:, S.TURN_LEFT], ent=ent, pfwd=p[:, S.FORWARD])


runs = {}


def evaluate(path):
    cfg, params = E.load(path)
    if "fns" not in runs:
        runs["fns"] = {c: E.make_rollout(cfg, E.get_transform(c), extra=extra if c == "normal" else None) for c in CONDS}
    row = {}
    for c in CONDS:
        rec = runs["fns"][c](params, jax.random.PRNGKey(7), args.envs, S.EPISODE_STEPS, env_seed=11)
        fe = E.first_episode(rec)
        row[f"died_{c}"], row[f"steps_{c}"] = fe["died"].mean(), fe["steps"].mean()
        if c == "normal":
            base = fe
            side = rec["has"] & np.isin(rec["face"], [1, 3])
            R = rec["face"] == 1
            match = np.where(R, rec["pr"], rec["pl"]); other = np.where(R, rec["pl"], rec["pr"])
            fmatch = np.where(R, rec["fr"], rec["fl"]); fother = np.where(R, rec["fl"], rec["fr"])
            eff = (match - other) - (fmatch - fother)
            wd = rec["word"]
            fw = np.isin(wd, [S.WORD_ID["food"], S.WORD_ID["water"]])
            for name, m in (("all", side), ("point", side & (rec["point"] == 1)), ("nopoint", side & (rec["point"] == 0)),
                            ("fw", side & fw), ("home", side & (wd == S.WORD_ID["home"])),
                            ("danger", side & (wd == S.WORD_ID["danger"])),
                            ("quiet", side & np.isin(wd, [S.NO_WORD, S.WORD_ID["look"]]))):
                row[f"n_{name}"] = int(m.sum())
                row[f"flip_{name}"] = eff[m].mean() if m.any() else np.nan
                row[f"pmatch_{name}"] = match[m].mean() if m.any() else np.nan
                row[f"pother_{name}"] = other[m].mean() if m.any() else np.nan
            a = rec["act"]
            row["p_turn"] = np.mean((a == S.TURN_LEFT) | (a == S.TURN_RIGHT))
            row["p_forward"] = np.mean(a == S.FORWARD)
            row["adult_visible"] = rec["has"].mean()
            row["ent_action"] = rec["ent"].mean()
        else:
            m, lo, hi = E.boot_ci(fe["died"].astype(float) - base["died"].astype(float), n=1000)
            row[f"dgap_{c}"], row[f"dgap_{c}_lo"], row[f"dgap_{c}_hi"] = m, lo, hi
    return row


for p in args.paths:
    if os.path.isdir(p):
        files, out = sorted(glob.glob(os.path.join(p, "snap", "u*.pkl"))), os.path.join(p, "snap_eval.csv")
    else:
        files, out = [p], os.path.join(os.path.dirname(p), "snap_eval.csv")
    done = set()
    if os.path.exists(out):
        done = {r["file"] for r in csv.DictReader(open(out))}
    for f in files:
        if os.path.basename(f) in done:
            continue
        row = {"file": os.path.basename(f), **{k: (round(float(v), 5) if not isinstance(v, int) else v) for k, v in evaluate(f).items()}}
        new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{f}: died normal {row['died_normal']:.3f} no_gaze {row['died_no_gaze']:.3f} no_words {row['died_no_words']:.3f} "
              f"no_adult {row['died_no_adult']:.3f} | flip all {row['flip_all']:+.3f} point {row['flip_point']:+.3f} "
              f"| food/water {row['flip_fw']:+.3f} home {row['flip_home']:+.3f} danger {row['flip_danger']:+.3f} quiet {row['flip_quiet']:+.3f} | turn {row['p_turn']:.2f} fwd {row['p_forward']:.2f}", flush=True)
