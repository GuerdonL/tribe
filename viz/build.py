"""Build the data the visualizer serves: for every saved model in the results repo, play it on a fixed
set of worlds and record every step.

    .venv/bin/python viz/build.py                 # sync with the Hugging Face repo, record what is new
    .venv/bin/python viz/build.py --only m1_gpu_xl_s1 --limit 2
    .venv/bin/python viz/build.py --offline       # only what is already in runs/

A "model" is one saved set of weights: a run's snapshot at some update (snap/uNNNNN.pkl), or its latest
weights (params.pkl). Every model plays the SAME worlds (same seeds, so the same maps), which is what lets
the site superimpose different models on one map. Runs on CPU; nothing here rents anything.

Writes viz/data/ (git-ignored):
    index.json            runs, models (with per-model statistics), field layout
    worlds.json           the maps
    log/<run>.json        the training log, by column
    ep/<model id>.json    the recorded episodes of one model
"""
import argparse
import base64
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import numpy as np

DATA = os.path.join(HERE, "data")
RUNS = os.path.join(ROOT, "runs")
N_WORLDS = 32            # changing this changes the maps (the seed split depends on it): re-record everything
ENV_SEED = 0
ACT_SEED = 1
N_FOOD = 6
VERSION = 2              # bump to force re-recording

FIELDS = ["lx", "ly", "lf", "act", "lpoint", "lsay", "alive", "sheltered", "hunger", "thirst", "warmth",
          "ax", "ay", "af", "apoint", "asay", "px", "py", "pactive", "heard", "heard_dir"] + [f"food{i}" for i in range(N_FOOD)]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def clean(x, nd=5):
    """float -> JSON-safe number with nd significant digits (None for nan/inf)."""
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{nd}g}")


# ------------------------------------------------------------ the repo ---
def list_remote(repo):
    """{run: {relative path: fingerprint}} for every file in the results repo."""
    from huggingface_hub import HfApi
    out = {}
    for f in HfApi().list_repo_tree(repo, recursive=True):
        if not hasattr(f, "size") or "/" not in f.path or f.path.startswith("_boxboy/"):
            continue
        run, rel = f.path.split("/", 1)
        lfs = getattr(f, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        out.setdefault(run, {})[rel] = sha or f.blob_id
    return out


def list_local():
    out = {}
    if not os.path.isdir(RUNS):
        return out
    for run in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, run)
        if not os.path.isdir(d) or run.startswith((".", "_")):
            continue
        for base, _, files in os.walk(d):
            if ".cache" in base:
                continue
            for fn in files:
                p = os.path.join(base, fn)
                rel = os.path.relpath(p, d)
                out.setdefault(run, {})[rel] = f"local:{os.path.getsize(p)}:{int(os.path.getmtime(p))}"
    return out


def fetch(repo, run, rel):
    """Local path of runs/<run>/<rel>, downloaded if a repo is given."""
    path = os.path.join(RUNS, run, rel)
    if repo:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, f"{run}/{rel}", local_dir=RUNS)
    return path


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in (rows[0].keys() if rows else []):
        try:
            cols[k] = [clean(r[k]) if r[k] not in ("", None) else None for r in rows]
        except ValueError:
            pass
    return cols


def read_snap_eval(path):
    """{weights file name: {column: value}} from a snap_eval.csv (rows are keyed by `file`: u00025.pkl, params.pkl).
    Sample counts (n_*) are left out."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            name = r.get("file")
            if not name:
                continue
            vals = {}
            for k, v in r.items():
                if k == "file" or k.startswith("n_"):
                    continue
                try:
                    vals[k] = clean(v)
                except (ValueError, TypeError):
                    pass
            out[os.path.basename(name)] = vals
    return out


# ------------------------------------------------------------ recording ---
class Recorder:
    """Plays models. One compile per distinct (model shape, world variant); weights are an argument."""

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import evallib as E
        from tribe import schema as S
        from tribe import jax_world as J
        self.jax, self.jnp, self.E, self.S, self.J = jax, jnp, E, S, J
        self.rollouts = {}

    def worlds(self):
        jax, J, S = self.jax, self.J, self.S
        env = J.reset_batch(jax.random.split(jax.random.PRNGKey(ENV_SEED), N_WORLDS), 1, 1)
        terrain = np.asarray(env.terrain, dtype=np.uint8)
        lair = np.asarray(env.predator_lair)
        ws = []
        for i in range(N_WORLDS):
            food = np.flatnonzero(terrain[i].reshape(-1) == S.FOOD)[:N_FOOD]
            ws.append(dict(terrain=b64(terrain[i]), lair=[int(lair[i, 0]), int(lair[i, 1])], food=[int(x) for x in food]))
        return dict(grid=S.GRID, n=N_WORLDS, env_seed=ENV_SEED, worlds=ws)

    def _extra(self, apply, frame, obs, logits, ctx):
        jnp, S, E = self.jnp, self.S, self.E
        e, out = ctx["env"], ctx["out"]
        n = e.t.shape[0]
        q = lambda x: jnp.clip(jnp.round(x * 255.0), 0, 255)
        fm = (e.terrain == S.FOOD).reshape(n, -1)
        idx = jnp.argsort(~fm, axis=1, stable=True)[:, :N_FOOD]       # food cells in row-major order
        stock = jnp.take_along_axis(e.food_stock.reshape(n, -1), idx, axis=1)
        cols = [e.pos[:, 0, 0], e.pos[:, 0, 1], e.facing[:, 0], out[:, 0], out[:, 1], out[:, 2],
                e.alive[:, 0], e.sheltered[:, 0], q(e.hunger[:, 0]), q(e.thirst[:, 0]), q(e.warmth[:, 0]),
                e.pos[:, 1, 0], e.pos[:, 1, 1], e.facing[:, 1], e.pointing[:, 1], e.saying[:, 1],
                e.predator_pos[:, 0], e.predator_pos[:, 1], e.predator_active,
                frame[:, E.I_HW], frame[:, E.I_HD]] + [q(stock[:, i]) for i in range(N_FOOD)]
        return dict(viz=jnp.stack([c.astype(jnp.uint8) for c in cols], axis=1))

    def record(self, pkl):
        """-> (cfg, episodes) where episodes is a list of dicts, one per world."""
        E, S, jax = self.E, self.S, self.jax
        cfg, params = E.load(pkl)
        key = (repr(cfg.model), getattr(cfg, "night_drain", None), getattr(cfg, "predator_move_every", None),
               getattr(cfg, "warmth_reward", None), getattr(cfg, "r_cold", None))
        if key not in self.rollouts:
            self.rollouts[key] = E.make_rollout(cfg, extra=self._extra, reset=False)
        rec = self.rollouts[key](params, jax.random.PRNGKey(ACT_SEED), N_WORLDS, S.EPISODE_STEPS, env_seed=ENV_SEED)
        viz = rec["viz"]                                   # (T, N, F)
        assert viz.shape[2] == len(FIELDS), viz.shape
        eps = []
        for i in range(N_WORLDS):
            done = rec["done"][:, i]
            steps = int(np.argmax(done)) + 1 if done.any() else S.EPISODE_STEPS
            cause = int(rec["cause"][steps - 1, i])
            v = viz[:steps, i]
            hung, thir, warm = (rec[k][:steps, i] for k in ("hunger", "thirst", "warmth"))
            night = rec["phase"][:steps, i] == S.NIGHT
            lx, ly, ax, ay = (v[:, FIELDS.index(k)].astype(int) for k in ("lx", "ly", "ax", "ay"))
            acts = np.bincount(v[:, FIELDS.index("act")], minlength=S.N_ACTIONS)[: S.N_ACTIONS]
            eps.append(dict(
                world=i, steps=steps, cause=E.CAUSES[cause], survived=cause == 0,
                ret=clean(rec["reward"][:steps, i].sum()),
                hunger=clean(hung.mean()), thirst=clean(thir.mean()), warmth=clean(warm.mean()),
                night_sheltered=clean((rec["sheltered"][:steps, i] & night).sum() / max(int(night.sum()), 1)),
                eats=int((np.diff(hung) > 0).sum()), drinks=int((np.diff(thir) > 0).sum()),
                adult_dist=clean((np.abs(lx - ax) + np.abs(ly - ay)).mean()),
                cells=int(len(set(zip(lx.tolist(), ly.tolist())))),
                heard_frac=clean((v[:, FIELDS.index("heard")] > 0).mean()),
                actions=[int(a) for a in acts],
                data=b64(v)))
        return cfg, eps


def model_stats(eps, S):
    """Per-model statistics over its recorded episodes."""
    f = lambda k: np.array([e[k] for e in eps], dtype=float)
    acts = np.array([e["actions"] for e in eps]).sum(0)
    st = dict(
        ep_steps=clean(f("steps").mean()), ep_survived=clean(np.mean([e["survived"] for e in eps])),
        ep_return=clean(f("ret").mean()), ep_hunger=clean(f("hunger").mean()), ep_thirst=clean(f("thirst").mean()),
        ep_warmth=clean(f("warmth").mean()), ep_night_sheltered=clean(f("night_sheltered").mean()),
        ep_eats=clean(f("eats").mean()), ep_drinks=clean(f("drinks").mean()),
        ep_adult_dist=clean(f("adult_dist").mean()), ep_cells=clean(f("cells").mean()),
        ep_heard_frac=clean(f("heard_frac").mean()))
    for c in ("hunger", "thirst", "cold", "hazard", "predator"):
        st[f"ep_died_{c}"] = clean(np.mean([e["cause"] == c for e in eps]))
    for i, name in enumerate(S.ACTION_NAMES):
        st[f"ep_act_{name}"] = clean(acts[i] / max(acts.sum(), 1))
    return st


def train_stats(log, update, window=10):
    """Mean of each training-log column over the `window` updates ending at `update`."""
    if not log or "update" not in log:
        return {}
    us = log["update"]
    idx = [i for i, u in enumerate(us) if u is not None and update - window < u <= update]
    out = {}
    for k, col in log.items():
        if k in ("update", "steps", "elapsed_s"):
            continue
        vals = [col[i] for i in idx if col[i] is not None]
        if vals:
            out[f"train_{k}"] = clean(np.mean(vals))
    if "train_gaze_follow_rate" in out and "train_gaze_chance" in out:
        out["train_gaze_gap"] = clean(out["train_gaze_follow_rate"] - out["train_gaze_chance"])
    return out


# ----------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="GuerdonL/tribe-runs")
    ap.add_argument("--offline", action="store_true", help="use only what is already in runs/")
    ap.add_argument("--only", nargs="*", default=None, help="run names to build")
    ap.add_argument("--limit", type=int, default=None, help="record at most this many new models, then stop")
    ap.add_argument("--force", action="store_true", help="record again even if unchanged")
    a = ap.parse_args()
    repo = None if a.offline else a.repo
    t0 = time.time()

    remote = list_local() if a.offline else list_remote(a.repo)
    old = {}
    try:
        prev = json.load(open(os.path.join(DATA, "index.json")))
        if prev.get("version") == VERSION and prev.get("n_worlds") == N_WORLDS:
            old = {m["id"]: m for m in prev["models"]}
            old_runs = {r["name"]: r for r in prev["runs"]}
    except (OSError, ValueError, KeyError):
        pass
    old_runs = locals().get("old_runs", {})

    rec = None
    runs, models, new = [], [], 0
    for run in sorted(remote):
        files = remote[run]
        if a.only and run not in a.only:
            if run in old_runs:                              # keep what was built before
                runs.append(old_runs[run])
                models += [m for m in old.values() if m["run"] == run]
            continue
        weights = sorted(r for r in files if r.startswith("snap/") and r.endswith(".pkl"))
        if "params.pkl" in files:
            weights.append("params.pkl")
        if not weights:
            continue
        log, snap_eval = {}, {}
        if "log.csv" in files:
            log = read_log(fetch(repo, run, "log.csv"))
            write_json(os.path.join(DATA, "log", f"{run}.json"), log)
        if "snap_eval.csv" in files:
            snap_eval = read_snap_eval(fetch(repo, run, "snap_eval.csv"))
        status = None
        if "STATUS.json" in files:
            try:
                status = json.load(open(fetch(repo, run, "STATUS.json")))
            except (OSError, ValueError):
                pass
        last_update = int(max((u for u in log.get("update", []) if u is not None), default=0))
        run_info = dict(name=run, updates=last_update, finished=bool(status and status.get("exit_code") == 0),
                        exit_code=status.get("exit_code") if status else None, has_log=bool(log))
        snap_updates = set()
        for rel in weights:
            final = rel == "params.pkl"
            update = last_update if final else int("".join(ch for ch in os.path.basename(rel) if ch.isdigit()))
            if final and update in snap_updates:
                continue                                   # the final weights are also a snapshot
            snap_updates.add(update)
            mid = f"{run}@final" if final else f"{run}@u{update:05d}"
            m = old.get(mid)
            ep_path = os.path.join(DATA, "ep", f"{mid}.json")
            fresh = m and m.get("sha") == files[rel] and os.path.exists(ep_path) and not a.force
            if not fresh:
                if a.limit is not None and new >= a.limit:
                    if m and os.path.exists(ep_path):
                        models.append(m)                   # keep the older recording for now
                    continue
                if rec is None:
                    rec = Recorder()
                    write_json(os.path.join(DATA, "worlds.json"), rec.worlds())
                t1 = time.time()
                cfg, eps = rec.record(fetch(repo, run, rel))
                write_json(ep_path, dict(id=mid, fields=FIELDS, episodes=eps))
                m = dict(id=mid, run=run, final=final, sha=files[rel], stats=model_stats(eps, rec.S),
                         episodes=[{k: v for k, v in e.items() if k not in ("data", "actions")} for e in eps],
                         world=dict(night_drain=clean(getattr(cfg, "night_drain", 1 / 35)),
                                    warmth_reward=getattr(cfg, "warmth_reward", "recover"),
                                    predator_move_every=int(getattr(cfg, "predator_move_every", 2))),
                         seed=int(getattr(cfg, "seed", 0)), lr=clean(getattr(cfg, "lr", 0)))
                new += 1
                print(f"recorded {mid}: mean {m['stats']['ep_steps']:.0f} steps, "
                      f"{100 * m['stats']['ep_survived']:.0f}% survived ({time.time() - t1:.0f} s)", flush=True)
            m["update"] = update
            m["stats"] = {k: v for k, v in m["stats"].items() if k.startswith("ep_")}
            m["stats"].update(train_stats(log, update))
            m["stats"].update({f"eval_{k}": v for k, v in snap_eval.get(os.path.basename(rel), {}).items() if v is not None})
            models.append(m)
        runs.append(run_info)

    from tribe import schema as S
    import evallib as E
    index = dict(version=VERSION, built_utc=time.strftime("%FT%TZ", time.gmtime()), n_worlds=N_WORLDS,
                 fields=FIELDS, runs=runs, models=sorted(models, key=lambda m: (m["run"], m["update"], m["final"])),
                 schema=dict(grid=S.GRID, episode_steps=S.EPISODE_STEPS, day_length=S.DAY_LENGTH,
                             dusk_start=S.DUSK_START, night_start=S.NIGHT_START, view_depth=S.VIEW_DEPTH,
                             view_half=S.VIEW_HALF, hear_radius=S.HEAR_RADIUS, types=S.TYPE_NAMES,
                             actions=S.ACTION_NAMES, words=S.WORDS, causes=E.CAUSES, facing_dxy=S.FACING_DXY))
    write_json(os.path.join(DATA, "index.json"), index)
    keep = {m["id"] + ".json" for m in models}
    ep_dir = os.path.join(DATA, "ep")
    for fn in os.listdir(ep_dir) if os.path.isdir(ep_dir) else []:
        if fn not in keep and not a.only:
            os.remove(os.path.join(ep_dir, fn))
    print(f"{len(models)} models in {len(runs)} runs; {new} recorded now; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
