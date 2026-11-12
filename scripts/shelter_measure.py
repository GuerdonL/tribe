"""How much do trained learners shelter at night? Rolls each model out on fresh episodes and reports sheltering
and survival with bootstrap 95% intervals. Runs on CPU (about 3 min per model at 512 episodes).
usage: python scripts/shelter_measure.py <out.json> [weights.pkl ...]
With no weights given: the update-600 snapshots of the shelter grid and of the two M1 seeds, from runs/."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax
import evallib as E
from tribe import schema as S
N = 512
out = {}
DEFAULT = [f"runs/{r}/snap/u00600.pkl" for r in ["m1_gpu_xl_s1", "m1_gpu_xl_s2", "sh_rec_d22", "sh_def_d35", "sh_def_d22", "sh_rec_d15", "sh_def_d15"]]
for path in sys.argv[2:] or DEFAULT:
    t0 = time.time()
    run = path
    cfg, params = E.load(path)
    rec = E.make_rollout(cfg, reset=False)(params, jax.random.PRNGKey(7), N, S.EPISODE_STEPS, env_seed=11)
    done = rec["done"]; steps = np.where(done.any(0), done.argmax(0) + 1, S.EPISODE_STEPS)
    T = np.arange(S.EPISODE_STEPS)[:, None]
    live = T < steps[None, :]
    night = (rec["phase"] == S.NIGHT) & live
    sh = rec["sheltered"].astype(bool) & live
    tod = T % S.DAY_LENGTH
    n1 = (T >= S.NIGHT_START) & (T < S.DAY_LENGTH)                      # the first night, steps 60..79
    reached_n1 = steps > S.NIGHT_START
    cause = rec["cause"][steps - 1, np.arange(N)]
    r = dict(
        night_sheltered=E.boot_ci((sh & night).sum(0) / np.maximum(night.sum(0), 1)),
        any_shelter_night1=E.boot_ci(((sh & n1).any(0))[reached_n1].astype(float)),
        sheltered_at_step_70=E.boot_ci(sh[70][steps > 70].astype(float)) if (steps > 70).any() else None,
        in_shelter_by_dusk_end=E.boot_ci(sh[S.NIGHT_START][reached_n1].astype(float)),
        day_sheltered=E.boot_ci((sh & live & (rec["phase"] == S.DAY)).sum(0) / np.maximum((live & (rec["phase"] == S.DAY)).sum(0), 1)),
        survived=E.boot_ci((cause == 0).astype(float)), survived_night1=E.boot_ci((steps > S.DAY_LENGTH).astype(float)),
        died_cold=float((cause == 3).mean()), died_predator=float((cause == 5).mean()), died_thirst=float((cause == 2).mean()),
        died_hunger=float((cause == 1).mean()), died_hazard=float((cause == 4).mean()), mean_steps=float(steps.mean()),
        enter_share=float((rec["act"][live] == S.ENTER).mean()),
        cold_death_given_sheltered_n1=float((cause == 3)[reached_n1 & (sh & n1).any(0)].mean()) if (reached_n1 & (sh & n1).any(0)).any() else None,
        cold_death_given_not=float((cause == 3)[reached_n1 & ~(sh & n1).any(0)].mean()) if (reached_n1 & ~(sh & n1).any(0)).any() else None,
        world=dict(night_drain=cfg.night_drain, warmth_reward=cfg.warmth_reward))
    out[run] = {k: ([float(x) for x in v] if isinstance(v, tuple) else v) for k, v in r.items()}
    print(run, f"{time.time() - t0:.0f}s", {k: (round(v[0], 3) if isinstance(v, list) else v) for k, v in out[run].items() if k != "world"}, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
