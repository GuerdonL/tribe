# CLAUDE.md — tribe

Read this first. It is the handoff from the session that built M0 and M1.

## What this project is

A test of one claim: an agent trained first on pre-verbal social survival tasks (joint attention, intent reading, joint commitment, testimony) will acquire grounded language from speaking "adults" where a randomly initialized RL agent cannot. Agents never invent language; they learn it from scripted (later model) adults inside a world they already understand. The full spec is in a Claude doc the owner has; the README summarizes the architecture and the frame schema.

Owner: Guerdon (GuerdonL on GitHub). He wants to move fast and prefers decisions made and flagged over questions asked. Commit messages end with a Co-Authored-By line for Claude.

## Where things stand

- **M0 passed** (hand-written policies, no model): gaze-following learner outlives gaze-ignoring one, +37.5 steps over 300 paired episodes, 95% CI [+27.7, +47.3]. Reproduced in the JAX world (+33.6, CI [+20.6, +46.3]).
- **M1 passed** (2026-11-10, branch `xl-memory`, `--preset gpu_xl`, 3000 updates / 393M steps in 2.9 h on a 4090; log `runs/m1_gpu_xl/log.csv`, weights `params.pkl`). `gaze_follow_rate` was above `gaze_chance` in every one of the last 500 updates (gap +0.071, per-update sd 0.011). Causal check (`scripts/gaze_eval.py`, 41k cue steps): flipping only the adult's observed facing, same memory, shifts the matching-turn preference by +0.11 / +0.12 (95% CIs +0.106..+0.116 and +0.117..+0.128). Facing predicts the turn within every adult-position stratum, so it is not "turn toward the adult".
- **Open question from M1:** the gap peaked at +0.23 around updates 200-300 and settled near +0.05..+0.07, while deaths per update fell 124 -> 7. Likely the learner leans on the adult less as it survives on its own; not tested. If M2 needs strong reliance on adults, this is the thing to look at (reward pressure, or cue value in harder worlds).
- Checkpoints pickle `cfg` as `__main__.TrainConfig` (training runs with `-m`); scripts that load them alias it (see `scripts/gaze_eval.py`).
- **Why not `--preset gpu`:** the windowed learner fits (37 GB with remat) but re-encodes a K=32 window for every rollout step and every training sample, about 120 s/update (about 100 h for 3000). It was stopped at update 11; its partial log is in `runs/m1_gpu/`. `arch="xl"` (`XLLearner`, `make_train_xl`) caches per-layer memory in the rollout and runs each env's segment once in training; `tests/test_train_xl.py` checks the training pass reproduces rollout log-probs exactly. Position is a relative bias, not absolute-in-window, so the two architectures are not checkpoint-compatible.
- CPU smoke run (300k params, 150k steps) trained stably: world CE 52 -> 19, heard CE 4.7 -> 0.7, gaze metric at chance. That only proves the plumbing.

## M1 pass criterion

In `runs/m1_gpu_xl/log.csv`, `gaze_follow_rate` reliably above `gaze_chance`. The metric: among steps where a pointing adult is visible and faces the learner's left or right, the fraction of learner actions that are the matching turn; chance is the learner's base rate of that turn. Also watch `gaze_cues` (how often the cue even occurs per update; if it is tiny the metric is noise and the fix is on the adult side, in `jax_world.adult_act`, not the model) and `heard` (word-prediction CE; it should fall well below ln(23) = 3.1 as the model learns what adults say when).

Measured on a 4090: about 1 min compile, then about 3.5 s/update for `gpu_xl` (3000 updates is about 3 h). Checkpoints every 50 updates to `ckpt.pkl`; `--resume` continues. XL minibatches are groups of envs (`n_envs` must divide by `minibatches`; 32 means 4096 samples per gradient step).

## If M1 fails (gaze stays at chance)

In order of suspicion:
1. Reward scale: `R_ALIVE` (0.02/step) may dominate; the agent can "succeed" by wandering. Try `R_ALIVE=0.0` with `R_NEED` kept, or a per-step penalty proportional to need deficit.
2. Prediction weight: `w_world` / `w_heard`. If world CE plateaus high, raise them; if PPO stalls while CE falls, raise `w_pi` or `lr`.
3. Cue frequency: `gaze_cues`. The adult must be in the learner's 90-degree cone to be a cue. `FOLLOW_DIST` and the viewpoint selection in `adult_act` control this.
4. Entropy: `ent_action` 0.02 may be too low for exploration or too high for commitment. Check `ent_action` in the log; healthy is roughly 1.0 to 1.6 (max is ln 7 = 1.95).
5. Context: K=32 frames. Gaze-following needs the adult's pointing to survive a turn (adult leaves the cone after the learner turns). K is plenty; but check `valid` masking isn't zeroing history (it resets per episode, which is correct).

Do not change the frame schema (`tribe/schema.py`, SCHEMA_VERSION 1) without bumping the version; everything downstream reads it.

## Code map

```
tribe/schema.py      frozen frame layout and constants. Read it first.
tribe/jax_world.py   the environment (batched JAX). adult_act is the scripted caretaker.
tribe/model.py       Learner (windowed GTrXL), XLLearner (cached-memory GTrXL: step / sequence), Predictor.
tribe/train.py       PPO + prediction losses, make_train (window) / make_train_xl, presets, resume, gaze metric.
tribe/world.py, frame.py, adult.py, policies.py, episode.py   NumPy reference world and M0 (keep in sync with jax_world semantics; tests compare them)
scripts/m0.py, m0_jax.py, replay.py, vast_setup.sh
scripts/fleet.py     rent Vast.ai instances, one job per instance (launch / status / logs / collect / destroy)
scripts/worker.sh, fleet_sync.py   what each instance runs; results go to a private Hugging Face repo
scripts/evallib.py + *_eval.py     ablation, word, facing and per-snapshot evaluations of trained weights
tests/               run `python -m pytest tests -q` (~1 min CPU) after any change
```

## Conventions

- Fixed-shape arrays everywhere in the JAX path; no Python objects per entity.
- Prediction losses train the trunk by backprop only. They must never enter the PPO advantage (dark-room problem). Keep it that way.
- The adult never judges. It looks, points, names, and acts in the world. Reward comes from the world.
- The learner has no word knowledge built in. The M0 follower's "danger = look, don't go" rule is the one exception and lives only in the hand-written policy, not the model.
- Log every run to `runs/<name>/log.csv`; never commit `runs/`.

## What comes after M1

M2: a tribe (3-6 learners, shared weights), rungs 2-3 (intent reading via two-agent game, joint commitment via shelter convergence), group-weighted reward. The NumPy world already supports multiple learners; the JAX world's `n_learners` arg does too but is untested beyond 1. Mobile game that needs two agents is not implemented in either world yet.

### M2 list (from the M1 follow-ups, 2026-11-10)

Fix before any M2 training:
- **Warmth reward pays for getting cold.** `world_step` rewards every positive meter change, including warmth
  recovering each morning (about 0.57 warmth = 1.1 reward per day, from the constants; not measured). A sheltered
  learner stays at full warmth and gets none of it. Pay for deficit avoided, not recovery.
- **Nights are harmless.** Seed-0 final weights: sheltered on 0.0% of night steps, zero cold deaths, 0.2% predator
  deaths. A night drains 0.57 warmth from 1.0 and the predator starts too far away to arrive. The shelter rung
  needs nights that cost something. Whether the reward term or the harmless night is what stops sheltering is untested.
- Consequence: "home" and "here" are ignored by the learner, and the adult says "home" about 40% of each day.

Both are now switchable, defaults unchanged (M1): `--warmth-reward deficit` (no pay for regained warmth; charge
`r_cold * (1 - warmth)` per step, 0.1 by default) and `--night-drain 1/15` (`J.WorldParams`, also `--predator-every`).
`tests/test_world_variants.py` covers them. Things measured while adding them (2026-11-11):
- A night is lethal exactly when 20 * drain >= 1: 1/20 or faster kills anyone who never shelters, 1/22 leaves 0.09.
- Seed-0 final learner at dusk start: median 14 cells from the shelter (p90 21, 99% within 25), adult 2-3 cells away.
  At 1/15 there are 12 dusk + 15 night steps to get there, so it is reachable for most, not all.
- The hand-written follower does not manage it (96% cold deaths at 1/15, sheltered 24% of night steps), so M0-style
  hand policies are not a baseline for lethal nights.
- `jobs/shelter.txt`: reward x night-drain grid, 600 updates each, seed 0. The M1 cell is seeds 1 and 2 on this box.

## Running on several instances

`python scripts/fleet.py launch jobs/<file>.txt --dry-run` shows the instances and hourly cost; without `--dry-run`
it asks, then rents. One job per instance, because a run cannot be split across machines and two runs sharing one
GPU each go at half speed (measured: 3.5 s/update alone, about 8 s/update with two on one 4090). Jobs must write to
`runs/<name>/`; that folder is uploaded to `<hf user>/tribe-runs` every 10 minutes and restored if the instance
restarts. Instances destroy themselves on success and stop on failure. The launch commit must be pushed.
Exercised end to end once (2026-11-10, 20-update smoke job on a rented 4090: clone, install, train at about
4 s/update, upload, self-destroy with the instance's own key). Restore-after-restart and the failure path (stop) are untested.
