# CLAUDE.md — tribe

Read this first. It is the handoff from the session that built M0 and M1.

## What this project is

A test of one claim: an agent trained first on pre-verbal social survival tasks (joint attention, intent reading, joint commitment, testimony) will acquire grounded language from speaking "adults" where a randomly initialized RL agent cannot. Agents never invent language; they learn it from scripted (later model) adults inside a world they already understand. The full spec is in a Claude doc the owner has; the README summarizes the architecture and the frame schema.

Owner: Guerdon (GuerdonL on GitHub). He wants to move fast and prefers decisions made and flagged over questions asked. Commit messages end with a Co-Authored-By line for Claude.

## Where things stand

- **M0 passed** (hand-written policies, no model): gaze-following learner outlives gaze-ignoring one, +37.5 steps over 300 paired episodes, 95% CI [+27.7, +47.3]. Reproduced in the JAX world (+33.6, CI [+20.6, +46.3]).
- **M1 is built, not yet run.** `python -m tribe.train --preset gpu --resume` on this box is the job. First attempt OOMed at 42 GB on a 48 GB card; fixed by gradient checkpointing (`ModelConfig.remat`) and 128 minibatches. Not yet confirmed to fit. If it still OOMs: `--minibatches 256`, then `--envs 1024`.
- CPU smoke run (300k params, 150k steps) trained stably: world CE 52 -> 19, heard CE 4.7 -> 0.7, gaze metric at chance. That only proves the plumbing.

## M1 pass criterion

In `runs/m1_gpu/log.csv`, `gaze_follow_rate` reliably above `gaze_chance`. The metric: among steps where a pointing adult is visible and faces the learner's left or right, the fraction of learner actions that are the matching turn; chance is the learner's base rate of that turn. Also watch `gaze_cues` (how often the cue even occurs per update; if it is tiny the metric is noise and the fix is on the adult side, in `jax_world.adult_act`, not the model) and `heard` (word-prediction CE; it should fall well below ln(23) = 3.1 as the model learns what adults say when).

Expected timeline on a 4090-class card: a minute or two of compile, then well under a second per update; 3000 updates is on the order of an hour. Checkpoints every 50 updates to `ckpt.pkl`; `--resume` continues.

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
tribe/model.py       Learner (frame encoder + GTrXL decoder + heads) and Predictor (next-frame).
tribe/train.py       PPO + prediction losses, one jitted update, presets, resume, gaze metric.
tribe/world.py, frame.py, adult.py, policies.py, episode.py   NumPy reference world and M0 (keep in sync with jax_world semantics; tests compare them)
scripts/m0.py, m0_jax.py, replay.py, vast_setup.sh
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
