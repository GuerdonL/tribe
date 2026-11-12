# Shelter grid — what makes the learner shelter at night

2026-11-11 and 12, branch `m1-followups`. One seed per cell. Run folders are in the results repo
(`GuerdonL/tribe-runs`); every model can be watched in the visualizer (`viz/`).

## Question

M1's learner never shelters (seed 0: sheltered on 0.0% of night steps), so "home" carries nothing. Two suspects from
the M1 follow-ups: the warmth reward pays for getting cold and recovering, and nights cost nothing. Which one stops
sheltering?

## Setup

- `jobs/shelter.txt`: `--preset gpu_xl --seed 0 --updates 600`, snapshots at 100, 200, 300, 400, 600.
  Warmth reward `recover` (M1) or `deficit`; night drain 1/35 (M1), 1/22 (harsh, survivable: leaves 0.09 warmth) or
  1/15 (lethal after 15 unsheltered night steps). The `recover` x 1/35 cell is M1 itself: seeds 1 and 2
  (`m1_gpu_xl_s1`, `_s2`), read at the same updates.
- Five 4090 instances through box-boy, about 40 min each, about $2 in total. Then `jobs/shelter_eval.txt`
  (`scripts/snapshot_eval.py`, 2048 episodes per snapshot, 33 min, $0.22).
- Sheltering is not in `snapshot_eval.py`. It was measured separately on CPU: `scripts/shelter_measure.py`, 512
  episodes per update-600 model. The distance analysis is `scripts/dusk_distance.py`, on the visualizer's 32
  recorded episodes per model.

## Result

**Under the M1 reward the learner does not shelter unless the night kills. Under the `deficit` reward it shelters
even when nights are harmless. Lethal nights are not solved in 600 updates, and most of those deaths are decided by
how far from the shelter the learner is when dusk starts.**

Update 600. Evaluation columns: 2048 episodes. Sheltering columns: 512 episodes, 95% bootstrap interval.

| run | reward | night drain | died | night steps sheltered | sheltered at some point in night 1 | in the shelter when night starts |
|---|---|---|---|---|---|---|
| M1 seed 1 | recover | 1/35 | 9.4% | 0.3% [0.1, 0.4] | 0.6% [0.0, 1.4] | 0.0% |
| M1 seed 2 | recover | 1/35 | 11.9% | 0.0% [0.0, 0.1] | 0.4% [0.0, 1.0] | 0.0% |
| `sh_rec_d22` | recover | 1/22 | 11.8% | 0.0% [0.0, 0.1] | 1.2% [0.4, 2.2] | 0.2% |
| `sh_def_d35` | deficit | 1/35 | 8.1% | 22.8% [20.7, 25.0] | 39.6% [35.4, 43.8] | 23.4% [19.7, 27.1] |
| `sh_def_d22` | deficit | 1/22 | 6.9% | 32.9% [30.3, 35.3] | 50.6% [46.3, 55.1] | 30.3% [26.4, 34.2] |
| `sh_rec_d15` | recover | 1/15 | 77.2% | 15.2% [14.1, 16.1] | 71.9% [68.0, 75.8] | 1.2% [0.4, 2.1] |
| `sh_def_d15` | deficit | 1/15 | 73.6% | 43.6% [40.4, 46.8] | 64.2% [60.1, 68.3] | 31.5% [27.6, 35.6] |

Reliance on the adult, update 600 (2048 episodes). "Flip" is the shift in the matching-turn preference when only the
adult's observed facing is reversed; "died, adult hidden" is the death rate with the adult removed from the frame.

| run | flip, all cues | flip on "food"/"water" | flip on "home" | flip on "danger" | died, adult hidden | extra deaths, adult hidden |
|---|---|---|---|---|---|---|
| M1 seed 1 | +0.164 | +0.589 | +0.064 | +0.047 | 71.1% | +0.617 |
| M1 seed 2 | +0.150 | +0.458 | +0.067 | +0.015 | 72.7% | +0.608 |
| `sh_rec_d22` | +0.132 | +0.466 | +0.022 | -0.050 | 73.3% | +0.615 |
| `sh_def_d35` | +0.129 | +0.361 | +0.061 | -0.079 | 65.9% | +0.578 |
| `sh_def_d22` | +0.278 | +0.474 | +0.271 | +0.131 | 65.9% | +0.589 |
| `sh_rec_d15` | +0.320 | +0.459 | +0.297 | +0.101 | 98.5% | +0.213 |
| `sh_def_d15` | +0.315 | +0.415 | +0.339 | +0.110 | 97.0% | +0.234 |

Flip on "home" over training:

| update | M1 s1 | M1 s2 | `rec_d22` | `def_d35` | `def_d22` | `rec_d15` | `def_d15` |
|---|---|---|---|---|---|---|---|
| 100 | +0.344 | +0.170 | +0.257 | +0.313 | +0.286 | +0.067 | +0.116 |
| 200 | +0.555 | +0.529 | +0.581 | +0.608 | +0.568 | +0.301 | +0.340 |
| 300 | +0.266 | +0.343 | +0.325 | +0.350 | +0.627 | +0.378 | +0.447 |
| 400 | +0.151 | +0.139 | +0.109 | +0.211 | +0.423 | +0.326 | +0.446 |
| 600 | +0.064 | +0.067 | +0.022 | +0.061 | +0.271 | +0.297 | +0.339 |

Death rate over training:

| update | M1 s1 | M1 s2 | `rec_d22` | `def_d35` | `def_d22` | `rec_d15` | `def_d15` |
|---|---|---|---|---|---|---|---|
| 100 | 72.1% | 71.0% | 70.7% | 71.4% | 68.0% | 98.7% | 99.0% |
| 200 | 36.9% | 36.2% | 38.8% | 30.9% | 32.9% | 96.4% | 96.3% |
| 300 | 22.5% | 20.5% | 19.7% | 15.3% | 20.4% | 94.1% | 88.1% |
| 400 | 18.4% | 16.6% | 14.7% | 11.8% | 13.9% | 85.8% | 81.7% |
| 600 | 9.4% | 11.9% | 11.8% | 8.1% | 6.9% | 77.2% | 73.6% |

## Observations

1. **A harsher but survivable night changes nothing by itself.** `sh_rec_d22` matches M1 on sheltering (0%), on
   "home" (+0.02) and on deaths (11.8%). The learner rides the night out and takes the reward for warming up again.
2. **The reward is what stops sheltering in a survivable world.** With `deficit` and M1's harmless nights the
   learner is in the shelter when night starts in 23% of episodes; at 1/22, 30%. It is patchy: in the 32 recorded
   worlds, `sh_def_d35` shelters on at least one night in 22 and `sh_def_d22` in 29, often one night of four.
3. **Sheltering under `deficit` at 1/35 does not go through the adult's facing.** `sh_def_d35` shelters, but its
   flip on "home" is M1's (+0.06). It reaches the shelter mostly when dusk finds it close (75% from within 8 steps,
   15% from 9 to 16). `sh_def_d22` keeps +0.27.
4. **Following the adult on "home" lasts only where the night costs something.** It peaks near update 200 in every
   survivable world (+0.53 to +0.61), then fades to +0.02..+0.07 in M1, `rec_d22` and `def_d35`. It holds near
   +0.3 in `def_d22` and both lethal cells. The same is true of the flip over all cues: the M1 fade repeats, and
   costly nights slow it (+0.13 against +0.28..+0.32).
5. **Lethal nights: sheltering in night 1 is surviving night 1.** `rec_d15`: 71.9% shelter, 71.7% survive it.
   `def_d15`: 64.2% and 64.1%. Of those who do not shelter in night 1, 99% die of cold. Of those who do, 54 to 62%
   die of cold anyway, almost all on a later night.
6. **Under `recover` at 1/15 the learner shelters late.** Only 1.2% are in the shelter when night starts, 72% enter
   during it, and it spends 15% of night steps sheltered (against 44% under `deficit`). It takes the cold it can
   survive, which the `recover` reward still pays for.
7. **In lethal worlds survival is the adult.** With the adult hidden 97 to 98.5% die, against 74 to 77% with it.
   The gap looks small (+0.21..+0.23) only because deaths are already near the ceiling.
8. **Death rates in the lethal cells were still falling at update 600** (99% -> 77% and 99% -> 74%).

## M1 seeds 1 and 2: the late pattern of seed 0 repeats

Both seeds finished 3000 updates on 2026-11-11. Final weights, 2048 episodes (`snap_eval.csv` in each run folder):

| seed | died | flip, all cues | flip on "food"/"water" | flip on "home" | flip on "danger" | died, gaze hidden | died, words hidden | died, adult hidden |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.4% | +0.057 | +0.414 | -0.021 | -0.111 | 51.3% | 39.6% | 69.0% |
| 1 | 2.2% | +0.101 | +0.471 | +0.030 | -0.009 | 51.1% | 35.7% | 66.2% |
| 2 | 2.6% | +0.063 | +0.349 | +0.014 | -0.057 | 52.8% | 33.7% | 64.8% |

In all three the learner follows the adult's facing on "food" and "water" and not on "home", and it dies 20 to 30
times as often with the adult hidden. The fade in gaze following is a narrowing to the words that matter in the M1
world, not independence from the adult.

## Distance at dusk decides the lethal night

The owner's observation from watching runs: a learner follows the adult well, goes shelter -> water -> food, is then
too far from the shelter to get back, and dies. Checked two ways (`scripts/dusk_distance.py`).

**The maps are possible.** In all 32 recorded worlds the loop shelter -> water -> food -> shelter is at most 53
steps (median 35, counting a turn as a step), against 74 steps a day outside the shelter. Water is more than 26
steps from the shelter in 1 world of 32, the nearest food in none.

**The learner cannot use them.** The adult says "home" from dusk start: 12 dusk steps plus 14 survivable night
steps is 26 steps. Nights that started outside the shelter, updates 400 and 600 of both lethal cells, 32 episodes each:

| steps from the shelter when dusk starts | `def_d15`: nights | reached the shelter | died of cold that night | `rec_d15`: nights | reached the shelter | died of cold that night |
|---|---|---|---|---|---|---|
| 0 to 8 | 77 | 94% | 6% | 73 | 95% | 7% |
| 9 to 16 | 38 | 63% | 34% | 54 | 57% | 41% |
| 17 to 26 | 23 | 4% | 96% | 20 | 5% | 90% |

After dusk starts the learner closes a median of 0.27 (`rec_d15`) to 0.43 (`def_d15`) steps of distance per step,
while hearing "home" on about 80% of those steps. In 26 steps that is 7 to 11 steps of distance. Water is more than
16 steps from the shelter in 13 of the 32 worlds. So a learner that follows the adult to water and then food is
often out of reach of home by the time the adult says so.

Not established: why homing is that slow (turn costs, waiting for the adult to come into view, or not committing).

## Open questions

1. **Is the late "home" call the cause of the lethal-night deaths?** The test would be a distance-aware call in
   `jax_world.adult_act` (say "home" when the time left is less than the learner's distance to the shelter divided
   by its real homing speed), then rerunning `rec_d15` and `def_d15`. **Decided 2026-11-12: not doing it.** The owner
   does not want the adult tuned further; the next step is M2 (several learners with shared weights).
2. **A model adult.** Raised by the owner: let the adult be a model that starts from the scripted behaviour and is
   rewarded on the learner's survival. Recommended form, if done: the adult model chooses among the script's
   intents (lead to water, lead to food, lead home, warn, idle) and the script still produces the pointing and the
   word, so word meanings stay fixed and the word evaluations keep their meaning. A fully free adult is emergent
   communication, which this project says it is not; it would be its own experiment, with a drift check against the
   script's word meanings. Not planned for now.
3. **Which cell does M2 build on?** `def_d22` is the only survivable cell that keeps the learner following "home"
   (+0.27) and it has the lowest death rate (6.9%). One seed.
4. **Do the lethal cells keep improving?** They were still falling at 600 updates. A 3000-update run would say
   whether they plateau at the distance limit above.
5. **Seeds.** Every cell here is seed 0 only. The M1 cell's two seeds agree closely at update 600 (deaths 9.4% and
   11.9%, flip on "home" +0.064 and +0.067), which is some evidence that one seed is not wildly off.
6. **`danger` goes negative** in `rec_d22` (-0.05) and `def_d35` (-0.08) at update 600: the learner turns away from
   where the adult faces on that word. Small, and it may be right behaviour (the adult faces the predator). Not looked at.

## Housekeeping

- `scripts/put_result.py` uploads a file to another run's folder from inside a job; `jobs/shelter_eval.txt` uses it
  so each run's `snap_eval.csv` sits in its own folder, where the visualizer reads it.
- box-boy changes that came out of these launches (CUDA floor from the image tag, final-upload retry, 15-minute
  checkpoint uploads) are on its `smoke-test-fixes` branch.
- One host refused this machine's SSH key although Vast reported it attached (`sh_def_d35`); the run was unaffected.
