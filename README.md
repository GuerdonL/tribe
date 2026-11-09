# tribe

Pre-verbal social pretraining for grounded language acquisition — stage one.

The bet: an agent trained first on pre-verbal social survival tasks (joint attention, intent reading, joint commitment, testimony) will acquire grounded language from speaking "adults" where a randomly initialized RL agent cannot. Language is not invented by the agents; it is learned from adults who already speak, inside a world the agent already understands. The spec lives in the project doc; this repo is the implementation.

## Status

**M0 passed.** A hand-written learner that follows adult gaze outlives one that ignores it, in identical worlds, with no model involved:

```
M0  episodes=300 steps=320 grid=32 view=5 day=80/60
  ignorer: mean steps  177.2  survived whole episode   5.0%
 follower: mean steps  214.6  survived whole episode  22.7%
paired: follower longer in 44% of worlds, shorter in 18%; mean gain +37.5 steps, 95% CI [+27.7, +47.3]
```

M0 exists to prove the environment can reward gaze-following at all. It says nothing about learning yet; that is M1.

## Layout

```
tribe/schema.py    frozen frame layout, token ids, world constants  (SCHEMA_VERSION = 1)
tribe/world.py     gridworld: reset, step, line of sight, egocentric geometry, ASCII render
tribe/frame.py     State -> fixed-size integer frame for one agent; decode/validate for tests
tribe/adult.py     scripted caretaker: looks, points, names; never judges
tribe/policies.py  hand-written learners for M0: GazeIgnorer vs GazeFollower
tribe/episode.py   run one episode
scripts/m0.py      the go/no-go comparison
scripts/replay.py  watch an episode in the terminal
tests/             schema, geometry, occlusion, frame round-trip
```

Pure NumPy, fixed-shape arrays, no per-entity objects, so the port to JAX for M1 is mechanical.

## The frame (schema v1)

Everything the learner perceives, hears, does and says is a small integer in a fixed-size frame, one per step. One word per step is the maximum speaking rate.

```
self  : hunger thirst warmth facing phase heard_word heard_dir          (7)
slots : 12 x [valid type rel_x rel_y rel_facing pointing saying]        (84)
out   : action pointing saying                                          (3)
```

Coordinates are egocentric: `rel_x` right(+)/left(-), `rel_y` forward(+). `rel_facing` is the other agent's facing relative to ours, so "behind" is literally `rel_y < 0` and "facing me" is `rel_facing == 2`. The view is a 90° cone 5 deep, occluded by trees and rocks. Words carry 6 cells through trees. The map edge shows as rock.

Vocabulary is the stage-one subset of the dictionary graph's concrete leaves: `look here there food water tree rock shelter danger come go wait get night home now together behind near far left right`.

## Run

```
python -m pytest tests -q
python scripts/m0.py --episodes 300
python scripts/replay.py --seed 7 --policy follower --every 2
```

## What M0 taught us about the world

Every one of these was a stall or a bug found by watching episodes, and each is a thing a learned agent will also have to cope with:

- A 90° cone drops an off-axis target the moment you turn toward it. Both M0 policies keep a 12-step dead-reckoned memory of the last seen target; a learned agent gets this from its context window.
- An adult standing directly in front of the learner blocks it; the adult now parks off-axis, two or more cells out, with line of sight.
- A warning ("danger") means look, not go. The follower treats it as turn-only. This is the one piece of word knowledge M0's follower has beyond gaze.
- The predator must be slower than the learner or warnings arrive too late to matter.
- Night must be short relative to day or thirst kills everyone before guidance can help. Day is 80 steps: 48 day, 12 dusk, 20 night.

## Next: M1

Single learner, scripted adult, the transformer from the spec (GTrXL recipe), world-prediction loss + PPO. Pass: gaze-following above chance within hours on one GPU. First step is the JAX port of `world.py` and `frame.py`.
