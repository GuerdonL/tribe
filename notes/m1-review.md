# M1 review — commentary on notes/m1.md and the follow-up experiments

2026-11-10. Written from the chat session that built M0/M1, after reading `notes/m1.md`, `scripts/gaze_eval.py`,
and the in-progress results (adult-hidden ablation, word ablation, word swap). Companion PRs:
`review/heard-spoken-metric` (log word CE on spoken steps) and `review/adult-names-what-it-points-at`
(decouple word identity from the learner's meters).

## The pass is sound

`gaze_eval.py` is the right test. Memory fixed, one field flipped, turn preference measured. A +0.11/+0.12 shift on
41k cue steps with those CIs, holding within every adult-position stratum, means the learner reads the adult's
facing. It is conservative: facing information also sits in memory from earlier frames and the intervention leaves
that alone. M1 passed as stated.

The XL memory path was the right fix for my windowed design (which re-encoded K frames per step). The test that the
training pass reproduces rollout log-probs exactly is the invariant to keep.

## The fade was the metric, not the learner

Adult hidden: deaths 3% -> 68%. Reliance did not fade; the turn metric stopped seeing it. Early policy: "see pointing
adult, turn to match," which the metric catches. Later: "hear word, orient to voice, follow gaze, walk," where the
left/right turn on the cue step is a small part of a longer routine. Two consequences:

1. **Make the ablation the standing pass criterion.** Survival drop with the adult hidden (and, separately, with
   words blanked and with pointing/facing blanked) says "uses the adult." The turn metric stays as a diagnostic.
   Cheap to run every N updates on the latest checkpoint.
2. **The fade is probably rational cue use, and that is testable.** Prediction: gaze-following is strong when the
   learner has not yet seen the pointed-at resource type this episode, weak when it has (it already holds the
   location in memory, so the cue is redundant). One stratification in `gaze_eval.py`: track per-env "seen
   food/water/shelter this episode" and split the cue steps. If it holds, the fade is evidence the learner uses
   testimony when informative, which is what the later rungs want. It only matters as a worry if later rungs need
   attention to adults regardless; the fix there is worlds where the adult's knowledge stays non-redundant.

## The word ablation is the first grounding result

"food" removed -> only hunger kills; "water" removed -> only thirst kills. If a word were redundant with self-state
the learner could read its own meter and lose nothing. So the word carries something the meters do not, and the
specificity means a word -> need-category mapping exists. Shallow, but it is the rung-one kind of grounding: the
word tells the learner which gaze is worth following. "home" and "here" having no effect fits: shelter is handled
from the learner's own memory, and "here" is rare.

**Suggested test: conjunctive or not.** Blank pointing/facing but keep words; blank words but keep pointing. If
survival needs both, the word is a conjunctive cue ("this gaze is about your hunger"). If words alone partly
suffice, the learner has learned something closer to "water means go toward the voice."

## The word swap (1/3 to 40% shift) is a real referential effect

Same conservative single-frame intervention as the gaze flip: memory still carries the unflipped context, so the
frame-level flip can only move the next-frame prediction so far. Call it partial referential expectation. Do not
spend time pushing the number up.

## Heard CE 0.09 and the confound in my stage-one design

Two issues, one small and one that changes M2:

- Small: `heard` mixes heard_word and heard_dir over every step; log the word CE on spoken steps only
  (`review/heard-spoken-metric`). In the CPU smoke run spoken steps are ~50%, so the split matters less than I
  first thought, but it should be a separate number.
- Large: the adult says "water" only when the learner's thirst < 0.5, which is readable off the learner's own
  `thirst` field. Word *prediction* can be solved from self-state with no reference to the world. The ablation
  shows the word is *used*, which is a different and better fact, but prediction-loss-falling is not evidence of
  grounding under this adult. Before the language ladder, the adult must name things whether or not the learner
  needs them, so that a word is predictable only from the world. `review/adult-names-what-it-points-at` does this
  behind `TRIBE_NAME_IDLE=1`: idle adult points at the nearest landmark and names it. Suggest the seed-2 or a
  fourth run use it, so the heard CE and the word-swap shift can be compared with and without the confound.

## Remaining open questions, my ranking

- Q6 seeds: yes (running). Two more at ~3 h is cheap and a single seed is a single seed.
- Q7 late clipfrac 0.35: linear lr decay is one line; put it on the seed runs rather than a separate experiment.
- Q5 architecture specificity: skip.
- Housekeeping: move `TrainConfig` to `tribe/config.py` so checkpoints stop pickling `__main__.TrainConfig`.

## What I would write into the spec's M2 section from this

- Pass criterion for any learned rung: survival (or task) drop under cue ablation, not a behavior-matching rate.
- Adults name referents independently of learner need, from stage two onward.
- The "turns vs knowledge" lesson: a behavioral metric that counts a specific action will fade as the policy
  reorganizes; measure dependence by intervention.
