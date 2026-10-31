# Tetris data flywheel: app games, relabeling and RL

The iPhone app (and the Mac demo) plays full games and can record them. This page describes how recorded games become a better
adapter. Nothing here changes the shipped adapter; it is the recipe for the next round.

```
app games --(export)--> games .jsonl / .jsonl.gz --relabel--> train/heldout rows --LoRA continue--> candidate --play eval
```

## What a recorded game contains

One JSON object per game (`schema: system1-ane-tetris-v1`): `lines`, `pieces`, `toppedOut`, `capped`, `explore`, and for every
decision the board before the move, the piece, the move played, the El-Tetris label, and the model's probability for **every**
legal placement (so the behaviour policy is known exactly). Options are not stored; `jeff_tetris.placements(board, piece)`
rebuilds them. No identifiers are stored.

## Rules must match the evaluation

`projects/tetris_training/dataset.py` ends a game when the centered, unrotated spawn is blocked (`topped_out`). The first app
builds ended a game only when no placement existed at all, so they kept playing past the real game over and the logs contain
boards the demo and the evaluation can never reach. The app now uses the same spawn rule (`Tetris.spawnBlocked`, checked
against Python on 600 boards), and `relabel.py` drops every decision at or after the first blocked spawn in older logs. Numbers
from the old app (about 40 lines per 120 pieces) are **not** comparable with the evaluation table in the training README.

## Step 1: relabel with a stronger teacher (distillation)

El-Tetris is a one-step greedy heuristic and the adapter imitates it, so imitation alone is capped at El-Tetris. `relabel.py`
re-scores every stored state with `search2`: the El-Tetris rating plus the discounted expected best rating after the next piece
(expectimax over the 7 pieces, the top 6 placements expanded). The student then learns the search result.

```sh
python3 -m projects.tetris_training.relabel ARCHIVE_DIR -o RUN/data --teacher search2 --mode distill
```

Per decision it also writes the model's **regret** (teacher value of the best move minus the move played). Near-ties have
about zero regret and teach nothing; confident mistakes have large regret. In `distill` mode the row weight is
`1 + k * min(regret, cap) / cap`, so real mistakes count up to 4x. `--min-regret` can drop near-ties entirely. The report shows
how often the search teacher differs from El-Tetris and how much of the data is "hard".

Games, not rows, are split into train and heldout, and repeated states are kept once (the empty-board openings repeat
constantly). Mix some of the original training rows back in (`MIX_PRIOR`, default 1:1) to avoid forgetting.

## Step 2: train

`scripts/jeff_lora_train.py` now reads an optional per-row `weight` (weighted cross-entropy; evaluation stays unweighted):

```sh
BASE_MODEL=... INIT_ADAPTER=... PRIOR_TRAIN=... scripts/tetris_flywheel.sh ARCHIVE_DIR RUN_DIR distill
```

It relabels, mixes, continues the adapter for one epoch at a low learning rate, and stops before gameplay evaluation, which
needs the ANE. Evaluate with `projects/tetris_training/play.py` on the reserved seeds and compare **lines and survival per
game**, not agreement with El-Tetris: a 31% disagreement rate on the first 8,669 app decisions coexisted with near-teacher
line rates because most disagreements were near-ties.

## Step 3: sequence-level RL (offline, advantage weighted)

A game is a sequence, so each decision also gets a reward (lines cleared), a discounted return over the next 20 moves (minus a
penalty if the game topped out inside the horizon) and an advantage (return minus the mean return of similar states, bucketed
by stack height and piece). `--mode awr` trains **advantage-weighted regression**: the label is the move that was played and
the weight is `exp(advantage / beta)`, clipped. It reuses the same weighted trainer, with the readout softmax as the policy.

AWR only learns from moves that differ from the top choice, so it needs exploratory data. The app has an **Explore** switch
(ellipsis menu) that samples each move from the model's own probabilities; those games set `explore: true`. Without
exploration, almost every logged move is the argmax and the advantages are mostly noise. Because every placement's
probability is stored, importance-sampled or PPO-style updates are possible later without changing the log format.

Expect AWR to be a second stage after distillation, and to give modest gains: rewards are sparse (lines are rare early in a
game), games are truncated at 120 pieces, and the piece sequence is random. The search teacher is the more reliable lever.
