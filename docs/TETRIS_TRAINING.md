# How the shipped Tetris adapter was trained

This page describes the **October 8, 2026 round-02 adapter** published as
[`anemll/jeff-ane-tetris-coreai`](https://huggingface.co/anemll/jeff-ane-tetris-coreai).
The [Colab notebook](../notebooks/03_tetris_training_colab.ipynb) is a separate,
smaller experiment. It has not been validated on a Colab GPU and does not
reproduce the shipped checkpoint or its measured results.

## Model and objective

- Base: [Jeff v1.3](https://huggingface.co/mstrasser/jeff-base), a fine-tune of
  Qwen3.5-0.8B. The tutorial pins revision
  `3cf045d8a52dc3ac2c6d0c842071235984d5fdf5` and checks its weight SHA-256.
- Trainable parameters: rank-16 LoRA, alpha 32, plus Jeff's 255-way readout.
  The original backbone weights stay frozen during training.
- Loss: cross-entropy on legal-option logits, targeting the El-Tetris teacher's
  chosen placement. The run used hard labels, without soft-score distillation or
  a reinforcement-learning objective.
- Prompt: the same text board, instructions, legal-placement descriptions and
  option order used by the demo.
- Implementation: [`scripts/jeff_lora_train.py`](../scripts/jeff_lora_train.py),
  invoked through `forge.py jeff-train-lora`, using PyTorch MPS on an Apple M5 Max.
  The shipped adapter was not trained on Colab or with the Unsloth runtime.

## Training sequence

1. **Pilot:** 1,024 teacher-labeled examples, LoRA LR `2e-4`, readout LR `5e-6`,
   batch 4. Data combines teacher games, games with random actions and synthetic
   garbage-recovery boards. Noisy behavior still receives teacher targets.
2. **Fresh pool:** 4,096 additional examples from games up to 300 pieces,
   excluding historical training states and the original offline split states.
3. **Corrections:** the current model plays on reserved training collection
   seeds. El-Tetris labels its visited boards, including losing games. The
   initial collection produced 1,379 unique corrections from 16 pilot games.
4. **Continuation:** rounds train on up to 2,048 rows, roughly half fresh,
   one quarter anchors and one quarter corrections. Round 02 used **1,024 fresh
   + 512 anchors + 512 corrections**, initialized from round 01. Settings:
   **LoRA LR `1e-4`**, readout LR `5e-6`, batch 4, one epoch, rank 16, alpha 32.
   Each continuation starts a fresh optimizer.
5. **Selection:** export candidates, check ANE compilation/cache placement and
   prompt parity, then compare greedy gameplay on eight development seeds.
   Rank by mean pieces, then mean lines. After a non-improving round, the
   controller returns to the best adapter and reduces the learning rate.

This is supervised imitation with teacher corrections from learner-visited
states. The shipped run did not keep only the longest games. Offline agreement
is a diagnostic; development gameplay determines acceptance.

The [subproject guide](../projects/tetris_training/README.md#overnight-continuation--october-8-2026)
contains commands, data audits and controller recovery details. Recovery saves
LoRA/readout weights, not optimizer state. The controller adds subprocess
deadlines; `--train-until` alone stops optimization between batches and does not
bound total runtime.

## Recorded results

Greedy ANE gameplay, browser-compatible spawn/top-out, 32 fixed seeds, 150-piece cap:

| Policy | Mean pieces | Mean lines | Reached cap |
| --- | ---: | ---: | ---: |
| Previous Tetris adapter | 59.84375 | 12.40625 | 0/32 |
| Pilot | 72.9375 | 15.125 | 1/32 |
| **Selected round 02** | **149.75** | **53.75** | **30/32** |
| El-Tetris teacher | 150 | 57.6875 | 32/32 |

Round-02 held-out agreement increased from **0.68359375 to 0.74609375** on 256
rows. The starting number is the round-01 adapter, not untouched Jeff base.
Warm server latency averaged **288.20 ms/move** across 4,792 visited boards;
this is not a controlled fixed-input throughput comparison.

These seeds had already been evaluated: this is a **repeated benchmark**, not a
pristine holdout. Round 02 passed four-prompt argmax parity after rebuilding its
readout cache. Round 03 saved weights but failed ANE load before parity/gameplay
measurement and was not promoted.

Evidence is outside Git under
`$HOME/Models/jeff-tetris-training/overnight-20261008/`: `best.json`,
`round-02/model/report.json`, `round-02/model/adapter/`, and
`final-test/report.json`. Documentation publication is not a new training or ANE run.

## Try the experimental Colab tutorial

[Open the notebook](../notebooks/03_tetris_training_colab.ipynb) or
[run it in Colab](https://colab.research.google.com/github/Anemll/system1-ane/blob/main/notebooks/03_tetris_training_colab.ipynb).

It uses the same trainer/prompt but two 1,024-row phases starting from Jeff base,
PyTorch corrections capped at 60 pieces, and no ANE checkpoint selection.
Continuation LR is `1e-4`. Fresh teacher rows fill correction shortfalls.
Speed, GPU memory fit and gameplay quality remain unmeasured for this notebook.

The default 28-minute planning target stops starting optimizer batches at minute
24. Loading, evaluation and saving can exceed it; this is not a hard runtime or
CU cap. Retries use fresh output directories, and saved weights are archived
before test evaluation. Recovery from periodic weights starts a fresh optimizer
and replays data. Automatic runtime deletion requires a verified Drive copy.

Begin with the notebook's small-run settings. Measure step times and GPU memory,
check adapter reload and correction collection, then scale up. Optional CUDA
development games measure policy survival on Colab. ANE parity, placement and
latency require the Mac export steps.

## Credits

- **Jeff:** Mathias Strasser / [mstrasser](https://huggingface.co/mstrasser/jeff-base).
- **Qwen3.5-0.8B:** [Qwen Team, Alibaba Cloud](https://huggingface.co/Qwen/Qwen3.5-0.8B).
  Jeff/Qwen model weights are Apache-2.0; this repository's source is MIT.
- **Teacher:** [El-Tetris by Islam El-Ashi](https://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/),
  building on Pierre Dellacherie's algorithm.
- **Training and ANE integration:** ANEMLL contributors.
- **Unsloth:** thanks to the [Unsloth team](https://github.com/unslothai/unsloth).
  The head training workflow was adopted from their notebooks. The main trainer
  is ANEMLL's PyTorch implementation. The default-off Unsloth appendix uses a
  different Clef prompt/head and remains untested for Tetris.
