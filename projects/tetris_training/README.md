# Jeff Tetris training

Read [How the shipped Tetris adapter was trained](../../docs/TETRIS_TRAINING.md) for the historical round-02 recipe and results. The [experimental Colab notebook](../../notebooks/03_tetris_training_colab.ipynb) trains a smaller new adapter; it has not yet been validated on a Colab GPU.

Build a reproducible supervised dataset for the existing **LoRA plus readout head** trainer. No RL framework, laya model, GPU or model inference is needed to build the initial data. Generation uses Python's standard library and the repository's El-Tetris oracle.

This is a starting dataset with broader board coverage; stronger gameplay must be established by training and ANE evaluation. No survival improvement is implied by passing the dataset audit.

## What the dataset contains

The default pilot has **1,024 training, 256 validation and 256 offline test rows**. Each split contains:

| Share | Boards visited | Label |
| --- | --- | --- |
| 50% | Teacher games, up to 150 pieces each | Best El-Tetris placement |
| 25% | Games with 10% uniformly random legal actions | Best El-Tetris placement, even when the executed action was random |
| 25% | Teacher recovery from 4–10 rows of synthetic garbage | Best El-Tetris placement |

Games run to the cap or the browser's centered-spawn top-out. No successful-game filtering is applied. Sampling balances occupied piece-type, stack-height and game-stage buckets within each source. Singleton-option decisions are excluded from training data. Candidate games are oversampled, and only the requested row count is retained.

The generator partitions games and piece seeds before collecting rows, then removes duplicate board/piece states within and across splits. Separate RNGs drive pieces, exploration and garbage, so changing exploration does not change the piece stream. Initial empty-board duplicates are removed from the offline sets; natural gameplay evaluations still start on an empty board.

Rows use `{state, options, label, instructions}` and the **current demo's exact prose options and text grid**. Teacher scores and collection metadata are outside the model input. Ordinary cross-entropy trains both LoRA and the readout; scores are retained for a future soft-target experiment and are ignored by the current trainer. Prompt shortening requires a coordinated change to generation, training and serving; this version preserves compatibility.

## Build and inspect

Run commands from the repository root. Store generated data and model outputs outside the checkout.

```sh
DATA="$HOME/Models/jeff-tetris-training/datasets/pilot-v1"
python3 -m projects.tetris_training.dataset build --output "$DATA"
python3 -m projects.tetris_training.dataset audit --dataset "$DATA"
```

If continuing an existing adapter, pass its historical training file when building the data:

```sh
python3 -m projects.tetris_training.dataset build --output "$DATA" \
  --exclude-dataset "$HOME/Models/jeff-tetris-mps/train.jsonl"
```

Choose one build command and a new output directory. Repeat `--exclude-dataset` for additional prior training files. This removes those states from all new splits and keeps validation clean when resuming the existing adapter. The exclusion list is saved with the dataset so auditing does not depend on those external files remaining available.

The output directory must be new. The builder writes:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: directly usable supervised rows.
- `games.jsonl`: full-game pieces, lines, termination reason, source mode and seed.
- `excluded_states.jsonl`: historical training states excluded from this release.
- `manifest.json`: configuration, checksums, source hashes, coverage statistics, 8 development gameplay seeds, 32 final gameplay seeds and 32 separate correction-collection seeds.

The audit recomputes labels, legal placements, scores, prompt text, option order and board statistics. It checks duplicate states, split isolation, game provenance and file checksums. A separate offline test set supports one-time regression checks; use validation for tuning.

To expand later, use a new output directory and, for example, `--train 4096 --validation 512 --test 512`. Keep the seed fixed to reproduce a dataset of the same configuration. Do not combine independently generated releases without checking for overlap.

## Train with the existing entry point

Use the existing Python environment containing PyTorch and Transformers. `python3` above may be the system interpreter; `python` below must be your training environment. This command starts a new adapter from the base:

```sh
DATA="$HOME/Models/jeff-tetris-training/datasets/pilot-v1"
RUN="$HOME/Models/jeff-tetris-training/runs/pilot-v1"
python forge.py jeff-train-lora \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --task tetris \
  --dataset "$DATA/train.jsonl" \
  --heldout-dataset "$DATA/validation.jsonl" \
  --output "$RUN" \
  --device mps --rank 16 --alpha 32 \
  --epochs 1 --batch 4 --lr 2e-4 --readout-lr 5e-6 \
  --eval-batch 16 --eval-train-rows 64
```

The explicit `--heldout-dataset` makes **all 1,024 train rows train**; otherwise the trainer takes an 80/20 row split. The trainer saves `adapter/`, `merged/` and `report.json`. It uses hard-label cross-entropy, not the stored teacher scores.

`--eval-train-rows 64` limits only the optional before/after training-accuracy checks to the same seeded sample. Training still uses every training row, and validation still uses all 256 validation rows. `--eval-batch 16` batches evaluation more efficiently on the 128 GB M5 Max; lower it if memory is limited. Both settings are recorded in the report. Omitting these flags retains the original full-training-set evaluation and training batch size for evaluation.

To continue the existing Tetris adapter, add `--init-adapter "$HOME/Models/jeff-tetris-mps/adapter"`. Preserve the original base, rank and alpha. Continuation restores the LoRA and readout weights and starts a fresh optimizer. For checkpoint selection, run one epoch at a time into new directories. The trainer saves its final adapter and, when enabled, intermediate recovery weights. Its `--play` option is Snake-only, so use the gameplay tool below for Tetris.

The local pilot below took **34.6 minutes for one training epoch**, excluding setup, evaluation, merge and export. Measure new prompt lengths before scaling data. Dataset generation is CPU-only and much cheaper than training.

## Measured local pilot — October 8, 2026

Machine: Apple M5 Max, 40 GPU cores, 128 GiB unified memory. PyTorch 2.14.0 on MPS, BF16, rank 16, alpha 32. The command above trained all 1,024 rows from the base checkpoint for one epoch. The full successful run, including evaluation and merge, took **44.5 minutes**. An earlier attempt spent 13.9 minutes in baseline evaluation and was stopped before training; its log is retained.

Held-out teacher agreement on 256 validation rows increased from **18.4% to 58.6%**. The compiled FP16 ANE build matched the PyTorch choice on all four saved parity prompts (mean KL 0.000387). All seven compiled packages passed the cache placement audit; that audit describes compiled placement, not live hardware profiling.

Greedy ANE gameplay on the same eight development seeds, using the centered-spawn rule and a 150-piece cap:

| Policy | Mean pieces | Mean lines | Reached cap |
| --- | ---: | ---: | ---: |
| Previous Tetris adapter | 47.375 | 8.125 | 0/8 |
| Pilot, one epoch | 93.125 | 22.25 | 1/8 |
| El-Tetris teacher | 150 | 57.25 | 8/8 |

The pilot improved survival on every development seed. These are development results; the larger held-out validation below gives a more cautious assessment. Pilot warm server latency averaged 290.8 ms/move (p50 249.2, p95 373.7). The previous adapter was measured while MPS training was active, so its latency is not a clean speed comparison.

Separate three-step training probes used the longest prompts (1,488 tokens), excluding the first step from timing:

| Batch | Seconds/step | Examples/second | Sampled Metal allocation |
| --- | ---: | ---: | ---: |
| 4 | 8.76 | 0.457 | 36.2 GiB |
| 8 | 17.55 | 0.456 | 69.9 GiB |

**Batch 8 fits the short probe but provides no throughput gain; retain batch 4 for this implementation.** Memory is sampled driver allocation, not an exact physical peak. A longer run with mixed shapes can retain additional caches. Batch 16 was not tested.

Local artifacts are under `$HOME/Models/jeff-tetris-training/`:

- `runs/pilot-v1/test_summary.json`: combined results and provenance.
- `runs/pilot-v1/{adapter,merged,report.json}`: trained weights and training metrics.
- `builds/pilot-v1/coreai/`: compiled candidate build.
- `evals/{previous-tetris-dev,pilot-v1-dev,teacher-dev}/report.json`: per-game results.
- `benchmarks/`: reusable batch probe, machine metadata, timings and memory samples.

### Actual browser game and 32-seed validation

The follow-up validation loaded the candidate as `tetris` and the previous build as `previous` on a separate server at port 8788. The actual unmodified browser demo played one natural, unseeded game with the candidate: **110 pieces, 28 lines**, ending by normal top-out with zero console errors. The screenshot and DOM-visible result are saved in `evals/pilot-v1-browser/`. This browser game is separate from the fixed-seed comparison.

All three policies were then evaluated on the same 32 reserved test seeds, greedily with the demo's top-out rule and a 150-piece cap. The learned policies ran sequentially against the ANE server, with browser gameplay stopped.

| Policy | Mean pieces | Median pieces | Minimum pieces | Mean lines | Reached cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous Tetris adapter | 59.84375 | 58 | 34 | 12.40625 | 0/32 |
| Pilot, one epoch | 72.9375 | 68.5 | 33 | 15.125 | 1/32 |
| El-Tetris teacher | 150 | 150 | 150 | 57.6875 | 32/32 |

The candidate's observed means increased about 22%, but it won survival on only **18/32** paired seeds and lost on 14. The mean paired gain was 13.09 pieces; a 10,000-resample paired bootstrap 95% interval was **-0.50 to 26.34 pieces**, including zero. These results do not yet establish a reliable improvement. The candidate still topped out before the cap in 31/32 games, so more training is needed for consistently long demos.

Candidate warm server latency averaged **285.3 ms/move**, with p50 249.3 and p95 375.1 ms across 2,334 moves. The previous adapter averaged 279.4 ms across 1,915 moves. Policies visit different boards, so this is not a fixed-input throughput comparison.

`evals/pilot-v1-validation.json` contains the combined result, paired per-seed differences, source hashes, server/build mapping, latency and browser evidence. Individual reports are in `evals/{previous-tetris-test,pilot-v1-test,teacher-test}/report.json`. The 32 test seeds have now been consumed by evaluation; use the development split for subsequent tuning. The existing port 8787 demo was kept separate from this validation.

## Evaluate on the demo rules

The subproject evaluator includes centered-spawn top-out, which the older `scripts/jeff_tetris_eval.py` omits. It uses greedy server choices and one question order. Invalid responses and server errors fail the evaluation rather than being counted as lost games.

First establish the CPU teacher reference:

```sh
python3 -m projects.tetris_training.play \
  --dataset "$DATA" --split dev --adapter teacher \
  --output "$HOME/Models/jeff-tetris-training/evals/teacher-dev"
```

Merge/export/compile and load the candidate using the existing [Jeff serving workflow](../../docs/SYSTEM1_ANE_SERVE.md). Once its ANE build is served on port 8796:

```sh
python3 -m projects.tetris_training.play \
  --dataset "$DATA" --split dev --adapter tetris --url http://127.0.0.1:8796 \
  --output "$HOME/Models/jeff-tetris-training/evals/pilot-dev"
```

Use `--split test` once for the selected candidate: that runs the 32 reserved final seeds. Compare teacher and candidate with the same split and `--max-pieces`. After success at 150 pieces, raise the cap for both for a longer soak. The report includes per-game results, mean/median/minimum pieces, lines, cap completions, server health, and warm p50/p95 server and HTTP wall latency. One warm-up request is reported separately; it is not asserted to be a cold model load. Warm timing covers the visited boards of that policy, so cross-policy differences can reflect different board distributions.

## Collect teacher corrections from Jeff

This makes requests to the already running server; it does not start or replace a model. Collection uses only the reserved training seeds, including losing games. Every visited state is labeled with the oracle, not with Jeff's action.

```sh
CORRECTIONS="$HOME/Models/jeff-tetris-training/datasets/corrections-v1"
python3 -m projects.tetris_training.play \
  --dataset "$DATA" --split train --adapter tetris --games 16 --collect \
  --output "$CORRECTIONS"
```

The result contains `corrections.jsonl`, a combined `train.jsonl` containing the original training rows plus unique corrections, and `report.json`. Duplicate states from any original split are excluded. Continue supervised training with this combined `train.jsonl`, the **original** validation file and `--init-adapter` pointing at the candidate's `adapter/`. This is a simple first correction round; the combined file is not a new three-split dataset release. If corrections vastly outnumber teacher rows, subsample them before training to retain a useful teacher/recovery mix.

## Overnight continuation — October 8, 2026

The bounded runner is `python -m projects.tetris_training.overnight`, launched from **system1-ane** with the training interpreter. It uses that same interpreter for training and `COREAI_PYTHON` for the Core AI SDK, matching `forge.py`. Export `COREAI_PYTHON=/path/to/coreai-sdk/bin/python` before running it, unless the training interpreter also contains the SDK. All child source paths resolve to this checkout; no sibling source checkout is required.

Artifacts default to `$HOME/Models/jeff-tetris-training/`; override with `JEFF_TETRIS_STORE`. The base checkpoint defaults to `$HOME/Models/jeff/jeff-base-v1.3`; override with `JEFF_BASE_MODEL`. Both variables accept `~` and relative paths. The controller retains the October 8 experiment's dataset layout (`pilot-v1`, `overnight-20261008-pool`, and `overnight-20261008-initial-corrections`) inside that store. It is a continuation recipe requiring the pilot, fresh pool, corrections and pilot development report, rather than a fresh-data bootstrap command.

The completed run is under `$HOME/Models/jeff-tetris-training/overnight-20261008/`. Its logs retain the original source paths for provenance. Copying the source into system1-ane does not resume that run or restart its servers.

- Fresh pool: `datasets/overnight-20261008-pool`, containing 4,096 new training examples from games up to 300 pieces. Historical training and all original offline split states were excluded when generating it.
- Initial corrections: `datasets/overnight-20261008-initial-corrections`, with 1,379 unique teacher-labeled states collected from 16 pilot games.
- Each round continues the best adapter on up to 2,048 rows: about half fresh examples, one quarter original training examples, and one quarter corrections. Both original and new offline validation/test states are excluded from every training mixture. Further corrections are collected from accepted models as time permits.
- Batch 4, rank 16, alpha 32, LoRA learning rate initially `1e-4`, readout learning rate `5e-6`. If a round fails to improve development gameplay, the next round returns to the best adapter and halves the LoRA learning rate, down to `2.5e-5`.
- `--checkpoint-every 64` atomically publishes LoRA/head recovery weights and retains the latest two checkpoints. `latest-recovery.json` points to the latest completed save. Optimizer state is not saved; recovery starts a fresh optimizer.
- `--train-until` stops optimization between batches. The trainer publishes the adapter, parity rows, merged checkpoint, and a `report.json` save marker before post-training evaluation; it later updates the report with evaluation metrics. The controller allows up to 20 minutes after the optimization cutoff for this work, within the final deadline. The October 8 schedule used a **5:15 AM Pacific optimization cutoff and 6:00 AM Pacific final deadline**. Creating a `STOP` file inside the run directory stops the active stage.
- Selection uses the original eight development seeds, ordered by mean pieces and then mean lines, after export, compilation, cache-placement audit and four-prompt parity. The selected model receives a final 32-game benchmark using the previously evaluated test seeds; this is a repeated benchmark, not a pristine new holdout.
- `status.json` records active stage, log, child PID, deadline, disk space and completed rounds. `best.json` points to the best adapter, merged checkpoint and ANE build. All final adapters and reports are retained; only this overnight run's superseded, rebuildable merged/build directories are pruned. Existing models are preserved.
- After a stopped run, `--resume-final` reopens that output directory, finishes a pending round whose adapter and report were saved, then evaluates the selected best checkpoint. If the trainer stopped during merge, the saved adapter is merged first. Pass the original `--deadline` and `--train-cutoff`; changing either is rejected. This mode can run after the optimization cutoff but before the final deadline, never calls training, and retains the failure history. An existing final-test output requires explicit review rather than being overwritten.
- The runner uses isolated port 8789 for collection/evaluation and leaves the selected model served there on successful completion. Its `base` selection is the chosen trained checkpoint. Existing servers on 8787 and 8788 remain separate.

The completed run used `caffeinate -i` and a thread heartbeat. It ended at 5:45 AM Pacific on October 8; the heartbeat is paused.

### Completed overnight results

Greedy ANE evaluation, the browser's spawn/top-out rules, the same 32 benchmark seeds, and a 150-piece cap:

| Policy | Mean pieces | Mean lines | Reached cap |
| --- | ---: | ---: | ---: |
| Previous Tetris adapter | 59.84375 | 12.40625 | 0/32 |
| Pilot before continuation | 72.9375 | 15.125 | 1/32 |
| **Round 02, selected checkpoint** | **149.75** | **53.75** | **30/32** |
| El-Tetris teacher | 150 | 57.6875 | 32/32 |

The selected adapter is `overnight-20261008/round-02/model/adapter`, with merged weights in the adjacent `merged/` and the exported build in `round-02/build/coreai`. `best.json`, `status.json`, and `final-test/report.json` identify the saved winner and per-game results. Warm server latency averaged 288.20 ms/move across 4,792 moves (p50 246.58 ms, p95 370.72 ms). These are visited-board timings, not a fixed-input speed comparison.

This is a **repeated benchmark**, not a pristine new holdout. Round 03 saved training weights, but its ANE load failed before numerical parity or gameplay could be measured; it was not promoted. The winning round 02 passed four-prompt argmax parity after rebuilding its readout cache, then completed the 32-game evaluation. These are recorded results from the original worktree; source migration did not retrain or rerun ANE gameplay.

## Checks

```sh
python3 -m unittest discover -s tests -p 'test_jeff_tetris*.py' -v
python -m unittest discover -s tests -p test_jeff_coreai.py -v
# Training Python with torch installed; uses only tiny CPU models:
python -m unittest discover -s tests -p test_jeff_training_deadline.py -v
```

These check CPU data generation, teacher labels, exact prompt compatibility with the browser port, split isolation, deterministic generation, top-out behavior, correction deduplication and the Forge training arguments. They do not prove trained policy quality, GPU/ANE parity or ANE placement.
