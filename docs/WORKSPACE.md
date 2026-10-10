# Jeff workspace handoff

## Primary repository

Use **system1-ane** for all new Jeff source work, reviews, and agent tasks. The October 8 Tetris work was ported from the `anemll-forge-jeff-serve` worktree on top of the standalone cleanup (`8af202e`). The original worktree is retained for provenance; it is not the destination for further Jeff development.

The port includes the previously uncommitted trainer/demo edits and all seven previously untracked project/test files. Existing standalone fixes, CUDA support, lazy SDK imports, and the Jeff-only launcher are retained. No 27B-only Forge code was restored.

## Review entry points

- `projects/tetris_training/dataset.py`: teacher/noisy/recovery examples, split isolation, checksums and audit.
- `projects/tetris_training/play.py`: browser-compatible top-out, fixed-seed evaluation and teacher corrections.
- `projects/tetris_training/overnight.py`: bounded continuation, saved recovery weights, candidate selection and final-evaluation recovery.
- `scripts/jeff_lora_train.py` and `forge.py`: normal LoRA plus readout training and forwarded controls.
- `scripts/jeff_demo.html`: trained adapter defaults and health-gated startup.
- `tests/test_jeff_tetris_dataset.py` and `tests/test_jeff_training_deadline.py`: data, launcher, deadline and recovery checks.

The [subproject README](../projects/tetris_training/README.md) records commands, results and limitations.

## External artifacts and environments

Keep weights, data and compiler caches outside Git:

| Default location | Contents |
| --- | --- |
| `$HOME/Models/jeff/jeff-base-v1.3` | Base checkpoint |
| `$HOME/Models/jeff-tetris-training` | Datasets, runs, builds, evaluation reports and logs |
| `$HOME/Models/jeff-tetris-training/overnight-20261008` | Completed overnight run; selected model and final 32-game benchmark |
| `$HOME/Models/jeff-coreai` | Existing base and adapter exports |
| `$HOME/Models/jeff-unsloth-snake/jeff_unsloth_snake_a100` | Separate A100 Snake export (Unsloth-trained). Unsloth Clef prompts and heads are handled by `coreai/clef_engine.py` for builds that ship a `clef_head/` folder; see [SYSTEM1_ANE_SERVE.md](SYSTEM1_ANE_SERVE.md#serving-the-unsloth-trained-stock-snake-adapter-optional) |

The selected Tetris checkpoint is `overnight-20261008/round-02/model/adapter`. Consult `best.json` and `final-test/report.json` in that run for evidence.

Run training with a Python environment containing the training dependencies. Set `COREAI_PYTHON` to the SDK interpreter for conversion/compile/serving. Existing external environments can be reused; they are dependencies, not alternative source checkouts. The controller uses the invoking Python for training, its own checkout for all source, and `JEFF_TETRIS_STORE` / `JEFF_BASE_MODEL` for optional artifact overrides.

The migration does not restart running servers or change their model mappings. A process launched from the old worktree keeps running that source until explicitly restarted from system1-ane. Historical logs retain their original paths. No training is automatically resumed by this handoff.

## Verification

Run with the training interpreter, from system1-ane:

```sh
python forge.py system1-ane-serve --help
python forge.py jeff-convert --help
python forge.py jeff-train-lora --help
python -m unittest discover -s tests
```

These checks do not establish new ANE gameplay results; see the recorded benchmark and parity evidence separately.

### Migration validation — October 8, 2026

The port was tested locally with the existing training environment (Python 3.11). Full unittest discovery ran 103 tests: **102 passed, 1 skipped**. The skipped upstream-client compatibility test requires Python 3.12 and the upstream Jeff client. The combined suite exposed an inference-helper import that globally disabled gradients; training now explicitly enables them within each training step and restores the caller's mode, with regression coverage.

All three launcher help commands and the dataset/play/overnight help commands passed. The demo JavaScript passed `node --check`; the HTML is byte-identical to the corrected page from the old worktree. Controller configuration was checked from an unrelated working directory, including artifact and SDK overrides. No new ANE compilation or full-model training/gameplay was run as part of this source migration.
