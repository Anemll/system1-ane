# Changelog

## Unreleased

### Added
- Optional Unsloth-trained stock Qwen3.5-0.8B Snake adapter (`snake-stock`): host-side Clef head engine (host fp32 default, optional fp16 ANE head), LoRA merge and verification scripts under `scripts/unsloth/`, serving how-to in `docs/SYSTEM1_ANE_SERVE.md`; prebuilt on Hugging Face as `anemll/system1-base-unsloth` (folder `snake-stock-qwen3.5-unsloth/`, `scripts/download_model.py snake-stock-qwen3.5-unsloth`). Jeff remains the default; the three original downloads are unchanged.

### Fixed
- `scripts/unsloth/make_snake_data.py` now writes a validation set with no board in the training set (5 of 256 validation boards used to appear in the 2,048 training rows; they are replaced in place, the other rows keep their index). The published Unsloth accuracies were measured on the earlier split (5 of 256 validation states also appeared in training); remeasurement on the deduplicated split pending.
- Clef (Unsloth) adapters: the prompt token count is checked against the server's max-tokens limit before the backbone runs (HTTP 422, same message as other builds); `score` and `noul` questions get a 422 (the joint-schema head is trained on choice fields only) instead of a 500.
- `jeff-train-lora --train-until` is applied to the two timed warm-up steps too (an expired deadline no longer enters the optimizer).
- `scripts/unsloth/run_stock_snake.py` reads `heldout.jsonl` only for `--parity`, `--heldout` and the default run.
- `scripts/download_model.py snake-stock-qwen3.5-unsloth` now lands in `$JEFF_MODELS_DIR/snake-stock-qwen3.5-unsloth/{coreai,model}` like the other builds (an older nested copy is flattened on the next run).
- README / docs: one example port (8796), the compile loop includes the stock build, Python 3.11 and the Core AI SDK venv setup, `python forge.py system1-ane-serve`.
- Decision-package loader (`coreai/jeff_coreai_runtime.py`), `forge.py` build detection and `graph_line` now accept manifest `"kind": "decision"` as well as the original `"jeff-decision"`, so published builds can carry a neutral manifest. Writers still emit `jeff-decision`.

## v0.1.0 — 2026-10-09 — first standalone release

First tagged release of System 1 ANE: Jeff (hybrid Qwen3.5-0.8B decision model) on the Apple Neural Engine via Core AI,
with prebuilt Snake / Tetris / base demo builds on Hugging Face.

### Changed
- Server renamed to **System 1 ANE**: `forge.py system1-ane-serve` (`jeff-serve` stays as a deprecated alias that prints a notice),
  `scripts/system1_ane_serve.py` (thin `scripts/jeff_serve.py` shim kept), `docs/SYSTEM1_ANE_SERVE.md`.
- Server environment variables are now `SYSTEM1_ANE_QUEUE_MS`, `_PREFILL`, `_API_KEY`, `_HOST`, `_MAX_TOKENS`, `_URL`; the old
  `JEFF_*` names still work as deprecated fallbacks. Model, adapter, build and training names are unchanged.
- Demo page branded "System 1 ANE" (model names unchanged); README reframed as System 1 ANE.

### Added
- **Tetris round-02 model** (October 8 overnight continuation): 30/32 benchmark games reach the 150-piece cap
  (149.75 pieces, 53.75 lines; El-Tetris teacher 57.69 lines), up from 59.8 pieces / 12.4 lines for the previous adapter.
  Re-validated on M5 Max on 2026-10-09: all packages `fully_ane`, 4/4 ANE-vs-PyTorch argmax parity.
- `scripts/download_model.py`: fetch the `tetris`, `snake` and `base` Core AI builds from Hugging Face
  (`anemll/jeff-ane-*-coreai`) into `$JEFF_MODELS_DIR` (default `~/Models/jeff-ane`); root `config.json` first.
- `hf/`: model cards, LICENSE/NOTICE and `hf/stage.py` to stage the upload folders (scrubs local paths).
- `notebooks/`: quick start (download → compile → serve → one decision per model), Tetris model, Snake demo.
- `docs/MODELS.md`: model distribution, sizes, licensing.

### Changed
- README quick start now starts from the prebuilt builds; training/conversion moved under "Build your own".
- Base checkpoint default is `$JEFF_BASE_MODEL` or `~/Models/jeff/jeff-base-v1.3` (no hardcoded home folder);
  docs and script examples use `$HOME` / `/path/to/coreai-sdk`.
- `requirements-inference.txt` adds `huggingface_hub`.

### Standalone repo (since anemll-forge)

- Port the remaining October 8 Tetris work from the Forge worktree: teacher/noisy/recovery datasets, collection/evaluation, and bounded continuation with final-evaluation recovery.
- Training steps explicitly enable gradients and restore the caller mode, even after inference-only helper imports.
- Trainer and launcher: evaluation batch/sample controls, synchronized timings, intermediate recovery weights, and timezone-aware optimization deadlines.
- Demo: select the trained Snake/Tetris adapters after health discovery, label them clearly, and start Snake after adapter initialization.
- Dataset/deadline tests and workspace handoff; controller source paths use this checkout, with external artifacts and `COREAI_PYTHON` configuration.
- Native README, docs (architecture, pipeline, API, adapters, results), HISTORY.
- `forge.py` keeps Jeff commands only: `jeff-train-lora`, `jeff-convert`, `jeff-smoke`, `jeff-serve`, `compile`, `doctor`.
- Shared Core AI graph/runtime files Jeff imports stay in place (origin: anemll-forge). 27B models, scripts, docs, and tests removed.
- `jeff-train-lora` launcher now forwards `--init-adapter`, `--heldout-dataset`, `--eval-only`, and `--device cuda`.
- Core AI SDK / coremltools imports are lazy so Linux can import kept modules and run `--help`.
- Samples: `examples/decide_http.py`, `examples/runtime_prompt.py`.
- `runtime_prompt.py` tokenizes via `jeff_tokenize.py` / `TOKENIZER_PYTHON` (Core AI venv has no transformers).
- README: `JEFF_QUEUE_MS=1500` live demo, `COREAI_PYTHON` required from the host venv, bridge dylib default, Tetris ~367 ms / 1430 tok.

## Before v0.1.0 — Jeff on anemll-forge (2026-10-07)

Recorded on the source branch `cursor/jeff-snake-lora-055e` and PRs #6–#9:

- Path B convert/compile/parity on M5 Max (PR #6).
- Decision server and Snake demo (PR #8).
- Snake + Tetris LoRA, published adapters, multi-width prefill, `JEFF_PREFILL` (PR #9 and follow-ups through `df7089e`).
