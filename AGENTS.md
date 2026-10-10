# System 1 ANE development workspace

System 1 ANE is a Core AI demo and server that runs small Qwen-based decision models on the Apple Neural Engine in milliseconds: the System 1 ANE Engine serving the Jeff base model and its Snake and Tetris adapters on Apple Silicon.
Work here, in this repository. Jeff is only the model and adapter name.

## Setup

```sh
python3.11 -m venv .venv && source .venv/bin/activate        # or reuse the selected training Python (3.11+)
python -m pip install -r requirements-inference.txt           # torch, transformers, safetensors, huggingface_hub, ...
export COREAI_PYTHON=/path/to/coreai-sdk/bin/python           # Core AI SDK interpreter: convert, compile, serve
bash coreai/swift_bridge/build.sh                             # once per Mac
python -m unittest discover -s tests                          # CPU only; no Core AI, no ANE
```

Demo server on a spare port, never on a port that is already in use (models: `python scripts/download_model.py tetris snake base`, then `python forge.py compile --build <dir>/coreai` once per Mac):

```sh
export JEFF_MODELS_DIR=/path/to/models                        # where scripts/download_model.py put the models
python forge.py system1-ane-serve --model "$JEFF_MODELS_DIR/base/model" --build "$JEFF_MODELS_DIR/base/coreai" \
  --adapter snake="$JEFF_MODELS_DIR/snake/coreai" --adapter tetris="$JEFF_MODELS_DIR/tetris/coreai" \
  --host 127.0.0.1 --port 8796
```

CI runs the same checks as `.github/workflows/tests.yml`: `forge.py system1-ane-serve|jeff-convert|jeff-train-lora --help`, `scripts/download_model.py --list`, and the unit tests.

## Layout

| Path | What |
| --- | --- |
| `forge.py` | CLI: `system1-ane-serve`, `jeff-convert`, `jeff-smoke`, `jeff-train-lora`, `compile`, `doctor` |
| `scripts/system1_ane_serve.py`, `scripts/jeff_demo.html` | decision server (`POST /v1/systemone`) and the browser demo |
| `coreai/` | Jeff checkpoint, prompt, readout, LoRA, Core AI export and runtime |
| `projects/tetris_training/` | Tetris data, gameplay evaluation, overnight training |
| `scripts/download_model.py`, `hf/`, `docs/MODELS.md` | model download and Hugging Face staging |
| `notebooks/`, `examples/` | quick start, Tetris and Snake notebooks; HTTP and runtime samples |
| `tests/` | unit tests |

Docs: [README](README.md), [CONTRIBUTING](CONTRIBUTING.md), [docs/WORKSPACE.md](docs/WORKSPACE.md) (external artifacts, limitations), [docs/SYSTEM1_ANE_SERVE.md](docs/SYSTEM1_ANE_SERVE.md) (server API), [docs/PIPELINE.md](docs/PIPELINE.md), [docs/RESULTS.md](docs/RESULTS.md).

## Rules

- **Do not disturb running services.** A live demo or other service may already be running on this Mac, served from the main worktree. Test on a spare port and stop what you started. Check with `lsof -nP -iTCP:<port> -sTCP:LISTEN` before launching anything heavy on the ANE.
- **Keep weights, datasets, builds and experiment logs out of Git.** Do not restart completed training runs.
- **Numbers in `docs/RESULTS.md` are marked "measured 7 Oct 2026; remeasurement pending".** Remeasure on a spare port without touching a live demo. Environment to match: M5 Max, macOS 27.2, ANE bonded compile mode 1, `p256_2k` FP16 packages. Do not change the numbers until a rerun.
- **The stock Unsloth Snake adapter uses a different prompt format and a Clef head**, not Jeff's readout. Serving it uses the host-side Clef engine (`coreai/clef_engine.py`, selected for builds that ship a `clef_head/` folder; `CLEF_HEAD_BACKEND=host|ane`) and is an optional extra adapter (`--adapter snake-stock=<build>/coreai`); Jeff remains the default. Prebuilt: `python scripts/download_model.py snake-stock-qwen3.5-unsloth` (Hugging Face `anemll/system1-base-unsloth`).
- **No commit, push, merge, tag, release or publish until Anemll approves.** Report status and ETAs; do not assume.
- Keep secrets, personal paths and hostnames out of the repo and its docs.
