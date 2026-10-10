# Adapters

Jeff v1.3 is adapter-first: a fixed base plus a LoRA, its own readout, and a fitted temperature. This repo **merges** adapters into weights and converts each merge to its own Core AI package. The server selects a loaded package; it does not apply LoRA at serve time.

## Sample LoRAs (this repo)

| Name | Trainer | Oracle | Typical prompt |
| --- | --- | --- | --- |
| `snake` | `jeff-train-lora --task snake` | Shortest safe path to food | 8×8 board in `latest` (`H` head, `#` body, `F` food) |
| `tetris` | `jeff-train-lora --task tetris` | El-Tetris placement | Board in `latest`; options are legal rotation+column sentences |

Code: `coreai/jeff_snake.py`, `coreai/jeff_tetris.py`, `coreai/jeff_lora.py`, `scripts/jeff_lora_train.py`.

Eval against a running server:

```sh
python scripts/jeff_snake_eval.py --url http://127.0.0.1:8787 --games 8 \
  --adapter base --adapter snake
python scripts/jeff_tetris_eval.py --url http://127.0.0.1:8787
```

## Published PEFT (mstrasser, v1.3, Apache-2.0)

| Name | Hub | What it does |
| --- | --- | --- |
| `triage` | `mstrasser/jeff-adapter-triage` | Route a support message |
| `tools` | `mstrasser/jeff-adapter-tools` | Answer vs tool |
| `guard` | `mstrasser/jeff-adapter-guard` | Injection / benign |
| `spam` | `mstrasser/jeff-adapter-spam` | Spam / phishing / ham |

```sh
python scripts/jeff_peft_merge.py \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --adapter "$HOME/Models/jeff/adapters/jeff-adapter-triage" \
  --output "$HOME/Models/jeff-published/triage"
python forge.py jeff-convert \
  --model "$HOME/Models/jeff-published/triage" \
  --output "$HOME/Models/jeff-coreai/adapters/triage"
python forge.py compile --build "$HOME/Models/jeff-coreai/adapters/triage/coreai"
```

`jeff-convert` stores the fitted temperature under `manifest["convert"]["temperature"]`. `JeffCoreAI` applies it so a served adapter uses its own temperature while embeddings stay on `--model`.

Score the merge before convert: `scripts/jeff_peft_torch.py`. Compare compiled head vs that merge: `scripts/jeff_lora_parity.py`.

## Serving several builds

The Tetris build below is the October 8 round-02 model (or `$JEFF_MODELS_DIR/tetris` from `scripts/download_model.py`). The older `jeff-coreai/adapters/tetris` export is superseded.

```sh
python forge.py system1-ane-serve \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --build "$HOME/Models/jeff-coreai/coreai" \
  --adapter snake="$HOME/Models/jeff-coreai/adapters/snake/coreai" \
  --adapter tetris="${JEFF_TETRIS_BUILD:-$HOME/Models/jeff-tetris-training/overnight-20261008/round-02/build}/coreai" \
  --adapter triage="$HOME/Models/jeff-coreai/adapters/triage/coreai" \
  --adapter tools="$HOME/Models/jeff-coreai/adapters/tools/coreai" \
  --adapter guard="$HOME/Models/jeff-coreai/adapters/guard/coreai" \
  --adapter spam="$HOME/Models/jeff-coreai/adapters/spam/coreai" \
  --host 127.0.0.1 --port 8787
```

Request field `"adapter": "snake"` or `"model": "snake"` selects a loaded build. Omitting both stays on `base`. `GET /health` lists `adapters`. The demo dropdown is filled from that list.

Loading six compiled builds at once can exhaust IOSurface during long Snake autoplay (`NDArray` allocation failure). A single decision on each adapter is fine. Prefer two or three resident engines for games.

## What is not in this tree

- **Runtime streamed LoRA** (ANE inputs, no recompile) lives on branch `cursor/jeff-lora-stream` (anemll-forge PR #10). This repo's serve path is merge-then-convert.
- **Prefix-cache policy** (live-last reuse) lives on `cursor/jeff-prefix-cache` (PR #7). The snapshot/resume **hook** is in `jeff_prefix.py` and the server; the cache object is not installed.
- The older checkpoint `Jeff-Qwen3.5-0.8B` (state-first prompt) is not a dropdown entry: this server always renders v1.3 live-last.
