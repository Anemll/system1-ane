# Build and export pipeline

```
checkpoint (jeff-base v1.3)
        │
        ├─ jeff-train-lora ──► adapter/ + merged/
        │                         │
        ├─ jeff_peft_merge.py ────┤  (published PEFT: triage/tools/guard/spam)
        │                         ▼
        │                    jeff-convert ──► model/ + coreai/*.aimodel
        │                         │
        │                    forge.py compile ──► ANE cache (this Mac + this Python)
        │                         │
        └──────────────────► system1-ane-serve --build … --adapter name=…
```

Weights are local. Nothing in this pipeline downloads Hugging Face repos.

## 1. Train (`jeff-train-lora`)

```sh
python forge.py jeff-train-lora \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --output "$HOME/Models/jeff-snake" \
  --task snake
```

Defaults: 256 train / 64 held-out synthetic rows, rank 16, alpha 32, LoRA lr `2e-4`, readout lr `5e-6`, 2 epochs, batch 4. `--device auto` is CUDA, then MPS, then CPU.

| Flag | Role |
| --- | --- |
| `--task snake\|tetris` | Oracle-labeled synthetic rows |
| `--dataset rows.jsonl` | `{state, options, label, instructions}` |
| `--init-adapter DIR` | Continue from a saved `adapter/` |
| `--heldout-dataset FILE` | Fixed held-out JSONL |
| `--eval-only` | Score and exit (no write) |
| `--eval-batch N` | Validation batch size; defaults to the training batch |
| `--eval-train-rows N` | Fixed sample for train-accuracy checks; 0 checks all training rows |
| `--checkpoint-every N` | Recovery LoRA/readout every N steps; latest two retained, no optimizer state |
| `--train-until ISO` | Stop optimization between batches at a timezone-aware deadline, then evaluate/save |
| `--skip-merge` | Keep factors only |
| `--play N` | PyTorch self-play (slow; prefer `jeff_snake_eval.py` on the server) |

Output: `adapter/` (LoRA + readout), `merged/` (full Jeff checkpoint), `report.json`, `parity_rows.json`.

`--dry-run` on `forge.py jeff-convert` prints the convert plan without the SDK.

For full-game teacher/recovery datasets, teacher corrections from Jeff's own games, and ANE gameplay selection, see the [Tetris training subproject](../projects/tetris_training/README.md). Its controller adds subprocess deadlines around training, export and evaluation; `--train-until` by itself only bounds optimization between batches.

## 2. Merge published PEFT

```sh
python scripts/jeff_peft_merge.py \
  --base "$HOME/Models/jeff/jeff-base-v1.3" \
  --adapter "$HOME/Models/jeff/adapters/jeff-adapter-triage" \
  --output "$HOME/Models/jeff-published/triage"
```

Folds `W += (alpha / rank) B A` the same way `jeff.lora.merge_adapter` does, and copies that adapter's readout and temperature.

## 3. Convert (`jeff-convert`)

```sh
python forge.py jeff-convert \
  --model "$HOME/Models/jeff-snake/merged" \
  --output "$HOME/Models/jeff-coreai/adapters/snake" \
  --ctx 2048 --prefill 256 --quant fp16
```

Needs `COREAI_PYTHON` (unless `--dry-run`). Writes:

```
<output>/
  model/     config, tokenizer, decision_config, embed_tokens_fp16.npy, readout
  coreai/    manifest.json, chunk_L00-03.aimodel, …, head_readout.aimodel
```

`manifest.json` has `"kind": "jeff-decision"`, `"dflash2": false`, FP16 KV, and `p{width}_{ctx}k` entries. `--prefill-extra 512,1024,1536,2048` adds more widths over the same weights (`TPS` in the manifest). `--quant int8` is per-channel INT8 on large projections (halves package size, same speed, 2–3× FP16 error on M5 Max).

The exporter lazy-imports `qwen38_coreai_build.py` after rebinding widths. It does not call `M.cfg()` on a 27B checkpoint.

## 4. Compile (`forge.py compile`)

```sh
python forge.py compile --build "$HOME/Models/jeff-coreai/coreai"
```

Loads each package once through the Swift bridge (`coreai_bridge`) so MPSGraph specializes it for this SoC and this Python. `--force` drops that cache first. First compile of the base `p256_2k` FP16 build was 79 s on M5 Max; later loads are ~0.1 s.

SoC policy (`docs/ANE_COMPILE_MODE_POLICY.md`): H17 / M5 family → bonded mode `1`; H18+ / M6 → `2`. Override with `MPSGRAPH_ANE_BONDED_COMPILE_MODE`.

Audit placement:

```sh
python coreai/inspect_coreai_cache.py \
  --model-dir "$HOME/Models/jeff-coreai/coreai" \
  --executable python --strict
```

`--executable python` must match the Core AI SDK interpreter name (the cache key).

## 5. Serve

```sh
export COREAI_PYTHON=/path/to/coreai-sdk/bin/python
python forge.py system1-ane-serve \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --build "$HOME/Models/jeff-coreai/coreai" \
  --adapter snake="$HOME/Models/jeff-coreai/adapters/snake/coreai" \
  --host 127.0.0.1 --port 8796
```

See [SYSTEM1_ANE_SERVE.md](SYSTEM1_ANE_SERVE.md) for the HTTP API and [ADAPTERS.md](ADAPTERS.md) for which builds to load.

## 6. Check

```sh
# Prompt port + FP32 reference probabilities (host venv)
python scripts/jeff_reference.py --model "$M" --jeff-src /path/to/jeff/src --out cases.json

# ANE vs those cases (COREAI_PYTHON)
python forge.py jeff-smoke --model "$M" --build "$HOME/Models/jeff-coreai/coreai" \
  --cases cases.json --bench 5

# Merged-adapter head vs PyTorch
$COREAI_PYTHON scripts/jeff_lora_parity.py \
  --model "$M" --build "$HOME/Models/jeff-coreai/adapters/snake/coreai" \
  --rows "$HOME/Models/jeff-snake/parity_rows.json"
```
