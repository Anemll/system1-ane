# Unsloth decision model on Qwen3.5-0.8B: Snake

This notebook trains a decision model with Unsloth on Qwen3.5-0.8B using the Snake decision dataset, evaluates it on held-out rows, and saves the LoRA adapter and decision head for later ANE conversion.

The cells (and the matching scripts in this folder) go data -> train -> evaluate/calibrate -> save `adapter_head` + fp16 LoRA + parity file. Converting the result to Core AI, serving it and verifying it on the ANE are separate steps: see "End-to-end run order" below.

Adapted from Unsloth's "Train your own Decision Model" guide (`FastDecisionModel` / `DecisionTrainer`, Clef-style head): https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth#train-with-code

Prebuilt build (optional download): [`anemll/system1-base-unsloth`](https://huggingface.co/anemll/system1-base-unsloth), folder `snake-stock-qwen3.5-unsloth/` (`python scripts/download_model.py snake-stock-qwen3.5-unsloth`). Serving how-to (build layout, `--adapter snake-stock=<build>/coreai`, head backends): [Serving the Unsloth-trained stock Snake adapter](../../docs/SYSTEM1_ANE_SERVE.md#serving-the-unsloth-trained-stock-snake-adapter-optional). Measured results: [docs/RESULTS.md](../../docs/RESULTS.md). The stock path is an optional extra adapter; it only appears in the server and the demo when its build is passed with `--adapter`.

Status: copied from the original Colab run folder on 2026-10-09; secrets and private paths were redacted. Trained adapters, bundles, logs and the local Colab orchestration helpers are not in this repository. The Snake data is regenerated with `make_snake_data.py`.

What is trained: an Unsloth `FastDecisionModel` / `DecisionTrainer` LoRA, r=16, alpha=16, lr 2e-4 cosine, 3 epochs, batch 8 x grad-accum 4, seed 3407, max length 1024, on 2048 Snake rows (256 held-out), base `unsloth/Qwen3.5-0.8B`. Training runs on a Google Colab GPU VM with files in `/content/snake/`.

## Files

| file | what it does |
|---|---|
| `../../notebooks/unsloth_snake_decision_model.ipynb` | T4 notebook (4-bit, fp16). Its cells are the contents of `run_decision.py` split into sections. |
| `run_decision.py` | T4 training script: Snake rows to decision format -> train/evaluate/calibrate -> save adapter+head, fp16 LoRA, 5-row parity file, pack tarball. 4-bit. |
| `run_decision_a100.py` | A100 variant: base not 4-bit (`load_in_4bit=False`, fp16 requested so Unsloth's forced-float32 path is used), extra precision logging. This is the recipe behind the 87.1% Colab held-out accuracy. |
| `post.py` / `post_a100.py` | After training: reload each saved adapter+head from disk, regenerate `parity_final.json` (5 held-out rows: token ids, spans, raw head logits, calibrated probabilities) and `template_example.json` (exact prompt template). Usage: `python post_a100.py qwen`. |
| `make_snake_data.py` | Regenerates `train.jsonl` (seed 0, 2048 rows) and `heldout.jsonl` (seed 1, 256 rows); verified byte-identical to the files used for training. |
| `prep_ckpt.py` | Merges the trained LoRA into the stock Qwen3.5-0.8B weights in fp32 and writes the checkpoint folder `forge.py jeff-convert` reads (zero placeholder readout, decision-code table, config/tokenizer files). |
| `assemble_stock_build.py` | After convert + compile: writes the fp16 language-model weights and config files into `<build>/model/` and the Clef head files into `<build>/coreai/clef_head/`. |
| `run_stock_snake.py` | Local verification without a server: parity vs the 5 Colab rows, sample states, live games, held-out accuracy, head/prefill/total latency. |
| `torch_reference.py` | PyTorch fp32 reference (merged checkpoint + Clef head) vs the same 5 Colab rows. |
| `serve_games.py` | Plays Snake games against a running server (`/v1/systemone`) and reports per-decision latency. |
| `data/stock_snake_parity_rows.json` | The 5 Colab parity rows (`parity_final.json` of the A100 run, 22 KB: token ids, spans, raw logits, calibrated probabilities). |

Repository code the flow relies on (outside this folder): `coreai/clef_head.py` (host head + prompt encoder), `coreai/clef_ane_head.py` (fp16 ANE head and its export), `coreai/clef_engine.py` (serve engine, `CLEF_HEAD_BACKEND=host|ane`), `scripts/convert_clef_head_fp16.py`, the `clef` request in the tokenizer worker and the `ClefEngine` hook in `scripts/system1_ane_serve.py`. Builds without a `clef_head/` folder behave exactly as before. Tests: `tests/test_clef_head.py`, `tests/test_unsloth_snake.py`. Background: `docs/FP16_HEAD_WRAPUP.md`.

## End-to-end run order

Steps 1-2 run on a Colab GPU VM; steps 3-8 run on an Apple-silicon Mac with the Core AI SDK Python (compile/serve/run) plus a Python with `torch`, `transformers` (Qwen3.5 support) and `safetensors` (convert/merge/tokenize). `COREAI_PYTHON` / `TOKENIZER_PYTHON` select the two interpreters where a script needs them. Every generated file below (checkpoints, builds, `.aimodel` packages, weights) stays outside Git.

1. **Data + train** (Colab). `pip install --upgrade unsloth unsloth_zoo`; put `train.jsonl` / `heldout.jsonl` (`python scripts/unsloth/make_snake_data.py OUT_DIR`) and the scripts in `/content/snake/`. Run the notebook cells in order (T4), or on an A100 `python run_decision_a100.py`.
2. **Post-process** (Colab): `python post_a100.py qwen`. Outputs land in `/content/snake/out/qwen/` (`adapter_head/`, `lora_fp16/`, `parity*.json`, `metrics.json`) and are tarred (copied to Google Drive if mounted). Download `adapter_head/`.
3. **Merge the LoRA into the stock weights** (fp32, about 1 minute):
   `STOCK_BASE_DIR=<local snapshot of unsloth/Qwen3.5-0.8B> DECISION_CONFIG=<models dir>/base/model/decision_config.json python scripts/unsloth/prep_ckpt.py <adapter_head> <ckpt dir>`
   (`DECISION_CONFIG` comes from `python scripts/download_model.py base`; only its answer-code table is used.)
4. **Convert to Core AI, prefill width 304**:
   `python forge.py jeff-convert --model <ckpt dir> --output <build> --ctx 2048 --prefill 304 --quant fp16`
   `--prefill` must be a multiple of 8 (300 is rejected). The stock prompt is 261-264 tokens for the usual boards (258-273 over the held-out rows), so a 256-row window needs two ANE calls and 304 needs one. No converter source changes were needed on top of the files in this branch.
5. **Compile for the ANE**: `python forge.py compile --build <build>/coreai`
6. **Assemble the build**: `python scripts/unsloth/assemble_stock_build.py <ckpt dir> <build> <adapter_head>`
7. **(Optional) fp16 ANE Clef head**: `python scripts/convert_clef_head_fp16.py <build>/coreai/clef_head`, then `python coreai/clef_ane_head.py <build>/coreai/clef_head` (Core AI SDK Python; writes `clef_head_ane.aimodel` + `ane_head_config.json`, about 48 MB, generated, not stored in Git).
8. **Serve** (default host fp32 head; `CLEF_HEAD_BACKEND=ane` uses the compiled head; prompts over 304 rows or 8 options fall back to the host head):
   `[CLEF_HEAD_BACKEND=ane] python forge.py system1-ane-serve --model <base build>/model --build <base build>/coreai --adapter snake-stock=<build>/coreai --port 8796`
   (`--model`/`--build` are the base build the server starts from, e.g. the downloaded `base`; the stock build is an additional `--adapter`, selected per request with `"adapter": "snake-stock"`.)
9. **Verify**:
   - `STOCK_BUILD=<build> STOCK_HELDOUT=<dir>/heldout.jsonl python scripts/unsloth/run_stock_snake.py` (add `--heldout 256` for accuracy, `--parity` for the Colab rows only);
   - `python scripts/unsloth/torch_reference.py <ckpt dir> <adapter_head>` for the fp32 PyTorch reference;
   - `python scripts/unsloth/serve_games.py http://127.0.0.1:8796 snake-stock 8` for latency through the server.

## Measured results (local, 9 Oct 2026, Apple M5 Max, machine not otherwise idle; not controlled benchmarks)

The accuracy rows were measured on the earlier split (5 of 256 validation states also appeared in training); remeasurement on the deduplicated split pending; `make_snake_data.py` now writes a validation set that shares no state with training.

| | value |
|---|---|
| Colab held-out accuracy, fp32 (256 rows) | 87.1% |
| ANE backbone + host head, held-out accuracy | 85.2% (256 rows) |
| Parity vs the 5 Colab rows | 5/5 answers match; max \|logit diff\| about 0.51 (fp16 backbone vs Colab fp32) |
| End to end per decision, host fp32 head | about 90 ms (head about 8 ms) |
| End to end per decision, fp16 ANE head | about 84.8 ms (head about 3 ms) |
| 256-row prefill window (two ANE calls) | about 139 ms end to end |
| Reference: the repository's smaller Snake build (214-token prompts) | 63.6 ms |

The head accuracy numbers with the ANE head: max \|dlogit\| 0.005 vs the host fp32 head on 261 rows, no answer flips.

## Reproduction check (local, 9 Oct 2026)

Re-run from the files in this folder on an Apple M5 Max, starting from the trained `adapter_head/` of the Colab A100 run: `make_snake_data.py` regenerated `heldout.jsonl` byte-identical to the 256 evaluation rows (the original `train.jsonl` was not available to compare in this check); `prep_ckpt.py` produced a merged checkpoint whose 488 tensors are bit-identical to the one used for the published numbers; `forge.py jeff-convert --prefill 304` (about 40 s), `forge.py compile` (about 1m45s, 7 of 7 packages) and `assemble_stock_build.py` produced a working build with no converter changes. The verification results of that fresh build are listed under "Fresh-build verification" below. Not re-run here: the Colab training itself (needs a GPU VM), the fp16 head export, the server path and `serve_games.py`/`torch_reference.py` on the fresh build (they were run on the earlier build of the same checkpoint).

Not included (large or third-party): trained adapters/heads, merged exports, bundles (.tgz), run logs, Colab outputs beyond the 5 parity rows, the Unsloth source copies (`decision.py`, `clef.py`, `joint_schema_model.py`; the scripts use the pip-installed `unsloth`). Training itself needs a GPU Colab VM and cannot be re-run locally.
