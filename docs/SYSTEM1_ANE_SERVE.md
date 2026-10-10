# Jeff decision server

`forge.py system1-ane-serve` loads one compiled Jeff Core AI package and answers decision requests. It speaks the same route as upstream Jeff's `jeff-serve`: `POST /v1/systemone`. (`jeff-serve` is still accepted as a deprecated alias of `system1-ane-serve`.) The process also serves a browser demo at `/`.

`forge.py` in this repo is Jeff-only (`jeff-convert`, `jeff-smoke`, `system1-ane-serve`, `jeff-train-lora`, `compile`, `doctor`). There is no 27B chat server.

## One command

Weights stay on the machine. The Core AI interpreter loads the packages. The interpreter that launches `forge.py` must have `transformers`, because that is what tokenizes Jeff's chat template (`TOKENIZER_PYTHON`).

```sh
export COREAI_PYTHON=/path/to/coreai-sdk/bin/python   # your Core AI SDK venv

python forge.py system1-ane-serve \
  --model "$HOME"/Models/jeff/jeff-base-v1.3 \
  --build "$HOME"/Models/jeff-coreai/coreai \
  --host 127.0.0.1 --port 8796
```

Then open http://127.0.0.1:8796/. Adding `?sample=1` also classifies the sample refund message on load.

| Page | What it does |
| --- | --- |
| `/` | Snake: every move is one Jeff decision, with per-option probabilities and tokenize / prefill / head / total time. Below it, a routing panel classifies a message into refunds, shipping, technical, or other. |
| `GET /health` | `ready`, model name, checkpoint, `max_options`, modalities `["text"]`, backend, whether a prefix cache is installed |
| `GET /v1/models` | `jeff`, `jeff-latest`, and the checkpoint name (`jeff-qwen3.5-0.8b` for this base) |
| `POST /v1/systemone` | One decision. `POST /v1/decide` is the same handler |

Server settings use `SYSTEM1_ANE_*` environment variables (`SYSTEM1_ANE_QUEUE_MS`, `SYSTEM1_ANE_PREFILL`, `SYSTEM1_ANE_API_KEY`, `SYSTEM1_ANE_HOST`, `SYSTEM1_ANE_MAX_TOKENS`, `SYSTEM1_ANE_URL` for clients). The earlier `JEFF_*` names of the same variables still work as deprecated fallbacks and print a one-line notice.

The package is loaded once. One request runs at a time. `SYSTEM1_ANE_QUEUE_MS` (default `0`) is how long a second request waits before `529` and `Retry-After: 1`. The live demo uses `SYSTEM1_ANE_QUEUE_MS=1500`. `SYSTEM1_ANE_API_KEY`, when set, requires `Authorization: Bearer …` on `/v1/systemone`, `/v1/decide`, and `/v1/models`. `/health` and the demo page stay open. `SYSTEM1_ANE_MAX_TOKENS` can lower the cap; it cannot exceed the build's KV rows (2,048 on the `p256_2k` package). Upstream Jeff's default of 8,192 does not fit this build.

The server is text only. A request with `images` is `422`.

## Request and response

The body matches upstream Jeff (`jeff.client` / the v1.3 request format in the firelex/jeff repo):

```json
{
  "model": "jeff-latest",
  "state": {"rules": "stable text", "latest": "the part that changes"},
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Pick the snake's next move.",
      "criteria": {"up": "one cell up", "down": "one cell down", "left": "one cell left", "right": "one cell right"}
    }
  }
}
```

`state` may be text, a JSON object, or a list. `questions` may also be `noul` or `score`. `orders: 2` answers each question again with its options reversed and averages the two distributions.

A short form is accepted when `questions` is omitted:

```json
{"state": "The disk is full.", "options": ["page", "wait", "ignore"], "instructions": "What next?"}
```

A list of strings becomes option keys `o1`, `o2`, … (the same keys `jeff.client.choose` uses for a list). An object is the criteria map, key to description (`null` means the key alone).

The response is Jeff's decision object plus timings:

```json
{
  "model": "jeff-qwen3.5-0.8b",
  "answers": {
    "move": {"type": "choice", "choice": "up", "probabilities": {"up": 0.4, "down": 0.2, "left": 0.2, "right": 0.2}, "confidence": 0.2}
  },
  "usage": {"input_tokens": 180, "output_tokens": 0, "orders": 1},
  "timings": {
    "tokenize_ms": 1.2,
    "calls_ms": [64.0],
    "prefill_ms": 64.0,
    "head_ms": 0.3,
    "total_ms": 66.0,
    "questions": [{"id": "move", "tokenize_ms": 1.2, "calls_ms": [64.0], "prefill_ms": 64.0, "head_ms": 0.3, "tokens": 180, "prefix_tokens": 0}]
  }
}
```

`calls_ms` is one number per 256-row prefill call. `prefill_ms` is their sum. `head_ms` is the readout. `total_ms` is the whole request inside the model lock (tokenize, prefill, head, and the small Python around them). `prefix_tokens` is how many leading tokens were resumed from a cache; it is 0 until a cache is installed.

Aliases `jeff` and `jeff-latest` (and the legacy name `jeff-qwen3.8-27b`) select the base build. A loaded adapter is selected with `"adapter": "snake"` or with `"model": "snake"` (the same name is listed by `GET /v1/models`). Omitting both stays on the base. The response includes `"adapter"`.

The demo's Snake panel has a dropdown filled from `/health` (`adapters`). It shows each move's probabilities, the food eaten this life, and the best life since the last switch. Switching adapters resets the board to the opening position. The routing panel stays on the base build.

An existing Jeff client points at this server with no other change, for text requests:

```python
from jeff import Client
jeff = Client("http://127.0.0.1:8796", model="jeff-latest")
jeff.choose("The disk is full.", {"page": "Page someone.", "wait": "Wait."})
```

## Sample LoRA (Snake)

Base Jeff has no Snake adapter, so the demo's first moves are soft (top probability about 0.34) and the snake misses the food. `jeff-train-lora` is a copyable sample: it builds rows of `{state, options, label, instructions}`, renders them with the same Jeff prompt the server uses, and trains a rank-16 LoRA (alpha 32) plus the readout. LoRA and the readout are separate AdamW groups (`2e-4` and `5e-6`). Loss is cross-entropy on the masked option-code logits. Temperature is applied only when the probabilities are formed, the same way `JeffCoreAI.decide` divides by `decision_config` temperature.

Snake rows are synthetic. An oracle plays a shortest safe path to the food (walls and the body are illegal; the tail cell is free because it vacates). The board text matches the demo: `H` head, `#` body, `F` food, `.` empty, and the state object's last field is `latest`.

```json
{"state": {"rules": "8 by 8 grid. …", "latest": {"board": "…", "head": "4,2", "food": "1,6"}},
 "options": {"up": "move the head one cell up", "down": "move the head one cell down",
             "left": "move the head one cell left", "right": "move the head one cell right"},
 "label": "up",
 "instructions": "Pick the snake's next move."}
```

`options` may be a list of strings. `label` is a key or a zero-based index. Pass your own file with `--dataset rows.jsonl` (an 80/20 split). The Snake generator is the default.

```sh
python forge.py jeff-train-lora \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --output "$HOME/Models/jeff-snake"
```

Default is 256 train rows, 64 held-out, rank 16, alpha 32, LoRA learning rate `2e-4`, readout learning rate `5e-6`, 2 epochs, batch 4. That split matches upstream `jeff-train --lora-rank 16 --lr 2e-4 --readout-lr 5e-6`. The loss is cross-entropy on the masked readout logits and does not divide by temperature; `JeffCoreAI.decide` still divides by `decision_config` temperature before the softmax. `--device auto` is MPS when it is available. An ANE compile does not move the job to CPU. The run prints train and held-out accuracy for the base and the adapter, and seconds per step. On Snake it also scores the base model with a hint prompt (food direction and the safe-move list added in front of `latest`). `--task tetris` uses the El-Tetris placement oracle instead. `--play N` adds PyTorch self-play; the default is 0 because each move is a full forward. Game scores for the served builds come from `scripts/jeff_snake_eval.py`.

The script writes:

| Path | What it is |
| --- | --- |
| `adapter/` | LoRA factors (`adapter_model.safetensors`), the trained readout, `adapter_config.json` |
| `merged/` | Full Jeff checkpoint with `W <- W + (alpha/rank) B A` folded in |
| `report.json` | Accuracy, optional games, device |
| `parity_rows.json` | A few held-out token rows and PyTorch probabilities |

This is not a PEFT folder and the server does not apply LoRA at runtime. Deploy by merging, then converting that checkpoint into its own Core AI directory so the package stays fully on the ANE:

```sh
export COREAI_PYTHON=/path/to/coreai-sdk/bin/python
python forge.py jeff-convert \
  --model "$HOME/Models/jeff-snake/merged" \
  --output "$HOME/Models/jeff-coreai/adapters/snake"
python forge.py compile --build "$HOME/Models/jeff-coreai/adapters/snake/coreai"
```

Check the merged build against the PyTorch probabilities:

```sh
$COREAI_PYTHON scripts/jeff_lora_parity.py \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --build "$HOME/Models/jeff-coreai/adapters/snake/coreai" \
  --rows "$HOME/Models/jeff-snake/parity_rows.json"
```

`--model` on the parity script is the base checkpoint. Embeddings stay there. This Snake build keeps the base temperature. A published adapter's fitted temperature is read from its build manifest, under `convert.temperature`. The Snake weights and readout are in the compiled build.

Serve both builds. `base` is `--build`. Each `--adapter` is `name=` plus that build's `coreai/` directory:

```sh
python forge.py system1-ane-serve \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --build "$HOME/Models/jeff-coreai/coreai" \
  --adapter snake="$HOME/Models/jeff-coreai/adapters/snake/coreai" \
  --adapter triage="$HOME/Models/jeff-coreai/adapters/triage/coreai" \
  --adapter tools="$HOME/Models/jeff-coreai/adapters/tools/coreai" \
  --adapter guard="$HOME/Models/jeff-coreai/adapters/guard/coreai" \
  --adapter spam="$HOME/Models/jeff-coreai/adapters/spam/coreai" \
  --host 127.0.0.1 --port 8796
```

Food eaten, survival steps, and per-decision latency, same opening boards for each name:

```sh
python scripts/jeff_snake_eval.py --url http://127.0.0.1:8796 --games 8 \
  --adapter base --adapter snake
```

The code behind the sample is `coreai/jeff_snake.py` (oracle and the row format), `coreai/jeff_lora.py` (the low-rank update and the merge), and `scripts/jeff_lora_train.py`.

## Prefix cache hook

Every request currently starts from an empty Gated DeltaNet state and position 0. A later live-last cache plugs in at two places:

1. **Runtime.** `JeffCoreAI.capture_state()` copies position, token ids, the last hidden row, and each chunk's GDN/conv state and KV. `JeffCoreAI.prefill(token_ids, prefix=snapshot)` and `decide(..., prefix=snapshot)` restore that snapshot and prefill only the suffix. The shape check lives in `coreai/jeff_prefix.py` (`resume_at`). A snapshot that already covers the prompt runs the readout on the cached hidden row and does not call the backbone.
2. **Server.** Set `app.prefix_cache` to an object with `lookup(token_ids) -> snapshot | None` and `store(token_ids, snapshot)`. After each question the server passes `lookup`'s snapshot into `decide` and then stores `capture_state()`. With `prefix_cache is None` (the default) the server does not copy KV.

The demo's snake state is an object whose last field is `latest` (the board). Question, options, and `rules` stay in front of that, which is the split a live-last cache would reuse. The cache policy itself is not in this branch. When more than one build is loaded, the hook is applied only to `base`, so a snapshot from one package is not restored into another.

## What stays on the ANE

`system1-ane-serve` loads the same `p256_2k` packages as `jeff-smoke`. Placement is a property of the cached specialization, checked with:

```sh
python coreai/inspect_coreai_cache.py \
  --model-dir "$HOME"/Models/jeff-coreai/coreai \
  --executable python --strict
```

`--executable python` matches the Core AI SDK venv (the cache key is the interpreter name). The head and every chunk entry should be `fully_ane`. The inspector originates from anemll-forge ([ORIGIN.md](ORIGIN.md)).

## Serving the Unsloth-trained stock Snake adapter (optional)

This is an **optional extra adapter, not the default**. Jeff stays the default model and adapter of the server and the demo. The stock option (`snake-stock`) exists only when you pass its build with `--adapter snake-stock=<build>/coreai`; without that flag the server, `/health` and the demo dropdown are exactly as before, and adapters without a `clef_head/` folder never import the Clef code.

The adapter is a stock `unsloth/Qwen3.5-0.8B` backbone with a LoRA and an Unsloth decision (Clef) head trained on Snake boards (`notebooks/unsloth_snake_decision_model.ipynb`, `scripts/unsloth/`). Its prompt layout and head differ from Jeff's, so the build carries its own backbone and a host-side head:

```text
<build>/
  coreai/                      # merged-LoRA backbone, compiled Core AI chunks + manifest.json (--prefill 304)
    clef_head/
      joint_head.safetensors        # the Unsloth head (required: this folder selects ClefEngine)
      joint_head_config.json
      unsloth_decision_config.json
      joint_head_fp16.safetensors   # optional, scripts/convert_clef_head_fp16.py
      clef_head_ane.aimodel         # optional, generated: python coreai/clef_ane_head.py <build>/coreai/clef_head
      ane_head_config.json          #   (about 48 MB, generated, never stored in Git)
  model/                       # fp16 backbone weights + tokenizer/config of the merged checkpoint
```

Prebuilt: the build is on Hugging Face as [`anemll/system1-base-unsloth`](https://huggingface.co/anemll/system1-base-unsloth) (folder [`snake-stock-qwen3.5-unsloth/`](https://huggingface.co/anemll/system1-base-unsloth/tree/main/snake-stock-qwen3.5-unsloth)); `python scripts/download_model.py snake-stock-qwen3.5-unsloth` downloads it (root `config.json` plus that folder), then run `forge.py compile --build "$JEFF_MODELS_DIR/snake-stock-qwen3.5-unsloth/coreai"` once per Mac (the build lands flat as `$JEFF_MODELS_DIR/snake-stock-qwen3.5-unsloth/{coreai,model}`, like the other builds; an older nested copy is flattened by the next download). Serving it needs `torch` in `COREAI_PYTHON` because the Clef head runs in PyTorch, and a Clef adapter answers `choice` questions only (a `score` or `noul` question gets HTTP 422); the same `max_tokens` limit as the other builds applies. Or build it yourself:

Flow (details and a run order in [`scripts/unsloth/README.md`](../scripts/unsloth/README.md)):

1. **Train** (Colab): the notebook or `scripts/unsloth/run_decision_a100.py`, then `post_a100.py`.
2. **Merge** the LoRA into the stock weights in fp32: `scripts/unsloth/prep_ckpt.py`.
3. **Convert**: `python forge.py jeff-convert --model <merged ckpt> --output <build> --ctx 2048 --prefill 304 --quant fp16`. The stock prompt is 261-264 tokens (258-273 seen), so a 256-row window needs two ANE calls and 304 needs one (`--prefill` is a multiple of 8).
4. **Compile**: `python forge.py compile --build <build>/coreai`, then `scripts/unsloth/assemble_stock_build.py` to add `model/` and `clef_head/`.
5. **Serve**: `python forge.py system1-ane-serve --model <base>/model --build <base>/coreai --adapter snake-stock=<build>/coreai --port 8796`. Add `--adapter snake=<snake build>/coreai` as usual; request it with `"adapter": "snake-stock"` (the Snake demo page lists it in the adapter dropdown only when it is loaded). Startup prints `adapter snake-stock: Unsloth Clef head (...)`.
6. **Verify**: `scripts/unsloth/run_stock_snake.py` (parity vs the 5 Colab rows, held-out accuracy, games, latency), `scripts/unsloth/torch_reference.py` (fp32 PyTorch reference), `scripts/unsloth/serve_games.py` (games through the running server).

Head backend: `CLEF_HEAD_BACKEND=host` (default) runs the head in PyTorch fp32 on the host CPU, about 8 ms per decision. `CLEF_HEAD_BACKEND=ane` runs it as a fixed-shape fp16 Core AI package (304 prompt rows x 8 options), about 3 ms (the ANE call itself about 2.2 ms); prompts over 304 rows or 8 options fall back to the host head, and builds without the package use the host head. Measured locally on an M5 Max: about 90.9 ms per decision with the host head, about 85 ms with the ANE head, 85.2% held-out accuracy (Colab fp32 87.1%); see [RESULTS.md](RESULTS.md) and [FP16_HEAD_WRAPUP.md](FP16_HEAD_WRAPUP.md). These are local measurements, not controlled benchmarks.

## Tests

Request and response tests use a fake engine, so they do not load Core AI:

```sh
python -m unittest discover -s tests
```

`tests.test_system1_ane_serve.UpstreamClientTests` runs the real `jeff.client` against that fake engine when `JEFF_SRC` (default `$HOME/Models/jeff/jeff-src/src`) is present.

## Measured latency

Apple M5 Max, the `p256_2k` FP16 package (measured 7 Oct 2026; remeasurement pending). Ten warm decisions after two warmup calls, median. An idle 27B `forge.py serve` was already resident on port 8766 and was not generating; no other Jeff ANE bench was running. The medians match the earlier single 256-row prefill (about 63 ms), so that idle server did not move them.

| Demo request | Tokens | Prefill calls | Tokenize | Prefill | Head | Total | HTTP wall | Decisions/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Snake, opening board | 213 | 1 × 63.6 ms | 0.47 ms | 63.6 ms | 0.34 ms | 64.7 ms | 65.3 ms | 15.5 |
| Route a refund message | 133 | 1 × 63.6 ms | 0.38 ms | 63.6 ms | 0.36 ms | 64.7 ms | 65.2 ms | 15.5 |
| Tetris placement | 1430 | 6 × 256 | — | ~367 ms | — | ~367 ms | — | — |

A 256-row call costs about 63 ms whether the prompt fills it or not, so both demo prompts land on the same latency. Decisions/s is `1000 / total_ms` inside the model lock. The browser adds the 160 ms pause between snake moves on top of that.

The refund message was classified `refunds` at probability 0.85. The snake's first moves were soft (top option about 0.34): this is the base checkpoint with no adapter.

## Sample Snake LoRA (M5 Max, 7 October 2026)

Rank 16, alpha 32, LoRA learning rate `2e-4`, readout learning rate `5e-6`, batch 4, 2 epochs, 256 train / 64 held-out. Backbone bf16 on MPS with bf16 autocast; LoRA factors and the readout stay fp32. Loss is cross-entropy on the masked 255-way logits. Temperature 1.075 is applied only when probabilities are formed.

| Device | Steady s/step | Notes |
| --- | --- | --- |
| CPU | 12.91 | Batch 4, rank 8, fp32, one learning rate on LoRA and the readout. 826 s for 64 steps. That run diverged (loss 1.34 to 2.05). |
| MPS | 1.58 | Batch 4, rank 16, bf16 autocast. Timed steady step after a 4.92 s warmup. Epoch steps were 1.46 s and 1.43 s. |

`jeff-src` `mlx_lora.py` applies PEFT adapters at serve time. It is not a trainer, so there is no MLX training step time.

| Split | Accuracy | Loss | n |
| --- | --- | --- | --- |
| Base, train | 0.309 | 1.337 | 256 |
| Base, held-out | 0.281 | 1.352 | 64 |
| Base, held-out, hint prompt | 0.484 | 1.121 | 64 |
| Adapter, train | 0.598 | 0.887 | 256 |
| Adapter, held-out | 0.594 | 0.932 | 64 |

The hint prompt adds food direction and the safe-move list in front of `latest`. It is scored on the base model only. The whole MPS run, including both evals and the merge, took 300 s. The merged checkpoint is `$HOME/Models/jeff-snake/merged`.

The merged checkpoint was converted to `$HOME/Models/jeff-coreai/adapters/snake` and compiled (`p256_2k`, FP16, bonded mode 1) in 10 m 18 s while another Jeff width compile was also using the ANE compiler. `inspect_coreai_cache.py --executable python --strict` reports every chunk and `head_readout` as `fully_ane` (one ANE region, no GPU region). The same 64 held-out rows, scored from the compiled head, are 38/64 = 0.594, the same accuracy as the PyTorch adapter.

Eight self-play games from the same openings, through `system1-ane-serve` on `127.0.0.1:8787` (`--adapter snake` beside the base build):

| Adapter | Food / game | Steps / game | Decisions | Mean latency |
| --- | --- | --- | --- | --- |
| base | 0.12 (1 total) | 3.0 | 32 | 67.0 ms |
| snake | 0.25 (2 total) | 3.5 | 36 | 67.7 ms |

Neither adapter survived to the 48-step cap. That measurement used a server whose `/health` listed `base` and `snake`. The same port now also loads the four published builds below, and the dropdown reads whatever `/health` returns.

Snake rows for a machine that can run upstream `jeff-train` use the adapter-kit shape (`id`, `suite`, `family`, `state`, `question`, `label`, `target`, `source`). `generate_kit_rows` writes that shape from the same oracle. One game is one family, so `jeff-kit split` can hold a game out. Upstream `jeff-train --lora-rank 16 --lr 2e-4 --readout-lr 5e-6` is the trainer in `jeff-src`. It calls `torch.cuda` before the first step, so it does not start on this Mac. The sample here is the MPS run above, with the same rank, alpha, and the two learning rates. `mlx_lora.py` applies PEFT adapters at serve time. It does not train.

```sh
# from a checkout of jeff-src, after the rows file exists
uv run jeff-kit check-rows snake-rows.jsonl
uv run jeff-train --lora-rank 16 --lr 2e-4 --readout-lr 5e-6 \
  --initial-checkpoint "$HOME/Models/jeff/jeff-base-v1.3" \
  --train snake-train.jsonl --development snake-dev.jsonl --temperature snake-cal.jsonl \
  --run runs/snake --output checkpoints/snake \
  --base-model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
```

## Published adapters

The same path imports any of the 13 official jeffhub adapters, not only the four below. See [Import any jeffhub adapter](ADAPTERS.md#import-any-jeffhub-adapter) for the list with licences, steps, limits and the missing-bridge-dylib fix. `sanctions` and `soc` are CC BY-NC 4.0 on Hugging Face.

`triage`, `tools`, `guard`, and `spam` are the Apache-2.0 PEFT adapters on `mstrasser/jeff-adapter-*` (revision `v1.3`). Each is a LoRA plus its own readout and fitted temperature. `scripts/jeff_peft_merge.py` folds one into a copy of jeff-base the same way `jeff.lora.merge_adapter` does (`W += (alpha / rank) B A`), then `jeff-convert` / `compile` write `$HOME/Models/jeff-coreai/adapters/<name>/`. `jeff-convert` stores that temperature under `manifest["convert"]["temperature"]`. `JeffCoreAI` reads it from there, so a served adapter uses its own temperature while embeddings still come from `--model`.

`scripts/jeff_peft_torch.py` scores the merged checkpoint (the weights the converter saw) on the published example plus two choice rows from `test.jsonl`. `scripts/jeff_lora_parity.py` compares those probabilities to the compiled head. On this Mac, 7 October 2026, every row that fits the 2048-token context matched argmax. One tools row was 3513 tokens, so it was scored in PyTorch only (it also matched its label).

| Adapter | Compile | Rows (ANE) | Argmax match | Mean KL | Temperature |
| --- | --- | --- | --- | --- | --- |
| triage | 2 m 20 s | 3 | 3 | 3.2e-6 | 0.794 |
| tools | 1 m 43 s | 2 | 2 | 7.3e-4 | 1.049 |
| guard | 1 m 19 s | 3 | 3 | 1.1e-6 | 0.975 |
| spam | 1 m 19 s | 3 | 3 | 1.5e-5 | 1.181 |

`inspect_coreai_cache.py --executable python --strict` reports every chunk and `head_readout` as `fully_ane` for each of the four builds (bonded mode 1, one ANE region, no GPU region). The eight labeled rows all matched their gold choice on the merged PyTorch checkpoint.

The same four official examples on base Jeff, at the base temperature, pick a different top option or a much lower confidence: triage `other` 0.60 versus adapter `k1` 1.00; tools `t3` 0.53 versus adapter `answer_directly` 0.68; guard `indirect_injection` 0.53 versus adapter 1.00; spam `spam` 0.49 versus adapter `phishing` 0.97. The demo dropdown loads a sample for the selected adapter and scores that question on base as well. On `127.0.0.1:8796` those page requests were triage `k1` 0.998 versus base `other` 0.392, tools `answer_directly` 0.650 versus base `t3` 0.453, guard `indirect_injection` 0.995 versus base `benign` 0.309, and spam `phishing` 0.939 versus base `spam` 0.242. The Snake opening on that server was base `up` at 0.117 and the snake adapter `left` at 0.222. `/health` lists `base`, `snake`, `triage`, `tools`, `guard`, and `spam`. The page starts Snake in Play. With all six builds loaded, a few hundred of those moves can exhaust IOSurface and kill the process (`NDArray` allocation failure). A single decision on each adapter succeeds, and the 27B server on port 8766 was left running.

```sh
python scripts/jeff_peft_torch.py \
  --merged "$HOME/Models/jeff-published" \
  --output "$HOME/Models/jeff-published" \
  --adapter triage="$HOME/Models/jeff/adapters/jeff-adapter-triage"
# then, with the Core AI interpreter:
python scripts/jeff_lora_parity.py \
  --model "$HOME/Models/jeff/jeff-base-v1.3" \
  --build "$HOME/Models/jeff-coreai/adapters/triage/coreai" \
  --rows "$HOME/Models/jeff-published/triage/parity_rows.json"
```

`--task tetris` uses `coreai/jeff_tetris.py`. One decision drops one piece. Each option is a legal rotation and column, written as a sentence (`T piece, rotated right, columns 4-6, lands on row 3, clears 1 line, leaves 0 holes`). The board is text in `latest`. Labels are the El-Tetris maximum. `generate_tetris_rows` builds the 256/64 split, and `generate_tetris_kit_rows` writes the same positions in the adapter-kit `rows.jsonl` shape. The demo Tetris controls sit above that board. The top-5 placement bars and the move log are fixed-height scroll boxes, same as Snake. `scripts/jeff_tetris_eval.py` plays identical seeds for the heuristic, base, and the `tetris` adapter.

MPS, batch 4, rank 16, 7 October 2026. Steady 8.40 s/step (prompts run about 500–1500 tokens). Epoch 1 was 519 s, loss 1.844, accuracy 0.402. Epoch 2 was 506 s, loss 1.485, accuracy 0.531. Wall time 1580 s, merged to `~/Models/jeff-tetris-mps/merged`, compiled `fully_ane` in 1 m 24 s.

| Player | Lines / game | Pieces survived | Notes |
| --- | --- | --- | --- |
| El-Tetris heuristic | 13.50 | 40.0 | all 8 games hit the 40-piece cap |
| Base, ANE | 0.125 | 26.6 | 1 line in 213 pieces, 247 ms/decision |
| Tetris adapter, ANE | 7.38 | 39.9 | 59 lines in 319 pieces, 7 of 8 games hit the cap, 299 ms/decision |

Held-out choice accuracy on the same 64 rows: base 0.062 (loss 2.955), adapter 0.531 (loss 1.686). Train accuracy 0.664. The ANE games loaded one build at a time; two resident Jeff engines plus these long prefills exhaust IOSurface after a few dozen decisions.

The older checkpoint `~/Models/jeff/Jeff-Qwen3.5-0.8B` is prompt layout `state-first` with its own temperature. This server renders every prompt with the v1.3 live-last layout, and a second full ANE compile is another gigabyte, so that checkpoint is not a dropdown entry.

The prefix cache on `cursor/jeff-prefix-cache` (PR #7) measured a Tetris suffix of about 85 tokens at 35.7 ms cached versus 58 ms cold. Wiring it in waits until that branch and this one have both landed. This server still ships with the cache off.

Placement audit of the base build this server loads (`inspect_coreai_cache.py --executable python --strict`, cache key `python`, OS build `26B5091g`): all six chunks and `head_readout` are `fully_ane`, bonded compile mode 1, one ANE region and no GPU region on every entry (`p256_2k`, head `h1`).
