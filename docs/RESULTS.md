# Measured results (Apple M5 Max)

All numbers below were recorded on an **Apple M5 Max** (H17, `Mac17,6`), macOS 27.2 (26B5091g), Xcode 27.2, ANE bonded compile mode **1**. **Measured 7 Oct 2026; remeasurement pending.** They are copied from the Jeff work on anemll-forge (PRs #6–#9 and follow-up commits on `cursor/jeff-snake-lora-055e`). This Linux checkout did **not** re-run ANE jobs.

## Convert and compile (base `jeff-base` v1.3)

| Build | Convert | Compile | Packages | Placement |
| --- | --- | --- | --- | --- |
| FP16, `p256_2k` (default) | 26.5 s | 79 s (13 s / 4-layer chunk) | 6 × 159 MB + 0.5 MB head; 505 MB host embedding | `fully_ane`, mode 1 |
| FP16, `p1024_2k` | 79 s | 6 min 07 s | 6 × 161 MB | GPU region on every chunk |
| INT8 per-channel, `p256_2k` | 26.6 s | 79 s | 6 × 80 MB | `fully_ane` |

Warm load from cache: 0.1 s. A 1,024-row entry is slower per token than four chained 256-row calls and is partly on the GPU — keep `--prefill 256`.

## Prefill parity vs PyTorch FP32

Real Jeff prompts (`jeff_reference.py` support-chat rows), median of 5 warm runs. Readout head adds 0.34 ms.

| Tokens | Options | Calls | Prefill | max \|dp\| | KL | Argmax |
| --- | --- | --- | --- | --- | --- | --- |
| 175 | 3 | 1 | 63.4 ms | 0.0075 | 1.1e-4 | same |
| 238 | 5 | 1 | 64.2 ms | 0.0012 | 4.6e-6 | same |
| 1,008 | 30 | 4 | 254.3 ms | 0.0091 | 6.6e-4 | same |
| 2,018 | 100 | 8 | 508.3 ms | 0.0131 | 8.3e-4 | near-tie flip (A 0.2460 vs B 0.2448) |

Host DecodeLayer vs FP32: max \|dp\| ≤ 2e-6. INT8 is the same speed and 2–3× FP16's error; FP16 stays the default. Details: [research/JEFF_DECISION_ANE.md](research/JEFF_DECISION_ANE.md).

## Server latency (demo prompts)

Ten warm decisions after two warmup calls. An idle 27B server was resident on another port and was not generating.

| Request | Tokens | Prefill | Tokenize | Head | Total | HTTP wall | Decisions/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Snake, opening board | 213 | 63.6 ms (1 call) | 0.47 ms | 0.34 ms | 64.7 ms | 65.3 ms | 15.5 |
| Route a refund message | 133 | 63.6 ms (1 call) | 0.38 ms | 0.36 ms | 64.7 ms | 65.2 ms | 15.5 |
| Tetris placement | 1430 | ~367 ms (6 × 256) | — | — | ~367 ms | — | — |

Refund classified `refunds` at 0.85. Base Snake first moves were soft (top option ~0.34).

## Snake LoRA (MPS, rank 16, 7 October 2026)

Steady 1.58 s/step after a 4.92 s warmup. Whole run including evals and merge: 300 s.

| Split | Accuracy | Loss | n |
| --- | --- | --- | --- |
| Base, train | 0.309 | 1.337 | 256 |
| Base, held-out | 0.281 | 1.352 | 64 |
| Base, held-out, hint prompt | 0.484 | 1.121 | 64 |
| Adapter, train | 0.598 | 0.887 | 256 |
| Adapter, held-out | 0.594 | 0.932 | 64 |

Compiled Snake adapter: 38/64 = 0.594 on the same held-out rows (matches PyTorch). Compile 10 m 18 s while another width compile was also using the ANE compiler. Placement `fully_ane`.

Eight self-play games through `system1-ane-serve`:

| Adapter | Food / game | Steps / game | Mean latency |
| --- | --- | --- | --- |
| base | 0.12 | 3.0 | 67.0 ms |
| snake | 0.25 | 3.5 | 67.7 ms |

## Published adapters (ANE vs merged checkpoint)

| Adapter | Compile | Rows (ANE) | Argmax match | Mean KL | Temperature |
| --- | --- | --- | --- | --- | --- |
| triage | 2 m 20 s | 3 | 3 | 3.2e-6 | 0.794 |
| tools | 1 m 43 s | 2 | 2 | 7.3e-4 | 1.049 |
| guard | 1 m 19 s | 3 | 3 | 1.1e-6 | 0.975 |
| spam | 1 m 19 s | 3 | 3 | 1.5e-5 | 1.181 |

One tools row was 3513 tokens (scored in PyTorch only). Official examples on base vs adapter: triage `other` 0.60 vs `k1` 1.00; tools `t3` 0.53 vs `answer_directly` 0.68; guard `indirect_injection` 0.53 vs 1.00; spam `spam` 0.49 vs `phishing` 0.97.

## Tetris LoRA (MPS, then ANE)

Steady 8.40 s/step. Wall 1580 s. Held-out accuracy: base 0.062 → adapter 0.531. Train 0.664. Compiled `fully_ane` in 1 m 24 s.

| Player | Lines / game | Pieces survived | Latency |
| --- | --- | --- | --- |
| El-Tetris heuristic | 13.50 | 40.0 | — |
| Base, ANE | 0.125 | 26.6 | 247 ms/decision |
| Tetris adapter, ANE | 7.38 | 39.9 | 299 ms/decision |

## Stock Qwen3.5-0.8B: Unsloth decision model (Snake; Colab training, local ANE conversion)

Training numbers come from Colab A100 runs and local run notes kept outside Git; the ANE numbers are `scripts/unsloth/run_stock_snake.py` runs on an Apple M5 Max (macOS 27.2, ANE bonded compile mode 1) on 9 October 2026, on a machine that was not otherwise idle. They are not controlled benchmarks and remeasurement is pending. This model is **optional and not the default** (prebuilt: [`anemll/system1-base-unsloth`](https://huggingface.co/anemll/system1-base-unsloth), folder `snake-stock-qwen3.5-unsloth/`): Jeff remains the default model and adapter in the serve path and the demo, and the stock option (`snake-stock`) appears only when its build is passed with `--adapter snake-stock=<build>/coreai`. Recipe, provenance and caveats: [STOCK_BASE_RESULTS.md](STOCK_BASE_RESULTS.md); run order: [scripts/unsloth/README.md](../scripts/unsloth/README.md); serving how-to: [SYSTEM1_ANE_SERVE.md](SYSTEM1_ANE_SERVE.md#serving-the-unsloth-trained-stock-snake-adapter-optional). No weights are stored in this repository.

> **Validation split.** The accuracies below (87.1% and 28.1-28.5% in Colab, 93.4% / 32.8% on the Jeff base, 85.2% on the ANE, and the ECE and calibration temperature fit on the same rows) were measured on the earlier split (5 of 256 validation states also appeared in training); remeasurement on the deduplicated split pending. Latency, placement and the 5-row parity check are not affected. `scripts/unsloth/make_snake_data.py` now regenerates a validation set with no state in training.

### Training (Colab)

Base `unsloth/Qwen3.5-0.8B` (stock), Unsloth `FastDecisionModel` + `DecisionTrainer`, LoRA rank 16 / alpha 16, lr 2e-4 cosine, 3 epochs (192 steps), seed 3407, max sequence length 1024, 2048 train / 256 held-out Snake rows, A100-40GB, base in bf16 (`load_in_4bit=False`, Unsloth's forced-float32 compute path). The calibration temperature is fit on the same 256 held-out rows that are scored, so the calibrated numbers are not an independent test.

| Item | Result | Source |
| --- | --- | --- |
| First run, 7 Oct 2026 | held-out accuracy **28.1% → 87.1%** (ECE 0.029 → 0.065) | local run notes (`metrics.json`, `README.md` of the run bundle), outside Git |
| Redo of the same recipe, Unsloth 2026.10.3, 457.7 s | **28.5% (73/256) → 87.1% (223/256)**, ECE after 0.044, loss 0.373: reproduces the first run | local run notes (`metrics.json`, `RESULT.md`), outside Git |
| 5-row parity file from the redo | token ids, option/question spans, raw head logits and calibrated probabilities for 5 held-out rows | `scripts/unsloth/data/stock_snake_parity_rows.json` (22 KB) |
| Same recipe on the Jeff v1.3 base (re-keyed to the Qwen3_5 layout) | 32.8% → 93.4% held-out (ECE 0.075 → 0.035), against 28.5% → 87.1% for the stock base above | local run notes (`metrics.json` of that run), outside Git |

### ANE conversion and serving (local, separate step)

Steps: `scripts/unsloth/prep_ckpt.py` (fp32 LoRA merge into the stock weights, zero placeholder readout) → `forge.py jeff-convert --prefill 304 --quant fp16` → `forge.py compile` → `scripts/unsloth/assemble_stock_build.py` → `system1-ane-serve --adapter snake-stock=<build>/coreai`. The stock prompt is 261-264 tokens for usual boards (258-273 seen over the held-out rows), so a 256-row prefill window needs two ANE calls and a 304-row window needs one (`--prefill` must be a multiple of 8). The conversion used the existing converter, with no changes to it. Serving needs the host-side Clef engine in this repository (`coreai/clef_head.py`, `clef_engine.py`, `clef_ane_head.py`, selected automatically for builds that ship a `clef_head/` folder; other adapters are unaffected). The Unsloth Clef head runs on the host in fp32 by default; `CLEF_HEAD_BACKEND=ane` runs it as a fixed-shape fp16 Core AI package (304 prompt rows x 8 options; larger prompts fall back to the host head).

| Item | Result | Source |
| --- | --- | --- |
| Placement of the 304-row stock build | 7 of 7 packages `fully_ane` | local `forge.py` placement audit |
| Parity vs the 5 Colab rows | 5/5 answers identical, same token ids; max \|logit diff\| 0.513 (fp16 backbone vs Colab fp32) | `run_stock_snake.py --parity` |
| Held-out accuracy on the ANE (fp16 backbone, host fp32 head) | **85.2%** (256 rows) vs 87.1% in Colab fp32 | `run_stock_snake.py --heldout 256` |
| Latency per decision, 256-row window before re-export (two ANE calls), 60 identical states, quiet machine | **135.5 ms** mean (p95 137.5; prefill 125.5, host head 8.3) | server-path measurement, local `measure_old_quiet.json` |
| Latency per decision, 304-row window (one ANE call), same 60 states, quiet machine | **90.9 ms** mean (p95 92.1; prefill 81.3, host fp32 head 8.0, 1 call at width 304) | server-path measurement, local `measure_p304_quiet.json` |
| Reference in the same runs: the repository's Snake adapter at the 256-row window (one call, 214-token prompts) | 63.7 ms mean | same two files |
| Latency per decision, 304-row window, fp16 ANE head | about 85 ms (84.8 ms; head 3.0 ms incl. host input preparation, ANE call about 2.2 ms; max \|dlogit\| 0.005 vs the host fp32 head on 261 rows, no answer flips) | `run_stock_snake.py`, `docs/FP16_HEAD_WRAPUP.md` |
| Earlier partial check: stock language-model weights converted with this pipeline (FP16, `p256_2k`, stock LM head) | 22 of 22 packages `fully_ane`; ANE vs FP32 KL 2.1e-4 to 1.8e-3, same top-1 on 5 prompts; untrained Snake accuracy 20.7% (256 rows), equal to PyTorch | local evaluation notes, outside Git (not a decision-model result) |

**Latency note.** The stock build is about 27 ms slower per decision than the one-call 256-row Snake adapter (90.9 vs 63.7 ms) because each call processes 304 rows for a longer prompt (about 262 tokens vs 214); the 304-row window removes the second call the stock prompt needed at 256 rows (135.5 → 90.9 ms). Compiling the 304-row build took about 108 s (1m47s in the compile log) vs about 83 s (1m23s) for the 256-row build. A figure of about 215 ms seen once under heavy machine load is excluded. With the head compiled to the ANE in fp16 (`CLEF_HEAD_BACKEND=ane`), the head takes 3.0 ms instead of 8.0 ms and `run_stock_snake.py` reports about 84.8 ms per decision (90.1 ms with the host head in the same runs); those runs were on a machine that was not idle, not a controlled benchmark. The server-path rows above were taken on a quiet machine; the two sets of runs differ by about 1 ms for the host-head build.

**Not run:** a controlled latency benchmark of the stock build; the Colab training itself from this repository (it needs a GPU Colab VM; the notebook and scripts are included).

## Prefix cache (other branch)

`cursor/jeff-prefix-cache` (PR #7) measured a Tetris suffix of about 85 tokens at **35.7 ms cached vs 58 ms cold**. This tree ships with the cache off; the snapshot API is present.

## Streamed LoRA (other branch)

`cursor/jeff-lora-stream` (PR #10) packed rank-16 factors as ANE inputs (28 program inputs). Not the serve path in this tree. See [HISTORY.md](../HISTORY.md).
