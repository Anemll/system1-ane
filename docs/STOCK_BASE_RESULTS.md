# Stock Qwen3.5-0.8B: training and ANE status

The resulting build is available as an optional extra adapter on Hugging Face: [`anemll/system1-base-unsloth`](https://huggingface.co/anemll/system1-base-unsloth), folder `snake-stock-qwen3.5-unsloth/` (`python scripts/download_model.py snake-stock-qwen3.5-unsloth`). Jeff remains the default model.

Jeff v1.3 (a Qwen3.5-0.8B fine-tune) is the primary model of this repository. This page records what has been checked on the **stock** `Qwen/Qwen3.5-0.8B` base (`unsloth/Qwen3.5-0.8B` carries the same weights). The evidence lives outside Git (run notes on the M5 Max and an external drive); nothing here was re-run for this release.

> **Validation split.** Every accuracy on this page (87.1% / 28.1-28.5% in Colab, 93.4% / 32.8% on the Jeff base, 85.2% on the ANE, ECE and calibration) was measured on the earlier split (5 of 256 validation states also appeared in training); remeasurement on the deduplicated split pending. Latency, placement and the 5-row parity result are not affected. `scripts/unsloth/make_snake_data.py` now regenerates a deduplicated validation set (the 5 overlapping rows, indexes 12, 48, 118, 150 and 218, are replaced; the other 251 rows and rows 0-4 used for parity are unchanged).

## Summary

| Question | Status |
| --- | --- |
| Can a stock Qwen3.5-0.8B base be trained for Snake with the Unsloth recipe? | **Verified**, twice (Colab A100: 28.1% → 87.1% on 7 Oct 2026, redo 28.5% → 87.1%) |
| Same recipe on Jeff v1.3 | 32.8% → 93.4% |
| Can the stock weights be converted to Core AI and run on the ANE? | **Yes**, for the stock language model and for the Snake decision model (see below) |
| ANE conversion and serving of a non-Jeff **decision model** (stock base + trained adapter and head) | **Verified on the ANE** (9 Oct 2026): converts with the existing converter, no converter changes; 7/7 packages `fully_ane`; 5/5 parity answers identical to Colab; ANE FP16 held-out accuracy 85.2% (256 rows) vs 87.1% in Colab. Served with the host-side Clef head engine for the Unsloth decision head. |

## Training (Unsloth recipe, Colab)

Snake LoRA plus Unsloth `FastDecisionModel` joint-schema head, base not quantized (`load_in_4bit=False`), Qwen3.5 forced-float32 compute path.

| | 7 Oct 2026 run | Redo |
| --- | --- | --- |
| GPU | Colab A100-SXM4-40GB | A100-40GB |
| Base | stock `unsloth/Qwen3.5-0.8B` | stock `unsloth/Qwen3.5-0.8B` |
| LoRA | rank 16 / alpha 16 | rank 16 / alpha 16 |
| Optimizer | lr 2e-4, cosine, warmup 10, weight decay 0.01 | lr 2e-4, cosine |
| Schedule | 3 epochs, 192 steps (batch 8 × accumulation 4), seed 3407 | 3 epochs, 192 steps, seed 3407 |
| Data | 2048 train / 256 held-out synthetic Snake rows | 2048 train / 256 held-out |
| Held-out accuracy | 28.1% → **87.1%** | 28.5% (73/256) → **87.1%** (223/256) |
| ECE after | 0.065 | 0.044 |
| Loss after | 0.350 | 0.373 |
| Training time | 459.7 s | 457.7 s |
| Unsloth | 2026.10.2 | 2026.10.3 |

Jeff v1.3 on the same recipe (re-keyed to the Qwen3_5 layout): 32.8% → **93.4%** held-out (ECE 0.075 → 0.035, loss 0.227).

The redo reproduced the 7 Oct accuracy exactly (87.1% after training). Calibration in these reports is fit on the same 256 held-out rows it is scored on.

## ANE conversion status

Result (9 Oct 2026, M5 Max, ANE bonded compile mode 1): the stock Qwen3.5-0.8B Unsloth Snake adapter (fp32 LoRA merged into the stock weights) converts with this repository's existing converter, with **no converter changes**, and runs fully on the ANE.

| Check | Result |
| --- | --- |
| Placement | 7 of 7 packages `fully_ane` (6 backbone chunks plus the readout head) |
| Parity on 5 Colab rows | 5/5 answers identical to Colab |
| ANE FP16 held-out accuracy | 85.2% (256 rows) vs 87.1% in Colab |
| Latency, 304-token prefill | 90.9 ms/decision (one ANE call) |
| Latency, 256-token prefill | 135.5 ms/decision (two ANE calls) |
| Jeff snake adapter, 256-token prefill, same run | 63.7 ms/decision (one ANE call) |

The stock Snake prompt is 261-264 tokens, so a 304-token prefill window covers it in one ANE call; at 256 it needs two. Latencies are the mean over 60 identical states in a quiet run (one run each for the 304 and 256 builds).

Serving: the decision head for this adapter is the Unsloth head, which runs on the host through the Clef head engine; the backbone runs on the ANE. The Jeff comparison above (93.4% vs 87.1% on the same recipe) is unchanged.

### Serving details and the fp16 ANE head

- **Optional, not the default.** Jeff stays the default model and adapter of the server and the demo. The `snake-stock` option appears (as `snake-stock` in the demo's Snake dropdown) only when its build is passed with `--adapter snake-stock=<build>/coreai`; without that flag nothing changes.
- **Recipe.** Merge the fp32 LoRA into the stock weights, convert with `python forge.py jeff-convert ... --prefill 304 --quant fp16`, compile, then serve as an extra adapter. Steps and run order: `scripts/unsloth/README.md`; serving how-to: `docs/SYSTEM1_ANE_SERVE.md` ("Serving the Unsloth-trained stock Snake adapter (optional)").
- **Head on the host (default).** The Unsloth head runs in fp32 on the host CPU, about 8 ms per decision, for about 90.9 ms end to end.
- **Head compiled to the ANE in fp16.** With `CLEF_HEAD_BACKEND=ane` the head runs as a fixed-shape fp16 Core AI package (about 3 ms; ANE call about 2.2 ms), for about 85 ms end to end (84.8 ms). Maximum |dlogit| against the host fp32 head is 0.005 on 261 rows, with no answer flips. Details: `docs/FP16_HEAD_WRAPUP.md`.
- **Comparison.** The Jeff Snake adapter shipped here takes 63.7 ms per decision at the 256-token window in the same runs. All ANE numbers are from local runs on an Apple M5 Max on 9 October 2026, on a machine that was not otherwise idle. They are not controlled benchmarks.
- **Where things are.** The weights are on Hugging Face as [`anemll/system1-base-unsloth`](https://huggingface.co/anemll/system1-base-unsloth), folder `snake-stock-qwen3.5-unsloth/` (`python scripts/download_model.py snake-stock-qwen3.5-unsloth`). Training logs and weights are not in this repository.

### Earlier partial check

Before the decision-model run (7 Oct 2026, local), the stock Qwen3.5-0.8B language-model weights were converted with this repository's converter (FP16, `p256_2k`, stock language-model head) and loaded on an M5 Max.

- 22 of 22 packages `fully_ane`.
- ANE vs FP32 KL 2.1e-4 to 1.8e-3, same top-1 token on 5 test prompts.
- About 59 ms cached prefill and about 5.5 ms per language-model-head step.
- Untrained Snake accuracy on the ANE 20.7% (256 rows), equal to PyTorch.

That check was not a decision-model result.

## Evidence locations (not in this repository)

| Item | Location |
| --- | --- |
| 7 Oct training run | `$HOME/Models/jeff-unsloth-snake/qwen_unsloth_snake_a100/` (`metrics.json`, `README.md`, `HEAD_NOTES.md`) |
| Jeff run, same recipe | `$HOME/Models/jeff-unsloth-snake/jeff_unsloth_snake_a100/` |
| Redo | `RESULT.md` from the redo run, kept offline (not in this repository); bundle sha256 `ed5f4dc5fa41cc856880e66156596f2f9974df1fc3dd683f4b04c1ee45f9bbd8`. Reported to us; not opened when this page was written. |
| Partial stock ANE check (7 Oct) | `$HOME/Models/qwen35-0.8b-stock-coreai/eval/` (`placement.json`, `ane_report.json`) |
