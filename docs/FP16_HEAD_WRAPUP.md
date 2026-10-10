# fp16 ANE-compiled Clef head for the stock Snake build

Measured 9 Oct 2026 on an Apple M5 Max with a local end-to-end run script (not in this tree), on a machine that was not otherwise idle. These are not controlled benchmarks; remeasurement is pending. Context: `docs/STOCK_BASE_RESULTS.md` on branch `docs/stock-clef-fp16-results` (not merged).

## Outcome

The Unsloth Clef joint-schema head of the stock Qwen3.5-0.8B Snake adapter (27.3 M parameters) can now run on the ANE as a fixed-shape fp16 Core AI package: about **3.0 ms** per head call (ANE call about 2.2 ms plus about 0.5 ms of host input preparation) instead of about 8.0 ms on the host CPU in fp32, and **84.8 ms** end to end instead of 90.1 ms. Accuracy is unchanged: 85.2% on the 256 held-out rows (Colab FP32 87.1%).

The host fp32 head stays the default and the fallback. `CLEF_HEAD_BACKEND=ane` enables the ANE head. Other adapters (builds without a `clef_head/` directory) are unaffected and never import the Clef code.

## What is in this tree

| File | Role |
| --- | --- |
| `coreai/clef_head.py` | Host PyTorch port of the head and the Clef prompt encoder. `CLEF_HEAD_DTYPE=fp32\|fp16\|bf16` and `CLEF_HEAD_DEVICE=cpu\|mps`; default fp32 on the CPU. fp16 loads `joint_head_fp16.safetensors` when present |
| `coreai/clef_ane_head.py` | ANE-layout head (`AneClefHead`), host input builder, `AneHeadRunner`, and the export entry point (`python coreai/clef_ane_head.py <clef_head dir>`) |
| `coreai/clef_engine.py` | Serve engine; `CLEF_HEAD_BACKEND=host\|ane` (default `host`). More than 304 prompt rows or more than 8 options fall back to the host head |
| `scripts/convert_clef_head_fp16.py` | Casts `joint_head.safetensors` (bf16 plus three fp32 scalars) to fp16 |
| the tokenizer worker, `scripts/system1_ane_serve.py` and the demo page (`scripts/*tokenize.py`, `scripts/*demo.html`) | `clef` tokenizer request, `ClefEngine` hook for adapters with a `clef_head/` directory, demo entry for `snake-stock` |
| `tests/test_clef_head.py` | Host head vs the ANE-layout port on random small weights, fp16 option, prompt spans |

**The ANE package is generated, not stored in Git.** The 48 MB `clef_head_ane.aimodel` is not committed. To rebuild it, with the Core AI SDK Python:

```
python scripts/convert_clef_head_fp16.py <build>/coreai/clef_head      # writes joint_head_fp16.safetensors
python coreai/clef_ane_head.py <build>/coreai/clef_head                # writes clef_head_ane.aimodel + ane_head_config.json
CLEF_HEAD_BACKEND=ane forge.py system1-ane-serve --adapter snake-stock=<build>/coreai ...
```

The first load specializes the package for the ANE (about 2 s). Revert by unsetting `CLEF_HEAD_BACKEND` (or `CLEF_HEAD_BACKEND=host`).

## Design

Every Linear is a 1x1 conv in the `(1, C, 1, T)` layout; T = 304 prompt rows and K = 8 options are fixed. Dynamic parts are host-built inputs: the post-norm hidden rows, a pooling matrix (question mean, last token, option means, 1/length baked in), the lexical option vectors from the host embedding gather, and additive masks. Attention is explicit broadcast-multiply and reduce, softmax is written out, LayerNorm uses the max-scaled form, GELU is exact (erf). Placement audit: `fully_ane`, no GPU regions.

## The divergence and its fix

The first export compiled fully on the ANE but returned wrong values, with no error. Stage-by-stage taps against the fp32 torch module (same inputs) located it in the attention output: the sum over keys is rank-3 `(heads, head_dim, K)`, and reshaping it to `(1, width, 1, K)` straight into the output-projection conv computes the wrong thing on the ANE. Scores, softmax and the weighted sum were all correct.

Fix (`MHA_OUT=kd` while debugging, now the only path): keep the sum rank-4 with `keepdim=True`, reshape to `(1, width, K, 1)`, run the conv in that layout, permute back to `(1, C, 1, K)`.

| Stage (max abs diff vs fp32 torch) | Before | After |
| --- | --- | --- |
| q, g, memory, first option rows | 0.002-0.004 | same |
| attention scores, softmax, weighted sum | 0.002 | same |
| attention output (after out-proj) | 1.34 | 0.0008 |
| evidence layer 0 / 1 output | 1.63 / 2.56 | 0.0023 / 0.0035 |
| field before final norm | 3.47 | 0.0157 |
| logits | 0.37-0.78 | 0.0049 |

## Accuracy

On 261 rows (5 Colab parity rows plus 256 held-out), with the cached ANE backbone output: max abs logit difference vs the host fp32 head 0.0049, 0 answer flips. Held-out accuracy 85.156% for every variant. The 5 parity rows give the same answers and token ids as Colab; the max logit difference vs Colab is 0.513, the same as with the fp32 host head (not caused by this change).

## Latency (about 488 decisions per variant: 5 parity rows, 8 games, 256 held-out rows)

| Variant | Head ms (mean) | End to end ms |
| --- | --- | --- |
| host fp32 CPU (default, fallback) | 8.01 | 90.1 |
| host fp16 CPU | 10.61 | 92.9 |
| host fp16 MPS | 10.53 | 92.5 |
| ANE fp16 package | 3.00 (ANE call about 2.15) | 84.8 |

fp16 on the CPU or MPS is slower than fp32 on the CPU (torch CPU fp16 is not fast; MPS pays for GPU wake-up after ANE calls) and is not used. For this adapter the end-to-end time was about 143 ms in the first evaluation and 139 ms on a 256-row build (two prefill calls), then 90.1 ms after the 304-row rebuild. The remaining time is dominated by the 304-wide prefill (about 81 ms for ~262 prompt tokens), not the head. For comparison, the repository's smaller Snake build with ~214-token prompts runs end to end in 63.6 ms.

## Not done

- Not merged. Nothing here changes the behavior of other adapters; the Clef modules are imported only for adapters that ship `clef_head/`.
- The ANE package and the stock Snake build are not in Git or on Hugging Face.
- No controlled benchmark; the numbers are from the runs described above.
