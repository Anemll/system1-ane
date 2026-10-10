# Jeff / Unsloth decision models on the ANE

- **Scope:** run Jeff (Qwen3.5-0.8B decision head) on the Apple Neural Engine through a prefill-only Core AI path.
- **Status:** Path B runs end to end on an M5 Max (measured 7 Oct 2026; remeasurement pending). `jeff-base` v1.3 FP16 converts in 27 s and compiles in 79 s. All six chunks and the readout head are cached fully on the ANE. A Jeff prompt prefills in 64 ms (up to 256 tokens), 254 ms (1,008 tokens) and 508 ms (2,018 tokens), and the option probabilities track the PyTorch FP32 reference within the FP16 noise band. See [M5 Max results](#m5-max-results-7-october-2026). `forge.py system1-ane-serve` answers Jeff's `/v1/systemone` route and serves a browser demo ([SYSTEM1_ANE_SERVE.md](../SYSTEM1_ANE_SERVE.md)). `JeffCoreAI.prefill(..., prefix=)` accepts a cached backbone snapshot; the live-last cache policy is still a separate piece. A sample LoRA trains in PyTorch and is merged into its own Core AI build (`jeff-train-lora` in that doc); the server selects builds, it does not apply LoRA factors at runtime. Not done: ANE temperature/ECE fit, the prefix-cache policy itself.
- **Branch:** `cursor/jeff-decision-ane-85f5` on anemll-forge; this repo is the standalone Jeff tree.

Some links in this note pointed at 27B Forge files that were removed from this repository. The convert/runtime paths it names still exist. Measured tables are copied into [RESULTS.md](../RESULTS.md).

Evidence labels: **Source-verified** (this checkout), **External doc** (Jeff / Unsloth / Qwen cards, not fetched as weights), **Inferred**.

## What Jeff is

Jeff is a small **decision** model, not a chat generator. You send a situation (`state`) and a list of questions with named options; one forward pass returns a calibrated probability for every option. There is no generated text to parse. **External doc:** [mstrasser/jeff-base](https://huggingface.co/mstrasser/jeff-base) (v1.3), [mstrasser/Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B) (v1.2 zero-shot), [firelex/jeff](https://github.com/firelex/jeff), [Unsloth FastDecisionModel](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth).

Shared serving shape (Jev / SystemOne / Clef / Jeff / Unsloth):

| Piece | Role |
| --- | --- |
| Backbone | Qwen3.5 text stack; Jeff-0.8B is a fine-tune of `Qwen/Qwen3.5-0.8B` |
| Prompt | Instructions + questions/options + changing `state` |
| Head | Scores the supplied options; does not emit next-token text |
| Calibration | Fitted temperature so stated confidence tracks hit rate (ECE) |
| API | `state` + `questions` (`choice` / `noul` / `score`), not `/v1/chat/completions` |

Jeff v1.3 (`jeff-base`) is adapter-first: LoRA adapters on a fixed base, **live-last** prompt (fixed instructions/options first, changing state last) so the prefix can be cached. Checkpoint extras: `readout.safetensors` (trained readout over answer codes A, B, … AA, …) and `decision_config.json` (codes, token ids, temperature, `prompt_layout: live-last`). **External doc.** v1.1+ Qwen Jeff models accept up to 254 options.

Unsloth `FastDecisionModel` trains a **Clef-style** head: one prompt of input + every question/option; a small head reads the backbone hidden states over those spans and scores options together. Same non-generative contract, different readout tensor than Jeff’s answer-code head. Both are in scope for an ANE spike.

Published Apple Silicon numbers are **MLX**, not Core AI: about 28 ms/decision on M4 Max for the 0.8B card, and ~0.25 s mean including adapters/prompt on the Jeff v1.3 comparison. Treat those as an MLX baseline, not an ANE forecast. **External doc.**

## Hypothesis check

The working hypothesis was: *most Qwen converter / Core AI graph code is reusable because the backbone is still Qwen3.5 dense; the new piece is the decision head and a different serve API.*

**Partly right, wrong about “dense.”**

Qwen3.5-0.8B is the **same hybrid family** as Forge’s Qwen3.8-27B: Gated DeltaNet layers plus gated full attention on a 3:1 cadence, `qwen3_5` / `qwen3_5_text` config. It is not a dense all-attention Qwen3. **External doc:** Qwen3.5-0.8B card (`6 × (3 × GDN → FFN, 1 × gated attention → FFN)`), Ollama `qwen3.5:0.8b` `layer_types`.

The older in-repo **Qwen3-0.6B** path (`scripts/qwen3_lut_common.py`) *is* dense all-attention. That is a different architecture. Its LUT/GPTQ helpers are reusable; its reference forward is **not** a Jeff graph template.

What actually holds:

- Reuse the **hybrid GDN + attention + SwiGLU** Core AI graphs if they are retargeted to 0.8B shapes.
- Do **not** assume a 27B `convert` / `serve` would accept Jeff (those commands gated 64 × 5120 and are gone from this tree).
- The genuinely new work is the decision/readout graph, a prefill-only serve path, prompt layout, and calibration — plus dropping DFlash2 / next-token decode for this model.

## Shape gap (must retarget)

| | Forge today (Qwen3.8-27B) | Jeff backbone (Qwen3.5-0.8B) |
| --- | --- | --- |
| Layers | 64 (48 GDN + 16 attn) | 24 (18 GDN + 6 attn) |
| Hidden / MLP | 5120 / 17408 | 1024 / 3584 |
| Full attention | 24 Q / 4 KV, `head_dim` 256 | 8 Q / 2 KV, `head_dim` 256 |
| GDN | config-driven (`linear_num_*`) | 16 QK + 16 V, dim 128 |
| Vocab | 248320, **untied** `lm_head` | 248320, **tied** embeddings |
| Layer pattern | `full_attention_interval` 4 | same 3:1 (`layer_types`) |
| Native context | 64K prepared ladder | 262K card; decisions are short |
| Weight prefix | `model.language_model.*` | same HF text prefix (confirm on download) |

27B figures were recorded in anemll-forge quantization notes (removed from this Jeff-only tree). 0.8B figures: **External doc.**

`coreai/qwen38_coreai_build.py` and `scripts/qwen38_ane_chunk.py` already read mixer/attention widths from `CFG` / `layer_types`. **Source-verified.** The old 27B launcher (`forge.py prepare()`, `hf_release.py`, DFlash2 taps `[5, 19, 33, 47, 61]`, vocab LUT4 `Head`) lived in anemll-forge and is not in this tree. `forge.py` here is Jeff-only.

## Inventory (Qwen-touching entry points)

| Stage | Path | What it does | Jeff fit |
| --- | --- | --- | --- |
| Launcher | [`forge.py`](../../forge.py) | Jeff-only: `jeff-convert` / `jeff-smoke` / `system1-ane-serve` / `jeff-train-lora` / `compile` / `doctor` | Current CLI. |
| 0.6B-era LUT | [`scripts/qwen3_lut_common.py`](../../scripts/qwen3_lut_common.py) | Dense Qwen3 fp32 ref (default `~/Models/Qwen3-0.6B`), tokenizer, Hadamard, LUT/GPTQ | Formats yes; graph no. |
| Core ML graph | [`scripts/qwen38_ane_chunk.py`](../../scripts/qwen38_ane_chunk.py), [`scripts/qwen38_ane_model.py`](../../scripts/qwen38_ane_model.py) | GDN/attn/MLP MIL; host embedding | Backbone primitives reused by convert; vocab head unused. |
| Core AI export | [`coreai/qwen38_coreai_build.py`](../../coreai/qwen38_coreai_build.py) | Hybrid GDN / attention / SwiGLU graphs; Jeff rebinds widths | 6 × 4-layer chunks, config-sized tensors, **no vocab head**. |
| Package helpers | [`scripts/qwen38_coreai_model.py`](../../scripts/qwen38_coreai_model.py) | `pick_package`, IOSurface buffers, compile-mode cache | Decode / DFlash2 classes were stripped. |
| Removed (anemll-forge only) | `hf_release.py`, `qwen3_mlp_gptq_model.py`, `qwen38_gptq_27b.py`, `qwen38_server.py`, `dflash2_*` | 27B bundle check, GPTQ/VQ, chat server, drafter | Not in this checkout. |

Packaging patterns that are model-size-agnostic: shared-weight multifunction packages, `manifest.json` context ladder, ANE compile cache + `[ANE compile]` readout, bonded compile mode ([docs/ANE_COMPILE_MODE_POLICY.md](../ANE_COMPILE_MODE_POLICY.md)), Swift bridge IOSurface views, host `embed_tokens_fp16.npy` lookup.

## Reusable

**Backbone graphs (Inferred, high confidence given config-driven widths).** GDN lazy-commit, gated full attention (RoPE with `partial_rotary_factor`, Q/K RMSNorm, GQA), SwiGLU + tanh-SiLU, scale-free RMSNorm (`1 + weight`), conv1d short conv, overflow-safe softplus. These are the 27B lessons that exist *because* Qwen3.5 is hybrid, not despite it. **Source-verified** builders: `qwen38_ane_chunk.py`, `qwen38_coreai_build.py` `LayerW` / `GDNW` / `AttnW`.

**Tokenizer / embeddings.** Qwen tokenizer files + host FP16 embedding table. 0.8B embeddings are ~248320 × 1024 × 2 ≈ 508 MB, vs ~2.5 GB at hidden 5120. Tied embeddings mean there is no separate 27B-style `lm_head.weight` unless the decision readout replaces it.

**Core AI packaging.** One `.aimodel` per chunk, multiple prefill/context entries sharing weights, compile-once cache, Swift vs Python runtime split. A 0.8B FP16 (or INT8) target should be a **small** package set: six 4-layer chunks, or one/two packages if ANEC size allows. 27B needed 16 chunks for resident-weight and compile limits, not because the math requires it.

**Prefix reuse.** `JeffCoreAI.prefill(..., prefix=)` restores a cached GDN/KV snapshot so a follow-up that shares a prefix can skip the fixed block. That is the live-last mechanism Jeff wants: cache instructions+options, prefill only the new `state`. **Source-verified.** Independent Jeff requests must still `reset()`; recurrent state must not leak across unrelated decisions. **External doc:** Jeff llama.cpp notes say the same. The live-last *policy* (when to cache) is still a separate piece; see [HISTORY.md](../../HISTORY.md) PR #7.

**Small-model conversion lessons (0.6B era).** `qwen3_lut_common.py` defaulted to `Qwen3-0.6B` and treated embeddings as tied with `lm_head`. LUT4 / INT8 / FP8 formats and GPTQ helpers still apply if a later spike wants light palettization. **Source-verified.** The 0.6B reference has **no** `linear_attn` / `layer_types`. Do not convert Jeff through that forward.

**Skip heavy VQ/GPTQ for 0.8B.** ~0.85B × 2 bytes ≈ 1.6 GB FP16; INT8 ≈ 0.8 GB. The 27B mix (`vector 2x16`, rank-64 residuals, 85-minute GPTQ) exists to fit 27B on 32 GB. **Inferred:** first Jeff spike should stay FP16 dense or INT8 per-channel.

## New

1. **Decision / readout graph.** Replace `Head` (final RMSNorm + LUT4 vocab `lm_head`, 8 rows → 248320 logits) with either Jeff’s `readout.safetensors` over answer-code positions, or a Clef/Unsloth span scorer. Then: gather N option scores, divide by the fitted temperature, softmax. Do not compile a 248k-way LM head just to throw it away.

2. **Non-autoregressive serve.** One prefill of the rendered prompt, optional prefix restore, readout, JSON probabilities. No token loop, no thinking budget, no DFlash2, no `/v1/chat/completions`. New route shaped like SystemOne / `system1-ane-serve` (`state`, `questions`, per-option probs).

3. **Prompt layout.** Build the Jeff / Unsloth string (v1.3 live-last: fixed block first, changing field last), apply the Qwen chat template with thinking **off**. Option keys must not be bare numbers. **External doc.**

4. **Calibration.** Jeff ships a temperature per model (and per GGUF format). An ANE FP16 or INT8 build needs its own fit on holdout rows; do not reuse the MLX or Q4_K_M number. Unsloth `FastDecisionModel.calibrate` is the same idea.

5. **Launcher / config.** Relax or bypass the 64 × 5120 gate; read `layer_types` and 24-layer plans; do not require a drafter pair.

6. **Optional later:** LoRA adapter load (v1.3), multiple questions per request, Unsloth Clef head vs Jeff readout A/B.

## Risks

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Treating 0.8B as “dense Qwen3” | Would send Jeff through `qwen3_lut_common.Qwen3` and drop GDN | Use `layer_types` / the 27B hybrid graph, retargeted |
| Hardcoded 27B serving | Old forge `serve` / `hf_release` / DFlash2 (removed from this tree) | Jeff-only `forge.py`; do not reintroduce 27B commands |
| Width / RoPE / GDN dim drift | 8 vs 24 Q heads; 16×128 vs 27B GDN; `partial_rotary_factor` 0.25 | Drive every tensor from `text_config`; parity vs MLX/PyTorch on a short prompt |
| Weight key / multimodal wrapper | 0.8B card includes a vision tower; Jeff may still ship `processor_config.json` | Confirm `model.language_model.*` after download; ignore vision for text Jeff |
| MLX ≠ Core AI | firelex `JEFF_BACKEND=mlx` is the supported Mac path | Core AI is a new backend: compile placement, fp16 numerics, ECE on-device |
| Recurrent leak | GDN state reused across unrelated decisions corrupts probs | `reset()` per request; snapshot only for live-last prefix |
| Calibration after quant | INT8/LUT shifts option logits | Fit temperature on the compiled ANE graph |
| Clef head vs Jeff readout | Unsloth span head ≠ answer-code readout | Spike Jeff readout first (published files); keep Clef as a second head package |
| Prefill entry shape | 27B uses T=8 verify + T=64 prefill | One 256-row prefill entry, chained for longer prompts. A 1,024-row entry was slower and partly on the GPU (measured below) |

## Implementation (Path B)

| Piece | Path | Role |
| --- | --- | --- |
| Checkpoint + prompt | [`coreai/jeff_coreai.py`](../../coreai/jeff_coreai.py) | Hybrid config, `JeffCheckpoint` (bf16 safetensors, `language_model.*` keys, vision tower skipped), 255×hidden `readout.safetensors`, host softmax. `decision_messages` / `prompt_ids`: an exact port of Jeff's prompt (system message, `Question:` / `State:` / `Options:` / `Latest:`, chat template with `enable_thinking=False`). Answer codes come from `decision_config.json` (A..Z, then single-token pairs: `BQ` is skipped, so index 68 is `BR`) |
| Core AI export | [`coreai/jeff_coreai_build.py`](../../coreai/jeff_coreai_build.py) | Rebinds `qwen38_coreai_build` widths to Jeff `text_config`; FP16 or light INT8; **no** LUT/VQ/palettize; prefill-only entries; `head_readout.aimodel` |
| Runtime | [`coreai/jeff_coreai_runtime.py`](../../coreai/jeff_coreai_runtime.py) | IOSurface `NDArray` I/O. Prompts longer than the entry run as chained TP-row calls: DeltaNet conv/recurrent state carried, every call's `k/v_new` rows written to the KV cache at its position, history mask opened up to it. T=1 readout on the last prompt row |
| Reference | [`scripts/jeff_reference.py`](../../scripts/jeff_reference.py) | Jeff's PyTorch forward (`Qwen3_5Model` FP32/bf16 + readout / temperature). Checks the prompt port against upstream `jeff/model.py` token for token. `--magnitudes` (DeltaNet FP16 headroom), `--chunk-dump` (per-chunk ANE error) |
| CLI | [`scripts/jeff_coreai_convert.py`](../../scripts/jeff_coreai_convert.py), [`scripts/jeff_coreai_smoke.py`](../../scripts/jeff_coreai_smoke.py) | Convert / host + Core AI smoke; `--cases` parity (max \|dp\|, KL, argmax, hidden cosine), `--bench`, `--dump` |
| Launcher | [`forge.py`](../../forge.py) `jeff-convert`, `jeff-smoke`, `system1-ane-serve`, `jeff-train-lora` | Jeff-only CLI. `COREAI_PYTHON` names the Core AI SDK interpreter for convert, `--build` smoke, `system1-ane-serve` and `compile`. `system1-ane-serve` tokenizes with `TOKENIZER_PYTHON` |
| Server | [`scripts/system1_ane_serve.py`](../../scripts/system1_ane_serve.py) | `POST /v1/systemone` (and `/v1/decide`), `GET /health`, `GET /v1/models`. One loaded package, one decision at a time. Optional `prefix_cache` calls `decide(..., prefix=)` |
| Demo | [`scripts/jeff_demo.html`](../../scripts/jeff_demo.html) | Snake (one decision per move) and a message-routing panel, served at `/` |

This Jeff-only tree has no 27B `convert` / `serve` / `hf_release`. DFlash2 is not built. `--quant int8` is per-channel INT8 on large projections only (not GPTQ/VQ).

Host smoke steps [`qwen38_decode_ref.DecodeLayer`](../../scripts/qwen38_decode_ref.py) (config-driven GDN + gated attention). It matches Jeff's PyTorch FP32 forward to 2e-6 in probability, so it is a portable golden for fixture tests. ANE uses tanh-SiLU, FP16 and the Core AI graph; it is compared against the FP32 reference, not expected to be bit-identical.

## Convert / compile / smoke (M5 Max)

Weights stay on the Mac. Default layout: `$HOME/Models/jeff/jeff-base-v1.3` (`config.json`, `model.safetensors`, `readout.safetensors`, `decision_config.json`, tokenizer). Do not download them in a cloud agent.

Two interpreters on the M5 Max:

- **Forge `.venv`** (Python 3.11): transformers, for the launcher, the reference and prompt token ids.
- **Core AI SDK venv** (Python 3.12): `coreai-torch`, `coreai-opt`, `coreai.runtime`, for convert, compile and `--build` smoke. Point `COREAI_PYTHON` at that venv's `bin/python`.

`COREAI_PYTHON` points `forge.py` at the SDK venv. Compile and smoke must run in the same interpreter: the ANE cache is keyed by the loading Python's identity. A framework Python caches as `org.python.python`; the SDK venv's caches as `python`. Compiling a `jeff-decision` build therefore also uses `COREAI_PYTHON`.

```sh
export COREAI_PYTHON=/path/to/coreai-sdk/bin/python   # your Core AI SDK venv
M="$HOME"/Models/jeff/jeff-base-v1.3

# 1) Plan only (any OS; no SDK)
python forge.py jeff-convert --model $M --output "$HOME"/Models/jeff-coreai \
  --ctx 2048 --prefill 256 --quant fp16 --dry-run

# 2) Export FP16 backbone + 255-way readout (macOS + Core AI SDK; runs in $COREAI_PYTHON)
python forge.py jeff-convert --model $M --output "$HOME"/Models/jeff-coreai \
  --ctx 2048 --prefill 256 --quant fp16          # light INT8 projections: --quant int8

# 3) Specialize for this Mac's ANE in $COREAI_PYTHON (no drafter)
python forge.py compile --build "$HOME"/Models/jeff-coreai/coreai

# 4) Golden cases: exact Jeff prompt ids + PyTorch FP32 probabilities (forge .venv; checks the port vs upstream)
python scripts/jeff_reference.py --model $M --jeff-src "$HOME"/Models/jeff/jeff-src/src \
  --out "$HOME"/Models/jeff/spike/parity/cases.json

# 5) Core AI prefill + readout on the ANE, parity vs the cases, timing (runs in $COREAI_PYTHON)
python forge.py jeff-smoke --model $M --build "$HOME"/Models/jeff-coreai/coreai \
  --cases "$HOME"/Models/jeff/spike/parity/cases.json --bench 5 \
  --out "$HOME"/Models/jeff/spike/parity/coreai_fp16_p256.json
#    add --host for the DecodeLayer reference; --row row.json or --state/--options for one Jeff row

# 6) Placement audit of the cached specializations the smoke used
python coreai/inspect_coreai_cache.py --model-dir "$HOME"/Models/jeff-coreai/coreai \
  --executable python --strict

# 7) Where the FP16 error enters: per-chunk ANE outputs vs HF layers, DeltaNet magnitudes
$COREAI_PYTHON scripts/jeff_coreai_smoke.py --model $M --build "$HOME"/Models/jeff-coreai/coreai \
  --cases "$HOME"/Models/jeff/spike/parity/cases.json --dump "$HOME"/Models/jeff/spike/parity/dump_p256
python scripts/jeff_reference.py --model $M --rows rows.json --magnitudes \
  --chunk-dump "$HOME"/Models/jeff/spike/parity/dump_p256 --out cases_layers.json
```

`--prefill` must be a multiple of 8 and greater than 8 (GDN sub-chunk). `--ctx` must be ≥ `--prefill`. A prompt may be up to the entry's KV rows (`pkv_len`, 2,048 here): the runtime chains `ceil(n / prefill)` calls. Output layout:

```
jeff-coreai/
  model/     config, tokenizer, decision_config, embed_tokens_fp16.npy, readout
  coreai/    manifest.json, chunk_L00-03.aimodel, …, head_readout.aimodel
```

`manifest.json` has `"kind": "jeff-decision"`, `"dflash2": false`, FP16 KV, and a single `p{prefill}_{ctx}k` entry per chunk. First `compile` / smoke load compiles each package for the ANE once.

Pass/fail for the Mac spike: convert writes packages; compile stays on the ANE (placement audit `fully_ane`); Jeff's exact prompt; Core AI option probabilities within the FP16 noise band of the FP32 reference.

## M5 Max results (7 October 2026)

*Measured 7 Oct 2026; remeasurement pending.*

Apple M5 Max, macOS 27.2 (26B5091g), Xcode 27.2 (27B5019j), ANE bonded compile mode 1 (M5 policy). Checkpoint `jeff-base` v1.3 (`decision_config` step 1113, temperature 1.0752, `prompt_layout: live-last`). Core AI SDK: coreai-core 1.0.0b2, coreai-torch 0.4.2, coreai-opt 0.2.1, torch 2.11. Reference: transformers 5.17, torch 2.14, PyTorch FP32 CPU (the `chunk_gated_delta_rule` reference kernel). Logs and JSON are in `$HOME/Models/jeff/spike/{logs,parity}/` on that Mac.

### Convert and compile

| Build | Convert | Compile (`forge.py compile`) | Packages | ANE placement |
| --- | --- | --- | --- | --- |
| FP16, `p256_2k` (recommended) | 26.5 s, 4.7 GB peak RSS | 79 s (13 s per 4-layer chunk) | 6 × 159 MB chunks + 0.5 MB head; 505 MB host embedding | `fully_ane`: every entry one ANE region, no GPU region, mode 1 |
| FP16, `p1024_2k` | 79 s | 6 min 07 s (61 s per chunk) | 6 × 161 MB | **`gpu_regions_present`**: every chunk has an ANE region and a GPU region; head `fully_ane` |
| FP16, `p256_2k`, `GDN_SQ=64` | 26.6 s | 80 s | same | not audited (numerics sweep) |
| INT8 per-channel, `p256_2k` | 26.6 s | 79 s | 6 × 80 MB | `fully_ane` |

A warm load from the cache takes 0.1 s.

### Prefill on the ANE

Real Jeff prompts (built-in support-chat rows of `jeff_reference.py`, live-last, chat template), median of 5 warm runs. The 255-way readout head adds 0.34 ms.

| Prompt tokens | Options | `p256_2k` calls | `p256_2k` prefill | tok/s | `p1024_2k` prefill |
| --- | --- | --- | --- | --- | --- |
| 175 | 3 | 1 | 63.4 ms | 2,760 | 318.5 ms |
| 238 | 5 | 1 | 64.2 ms | 3,705 | 319.6 ms |
| 1,008 | 30 | 4 | 254.3 ms | 3,964 | 319.7 ms (1 call) |
| 2,018 | 100 | 8 | 508.3 ms | 3,970 | 639.4 ms (2 calls) |

A 256-row call costs 63 ms whether full or not (6 chunks); a prompt costs about 63 ms × `ceil(n / 256)`. The single 1,024-row entry is slower per token than four chained 256-row calls and partly runs on the GPU, so `--prefill 256` is the build to use. Per-call overhead below 256 rows was not measured (no `p64` / `p128` build).

### Parity against Jeff's PyTorch FP32 forward

Same token ids (port checked against upstream `jeff/model.py` on every row; unit test `test_prompt_ids_match_upstream_jeff`). Probabilities are `softmax(readout[:n] @ h / T)`.

| Prompt | FP32 answer (p) | ANE answer (p) | max \|dp\| | KL(FP32 ‖ ANE) | hidden cosine | bf16 max \|dp\| / KL / argmax |
| --- | --- | --- | --- | --- | --- | --- |
| 175 tok, 3 opt | B 0.4639 | B 0.4564 | 0.0075 | 1.1e-4 | 0.999975 | 0.0089 / 2.4e-4 / same |
| 238 tok, 5 opt | C 0.4525 | C 0.4537 | 0.0012 | 4.6e-6 | 0.999974 | 0.0092 / 2.8e-4 / same |
| 1,008 tok, 30 opt | B 0.2794 | B 0.2749 | 0.0091 | 6.6e-4 | 0.999913 | 0.0023 / 1.3e-4 / **flips** (A) |
| 2,018 tok, 100 opt | A 0.2460 | **B** 0.2433 | 0.0131 | 8.3e-4 | 0.999879 | 0.0023 / 1.6e-4 / same |

- **Host DecodeLayer** vs FP32 (175 and 238 tokens): max \|dp\| 1.1e-6 and 1.9e-6, KL ≈ 1e-11.
- **The one ANE argmax flip** is a near tie: FP32 has A 0.2460 against B 0.2448, a 0.0012 margin under the ANE's 0.013 error. bf16, the precision Jeff serves in on GPU / MLX, flips a different near tie (the 1,008-token row).
- **The readout head is not the error source.** The same readout applied on the host to the ANE hidden state gives the same distribution (`host_head` in the JSON).
- **The ANE build is inside bf16's absolute band** (max \|dp\| ≤ 0.013 vs ≤ 0.009). Its KL is about 3–5× bf16's on the 1K–2K prompts.

**INT8** (`--quant int8`, per-channel on the large projections) halves the packages at the same speed (63.2 / 63.3 / 252.0 / 504.7 ms). Its error is 2–3× FP16's:

- max \|dp\|: 0.007 / 0.011 / 0.019 / 0.022
- KL: 9.7e-5 / 3.2e-4 / 2.2e-3 / 2.8e-3
- hidden cosine: at least 0.99957
- argmax: same four answers as FP16, including the near-tie flip

FP16 stays the default: a 0.8B model has no memory pressure, and INT8 buys no speed.

The 1.3 base without an adapter is uncertain on these rows (top probability 0.25–0.46). Parity on adapter-served prompts, where Jeff is confident, is still to be measured.

### Where the FP16 error comes from

`--chunk-dump` compares every chunk's ANE output with the HF layer output on all rows:

| After layer | rel L2 (175 tok) | rel L2 (2,018 tok) | last-row rel L2 (2,018 tok) | max \|x\| |
| --- | --- | --- | --- | --- |
| 3 | 0.019 | 0.018 | 0.024 | 1 |
| 7 | 0.018 | 0.019 | 0.019 | 2 |
| 11 | 0.021 | 0.025 | 0.019 | 1 |
| 15 | 0.019 | 0.021 | 0.018 | 3 |
| 19 | 0.014 | 0.016 | 0.016 | 7 |
| 23 | 0.010 | 0.012 | 0.009 | 27 |

The error enters in the first four layers, where the residual stream is about 1, and shrinks as the residual grows. It grows only mildly with length (last row after layer 23: 0.005 → 0.008 → 0.009 at 175 / 1,008 / 2,018 tokens). Chaining eight calls adds no step.

DeltaNet magnitudes at 2,018 tokens (`--magnitudes`), in the ANE graph's scaled units (`GDN_SQ` 16 × `GDN_SV` 64 = 1,024 on q·S, 64 on the state):

- **Output:** max 477 (layer 0), so at least 137× FP16 headroom.
- **State:** max 692 (layer 4), at least 94× headroom.
- **Low tail:** the 1st percentile of \|q·S\| is at least 9.4e-5 (layer 1), just above the FP16 normal minimum 6.1e-5.

A `GDN_SQ=64` rebuild moved no chunk error (layer 3 last row 0.0268 vs 0.0264). Final KL was 4.9e-5 / 4.1e-5 / 7.2e-4 / 7.4e-4 against 1.1e-4 / 4.6e-6 / 6.6e-4 / 8.3e-4: noise, not a fix. The 27B scales are adequate for Jeff and the default is unchanged. The early-layer error is ordinary FP16 rounding on small activations; next suspects are the layer-0..3 projections and RMSNorm, not overflow.

### Gaps and next steps

1. **LoRA adapters.** v1.3 is a base for adapters (per-adapter LoRA, readout and temperature). Merging an adapter into the FP16 weights before `jeff-convert` is the simplest path. A shared base with runtime LoRA is not built.
2. **ANE temperature / ECE.** The FP32 temperature is reused. Fit it on holdout rows with the compiled build.
3. **Live-last prefix cache.** `capture_state()` / `prefill(..., prefix=)` copy and resume GDN, KV and the last hidden row. The server forwards a snapshot when `App.prefix_cache` is set. Still open: the policy that snapshots the fixed question and options and replays only `Latest:`.
4. **Longer prompts.** Jeff trains at up to 8,192 tokens. Build `--ctx 8192` (same 256-row entry, 8,192-row KV cache) and recheck placement and parity.
5. **Serving route.** Done: `forge.py system1-ane-serve`. On the demo's prompts (133 and 213 tokens, one 256-row call) a decision is 65 ms, about 15.5/s, and the loaded packages stay `fully_ane`. See [SYSTEM1_ANE_SERVE.md](../SYSTEM1_ANE_SERVE.md). The runtime method a prefix cache should call is `JeffCoreAI.prefill(token_ids, prefix=capture_state())`.

The spike's Core ML readout head in `$HOME/Models/jeff/spike/coreml/` is not used here. Its normalization multiplies `amax` back in and is scale-incorrect; the Core AI head uses `rms_hidden`. The live-last layout comes from `decision_config.json` (the external feasibility note's "state-first" was wrong for v1.3).

## What was verified where

- **Linux cloud VM:** implemented Path B and ran the unit tests (no Core AI SDK, no Jeff weights).
- **M5 Max (this section):** convert, compile, placement audit, prefill timing and FP32 parity with the local weights. Unit tests: `python -m unittest tests.test_jeff_coreai tests.test_launcher tests.test_checkpoint` (34 tests, including the upstream token-id check).

## See also

- [SYSTEM1_ANE_SERVE.md](../SYSTEM1_ANE_SERVE.md) — decision server
- [RESULTS.md](../RESULTS.md) — M5 Max tables
- [ORIGIN.md](../ORIGIN.md) — shared anemll-forge files
