# Architecture

Jeff on this repo is a **prefill-only** decision stack. One request tokenizes a Jeff prompt, runs the hybrid Qwen3.5 backbone on the ANE, and scores options with a 255-way readout. There is no next-token loop, no DFlash2 drafter, and no `/v1/chat/completions`.

## Pieces

| Piece | Path | Role |
| --- | --- | --- |
| Checkpoint | `coreai/jeff_coreai.py` | Hybrid `text_config`, `JeffCheckpoint` (bf16 safetensors, vision/MTP skipped), `readout.safetensors`, `decision_config.json` (codes, temperature, `prompt_layout: live-last`) |
| Prompt | `decision_messages` / `prompt_ids` | Exact port of upstream Jeff: system + `Question:` / `State:` / `Options:` / `Latest:`, chat template with `enable_thinking=False` |
| Export | `coreai/jeff_coreai_build.py` | Rebinds the shared GDN / gated-attention / SwiGLU graphs in `qwen38_coreai_build.py` to Jeff widths; writes 4-layer chunks + `head_readout.aimodel` |
| Runtime | `coreai/jeff_coreai_runtime.py` | IOSurface `NDArray` I/O, chained or multi-width prefill (`SYSTEM1_ANE_PREFILL`), T=1 readout on the last prompt row |
| Server | `scripts/system1_ane_serve.py` | `POST /v1/systemone`, tokenizer subprocess, optional `--adapter` builds, browser demo |
| Prefix hook | `coreai/jeff_prefix.py` | `resume_at` + `capture_state()` / `prefill(..., prefix=)`. The cache **policy** is not installed by default |
| LoRA | `coreai/jeff_lora.py` | Rank-16 update, merge `W += (alpha/rank) B A`, PEFT fold (`jeff_peft_merge.py`) |
| Games | `coreai/jeff_snake.py`, `coreai/jeff_tetris.py` | Oracles, row format, self-play helpers |

Shared files Jeff imports (`qwen38_*`, Swift bridge, compile helpers) keep their anemll-forge names. See [ORIGIN.md](ORIGIN.md).

## Why prefill-only

A Jeff decision is one forward of a short prompt (typically 100–400 tokens for Snake/routing; Tetris placements can be 500–1500). The compiled default is a **256-row** prefill entry and a **2048-row** KV cache (`p256_2k`). Longer prompts chain calls: each call carries Gated DeltaNet conv/recurrent state and writes new KV rows at its position.

A 256-row call costs about 63 ms on M5 Max whether the prompt fills it or not. `SYSTEM1_ANE_PREFILL` chooses how a prompt is split when the build has extra widths (`--prefill-extra`):

| Mode | Behavior |
| --- | --- |
| `chain` | Consecutive TP-row calls (single-width path) |
| `fit` | One call in the narrowest compiled entry that holds the rows, else chain |
| `cheapest` | Chain of compiled widths with the lowest summed per-call cost (default) |

Costs start from a measured table (`DEFAULT_CALL_MS` in `jeff_coreai_runtime.py`) until `calibrate()` times the loaded build.

## Adapters are whole packages

The server does **not** apply LoRA factors at runtime. Train or merge, convert that checkpoint, compile a second `coreai/` directory, and pass `--adapter name=path`. Each adapter can carry its own readout and fitted temperature (`manifest["convert"]["temperature"]`). Embeddings still come from `--model`.

A live-last prefix cache (instructions + options first, changing `state` last) can reuse GDN/KV across moves. `JeffCoreAI.capture_state()` copies the snapshot; the server calls `app.prefix_cache.lookup/store` when that object is set. This tree ships with the hook off.

## Interpreters

| Process | Interpreter | Why |
| --- | --- | --- |
| `forge.py`, train, tokenize | Host `.venv` | `transformers`, torch, the launcher |
| convert, compile, `system1-ane-serve`, ANE smoke | `COREAI_PYTHON` | `coreai_torch` / `coreai.runtime`; ANE cache is keyed by this Python |

`system1-ane-serve` sets `TOKENIZER_PYTHON` to the host interpreter so the Core AI process can tokenize via a subprocess (`scripts/jeff_tokenize.py`).

## Swift bridge

`coreai/swift_bridge/` (`CoreAIBridge.swift`, `coreai_bridge.py`, `build.sh`) is the native Core AI load path used by `forge.py compile`. Set `COREAI_BRIDGE_LIB` or `ANEMLL_CORE_AI_BRIDGE_LIB` to the dylib if it is not next to `coreai_bridge.py`. Decision I/O in `JeffCoreAI` uses the Python `coreai.runtime` binding (IOSurfaces freed after every call).
