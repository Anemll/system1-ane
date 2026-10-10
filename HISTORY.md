# History

Jeff on the ANE was built as a series of branches on [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge). This private repo's `main` was copied from `cursor/jeff-snake-lora-055e` at `df7089e`. Other copied branches: `cursor/jeff-decision-ane-85f5`, `cursor/jeff-prefix-cache`, `cursor/jeff-lora-stream`, `cursor/jeff-serve-demo-055e`.

Upstream PRs (all draft against anemll-forge as of 8 October 2026):

## PR #6 — Core AI convert path (`cursor/jeff-decision-ane-85f5`)

https://github.com/Anemll/anemll-forge/pull/6

Path B: `jeff-base` v1.3 as a prefill-only Core AI package. End to end on M5 Max.

- `jeff-convert` / `jeff-smoke` bypass the 27B 64×5120 gate
- Exact Jeff prompt port (`decision_messages` + chat template); codes from `decision_config`
- IOSurface runtime; prompts longer than 256 rows chain calls
- `COREAI_PYTHON` so convert/compile/smoke share one ANE cache key
- Result: FP16 convert 26.5 s, compile 79 s, `fully_ane`, 63–508 ms prefill, FP32 parity inside the FP16 band

Not in that PR: LoRA, serve, prefix-cache policy.

## PR #7 — Prefix cache (`cursor/jeff-prefix-cache`)

https://github.com/Anemll/anemll-forge/pull/7

Stacked on #6. Live-last snapshots of GDN/KV; `prepare_prefix` / `decide(handle, suffix)`. Also INT8 audit and wide-prefill cliff (rows/s drops at 512 and 1024). A Tetris suffix of ~85 tokens was 35.7 ms cached vs 58 ms cold. **This `main` has the snapshot hook (`capture_state`, `prefix=`) but does not install a cache.**

## PR #8 — Decision server and demo (`cursor/jeff-serve-demo-055e`)

https://github.com/Anemll/anemll-forge/pull/8

`forge.py jeff-serve`: `POST /v1/systemone`, `GET /health`, browser Snake + routing panel. Demo latency ~65 ms on the opening board and a refund message. Prefix cache left off pending #7.

## PR #9 — Snake LoRA and published adapters (`cursor/jeff-snake-lora-055e`)

https://github.com/Anemll/anemll-forge/pull/9

This is the branch `main` was copied from (plus later commits on the same line).

- `jeff-train-lora` (Snake, then Tetris)
- `--adapter name=path` and live dropdown
- Merge `mstrasser/jeff-adapter-{triage,tools,guard,spam}`
- Tetris mode, timing readout, IOSurface free-after-call
- `JEFF_QUEUE_MS`, `--prefill-extra`, `JEFF_PREFILL` chain/fit/cheapest
- `--init-adapter`, `--heldout-dataset`, `--eval-only`, CUDA device

## PR #10 — Streamed LoRA (`cursor/jeff-lora-stream`)

https://github.com/Anemll/anemll-forge/pull/10

Experimental: pack LoRA factors as ANE inputs so an adapter needs no recompile (28 program inputs, 43-input cliff). Snake workflows with Jeff and Unsloth heads. **Not merged into this `main`.** Branch kept in the repo.

## After the copy

This repo drops the 27B toolkit and keeps the Jeff tree, with documentation that matches the code. See [CHANGELOG.md](CHANGELOG.md).
