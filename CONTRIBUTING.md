# Contributing to System 1 ANE

This is a standalone Jeff + Core AI tree. Keep changes in the existing layout (`coreai/`, `scripts/`, `forge.py`). Do not reintroduce the 27B Qwen3.8 toolkit.

## Get started

Follow the [README](README.md) on an Apple Silicon Mac with Core AI. Use a branch. Weights and compiled `.aimodel` files stay outside the checkout (see `.gitignore`).

## Report a bug

Open an issue with chip, unified memory, macOS, Xcode/Swift, Python, `COREAI_PYTHON`, the Jeff commit, checkpoint revision, build directory, and the command. For latency, report tokenize / prefill / head / total, token count, `JEFF_PREFILL`, and how many builds were loaded.

## Submit a pull request

1. Explain the behavior change and how to reproduce it.
2. Keep anemll-forge origin notes on shared `qwen38_*` / bridge files if you edit them.
3. Update README or `docs/` when the API, env vars, or pipeline change.
4. Run:

```sh
python forge.py system1-ane-serve --help
python forge.py jeff-convert --help
python forge.py jeff-train-lora --help
python -m unittest discover -s tests
```

These tests do not load Core AI or prove ANE placement. On a Mac, add `jeff-smoke` / `inspect_coreai_cache.py` when you touch convert or runtime.

## Keep secrets out

Do not commit credentials, unpublished weights, or personal model paths. `JEFF_DEFAULT` in `jeff_coreai.py` is a local default on the author's Mac; commands should take `--model`.
