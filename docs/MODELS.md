# Models and distribution

**Source / code:** https://github.com/Anemll/system1-ane. Every Hugging Face card, `NOTICE` and `config.json` below links back to it (`hf/stage.py` refuses to stage a card without that link).

Weights are not stored in Git. The three demo builds are published on Hugging Face under the `anemll` org as
**exported** Core AI packages (compile once per Mac). Every repo has a root-level `config.json` (Hugging Face counts
downloads by it) and `scripts/download_model.py` fetches that file first.

| Name | Hugging Face repo | Source build (maintainer's Mac) | Size | Origin |
| --- | --- | --- | --- | --- |
| `tetris` | `anemll/jeff-ane-tetris-coreai` | `$HOME/Models/jeff-tetris-training/overnight-20261008/round-02/build` | 1.53 GB | Oct 8 overnight continuation, round 02 (merged LoRA + readout) |
| `snake` | `anemll/jeff-ane-snake-coreai` | `$HOME/Models/jeff-coreai/adapters/snake` | 1.53 GB | Snake LoRA rank 16 (Oct 7), merged |
| `base` | `anemll/jeff-ane-base-coreai` | `$HOME/Models/jeff-coreai` + `jeff/jeff-base-v1.3/model.safetensors` | 3.24 GB | `mstrasser/jeff-base` v1.3, unchanged |

The optional Unsloth-trained stock Snake adapter (not one of the three above, not in the default download set; Jeff remains the default) lives in a folder of a separate repo: [`anemll/system1-base-unsloth`](https://huggingface.co/anemll/system1-base-unsloth), folder [`snake-stock-qwen3.5-unsloth/`](https://huggingface.co/anemll/system1-base-unsloth/tree/main/snake-stock-qwen3.5-unsloth) plus a root `config.json`. Fetch it with `python scripts/download_model.py snake-stock-qwen3.5-unsloth` (or `snapshot_download(repo_id='anemll/system1-base-unsloth', allow_patterns=['config.json', 'snake-stock-qwen3.5-unsloth/*'])`); serve it with `--adapter snake-stock=<folder>/coreai`.

Repo layout (identical for all three):

```
config.json            backbone config (copy of model/config.json)
README.md              model card (hf/<name>/README.md in this repo)
LICENSE, NOTICE        Apache-2.0 + attribution (hf/LICENSE, hf/NOTICE)
coreai/                chunk_L00-03 … chunk_L20-23, head_readout (.aimodel), manifest.json
model/                 tokenizer.json, tokenizer_config.json, chat_template.jinja, config.json,
                       decision_config.json, readout.safetensors, embed_tokens_fp16.npy
```

`system1-ane-serve --model` loads the full backbone checkpoint (embeddings, final norm, shape checks), so only the **base** repo carries `model/model.safetensors` (1.7 GB, unchanged from jeff-base v1.3). Snake and Tetris are served as `--adapter` beside the base; their `model/` folders are for reference/standalone tooling. `--build` / `--adapter` take `coreai/`.

## Download

The Core AI packages come from `pip install` of `coreai-core` (compile + serve; it provides `coreai.runtime`), plus `coreai-torch` and `coreai-opt` for convert (`requirements-conversion.txt`). `coreai-core==1.0.0b2` is on PyPI for Python 3.11-3.13 on macOS arm64; the macOS system Python (3.9) cannot install it, so create the SDK venv with Python 3.11 (README, Quick start step 2). Newer releases exist (coreai-core 1.0.0b3 from 24 Sep 2026, which also adds Python 3.14 wheels; coreai-torch 0.4.3 pins it); this repository has only been run with the pins in `requirements-conversion.txt`, and what b3 changes for these builds has not been checked. Fresh-machine check on an M6 (macOS 27.0.1, compile mode 2): base, snake, tetris and snake-stock all loaded together in one server.

```sh
python scripts/download_model.py                  # tetris, snake, base -> $JEFF_MODELS_DIR (default ~/Models/jeff-ane)
python scripts/download_model.py tetris --revision v0.1.0
```

## Maintainers: stage and upload (needs approval; nothing here runs automatically)

```sh
python hf/stage.py --out ~/jeff_release_prep/hf_staging \
  --tetris "$HOME/Models/jeff-tetris-training/overnight-20261008/round-02/build" \
  --snake  "$HOME/Models/jeff-coreai/adapters/snake" \
  --base   "$HOME/Models/jeff-coreai" \
  --base-weights "$HOME/Models/jeff/jeff-base-v1.3/model.safetensors"
# stage.py replaces absolute paths in manifest.json and refuses to finish if a home path remains.

hf repo create anemll/jeff-ane-tetris-coreai --repo-type model   # repeat for snake, base
hf upload anemll/jeff-ane-tetris-coreai ~/jeff_release_prep/hf_staging/tetris . --repo-type model
hf upload anemll/jeff-ane-snake-coreai  ~/jeff_release_prep/hf_staging/snake  . --repo-type model
hf upload anemll/jeff-ane-base-coreai   ~/jeff_release_prep/hf_staging/base   . --repo-type model
# optional: tag each repo v0.1.0 so download_model.py --revision v0.1.0 is reproducible
```

Not published: the PyTorch merged checkpoints (1.6 GB each), the raw PEFT adapters, training datasets and logs.
The Tetris PEFT adapter (`round-02/model/adapter`, 40 MB) could be added later as `anemll/jeff-adapter-tetris`.

## Licensing

Qwen3.5-0.8B (Qwen Team / Alibaba Cloud) and jeff-base v1.3 (mstrasser) are Apache-2.0, so the derived builds are
Apache-2.0 with attribution in `NOTICE`. Snake and Tetris training data are generated by this repo's heuristics
(El-Tetris oracle, Snake search oracle); no third-party dataset is used. The published PEFT adapters used in
`docs/ADAPTERS.md` (triage / tools / guard / spam) are Apache-2.0; `jeff-adapter-sanctions` and `jeff-adapter-soc`
are CC-BY-NC-4.0 and are not redistributed here.
