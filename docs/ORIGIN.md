# Origin of shared files

This repository is a standalone Jeff tree. It started as a copy of [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge) `cursor/jeff-snake-lora-055e` (commit `df7089e`). Jeff convert and runtime still need the hybrid Qwen3.5 graphs and Core AI package helpers that were written for Forge's 27B path.

**Judgment:** keep those files in-tree (same paths) rather than depending on anemll-forge as a package. A package dependency would force a second checkout, pin a 27B-oriented release, and still require the same modules. The files below are marked in their module docstrings.

| Path | What Jeff uses |
| --- | --- |
| `coreai/qwen38_coreai_build.py` | `LayerW`, `Entry`, `rms_hidden`, `kv_len` after `bind_builder_cfg` |
| `scripts/qwen38_ane_model.py` | `cfg` hook, checkpoint layout |
| `scripts/qwen38_ane_chunk.py` | Imported by the model module (MIL graph helpers) |
| `scripts/qwen38_decode_ref.py` | Host `DecodeLayer` for `jeff-smoke --host` |
| `scripts/qwen3_lut_common.py` | Imported by the chunk module |
| `scripts/qwen38_coreai_model.py` | `pick_package`, `buffer`, `writable`, `_spec`, `graph_line` |
| `scripts/qwen38_kv_cache.py` | `put_rows` (and inspect/compile metadata) |
| `scripts/ane_compile_mode.py` | SoC → bonded compile mode |
| `scripts/coreai_compile.py` / `coreai_compile_guide.py` | `forge.py compile` |
| `scripts/soc_variant.py` | No-op `prepare()` on Jeff manifests |
| `coreai/inspect_coreai_cache.py` | Placement audit |
| `coreai/swift_bridge/` | `CoreAIBridge.swift`, `coreai_bridge.py`, `build.sh` |

`qwen38_ane_chunk.py` still contains unused MIL helpers (`gdn_lazy_block`) that used to import `dflash2_gdn_lazy` (removed). Those functions now raise `ImportError`. Jeff convert uses the torch graphs in `qwen38_coreai_build.py`.

27B-only scripts, docs, tests, DFlash2, Hugging Face release helpers, and quantization plans were removed. Git history on `main` is unchanged; this cleanup is ordinary commits on top.
