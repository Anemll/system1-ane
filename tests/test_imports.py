"""Import every kept Python module. Core AI / coremltools must fail only at use time."""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "coreai", ROOT / "coreai" / "swift_bridge"):
    sys.path.insert(0, str(extra))

MODULES = [
    "forge",
    "ane_compile_mode",
    "coreai_compile",
    "coreai_compile_guide",
    "qwen38_kv_cache",
    "qwen38_coreai_model",
    "qwen38_decode_ref",
    "qwen3_lut_common",
    "qwen38_ane_chunk",
    "qwen38_ane_model",
    "qwen38_coreai_build",
    "jeff_coreai",
    "jeff_coreai_runtime",
    "jeff_coreai_build",
    "jeff_prefix",
    "jeff_lora",
    "jeff_snake",
    "jeff_tetris",
    "system1_ane_serve",
    "jeff_serve",  # deprecated shim
    "system1_env",
    "jeff_coreai_convert",
    "jeff_coreai_smoke",
    "jeff_lora_train",
    "jeff_peft_merge",
    "jeff_peft_torch",
    "jeff_reference",
    "jeff_tokenize",
    "jeff_snake_eval",
    "jeff_tetris_eval",
    "jeff_lora_parity",
    "jeff_demo_soak",
    "soc_variant",
    "coreai_bridge",
    "inspect_coreai_cache",
]


class ImportTests(unittest.TestCase):
    def test_kept_modules_import(self):
        for name in MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_runtime_is_lazy_without_coreai(self):
        import jeff_coreai_runtime
        self.assertIsNone(jeff_coreai_runtime.AIModel)


if __name__ == "__main__":
    unittest.main()
