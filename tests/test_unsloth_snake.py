"""Cheap checks for the Unsloth Snake notebook/scripts: files compile, notebook cells match run_decision.py, LoRA merge math."""
import importlib.util
import json
import py_compile
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UN = ROOT / "scripts" / "unsloth"
NB = ROOT / "notebooks" / "unsloth_snake_decision_model.ipynb"
# the only repository module / file / command names these files may contain
ALLOWED = re.compile(r"jeff_snake|jeff_coreai_runtime|jeff_tokenize\.py|JeffCoreAI|jeff-convert", re.I)


def load(name):
    spec = importlib.util.spec_from_file_location(name, UN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class UnslothSnakeFiles(unittest.TestCase):
    def test_scripts_compile(self):
        with tempfile.TemporaryDirectory() as d:
            for f in sorted(UN.glob("*.py")):
                py_compile.compile(str(f), cfile=str(Path(d) / (f.name + "c")), doraise=True)

    def test_notebook_matches_training_script(self):
        nb = json.loads(NB.read_text())
        self.assertEqual(nb["nbformat"], 4)
        code = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                         for c in nb["cells"] if c["cell_type"] == "code")
        script = (UN / "run_decision.py").read_text()
        missing = [l for l in script.splitlines()
                   if l.strip() and not l.startswith("#") and l not in code and "__file__" not in l]
        self.assertEqual(missing, [])

    def test_only_allowed_repo_names(self):
        files = [NB, UN / "README.md", *UN.glob("*.py"), ROOT / "docs" / "FP16_HEAD_WRAPUP.md"]
        for f in files:
            for n, line in enumerate(f.read_text().splitlines(), 1):
                rest = ALLOWED.sub("", line)
                self.assertNotIn("jeff", rest.lower(), f"{f.name}:{n}")

    def test_no_private_paths_or_secrets(self):
        bad = re.compile(r"/Users/|/Volumes/|192\.168\.|hf_[A-Za-z0-9]{20}|ghp_|sk-[A-Za-z0-9]{20}|\.local\b")
        for f in [NB, *UN.rglob("*")]:
            if f.is_file() and f.suffix in {".py", ".md", ".json", ".ipynb"}:
                self.assertIsNone(bad.search(f.read_text()), f.name)

    def test_parity_rows_shape(self):
        d = json.loads((UN / "data" / "stock_snake_parity_rows.json").read_text())
        self.assertEqual(len(d["rows"]), 5)
        for r in d["rows"]:
            self.assertEqual(len(r["raw_logits"]), len(r["option_keys_in_logit_order"]))
            self.assertEqual(len(r["input_ids"]), r["n_tokens"])


class MergeLora(unittest.TestCase):
    def test_merge_adds_scaled_product(self):
        import torch
        mod = load("prep_ckpt")
        torch.manual_seed(0)
        w = torch.randn(6, 4)
        a, b = torch.randn(2, 4), torch.randn(6, 2)
        tensors = {"m.weight": w.clone(), "other.weight": torch.ones(2)}
        n, mx = mod.merge_lora(tensors, {"m": {"A": a.half(), "B": b.half()}}, 0.5)
        expect = w + 0.5 * (b.half().float() @ a.half().float())
        self.assertEqual(n, 1)
        self.assertTrue(torch.allclose(tensors["m.weight"], expect, atol=1e-6))
        self.assertTrue(torch.equal(tensors["other.weight"], torch.ones(2)))
        self.assertAlmostEqual(mx, (expect - w).abs().max().item(), places=5)

    def test_load_adapter_scale(self):
        import torch
        from safetensors.torch import save_file
        mod = load("prep_ckpt")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "adapter_config.json").write_text(json.dumps(
                {"peft_type": "LORA", "bias": "none", "lora_alpha": 32, "r": 16}))
            save_file({"base_model.model.x.y.lora_A.weight": torch.zeros(2, 3),
                       "base_model.model.x.y.lora_B.weight": torch.zeros(4, 2)}, str(d / "adapter_model.safetensors"))
            scale, layers = mod.load_adapter(d)
        self.assertEqual(scale, 2.0)
        self.assertEqual(list(layers), ["x.y"])


if __name__ == "__main__":
    unittest.main()
