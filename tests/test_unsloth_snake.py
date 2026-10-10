"""Cheap checks for the Unsloth Snake notebook/scripts: files compile, notebook cells match run_decision.py, LoRA merge math."""
import importlib.util
import json
import py_compile
import re
import sys
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


class SnakeDataSplit(unittest.TestCase):
    def test_validation_boards_are_not_in_training(self):
        sys.path.insert(0, str(ROOT))
        from coreai.jeff_snake import generate_snake_rows, row_key
        train, held = load("make_snake_data").build_splits()
        self.assertEqual((len(train), len(held)), (2048, 256))
        self.assertEqual({row_key(r) for r in train} & {row_key(r) for r in held}, set())
        self.assertEqual(len({row_key(r) for r in held}), 256)
        # deterministic; train is the original seed-0 set; only the 5 boards that overlapped training were replaced
        self.assertEqual(train, generate_snake_rows(2048, 0))
        self.assertEqual(held, load("make_snake_data").build_splits()[1])
        old = generate_snake_rows(256, 1)
        self.assertEqual([i for i, (a, b) in enumerate(zip(old, held)) if a != b], [12, 48, 118, 150, 218])

    def test_exclude_skips_boards_without_changing_the_default(self):
        sys.path.insert(0, str(ROOT))
        from coreai.jeff_snake import generate_snake_rows, row_key
        base = generate_snake_rows(20, 3)
        self.assertEqual(base, generate_snake_rows(20, 3, exclude=None))
        skipped = generate_snake_rows(20, 3, exclude={row_key(base[0])})
        self.assertNotIn(row_key(base[0]), {row_key(r) for r in skipped})


class RunStockSnakeHeldout(unittest.TestCase):
    """--games / --state-json must not open heldout.jsonl; --parity / --heldout / the default run must."""

    def run_main(self, argv):
        import os
        from types import SimpleNamespace
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        (tmp / "coreai").mkdir()
        sys.path.insert(0, str(ROOT))
        env = {"STOCK_BUILD": str(tmp), "STOCK_HELDOUT": str(tmp / "missing" / "heldout.jsonl"),
               "STOCK_REPORT": str(tmp / "report.json"), "COREAI_PYTHON": sys.executable}
        with mock.patch.dict(os.environ, env), mock.patch.object(sys, "argv", ["run_stock_snake.py", *argv]):
            mod = load("run_stock_snake")
        move = {"move": "up", "total_ms": 1.0, "prefill_ms": 1.0, "head_ms": 0.0, "confidence": 1.0, "tokens": 1,
                "tokenize_ms": 0.0, "probabilities": {"up": 1.0}, "logits": [0.0], "enc": {"input_ids": []}}
        eng = mock.Mock(head=SimpleNamespace(t_choice=1.0, decision={}), head_desc="x")
        with mock.patch.object(mod, "JeffCoreAI", return_value=mock.Mock(entry="e")), \
                mock.patch.object(mod, "Tokenizer"), mock.patch.object(mod, "ClefEngine", return_value=eng), \
                mock.patch.object(mod, "decide", return_value=move), mock.patch.object(sys, "argv", ["run_stock_snake.py", *argv]), \
                mock.patch("builtins.print"):
            mod.main()

    def test_games_and_state_json_do_not_need_heldout(self):
        self.run_main(["--games", "1", "--max-steps", "2"])
        state = Path(tempfile.mkdtemp()) / "s.json"
        state.write_text(json.dumps({"snake": [[4, 2], [4, 1], [4, 0]], "food": [1, 6]}))
        self.run_main(["--state-json", str(state)])

    def test_heldout_modes_still_read_the_file(self):
        for argv in (["--heldout", "4"], ["--parity"]):
            with self.assertRaises(FileNotFoundError):
                self.run_main(argv)


if __name__ == "__main__":
    unittest.main()
