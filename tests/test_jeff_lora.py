"""LoRA wrap, forward, and merge without loading Jeff."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_lora import LoRALinear, attach_lora, merge_peft_adapter, save_merged_checkpoint


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 3, bias=False)
        self.in_proj_a = nn.Linear(4, 2, bias=False)
        self.visual = nn.Module()
        self.visual.q_proj = nn.Linear(4, 3, bias=False)


class LoRATests(unittest.TestCase):
    def test_forward_and_merge_match_the_low_rank_update(self):
        torch.manual_seed(0)
        toy = Toy()
        with torch.no_grad():
            toy.q_proj.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        layers = attach_lora(toy, rank=2, alpha=4)
        self.assertEqual([layer.key for layer in layers], ["q_proj.weight"])
        self.assertIsInstance(toy.q_proj, LoRALinear)
        self.assertIsInstance(toy.visual.q_proj, nn.Linear)
        self.assertIsInstance(toy.in_proj_a, nn.Linear)
        layer = layers[0]
        with torch.no_grad():
            layer.A.fill_(1)
            layer.B.fill_(1)
        x = torch.ones(1, 4)
        delta = layer.scale * (layer.B.detach() @ layer.A.detach())
        expected = x @ (toy.q_proj.base.weight.detach() + delta).T
        self.assertTrue(torch.allclose(layer(x), expected))
        self.assertTrue(torch.allclose(layer.merged_weight(), (toy.q_proj.base.weight.detach() + delta).cpu()))

    def test_save_merged_replaces_only_the_adapted_tensor(self):
        toy = Toy()
        with torch.no_grad():
            toy.q_proj.weight.zero_()
        layers = attach_lora(toy, rank=2, alpha=2)
        with torch.no_grad():
            layers[0].A.fill_(1)
            layers[0].B.fill_(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            dest = root / "dst"
            source.mkdir()
            original = torch.zeros(3, 4)
            other = torch.ones(2)
            save_file({"q_proj.weight": original, "frozen.weight": other}, str(source / "model.safetensors"))
            (source / "decision_config.json").write_text("{}\n")
            readout = nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                readout.weight.fill_(0.5)
            save_merged_checkpoint(source, dest, layers, readout)
            merged = load_file(str(dest / "model.safetensors"))
            self.assertTrue(torch.equal(merged["frozen.weight"], other))
            self.assertGreater(float(merged["q_proj.weight"].abs().sum()), 0)
            self.assertEqual(load_file(str(dest / "readout.safetensors"))["weight"].shape, (2, 4))
            self.assertTrue((dest / "decision_config.json").is_file())

    def test_peft_merge_uses_alpha_over_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter, dest = root / "base", root / "adapter", root / "merged"
            base.mkdir()
            adapter.mkdir()
            save_file({"language_model.layers.0.q_proj.weight": torch.zeros(3, 4)}, str(base / "model.safetensors"))
            save_file({"weight": torch.zeros(2, 4)}, str(base / "readout.safetensors"))
            (base / "decision_config.json").write_text(json.dumps({"temperature": 1.0, "codes": ["A"]}))
            (base / "notes.txt").write_text("keep")
            save_file({
                "base_model.model.language_model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4),
                "base_model.model.language_model.layers.0.q_proj.lora_B.weight": torch.ones(3, 2),
            }, str(adapter / "adapter_model.safetensors"))
            save_file({"weight": torch.ones(2, 4)}, str(adapter / "readout.safetensors"))
            (adapter / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "bias": "none", "r": 2, "lora_alpha": 4,
            }))
            (adapter / "decision_config.json").write_text(json.dumps({"temperature": 0.5}))
            info = merge_peft_adapter(base, adapter, dest)
            merged = load_file(str(dest / "model.safetensors"))["language_model.layers.0.q_proj.weight"]
            self.assertTrue(torch.equal(merged, torch.full((3, 4), 4.0)))
            self.assertEqual(info["temperature"], 0.5)
            decision = json.loads((dest / "decision_config.json").read_text())
            self.assertEqual(decision["temperature"], 0.5)
            self.assertEqual(decision["codes"], ["A"])
            self.assertEqual((dest / "notes.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
