"""Clef (Unsloth joint-schema) head: host port vs the fixed-shape ANE-layout port, fp16 option, prompt encoder.

Random small weights, CPU only; no Core AI SDK, model or ANE package is touched.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "coreai"), str(ROOT / "scripts")]

try:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    import clef_head
    import clef_ane_head
except ImportError:  # pragma: no cover
    torch = None

CFG = {"hidden_size": 1024, "width": 64, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 128}
T, K = 24, 8


class FakeTokenizer:
    """One token per character; enough to check spans."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 997 for c in text]}


def make_head_dir(tmp: Path) -> Path:
    (tmp / "joint_head_config.json").write_text(json.dumps(CFG))
    (tmp / "unsloth_decision_config.json").write_text(json.dumps({"temperature": [1.0, 1.0, 1.0]}))
    with mock.patch.object(clef_head, "load_file", return_value={}), mock.patch.object(clef_head.ClefHead, "load_state_dict"):
        skeleton = clef_head.ClefHead(tmp)  # randomly initialised modules of the right shapes
    g = torch.Generator().manual_seed(0)
    sd = {k: (torch.randn(v.shape, generator=g) * 0.05).to(torch.bfloat16) if v.ndim else torch.tensor(0.1)
          for k, v in skeleton.state_dict().items()}
    save_file(sd, str(tmp / "joint_head.safetensors"))
    return tmp


@unittest.skipIf(torch is None, "torch / safetensors not installed")
class ClefHeadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = make_head_dir(Path(self.tmp.name))
        g = torch.Generator().manual_seed(1)
        self.hidden = torch.randn(20, 1024, generator=g)
        self.embed = torch.randn(50, 1024, generator=g) * 0.02
        self.ids = [int(i) % 50 for i in range(20)]
        self.qspan, self.ospans = [2, 6], [[8, 11], [11, 14], [14, 18]]
        self.sd = clef_head.load_file(str(self.dir / "joint_head.safetensors"))

    def tearDown(self):
        self.tmp.cleanup()

    def host(self, dtype=None):
        return clef_head.ClefHead(self.dir, dtype, "cpu")

    def ane_port(self, t=T):
        m = clef_ane_head.AneClefHead(self.sd, CFG, T=t, K=K).eval()
        ins = clef_ane_head.build_inputs(self.hidden.numpy(), self.ids, self.embed, self.qspan, self.ospans, t, K)
        with torch.no_grad():
            return m(*[torch.from_numpy(ins[k]) for k in clef_ane_head.INPUT_NAMES]).numpy()[0][:len(self.ospans)]

    def test_encode_record_spans(self):
        enc = clef_head.encode_record(FakeTokenizer(), {"a": 1}, "pick one", {"up": "u", "down": "d"})
        self.assertEqual(enc["option_keys"], ["down", "up"])
        a, b = enc["question_span"]
        self.assertEqual(b - a, len("pick one"))
        self.assertEqual(len(enc["option_spans"]), 2)
        self.assertTrue(all(0 <= s < e <= len(enc["input_ids"]) for s, e in enc["option_spans"]))

    def test_defaults_are_host_fp32_cpu(self):
        h = self.host()
        self.assertEqual((h.dtype, h.device.type, h.weights_file), (torch.float32, "cpu", "joint_head.safetensors"))

    def test_fp16_host_head_matches_fp32(self):
        ref = self.host("fp32")(self.hidden, self.ids, self.embed, self.qspan, self.ospans)
        half = self.host("fp16")(self.hidden, self.ids, self.embed, self.qspan, self.ospans)
        self.assertEqual(int(ref.argmax()), int(half.argmax()))
        self.assertLess(float((ref - half).abs().max()), 0.05)

    def test_ane_layout_port_matches_host_head(self):
        ref = self.host()(self.hidden, self.ids, self.embed, self.qspan, self.ospans).numpy()
        out = self.ane_port()
        self.assertLess(float(np.abs(out - ref).max()), 1e-3)
        self.assertEqual(int(out.argmax()), int(ref.argmax()))

    def test_padding_rows_do_not_change_logits(self):
        self.assertLess(float(np.abs(self.ane_port(T) - self.ane_port(T + 8)).max()), 1e-4)


@unittest.skipIf(torch is None, "torch / safetensors not installed")
class ClefEngineQuestionTypes(unittest.TestCase):
    """ClefEngine answers choice questions; score and noul get a ValueError (HTTP 422), not a crash (HTTP 500)."""

    def engine(self):
        import clef_engine
        eng = object.__new__(clef_engine.ClefEngine)
        eng.tokenize = lambda state, instructions, options: {
            "input_ids": [1, 2, 3], "option_keys": sorted(options), "question_span": [0, 1], "option_spans": []}
        eng.head = mock.Mock(t_choice=1.0)
        eng.logits_for = lambda enc: (np.zeros(len(enc["option_keys"]), np.float32),
                                      {"calls_ms": [1.0], "head_ms": 0.5, "call_widths": [256]})
        return eng

    def test_choice_still_works_and_keeps_option_order(self):
        eng = self.engine()
        out = eng.decide_row("s", {"type": "choice", "criteria": {"up": "a", "down": None}})
        self.assertEqual(len(out["option_probabilities"]), 2)
        self.assertAlmostEqual(sum(out["option_probabilities"]), 1.0, places=5)
        self.assertEqual(out["tokens"], 3)

    def test_score_and_noul_are_rejected_with_value_error(self):
        eng = self.engine()
        for question in ({"type": "score", "criteria": ["bad", "ok", "good"]},
                         {"type": "noul", "criteria": {"true": "yes"}}, {"type": "noul"}):
            for call in (eng.encode_row, eng.decide_row):
                with self.assertRaisesRegex(ValueError, "choice questions only"):
                    call("s", question)


if __name__ == "__main__":
    unittest.main()
