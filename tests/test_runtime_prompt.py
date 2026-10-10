"""examples/runtime_prompt.py tokenizes via jeff_tokenize.py, not transformers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

import runtime_prompt  # noqa: E402


STUB = """\
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    if not line.strip():
        continue
    msg = json.loads(line)
    if msg.get("cmd") == "stop":
        break
    print(json.dumps({"ids": [11, 22, 33]}), flush=True)
"""


class RuntimePromptTests(unittest.TestCase):
    def test_sample_does_not_import_transformers(self):
        src = (ROOT / "examples" / "runtime_prompt.py").read_text()
        self.assertNotIn("import transformers", src)
        self.assertNotIn("from transformers", src)
        self.assertNotIn("prompt_ids", src)

    def test_tokenizer_python_honors_env(self):
        old = os.environ.get("TOKENIZER_PYTHON")
        os.environ["TOKENIZER_PYTHON"] = "/opt/host/bin/python"
        try:
            self.assertEqual(runtime_prompt.tokenizer_python(), "/opt/host/bin/python")
        finally:
            if old is None:
                del os.environ["TOKENIZER_PYTHON"]
            else:
                os.environ["TOKENIZER_PYTHON"] = old

    def test_tokenize_ids_uses_worker_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "tok.py"
            script.write_text(STUB)
            ids = runtime_prompt.tokenize_ids(
                Path(tmp) / "model",
                {"state": "x", "question": {"type": "choice", "criteria": {"a": None}}},
                python=sys.executable,
                script=script,
            )
        self.assertEqual(ids, [11, 22, 33])

    def test_tokenize_ids_reports_missing_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "tok.py"
            script.write_text("import sys; sys.exit(0)\n")
            with self.assertRaises(RuntimeError) as ctx:
                runtime_prompt.tokenize_ids(
                    Path(tmp) / "model", {"state": "x"},
                    python=sys.executable, script=script)
        self.assertIn("TOKENIZER_PYTHON", str(ctx.exception))

    def test_main_exits_without_coreai(self):
        with patch.object(runtime_prompt, "AIModel", None):
            self.assertEqual(
                runtime_prompt.main(["--model", "/tmp/m", "--build", "/tmp/b"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
