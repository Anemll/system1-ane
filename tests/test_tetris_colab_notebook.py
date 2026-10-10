"""Exercise notebook recovery/accumulation without Colab, downloads or a GPU."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jeff_lora_train as trainer

NOTEBOOK = json.loads((ROOT / "notebooks/03_tetris_training_colab.ipynb").read_text())


def source(cell_id):
    return "".join(next(c for c in NOTEBOOK["cells"] if c["id"] == cell_id)["source"])


def helper_source(name):
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text" and isinstance(node.func.value, ast.BinOp)
                    and isinstance(node.func.value.right, ast.Constant)
                    and node.func.value.right.value == name):
                return ast.literal_eval(node.args[0])
    raise AssertionError(name)


def definitions(text, namespace, names=None):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)
             and (names is None or n.name in names)]
    # Keep only top-level requested functions; nested function definitions belong
    # to their enclosing function and must not execute at module scope.
    if names is None:
        nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-functions", "exec"), namespace)


def fake_adapter(path):
    path.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "readout.safetensors", "adapter_config.json"):
        (path / name).write_text("{}")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)
        self.requires_grad_(False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class NotebookTests(unittest.TestCase):
    def test_all_cells_and_embedded_helpers_parse_without_outputs(self):
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        for name in ("run_train.py", "collect_corrections.py", "run_unsloth_tetris.py"):
            ast.parse(helper_source(name))

    def test_accumulation_matches_full_batches_including_short_tail(self):
        ns = {"T": trainer, "torch": torch, "math": math, "time": time, "accum": 2,
              "_orig": trainer.run_epoch}
        definitions(helper_source("run_train.py"), ns, {"run_epoch_accum"})
        for count in (3, 7, 8):
            with self.subTest(rows=count), contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(3)
                model = Tiny()
                head = torch.nn.Linear(4, 2, bias=False)
                other = copy.deepcopy(head)
                rows = [{"ids": [i % 7 + 1], "label": i % 2, "n": 2} for i in range(count)]
                torch.manual_seed(4)
                trainer.run_epoch(model, head, rows, 4, 0, "cpu", torch.optim.SGD(head.parameters(), lr=.1))
                torch.manual_seed(4)
                ns["run_epoch_accum"](model, other, rows, 2, 0, "cpu", torch.optim.SGD(other.parameters(), lr=.1))
                torch.testing.assert_close(head.weight, other.weight, atol=1e-7, rtol=1e-6)

    def namespace(self, temp):
        work = Path(temp)
        run = work / "runs"
        run.mkdir()
        (work / "recipe.json").write_text("{}")
        ns = {**globals(), "WORK": work, "RUN": run, "BASE": work / "base", "MAIN": work / "data",
              "REPO": ROOT, "HELP": work / "helpers", "PY": sys.executable,
              "DEVICE": "cpu", "RANK": 16, "ALPHA": 32, "BATCH": 2, "GRAD_ACCUM": 2,
              "READOUT_LR": 5e-6, "SEED": 0, "EVAL_BATCH": 2, "EVAL_TRAIN_ROWS": 4,
              "train_until_iso": lambda: "2099-01-01T00:00:00+00:00",
              "time_left_for_training": lambda: 600}
        definitions(source("tetris-colab-03"), ns)
        # Settings definitions require these globals; don't execute installs/settings.
        ns["time_left_for_training"] = lambda: 600
        ns["train_until_iso"] = lambda: "2099-01-01T00:00:00+00:00"
        definitions(source("tetris-colab-10"), ns)
        return ns

    def test_checkpoint_retry_uses_fresh_output_and_saved_weights(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            ns = self.namespace(temp)
            calls = []

            def fake_train(args, **kwargs):
                calls.append(args)
                out = Path(args[args.index("--output") + 1])
                self.assertFalse(out.exists())
                adapter = out / ("checkpoints/step-64" if len(calls) == 1 else "adapter")
                fake_adapter(adapter)
                if len(calls) == 1:
                    (out / "latest-recovery.json").write_text(json.dumps({"adapter": str(adapter)}))
                    return 1, "CUDA out of memory"
                (out / "report.json").write_text(json.dumps({"save_state": "complete", "history": []}))
                return 0, ""

            ns["sh"] = fake_train
            first = ns["run_phase"]("phaseA", Path(temp) / "rows.jsonl", 1, 2e-4)
            self.assertTrue(first["_partial"])
            self.assertTrue((Path(temp) / "phaseA-recovery.tgz").is_file())
            second = ns["run_phase"]("phaseA", Path(temp) / "rows.jsonl", 1, 2e-4)
            self.assertNotEqual(first["_output"], second["_output"])
            self.assertEqual(calls[1][calls[1].index("--init-adapter") + 1], first["_adapter"])
            ns["run_phase"]("phaseA", Path(temp) / "rows.jsonl", 1, 2e-4)
            self.assertEqual(len(calls), 2, "saved final adapter must be reused without repeating training")

    def test_failed_test_eval_still_bundles_weights_and_blocks_unsafe_disconnect(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            ns = self.namespace(temp)
            out = ns["RUN"] / "phaseA" / "attempt-0001"
            fake_adapter(out / "adapter")
            report = {"save_state": "adapter_saved", "history": [], "_output": str(out),
                      "_adapter": str(out / "adapter"), "_partial": True}
            ns.update({"repA": report, "repB": None, "final_report": report, "FINAL": "phaseA",
                       "final_adapter": out / "adapter", "sh": lambda *a, **k: (1, "out of memory"),
                       "NB_START": time.time(), "TIME_BUDGET_MIN": 28, "gpu_name": "CPU test"})
            for key in ("LORA_LR", "PHASE_B_LORA_LR", "EPOCHS_A", "EPOCHS_B", "DATA_SEED", "N_ANCHOR",
                        "N_VAL", "N_TEST", "POOL_SEED", "N_FRESH", "N_ANCHOR_REPLAY", "N_CORR",
                        "COLLECT_CORRECTIONS", "REPO_SHA", "BASE_REPO", "BASE_REV", "BASE_SHA256"):
                ns.setdefault(key, 1)
            exec(source("tetris-colab-16"), ns)
            self.assertIsNone(ns["results"]["test_after_final"])
            self.assertIsNotNone(ns["results"]["test_error"])
            ns.update({"POOL": Path(temp) / "pool", "SAVE_TO_DRIVE": False,
                       "DOWNLOAD_TO_BROWSER": False, "DISCONNECT_WHEN_DONE": True})
            for key in ("MAIN", "POOL", "BASE", "HELP"):
                ns[key].mkdir()
            (ns["POOL"] / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(AssertionError, "no verified persistent archive"):
                exec(source("tetris-colab-18"), ns)
            archive = Path(temp) / "jeff_tetris_colab_adapter_bundle.tgz"
            with tarfile.open(archive) as tf:
                self.assertIn("jeff_tetris_colab/adapter/readout.safetensors", tf.getnames())
                self.assertIn("jeff_tetris_colab/NOTICE", tf.getnames())
                self.assertIn("jeff_tetris_colab/LICENSE", tf.getnames())
            expected = archive.with_name(archive.name + ".sha256").read_text().split()[0]
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), expected)

    def test_silent_subprocess_timeout_is_effective_and_keeps_log(self):
        ns = dict(globals())
        definitions(source("tetris-colab-03"), ns, {"sh"})
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            log = Path(temp) / "child.log"
            started = time.monotonic()
            rc, _ = ns["sh"]([sys.executable, "-c", "import time; time.sleep(30)"],
                              log=log, check=False, timeout=.2)
            self.assertNotEqual(rc, 0)
            self.assertLess(time.monotonic() - started, 5)
            self.assertTrue(log.exists())


if __name__ == "__main__":
    unittest.main()
