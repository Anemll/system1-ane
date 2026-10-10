"""Dataset integrity, browser semantics and compatibility with the normal Jeff trainer."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from projects.tetris_training import dataset as ds
from projects.tetris_training import play
from projects.tetris_training.play import deduplicate_corrections, request_body
from jeff_demo_soak import t_placements, t_topped_out
from jeff_snake import as_decision
import forge


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = {"train": 48, "validation": 16, "test": 16}
        cls.splits, cls.games = ds.build_splits(cls.counts, seed=7, max_pieces=40, noise=.5)

    def test_split_isolation_labels_and_later_recovery_states(self):
        summary = ds.validate_splits(self.splits)
        for split, count in self.counts.items():
            self.assertEqual(summary[split]["rows"], count)
            self.assertGreater(summary[split]["rows_after_12_pieces"], 0)
            self.assertGreater(summary[split]["rows_with_holes"], 0)
        self.assertEqual(summary["train"]["modes"], {"teacher": 24, "noisy": 12, "garbage": 12})
        repeated, games = ds.build_splits(self.counts, seed=7, max_pieces=40, noise=.5)
        self.assertEqual(repeated, self.splits)
        self.assertEqual(games, self.games)

    def test_noise_changes_behavior_but_never_the_teacher_label(self):
        rows, _ = ds.rollout(5, "noisy", 60, 1.0, "noise-test")
        self.assertTrue(any(r["label"] != r["source"]["behavior_action"] for r in rows))
        for row in rows:
            board, piece = ds.board_from_row(row)
            self.assertEqual(row["label"], ds.teacher_pick(ds.placements(board, piece))["key"])

    def test_prior_training_states_are_excluded_from_every_split(self):
        excluded = {ds.state_key(row) for row in self.splits["validation"]}
        splits, _ = ds.build_splits(self.counts, seed=7, max_pieces=40, noise=.5, excluded_states=excluded)
        ds.validate_splits(splits)
        self.assertFalse(excluded & {ds.state_key(row) for rows in splits.values() for row in rows})

    def test_prompt_options_and_topout_match_browser_port(self):
        for row in self.splits["train"]:
            board, piece = ds.board_from_row(row)
            self.assertEqual(ds.topped_out(board, piece), t_topped_out(board, piece))
            self.assertEqual(list(row["options"].items()),
                             [(p["key"], p["text"]) for p in t_placements(board, piece)])
            decision, index = as_decision(row)
            self.assertEqual(list(decision["question"]["criteria"])[index], row["label"])
            self.assertNotIn("teacher_scores", decision["state"])
            body = request_body(board, piece, ds.placements(board, piece), "tetris")
            self.assertEqual(body["questions"]["move"], decision["question"])
        board = ds.empty_board()
        board[0][4] = 1
        self.assertTrue(ds.placements(board, "O"))
        self.assertTrue(ds.topped_out(board, "O"))
        self.assertTrue(t_topped_out(board, "O"))
        with patch.object(ds, "empty_board", return_value=board):
            # Select a seed that spawns O to exercise termination before the policy.
            import random
            seed = next(s for s in range(100) if list(ds.PIECES)[random.Random(s).randrange(7)] == "O")
            rows, game = ds.rollout(seed, "teacher", 10, 0, "blocked")
        self.assertEqual(rows, [])
        self.assertEqual(game["terminal"], "spawn_blocked")

    def test_audit_rejects_duplicate_wrong_label_and_option_reordering(self):
        damaged = copy.deepcopy(self.splits)
        damaged["train"][0] = copy.deepcopy(damaged["validation"][0])
        with self.assertRaisesRegex(ValueError, "duplicate state"):
            ds.validate_splits(damaged)
        damaged = copy.deepcopy(self.splits)
        row = damaged["train"][0]
        row["label"] = next(k for k in row["options"] if k != row["label"])
        with self.assertRaisesRegex(ValueError, "label"):
            ds.validate_splits(damaged)
        damaged = copy.deepcopy(self.splits)
        damaged["train"][0]["options"] = dict(reversed(list(damaged["train"][0]["options"].items())))
        with self.assertRaisesRegex(ValueError, "option text/order"):
            ds.validate_splits(damaged)

    def test_collection_excludes_all_existing_splits_and_duplicate_visits(self):
        existing = [row for rows in self.splits.values() for row in rows]
        new, _ = ds.rollout(9876, "teacher", 50, 0, "new")
        corrected = deduplicate_corrections(existing + new + new, existing)
        keys = [ds.state_key(row) for row in corrected]
        self.assertTrue(keys)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(set(keys) & {ds.state_key(row) for row in existing})

    def test_forge_forwards_explicit_split_and_adapter_without_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"text_config": {
                "num_hidden_layers": 2, "hidden_size": 8,
                "layer_types": ["linear_attention", "full_attention"]}}))
            argv = ["jeff-train-lora", "--model", str(model), "--output", str(Path(tmp) / "out"),
                    "--dataset", str(Path(tmp) / "train.jsonl"),
                    "--heldout-dataset", str(Path(tmp) / "validation.jsonl"),
                    "--init-adapter", str(Path(tmp) / "adapter"), "--eval-only", "--eval-batch", "16",
                    "--eval-train-rows", "64", "--checkpoint-every", "64",
                    "--train-until", "2026-10-08T12:15:00+00:00", "--dry-run"]
            args = forge.parser().parse_args(argv)
            command, _ = forge.prepare(args)
            for flag in ("--dataset", "--heldout-dataset", "--init-adapter"):
                self.assertEqual(command[command.index(flag) + 1], str(Path(argv[argv.index(flag) + 1]).resolve()))
            self.assertIn("--eval-only", command)
            self.assertEqual(command[command.index("--eval-batch") + 1], "16")
            self.assertEqual(command[command.index("--eval-train-rows") + 1], "64")
            self.assertEqual(command[command.index("--checkpoint-every") + 1], "64")
            self.assertEqual(command[command.index("--train-until") + 1], "2026-10-08T12:15:00+00:00")

    def test_dataset_files_and_served_collection_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            data = Path(tmp) / "data"
            out = Path(tmp) / "collection"
            self.assertEqual(ds.main(["build", "--output", str(data), "--train", "16",
                                      "--validation", "8", "--test", "8", "--max-pieces", "30"]), 0)
            self.assertEqual(ds.audit(data)["train"]["rows"], 16)

            def mock_response(request, timeout):
                if isinstance(request, str):
                    return io.BytesIO(b'{"status":"ready","backend":"test-double"}')
                body = json.loads(request.data)
                row = {"state": body["state"]}
                board, piece = ds.board_from_row(row)
                options = ds.placements(board, piece)
                # A deliberately weak behavior policy. Collection must still label with the teacher.
                action = min(options, key=lambda item: item["rating"])["key"]
                return io.BytesIO(json.dumps({"timings": {"total_ms": 1.0},
                                             "answers": {"move": {"choice": action}}}).encode())

            with patch.object(play.urllib.request, "urlopen", side_effect=mock_response):
                self.assertEqual(play.main(["--dataset", str(data), "--split", "train", "--collect",
                                            "--adapter", "test", "--games", "1", "--max-pieces", "30",
                                            "--output", str(out)]), 0)
            report = json.loads((out / "report.json").read_text())
            rows = ds.read_jsonl(out / "corrections.jsonl")
            self.assertTrue(rows)
            self.assertEqual(report["warm_server_ms"]["n"], report["games"][0]["pieces"])
            self.assertTrue(any(r["label"] != r["source"]["behavior_action"] for r in rows))
            for row in rows:
                board, piece = ds.board_from_row(row)
                self.assertEqual(row["label"], ds.teacher_pick(ds.placements(board, piece))["key"])
            combined = {s: ds.read_jsonl(data / f"{s}.jsonl") for s in ds.SPLITS}
            combined["train"] = ds.read_jsonl(out / "train.jsonl")
            ds.validate_splits(combined)
            with (data / "train.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                ds.audit(data)


if __name__ == "__main__":
    unittest.main()
