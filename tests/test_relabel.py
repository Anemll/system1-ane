"""relabel: stronger teacher, regret, RL fields and trainer rows, on synthetic games (no real data, no model)."""
import gzip
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT))
import jeff_tetris as T  # noqa: E402
from projects.tetris_training import relabel as R  # noqa: E402


def synthetic_game(seed: int, moves: int = 8, explore: float = 0.0) -> dict:
    rng = random.Random(seed)
    board, decisions = T.empty_board(), []
    for _ in range(moves):
        piece = rng.choice(list(T.PIECES))
        options = T.placements(board, piece)
        if not options:
            break
        scores = [o["rating"] for o in options]
        top = max(scores)
        w = [math.exp((s - top) / 2.0) for s in scores]
        probs = [x / sum(w) for x in w]
        i = rng.choices(range(len(options)), probs)[0] if rng.random() < explore else max(range(len(options)), key=probs.__getitem__)
        decisions.append({"board": T.board_text(board), "piece": piece, "pick": options[i]["key"], "oracle": T.oracle_key(board, piece),
                          "probs": [round(p, 3) for p in probs], "ms": 300, "tokens": 800})
        board = options[i]["board"]
    return {"schema": "system1-ane-tetris-v1", "id": f"{seed:08x}-0000-4000-8000-000000000000", "app": "t", "model": "t",
            "mode": "Fast", "lines": 0, "pieces": len(decisions), "toppedOut": False, "capped": True, "decisions": decisions}


class Relabel(unittest.TestCase):
    def test_q_values_and_regret_are_consistent(self):
        board = T.empty_board()
        options = T.placements(board, "T")
        q = R.q_values(board, "T", options, "search2", 6, {options[0]["key"]})
        self.assertIn(options[0]["key"], q)                       # the played move is always evaluated
        self.assertEqual(sum(1 for o in options if o["key"] in q) >= 6, True)
        el = R.q_values(board, "T", options, "el", 6, set())
        self.assertEqual(len(el), len(options))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            games = [synthetic_game(s, explore=0.5) for s in range(1, 9)]
            with gzip.open(tmp / "games-2026-10-10.jsonl.gz", "wt") as f:
                for g in games:
                    f.write(json.dumps(g) + "\n")
            for mode in ("distill", "awr"):
                out = tmp / mode
                subprocess.run([sys.executable, "-m", "projects.tetris_training.relabel", str(tmp / "games-2026-10-10.jsonl.gz"),
                                "-o", str(out), "--mode", mode, "--workers", "1", "--heldout", "0.25"], cwd=ROOT, check=True,
                               capture_output=True)
                rep = json.loads((out / "report.json").read_text())
                self.assertEqual(rep["games"], 8)
                self.assertGreaterEqual(rep["mean_model_regret"], 0.0)
                rows = [json.loads(l) for s in ("train", "heldout") for l in (out / f"{s}.jsonl").read_text().splitlines()]
                self.assertTrue(rows)
                train_fam = {json.loads(l)["family"] for l in (out / "train.jsonl").read_text().splitlines()}
                held_fam = {json.loads(l)["family"] for l in (out / "heldout.jsonl").read_text().splitlines()}
                self.assertFalse(train_fam & held_fam)           # a game never straddles the split
                for r in rows:
                    self.assertIn(r["label"], r["question"]["criteria"])   # label is a real option key
                    self.assertGreater(r["weight"], 0)
                    self.assertEqual(r["source"]["dataset"], "system1-ane-app")
                if mode == "distill":
                    self.assertTrue(all(1.0 <= r["weight"] <= 4.0 for r in rows))
                else:
                    self.assertTrue(all(0.1 <= r["weight"] <= 5.0 for r in rows))

    def test_decisions_after_spawn_block_are_dropped(self):
        g = synthetic_game(5, moves=4)
        full = [[1] * 10 for _ in range(20)]                  # every spawn cell is filled
        g["decisions"].insert(2, {"board": T.board_text(full), "piece": "T", "pick": "r0c0", "oracle": "r0c0", "probs": [1.0],
                                  "ms": 1, "tokens": 1})
        res = R.label_game((g, "el", 6, 20, 10.0))
        self.assertTrue(res["spawn_blocked"])
        self.assertEqual(len(res["decisions"]), 2)           # only the states before the blocked spawn
        self.assertEqual(res["dropped"], len(g["decisions"]) - 2)

    def test_return_to_go_and_topout_penalty(self):
        g = synthetic_game(3, moves=6)
        a = R.label_game((g, "el", 6, 20, 10.0))
        g2 = dict(g, toppedOut=True)
        b = R.label_game((g2, "el", 6, 20, 10.0))
        self.assertLess(b["decisions"][-1]["return"], a["decisions"][-1]["return"])   # topping out is penalised
        self.assertEqual(a["decisions"][0]["return"], sum(R.GAMMA ** k * d["reward"] for k, d in enumerate(a["decisions"])))


if __name__ == "__main__":
    unittest.main()
