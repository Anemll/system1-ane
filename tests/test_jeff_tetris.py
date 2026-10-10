"""Tetris placements and the El-Tetris label."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_snake import as_decision
from jeff_tetris import (
    describe_placement, empty_board, generate_tetris_kit_rows, generate_tetris_rows,
    oracle_key, placements, play_game, tetris_row,
)


class TetrisTests(unittest.TestCase):
    def test_line_clear_beats_stacking_on_the_pile(self):
        board = empty_board()
        for col in range(4, 10):
            board[19][col] = 1
        options = {item["key"]: item for item in placements(board, "I")}
        self.assertIn("r0c0", options)
        self.assertGreater(options["r0c0"]["rating"], options["r0c6"]["rating"])
        self.assertEqual(options["r0c0"]["features"]["eroded"], 4)
        self.assertEqual(options["r0c0"]["features"]["lines"], 1)
        self.assertEqual(oracle_key(board, "I"), "r0c0")
        text = describe_placement("I", options["r0c0"])
        self.assertIn("I piece, unrotated, columns 0-3", text)
        self.assertIn("clears 1 line", text)

    def test_row_matches_the_generic_format(self):
        board = empty_board()
        label = oracle_key(board, "T")
        row = tetris_row(board, "T", label)
        self.assertIn(label, row["options"])
        decision, index = as_decision(row)
        self.assertEqual(list(decision["question"]["criteria"])[index], label)
        self.assertEqual(decision["state"]["latest"]["board"].count("\n"), 19)
        self.assertEqual(decision["state"]["latest"]["piece"], "T")

    def test_generator_labels_are_legal(self):
        rows = generate_tetris_rows(6, seed=1)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertIn(row["label"], row["options"])
            self.assertGreaterEqual(len(row["options"]), 2)
            self.assertIn("lands on row", row["options"][row["label"]])
        games = play_game(lambda board, piece: oracle_key(board, piece), seed=2, max_pieces=12)
        self.assertGreater(games["pieces"], 0)
        kit = generate_tetris_kit_rows(4, seed=3)
        self.assertEqual(len(kit), 4)
        self.assertEqual(kit[0]["suite"], "tetris")
        self.assertEqual(kit[0]["label"], kit[0]["target"])


if __name__ == "__main__":
    unittest.main()
