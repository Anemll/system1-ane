"""Snake oracle, demo board text, and the generic decision-row format."""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import decision_messages
from jeff_snake import (OPENING_FOOD, OPENING_SNAKE, OPTIONS, RULES, as_decision, board_text, generate_kit_rows,
                        generate_snake_rows, oracle_move, play_game, safe_moves, snake_row)


class SnakeTests(unittest.TestCase):
    def test_opening_board_matches_the_demo(self):
        lines = board_text(OPENING_SNAKE, OPENING_FOOD).split("\n")
        self.assertEqual(len(lines), 8)
        self.assertTrue(all(len(line) == 8 for line in lines))
        self.assertEqual(lines[1], "......F.")
        self.assertEqual(lines[4], "##H.....")

    def test_opening_oracle_steps_toward_the_food(self):
        self.assertEqual(oracle_move(OPENING_SNAKE, OPENING_FOOD), "up")
        self.assertNotIn("up", safe_moves([(0, 3), (1, 3), (2, 3)], (0, 6)))
        self.assertEqual(oracle_move([(0, 3), (1, 3), (2, 3)], (0, 6)), "right")

    def test_tail_cell_is_safe_and_the_body_is_not(self):
        snake = [(1, 0), (0, 0), (0, 1), (1, 1)]
        moves = safe_moves(snake, (5, 5))
        self.assertIn("right", moves)
        self.assertNotIn("up", moves)

    def test_row_uses_the_demo_prompt_fields(self):
        row = snake_row(OPENING_SNAKE, OPENING_FOOD, "up")
        self.assertEqual(row["options"], OPTIONS)
        self.assertEqual(row["instructions"], "Pick the snake's next move.")
        self.assertEqual(row["state"]["rules"], RULES)
        decision, index = as_decision(row)
        self.assertEqual(index, 0)
        text = decision_messages(decision, ["A", "B", "C", "D"], "live-last")[1]["content"][0]["text"]
        self.assertIn("Latest:", text)
        self.assertIn("##H.....", text)
        self.assertLess(text.index("Question:"), text.index("Latest:"))
        hinted, _ = as_decision(snake_row(OPENING_SNAKE, OPENING_FOOD, "right", hint=True))
        hint_text = decision_messages(hinted, ["A", "B", "C", "D"], "live-last")[1]["content"][0]["text"]
        self.assertIn("Safe moves:", hint_text)
        self.assertLess(hint_text.index("guide"), hint_text.index("Latest:"))

    def test_list_options_and_index_label(self):
        decision, index = as_decision({"state": "disk full", "options": ["page", "wait"], "label": 1})
        self.assertEqual(index, 1)
        self.assertEqual(list(decision["question"]["criteria"]), ["page", "wait"])

    def test_kit_rows_use_the_oracle_label(self):
        rows = generate_kit_rows(2, seed=1, steps=3)
        self.assertGreaterEqual(len(rows), 2)
        families = {row["family"] for row in rows}
        self.assertGreaterEqual(len(families), 1)
        for row in rows:
            self.assertEqual(row["label"], row["target"])
            self.assertIn(row["label"], row["question"]["criteria"])
            decision, index = as_decision(row)
            self.assertEqual(list(decision["question"]["criteria"])[index], row["label"])
            self.assertEqual(set(row), {"id", "suite", "family", "state", "question", "label", "target", "source"})

    def test_oracle_survives_and_eats(self):
        def choose(snake, food):
            move = oracle_move(snake, food)
            if move is None:
                raise AssertionError("oracle died")
            return move

        game = play_game(choose, OPENING_SNAKE, OPENING_FOOD, random.Random(0), max_steps=40)
        self.assertGreaterEqual(game["food"], 1)
        self.assertGreaterEqual(game["steps"], 7)

    def test_generator_labels_are_legal(self):
        rows = generate_snake_rows(5, seed=3)
        self.assertEqual(len(rows), 5)
        boards = set()
        for row in rows:
            self.assertIn(row["label"], safe_moves(row["snake"], row["food"]))
            boards.add(row["state"]["latest"]["board"])
        self.assertEqual(len(boards), 5)


if __name__ == "__main__":
    unittest.main()
