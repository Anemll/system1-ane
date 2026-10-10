"""Hugging Face model-card templates (hf/) point at the source repo and use the current command names."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/Anemll/system1-ane"
FILES = [*(ROOT / "hf" / n / "README.md" for n in ("base", "snake", "tetris")), ROOT / "hf" / "NOTICE"]


class HfCards(unittest.TestCase):
    def test_every_card_links_the_source_repo(self):
        for f in FILES:
            self.assertIn(SOURCE, f.read_text(), f.name)
        for f in FILES[:3]:
            self.assertIn("**Source / code:**", f.read_text(), f.parent.name)
        self.assertIn(SOURCE, (ROOT / "docs" / "MODELS.md").read_text())

    def test_no_old_github_name_or_command(self):
        old = re.compile(r"Anemll/jeff-ane|jeff-serve")
        for f in [*FILES, ROOT / "docs" / "MODELS.md", ROOT / "hf" / "stage.py"]:
            self.assertIsNone(old.search(f.read_text()), f.name)
        self.assertIn("jeff-ane/coreai-package", (ROOT / "hf" / "stage.py").read_text())  # config_type is kept


if __name__ == "__main__":
    unittest.main()
