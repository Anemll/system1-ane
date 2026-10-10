"""scripts/download_model.py layout: every build, including the optional Unsloth extra, lands in <dir>/<name>/{coreai,model}."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_model", ROOT / "scripts" / "download_model.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

FOLDER = dm.EXTRA
FILES = dm.EXTRA_NEED


def fake_hub(calls):
    def hf_hub_download(repo, filename, revision=None, local_dir=None):
        (Path(local_dir) / filename).write_text("{}")

    def snapshot_download(repo, revision=None, local_dir=None, allow_patterns=None):
        calls.append(local_dir)
        for rel in FILES:  # the repo stores the build inside a folder named like the build
            path = Path(local_dir) / FOLDER / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        (Path(local_dir) / "config.json").write_text("{}")

    return types.SimpleNamespace(hf_hub_download=hf_hub_download, snapshot_download=snapshot_download)


class StockDownloadLayout(unittest.TestCase):
    def download(self, root):
        calls = []
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub(calls)}):
            return dm.download(FOLDER, root, None), calls

    def test_build_is_flat_like_the_other_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            build, calls = self.download(Path(tmp))
            self.assertEqual(build, Path(tmp) / FOLDER)
            for rel in FILES:
                self.assertTrue((build / rel).is_file(), rel)
            self.assertFalse((build / FOLDER).exists())  # no nested <name>/<name>/
            self.assertEqual([p.name for p in Path(tmp).iterdir()], [FOLDER])  # staging dir removed
            _, again = self.download(Path(tmp))  # a complete build is not downloaded again
            self.assertEqual(len(calls), 1)
            self.assertEqual(again, [])

    def test_old_nested_layout_is_flattened_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rel in FILES:
                path = Path(tmp) / FOLDER / FOLDER / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old")
            build, calls = self.download(Path(tmp))
            self.assertEqual(calls, [])
            self.assertTrue((build / "coreai" / "manifest.json").is_file())
            self.assertFalse((build / FOLDER).exists())


if __name__ == "__main__":
    unittest.main()
