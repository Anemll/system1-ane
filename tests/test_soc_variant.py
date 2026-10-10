"""Legacy SoC function-set handling must fail clearly without its optional helper."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import soc_variant


class SocVariantTests(unittest.TestCase):
    def test_jeff_fp16_manifest_needs_no_legacy_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'manifest.json').write_text(json.dumps({'chunks': [{'file': 'chunk.aimodel'}]}))
            with patch.dict(sys.modules, {'strip_functions': None}):
                self.assertEqual(soc_variant.prepare(root, 'm5'), root)

    def test_soc_manifest_reports_missing_optional_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'manifest.json').write_text(json.dumps({
                'chunks': [{'file': 'chunk.aimodel', 'entries_by_soc': {'m5': {'a': 'a_m5'}}}]}))
            with patch.dict(sys.modules, {'strip_functions': None}):
                with self.assertRaisesRegex(RuntimeError, 'python -m pip install coreai-core'):
                    soc_variant.prepare(root, 'm5', state=root/'state')


if __name__ == '__main__':
    unittest.main()
