import contextlib
import io
from unittest.mock import patch
import importlib.util
import json
import plistlib
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'coreai' / 'inspect_coreai_cache.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CacheAuditTests(unittest.TestCase):
    def test_selectable_audit_uses_only_selected_physical_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            man = {'ctxs': [8192], 'chunks': [{'file': 'chunk.aimodel',
                    'entries': ['v8_8k', 'v8_8k_kvv8'],
                    'entries_by_kv': {'fp16': {'v8_8k': 'v8_8k'}, 'v8': {'v8_8k': 'v8_8k_kvv8'}}}],
                   'head': {'file': 'head.aimodel'},
                   'kv_cache': {'format': 'selectable', 'default': 'fp16', 'formats': {
                       'fp16': {'format': 'fp16', 'keys': 'float16', 'values': 'float16'},
                       'v8': {'format': 'v8', 'keys': 'float16', 'values': 'int8',
                              'scales': 'float16', 'scale_granularity': 'token_head'}}}}
            (root / 'manifest.json').write_text(json.dumps(man))
            self.assertEqual(audit.package_entries(root)[0][1], ['v8_8k'])
            self.assertEqual(audit.package_entries(root, kv_cache_dtype='v8')[0][1], ['v8_8k_kvv8'])

    def fixture(self, root, attrs, graph=b'', entry='v8_8k_hash_0'):
        package = root / 'chunk.aimodel'
        package.mkdir()
        (package / 'main.hash').write_bytes(b'hash')
        cache = root / 'cache'
        p = cache / 'TEST' / 'my-python' / b'hash'.hex() / 'spec' / 'model.aimodelx' / 'x.mpsgraphpackage'
        p.mkdir(parents=True)
        manifest = {'Package Version': {'7.0': {'GPU adapter present': 'NO', 'Optimized Modules': {'preferredANE_mode2': {
            'Entry Function Attributes': {entry: attrs}, 'File Name': 'model.mpsgraph'}}}}}
        (p / 'manifest.plist').write_bytes(plistlib.dumps(manifest))
        (p / 'model.mpsgraph').write_bytes(graph)
        return package, cache

    def test_fully_ane(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'], b'v8_8k_hash_0_ANE_region_0')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', '/bin/my_python')['status'], 'fully_ane')

    def test_gpu_despite_preferred_ane_and_adapter_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.aneAlignedIO'], b'v8_8k_hash_0_GPU_region_0')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', '/bin/my_python')['status'], 'gpu_regions_present')

    def test_gpu_symbols_override_fully_ane_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'], b'v8_8k_hash_0_GPU_region_0')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', 'my_python')['status'], 'gpu_regions_present')

    def test_missing_graph_is_unknown_despite_full_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'])
            next(c.rglob('model.mpsgraph')).unlink()
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', 'my_python')['status'], 'unknown')

    def test_empty_or_unrecognized_graph_is_unknown(self):
        for graph in (b'', b'opaque newer compiler format', b'other_entry_ANE_region_0'):
            with self.subTest(graph=graph), tempfile.TemporaryDirectory() as tmp:
                p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'], graph)
                self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', 'my_python')['status'], 'unknown')

    def test_compiled_override_cannot_be_hidden_by_good_source_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, c = self.fixture(root, ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'], b'v8_8k_hash_0_ANE_region_0')
            compiled = root / 'custom-override.aimodelc'
            compiled.mkdir()
            result = audit.inspect_package(p, ['v8_8k'], c, 'TEST', 'my_python', compiled)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['compiled_override'], str(compiled))

    def test_manifest_custom_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps({'chunks': [{'file': 'chunk.aimodel', 'entries': ['v8_8k'],
                                                                     'compiled': 'custom.aimodelc'}],
                                                           'head': {'file': 'head.aimodel'}}))
            self.assertEqual(audit.package_entries(root)[0][2], root / 'custom.aimodelc')
            self.assertEqual(audit.package_entries(root)[1][2], root / 'head.aimodelc')

    def test_missing_expected_entry_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'])
            self.assertEqual(audit.inspect_package(p, ['v8_16k'], c, 'TEST', 'my_python')['status'], 'unknown')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'OTHER', 'my_python')['status'], 'unknown')

    def test_strict_missing_returns_one_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(audit, 'package_entries', return_value=[(root / 'absent.aimodel', ['v8_8k'], None)]):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    code = audit.main(['--model-dir', tmp, '--os-build', 'TEST', '--strict'])
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(output.getvalue())['packages'][0]['status'], 'unknown')

    def test_drafter_sidecar_dict_uses_actual_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps({'chunks': [], 'head': {'file': 'head.aimodel'}}))
            drafter = root / 'drafter.aimodel'
            drafter.with_suffix('.json').write_text(json.dumps({'entries': {'draft': {}, 'ctx64': {}}}))
            self.assertEqual(audit.package_entries(root, drafter)[-1][1], ['draft', 'ctx64'])


if __name__ == '__main__':
    unittest.main()
