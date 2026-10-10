"""Buffer lifetime tests with tiny host allocations; no Core AI, dylib, or model loads."""
import ctypes
import gc
import importlib.util
from pathlib import Path
import unittest
import weakref
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None


class MockNative:
    def __init__(self):
        self.allocations = {}
        self.releases = []
        self.next_handle = 1

    def cai_buffer_create(self, dtype, rank, shape, strides, error, capacity):
        assert dtype == 0, "fixtures use float16"
        count = 1
        for i in range(rank):
            count *= shape[i]
        handle = self.next_handle
        self.next_handle += 1
        self.allocations[handle] = ctypes.create_string_buffer(count * 2)
        return handle

    def cai_buffer_address(self, handle):
        return ctypes.addressof(self.allocations[handle])

    def cai_buffer_nbytes(self, handle):
        return ctypes.sizeof(self.allocations[handle])

    def cai_release(self, handle):
        self.releases.append(handle)
        del self.allocations[handle]


@unittest.skipIf(np is None, "NumPy is required for bridge ownership tests")
class BufferOwnershipTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "coreai/swift_bridge/coreai_bridge.py"
        spec = importlib.util.spec_from_file_location("bridge_ownership_test", source)
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)
        self.native = MockNative()
        self.bridge._lib = self.native
        # Fail explicitly if a change accidentally opens a real library.
        self.no_dylib = patch.object(ctypes, "CDLL", side_effect=AssertionError("native loads prohibited"))
        self.no_dylib.start()
        self.addCleanup(self.no_dylib.stop)

    def test_buffer_without_external_views_is_released(self):
        buffer = self.bridge.Buffer((2, 3))
        ref = weakref.ref(buffer)
        del buffer
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(self.native.releases, [1])
        self.assertEqual(self.native.allocations, {})

    def test_external_views_keep_storage_alive_until_last_view_dropped(self):
        buffer = self.bridge.Buffer((2, 3))
        first, second = buffer.np, buffer.np
        self.assertIs(first.base, buffer)
        self.assertIs(second.base, buffer)
        self.assertTrue(np.shares_memory(first, second))
        first[0, 1] = 17
        self.assertEqual(float(second[0, 1]), 17)
        ref = weakref.ref(buffer)
        del buffer, first
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertEqual(self.native.releases, [])
        self.assertEqual(float(second[0, 1]), 17)
        del second
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(self.native.releases, [1])

    def test_derived_slice_retains_owner_and_releases_once(self):
        buffer = self.bridge.Buffer((2, 3))
        view = buffer.np[1:, 1:]
        ref = weakref.ref(buffer)
        del buffer
        gc.collect()
        view[:] = 9
        self.assertTrue(np.all(view == 9))
        self.assertIsNotNone(ref())
        self.assertEqual(self.native.releases, [])
        del view
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(self.native.releases, [1])

    def test_replaced_buffer_and_view_pairs_do_not_accumulate(self):
        current = None
        for _ in range(20):
            buffer = self.bridge.Buffer((2, 3))
            current = (buffer, buffer.np)
            del buffer
            gc.collect()
            self.assertEqual(len(self.native.allocations), 1)
        del current
        gc.collect()
        self.assertEqual(self.native.allocations, {})
        self.assertEqual(self.native.releases, list(range(1, 21)))


if __name__ == "__main__":
    unittest.main()
