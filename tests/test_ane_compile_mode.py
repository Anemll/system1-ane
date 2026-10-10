"""SoC -> ANE bonded compile mode policy (docs/ANE_COMPILE_MODE_POLICY.md).

No model, package or ANE compile is touched: detect_soc() is mocked or driven through
COREAI_ARCH, which short-circuits before any ioreg/sysctl call.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ane_compile_mode as SOC


def info(klass, generation=None, variant="", source="test"):
    return SOC.SocInfo(klass, generation, variant, source)


class ClassifyTests(unittest.TestCase):
    def test_generation_mapping(self):
        self.assertEqual(SOC._from_generation(16, "x").klass, "pre_m5")
        self.assertEqual(SOC._from_generation(17, "x").klass, "m5")
        self.assertEqual(SOC._from_generation(17, "x", "pro").klass, "m5_pro_max")
        self.assertEqual(SOC._from_generation(17, "x", "max").klass, "m5_pro_max")
        self.assertEqual(SOC._from_generation(18, "x").klass, "m6")
        self.assertEqual(SOC._from_generation(19, "x").klass, "newer")

    def test_arch_override_detection(self):
        with mock.patch.dict(os.environ, {"COREAI_ARCH": "h17c"}, clear=False):
            soc = SOC.detect_soc()
        self.assertEqual(soc.klass, "m5")
        self.assertEqual(soc.generation, 17)
        self.assertIn("COREAI_ARCH", soc.source)

    def test_arch_override_m6(self):
        with mock.patch.dict(os.environ, {"COREAI_ARCH": "h18g"}, clear=False):
            self.assertEqual(SOC.detect_soc().klass, "m6")


class PolicyModeTests(unittest.TestCase):
    def test_modes(self):
        self.assertIsNone(SOC.policy_mode(info("pre_m5")))
        self.assertEqual(SOC.policy_mode(info("m5")), 1)
        self.assertEqual(SOC.policy_mode(info("m5_pro_max")), 1)
        self.assertEqual(SOC.policy_mode(info("m6")), 2)
        self.assertEqual(SOC.policy_mode(info("newer")), 2)
        self.assertIsNone(SOC.policy_mode(info("unknown")))


class ApplyTests(unittest.TestCase):
    def _run(self, klass, strict=False, **env):
        """Apply inside a clean env; return (mode, env_value) captured before restore."""
        base = {k: v for k, v in os.environ.items() if k != SOC.MODE_ENV}
        base.update(env)
        with mock.patch.dict(os.environ, base, clear=True), \
             mock.patch.object(SOC, "detect_soc", return_value=info(klass)):
            mode = SOC.apply(strict=strict, log=lambda m: None)
            return mode, os.environ.get(SOC.MODE_ENV)

    def test_m5_sets_mode_1(self):
        self.assertEqual(self._run("m5"), (1, "1"))

    def test_m5_pro_max_sets_mode_1(self):
        self.assertEqual(self._run("m5_pro_max"), (1, "1"))

    def test_m6_sets_mode_2(self):
        self.assertEqual(self._run("m6"), (2, "2"))

    def test_newer_sets_mode_2(self):
        self.assertEqual(self._run("newer"), (2, "2"))

    def test_pre_m5_non_strict_returns_none(self):
        self.assertEqual(self._run("pre_m5"), (None, None))

    def test_pre_m5_strict_raises(self):
        with self.assertRaises(SOC.UnsupportedSocError):
            self._run("pre_m5", strict=True)

    def test_unknown_strict_raises(self):
        with self.assertRaises(SOC.UnsupportedSocError):
            self._run("unknown", strict=True)

    def test_unknown_allowed_assumes_m6(self):
        self.assertEqual(self._run("unknown", **{SOC.ALLOW_UNKNOWN_ENV: "1"}), (2, "2"))

    def test_explicit_override_wins_on_m5(self):
        self.assertEqual(self._run("m5", **{SOC.MODE_ENV: "2"}), (2, "2"))

    def test_pre_m5_override_still_fails(self):
        with self.assertRaises(SOC.UnsupportedSocError):
            self._run("pre_m5", strict=True, **{SOC.MODE_ENV: "1"})

    def test_bad_override_rejected(self):
        with self.assertRaises(SOC.UnsupportedSocError):
            self._run("m5", strict=True, **{SOC.MODE_ENV: "bonded"})


if __name__ == "__main__":
    unittest.main()
