"""ANE bonded compile mode policy by SoC generation (see docs/ANE_COMPILE_MODE_POLICY.md).

Origin: Anemll/anemll-forge. Jeff compile and ``JeffCoreAI`` apply this before loading packages.

Policy:
  H17 (M5 family: M5 / M5 Pro / M5 Max)  -> MPSGRAPH_ANE_BONDED_COMPILE_MODE=1
  H18 (M6)                               -> 2   (measured release default)
  H19+ (newer than M6)                   -> 2   (assume M6 settings; logged as unvalidated)
  < H17 (pre-M5)                         -> unsupported; fail on model-loading paths
  undetectable                           -> unsupported unless COREAI_ALLOW_UNKNOWN_SOC=1 (then 2)

An explicit MPSGRAPH_ANE_BONDED_COMPILE_MODE in the environment always wins, but the
effective mode is always reported by the startup log and GET /health.

Detection order: COREAI_ARCH code -> `ioreg` soc-generation -> `machdep.cpu.brand_string`
-> `hw.model` class. The helper is centralized so forge.py, coreai_compile.py and the
server runtime agree on one classification.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

MODE_ENV = "MPSGRAPH_ANE_BONDED_COMPILE_MODE"
ALLOW_UNKNOWN_ENV = "COREAI_ALLOW_UNKNOWN_SOC"

# H-number = M-generation + 12 (M5 -> H17, M6 -> H18), confirmed via ioreg.
_M_GENERATION_OFFSET = 12


class UnsupportedSocError(RuntimeError):
    """Raised on a strict policy check for a pre-M5 or undetectable SoC."""


@dataclass(frozen=True)
class SocInfo:
    klass: str          # pre_m5 | m5 | m5_pro_max | m6 | newer | unknown
    generation: int | None   # H-number when known
    variant: str        # base | pro | max | ""
    source: str         # where the classification came from


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _from_generation(h: int, source: str, variant: str = "") -> SocInfo:
    if h < 17:
        klass = "pre_m5"
    elif h == 17:
        klass = "m5_pro_max" if variant in ("pro", "max") else "m5"
    elif h == 18:
        klass = "m6"
    else:
        klass = "newer"
    return SocInfo(klass, h, variant, source)


def detect_soc() -> SocInfo:
    """Classify this Mac's SoC generation without loading any model."""
    arch = os.environ.get("COREAI_ARCH", "")
    m = re.match(r"h(\d+)", arch.strip().lower())
    if m:
        return _from_generation(int(m.group(1)), f"COREAI_ARCH={arch}")
    out = _run(["ioreg", "-l"])
    m = re.search(r'"soc-generation"\s*=\s*<"?H(\d+)"?>', out)
    if m:
        return _from_generation(int(m.group(1)), "ioreg soc-generation")
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    m = re.match(r"Apple M(\d+)(?:\s+(Pro|Max))?", brand)
    if m:
        variant = (m.group(2) or "").lower()
        return _from_generation(int(m.group(1)) + _M_GENERATION_OFFSET, f"brand {brand}", variant)
    model = _run(["sysctl", "-n", "hw.model"]).strip()
    m = re.match(r"Mac(\d+),", model)
    if m:
        # Mac17 -> M5 (H17), Mac18 -> M6 (H18). Observed on anemll-forge M5 Max notes.
        return _from_generation(int(m.group(1)) + _M_GENERATION_OFFSET, f"hw.model {model}")
    return SocInfo("unknown", None, "", "undetected")


def policy_mode(soc: SocInfo) -> int | None:
    """Policy mode for a known generation; None when unsupported."""
    if soc.klass in ("m5", "m5_pro_max"):
        return 1
    if soc.klass in ("m6", "newer"):
        return 2
    return None


def apply(strict: bool = False, log=None, soc: SocInfo | None = None) -> int | None:
    """Set and return the effective ANE bonded compile mode.

    Honors an explicit mode override. Sets os.environ[MODE_ENV] for a supported (or
    explicitly-allowed unknown) SoC. Raises UnsupportedSocError when strict and the SoC
    is pre-M5 or undetectable; otherwise warns and returns None.
    """
    log = log or (lambda m: print(m, file=sys.stderr))
    soc = soc or detect_soc()
    if soc.klass == "pre_m5":
        msg = (f"Unsupported ANE target: pre-M5 SoC ({soc.source or 'generation below H17'}). "
               f"This build supports the M5 family (H17) and M6+ (H18).")
        if strict:
            raise UnsupportedSocError(msg)
        log("warning: " + msg)
        return None
    override = os.environ.get(MODE_ENV)
    if override not in (None, ""):
        try:
            mode = int(override)
        except ValueError:
            raise UnsupportedSocError(f"{MODE_ENV}={override!r} is not an integer") from None
        log(f"ANE compile mode policy: {soc.klass} ({soc.source}) -> bonded mode {mode} (explicit override)")
        return mode
    mode = policy_mode(soc)
    if mode is None:
        allow_unknown = os.environ.get(ALLOW_UNKNOWN_ENV) == "1"
        if allow_unknown and soc.klass == "unknown":
            os.environ[MODE_ENV] = "2"
            log(f"ANE compile mode policy: {soc.klass} ({soc.source}) -> bonded mode 2 "
                f"(assumed M6 settings; {ALLOW_UNKNOWN_ENV}=1)")
            return 2
        msg = (f"Unsupported ANE target: {soc.klass} ({soc.source or 'generation undetected'}). "
               f"This build supports the M5 family (H17) and M6+ (H18). "
               f"Set {MODE_ENV} explicitly, or {ALLOW_UNKNOWN_ENV}=1 to assume M6 settings on an unclassified chip.")
        if strict:
            raise UnsupportedSocError(msg)
        log("warning: " + msg)
        return None
    os.environ[MODE_ENV] = str(mode)
    suffix = " (unvalidated generation; assuming M6 settings)" if soc.klass == "newer" else ""
    log(f"ANE compile mode policy: {soc.klass} ({soc.source}) -> bonded mode {mode}{suffix}")
    return mode


if __name__ == "__main__":  # quick manual classification, e.g. `python scripts/ane_compile_mode.py`
    info = detect_soc()
    print(f"{info.klass} generation={info.generation} variant={info.variant or 'base'} source={info.source}")
    print(f"policy mode = {policy_mode(info)}")
