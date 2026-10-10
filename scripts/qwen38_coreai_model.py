"""Core AI package helpers used by Jeff (load, IOSurface buffers, compile-mode cache).

Origin: Anemll/anemll-forge. Jeff's runtime only needs pick_package / buffer / writable / _spec /
graph_line. The 27B AneQwen3 decode/DFlash2 classes were removed from this checkout.
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
from pathlib import Path

import numpy as np
import coreai_compile_guide as G
import ane_compile_mode as SOC

try:
    from coreai.runtime import AIModel, NDArray
    from coreai.runtime._ndarray import StorageKind
    IOS = StorageKind.IO_SURFACE
except ImportError:  # Linux / no Core AI SDK
    AIModel = NDArray = StorageKind = IOS = None

CACHE = Path.home() / "Library/Caches/coreai-cache"
MODE_ENV = SOC.MODE_ENV
BRIDGE_DIR = Path(os.path.expanduser(os.environ.get(
    "COREAI_BRIDGE_DIR", str(Path(__file__).resolve().parents[1] / "coreai" / "swift_bridge"))))
f16 = np.float16


def _vtuple(v: str):
    return tuple(int(x) for x in v.split(".") if x.isdigit())


def _os_mpsgraph() -> str | None:
    """MPSGraph package version this OS reads (MetalPerformanceShadersGraph's short version; env COREAI_MPSGRAPH_MAX)."""
    if os.environ.get("COREAI_MPSGRAPH_MAX"):
        return os.environ["COREAI_MPSGRAPH_MAX"]
    import plistlib
    p = Path("/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/Resources/version.plist")
    try:
        return plistlib.loads(p.read_bytes()).get("CFBundleShortVersionString")
    except OSError:
        return None


def _package_mpsgraph(path: Path) -> str | None:
    """Highest MPSGraph package version inside a compiled package (None: no MPSGraph delegate)."""
    import plistlib
    best = None
    for mf in path.rglob("manifest.plist"):
        if ".mpsgraphpackage" not in str(mf):
            continue
        try:
            for v in plistlib.loads(mf.read_bytes()).get("Package Version", {}):
                if best is None or _vtuple(v) > _vtuple(best):
                    best = v
        except Exception:  # noqa: BLE001
            continue
    return best


def readable(compiled: Path) -> tuple[bool, str]:
    """Whether this OS can load a compiled package (a too-new MPSGraph package segfaults the load)."""
    pv, ov = _package_mpsgraph(compiled), _os_mpsgraph()
    if pv and ov and _vtuple(pv) > _vtuple(ov):
        return False, f"MPSGraph package {pv} > OS {ov}"
    return True, ""


def _modes(manifests) -> set[int]:
    """aneBondedCompileMode values in MPSGraph package manifests."""
    out = set()
    for mf in manifests:
        if ".mpsgraphpackage" in str(mf):
            out.update(int(m) for m in re.findall(rb"aneBondedCompileMode\W+(\d+)", mf.read_bytes()))
    return out


def _cache_entries(path: Path) -> list[tuple[Path, set[int]]]:
    """(folder, compile modes) of this process's cached specializations of an .aimodel."""
    d = G.cache_dir(path)
    if d is None or not d.is_dir():
        return []
    return [(e, _modes(e.rglob("manifest.plist"))) for e in d.iterdir() if e.is_dir()]


def _align_mode(path: Path, log=print) -> None:
    """Drop cached specializations compiled in another ANE bonded mode."""
    want = int(os.environ.get(MODE_ENV, "0"))
    stale = [(d, m) for d, m in _cache_entries(path) if m and m != {want}]
    for d, _ in stale:
        shutil.rmtree(d, ignore_errors=True)
    if stale:
        log(f"{path.name}: cached in ANE compile mode {sorted(set().union(*(m for _, m in stale)))}, loading with "
            f"{want}; purged {len(stale)} cache entries, recompiling (several minutes)")


def pick_package(root: Path, file: str, compiled: str | None, log=print) -> tuple[Path, Path]:
    """(package to load, source .aimodel): the .aimodelc when present, readable and compiled in the requested ANE
    mode, else the source (its cache first aligned to that mode)."""
    path = root / file
    comp = root / compiled if compiled else path.with_suffix(".aimodelc")
    if comp.exists():
        ok, why = readable(comp)
        modes, want = _modes(comp.rglob("manifest.plist")), int(os.environ.get(MODE_ENV, "0"))
        if ok and modes and modes != {want}:
            ok, why = False, f"compiled in ANE mode {sorted(modes)}, want {want}"
        if ok:
            return comp, path
        log(f"{comp.name}: {why}; loading {path.name} (specialized on first load, then cached)")
    _align_mode(path, log)
    return path, path


def writable(nd) -> np.ndarray:
    """Writable zero-copy numpy view of an NDArray's own storage."""
    if NDArray is None:
        raise RuntimeError("coreai.runtime is not installed; load a Core AI SDK Python to run Jeff on the ANE")
    mv = memoryview(nd._tensor)  # noqa: SLF001
    ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    return np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr)).view(f16).reshape(nd.shape)


def buffer(shape) -> tuple:
    if NDArray is None or IOS is None:
        raise RuntimeError("coreai.runtime is not installed; load a Core AI SDK Python to run Jeff on the ANE")
    nd = NDArray(np.zeros(shape, f16), IOS)
    return nd, writable(nd)


def _spec():
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    return SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())


def target_graph(man: dict) -> dict:
    """Graph options recorded on the convert manifest's chunk numerics."""
    nums = [c.get("numerics") or {} for c in man.get("chunks", [])]
    recorded = bool(nums) and all("GDN_FAST" in n and "ATT_BLOCK" in n for n in nums)
    gdn = sorted({bool(n.get("GDN_FAST", False)) for n in nums}) or [False]
    att = sorted({int(n.get("ATT_BLOCK", 16384)) for n in nums}) or [16384]
    attp = sorted({int(n.get("ATT_BLOCK_PREFILL", n.get("ATT_BLOCK", 16384))) for n in nums}) or [16384]
    one = lambda v: v[0] if len(v) == 1 else v  # noqa: E731
    att8 = sorted({n.get("ATT_INT8MM", "") for n in nums}) or [""]
    by_layer = {k: v for n in nums for k, v in (n.get("ATT_INT8MM_BY_LAYER") or {}).items()}
    pf8 = sorted({n["ATT_PF8_UNIT"] for n in nums if "ATT_PF8_UNIT" in n})
    return {"gdn_fast": one(gdn), "att_block": one(att), "att_block_prefill": one(attp), "recorded": recorded,
            "att_int8mm": one(att8), "att_int8mm_by_layer": by_layer, "att_pf8_unit": one(pf8) if pf8 else None}


def graph_line(man: dict, root: Path) -> str:
    if man.get("kind") in ("decision", "jeff-decision"):
        widths = man.get("TPS") or [man.get("TP")]
        return (f"jeff-decision {man.get('quant', 'fp16')} ctx={man.get('ctxs')} "
                f"prefill={widths} | build {root}")
    g = target_graph(man)
    fast = g["gdn_fast"] if isinstance(g["gdn_fast"], list) else int(g["gdn_fast"])
    note = "" if g["recorded"] else " (not recorded in manifest: release defaults)"
    pre = "" if g["att_block_prefill"] == g["att_block"] else f" ATT_BLOCK_PREFILL={g['att_block_prefill']}"
    return f"target graph: GDN_FAST={fast} ATT_BLOCK={g['att_block']}{pre}{note} | build {root}"
