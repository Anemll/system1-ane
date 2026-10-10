"""Per-chip function sets in one Core AI target build.

Origin: Anemll/anemll-forge. Jeff FP16 packages have one function set, so ``prepare()`` is a no-op.

A build exported with ATT_INT8MM_M5 (qwen38_coreai_build.py) holds every entry twice in each chunk package, sharing the
weights: <entry> (M6, FP8 softmax and PV) and <entry>_m5 (no FP8; chunk manifest entries_by_soc). Core AI specializes
a package as a whole and the M5 ANE compiler rejects FP8, so on an M5 the runtime first derives that chip's build: each
chunk with only its functions, renamed to the canonical names, written once to $ANEMLL_FORGE_STATE/builds/ (the
download stays read-only; about 3 s per chunk and one more copy of the chunks on disk), the head linked, the manifest
written last. Other chips use the build as it is. Needs coreai-core (PyPI) on the chip that derives."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FP8_FORMS = ("sm8", "pvf8", "pvf5")


def soc_sets(man: dict) -> list[str]:
    """The chip names with their own function set in this build (e.g. ["m5"])."""
    return sorted({k for c in man.get("chunks", []) for k in (c.get("entries_by_soc") or {})})


def derived_manifest(man: dict, soc: str, source: Path | None = None) -> dict:
    """The manifest of soc's build: canonical entries, that set's attention forms in the numerics, no SoC map."""
    m = copy.deepcopy(man)
    for c in m["chunks"]:
        sets = c.pop("entries_by_soc", None) or {}
        if soc not in sets:
            raise ValueError(f"{c.get('file')}: no {soc} function set")
        c["entries"] = list(sets[soc])
        c.pop("compiled", None)
        n = c.setdefault("numerics", {})
        forms = n.pop(f"ATT_INT8MM_{soc.upper()}", None)
        if forms is not None:
            n["ATT_INT8MM"] = forms
            if not any(f in forms.split(",") for f in FP8_FORMS):
                n.pop("ATT_PF8_UNIT", None)
        n["SOC_FUNCTIONS"] = soc
    m["derived"] = {"soc": soc, **({"from": str(source)} if source else {})}
    return m


def prepare(root: Path, soc: str, log=print, state: Path | None = None) -> Path:
    """root itself, or soc's derived build (created on first use) when the build has a set for soc."""
    root = Path(root)
    raw = (root / "manifest.json").read_bytes()
    man = json.loads(raw)
    if soc not in soc_sets(man):
        return root
    state = Path(state or os.environ.get("ANEMLL_FORGE_STATE") or Path.home() / ".anemll-forge")
    out = state / "builds" / f"{root.name}-{soc}-{hashlib.sha256(raw).hexdigest()[:12]}"
    if (out / "manifest.json").exists():
        log(f"[{soc}] using the {soc} build derived from {root}: {out}")
        return out
    try:
        sys.path.insert(0, str(ROOT / "coreai"))
        import strip_functions
        strip_functions.require()
    except ImportError as e:
        raise RuntimeError(f"{root} has {soc} functions that are extracted once on this chip, which needs the Core AI "
                           f"authoring package: python -m pip install coreai-core ({e})") from None
    need = sum(f.stat().st_size for c in man["chunks"] for f in (root / c["file"]).rglob("*") if f.is_file())
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if free < need * 1.05:
        raise RuntimeError(f"deriving the {soc} build needs {need / 1e9:.1f} GB in {out}; {free / 1e9:.1f} GB free")
    log(f"[{soc}] deriving the {soc} build (its functions only, same weights) into {out}: "
        f"{len(man['chunks'])} chunks, {need / 1e9:.1f} GB")
    t0 = time.time()
    for c in man["chunks"]:
        dst = out / c["file"]
        if dst.exists():
            continue
        tmp = dst.with_name(dst.stem + ".partial.aimodel")
        shutil.rmtree(tmp, ignore_errors=True)
        sets = c["entries_by_soc"][soc]  # canonical -> physical
        strip_functions.strip(root / c["file"], tmp, {phys: canon for canon, phys in sets.items()})
        tmp.rename(dst)
    head = man["head"]["file"]
    if not (out / head).exists():
        (out / head).symlink_to((root / head).resolve(), target_is_directory=True)
    (out / "manifest.json").write_text(json.dumps(derived_manifest(man, soc, root), indent=1))
    log(f"[{soc}] {soc} build ready in {time.time() - t0:.0f}s")
    return out
