"""Guided first load of Core AI packages.

Origin: Anemll/anemll-forge. Used by ``forge.py compile`` and crash recovery.

Guided first load of Core AI packages: the first start of a build on a macOS build compiles every package for the
ANE once (then it is cached and later starts take seconds). This module tells the user what is being compiled, how
far along it is, how long is left, that stopping is safe, and which build options compile faster.

Standard library only; the package loads are passed in as callables. A cold load runs in a worker thread so the main
thread can print a heartbeat and react to Ctrl-C immediately (finished packages stay cached; the next start resumes).
Estimates come from cold compiles measured on the M6 (anemll-forge research notes, not copied) and are
re-scaled by the compile times measured during the run."""
from __future__ import annotations

import functools
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

CACHE = Path.home() / "Library/Caches/coreai-cache"
TAG = "[ANE compile]"
HEARTBEAT_S = float(os.environ.get("ANE_COMPILE_HEARTBEAT_S", "30"))
HEAD_S, DRAFTER_S = 30.0, 60.0          # rough cold compile of the head / DFlash2 drafter packages
STATE_ENV = "ANE_COMPILE_STATE"         # file naming the package being loaded, for a parent to recover from a crash
CRASH_SIGNALS = (6, 4, 5, 10, 11)       # SIGABRT (MPSGraph assertion), SIGILL, SIGTRAP, SIGBUS, SIGSEGV


def os_build() -> str:
    return subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip() or "unknown"


@functools.lru_cache(maxsize=1)
def process_key() -> str:
    """Core AI's per-process cache folder: the main bundle identifier when the process has one (a framework Python,
    such as Homebrew's, runs inside Python.app: org.python.python), else the executable name with '.' and '_' as '-'
    (a uv/standalone python3.12 -> python3-12). The executable name alone misses the framework case."""
    try:
        import ctypes
        import ctypes.util
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        cf.CFBundleGetMainBundle.restype = ctypes.c_void_p
        cf.CFBundleGetIdentifier.argtypes, cf.CFBundleGetIdentifier.restype = [ctypes.c_void_p], ctypes.c_void_p
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        ident, buf = cf.CFBundleGetIdentifier(cf.CFBundleGetMainBundle()), ctypes.create_string_buffer(1024)
        if ident and cf.CFStringGetCString(ident, buf, len(buf), 0x08000100):   # kCFStringEncodingUTF8
            return buf.value.decode()
    except (OSError, AttributeError, TypeError):
        pass
    return re.sub(r"[._]", "-", Path(sys.executable).name)


def cache_dir(path: Path) -> Path | None:
    """This process's Core AI cache folder for a package: <os build>/<process_key()>/<main.hash>."""
    h = Path(path) / "main.hash"
    if not h.is_file():
        return None
    return CACHE / os_build() / process_key() / h.read_bytes().hex()


def purge(path: Path) -> int:
    """Remove this process's cached specializations of a package (the next load recompiles it); returns how many."""
    d = cache_dir(path)
    entries = [e for e in d.iterdir() if e.is_dir()] if d is not None and d.is_dir() else []
    for e in entries:
        shutil.rmtree(e, ignore_errors=True)
    return len(entries)


def mark_loading(path: Path | None):
    """Name the package being loaded in $ANE_COMPILE_STATE (cleared after the load), so a parent process can purge
    its cache and retry when an uncatchable abort kills the load (forge.py compile / serve)."""
    f = os.environ.get(STATE_ENV)
    if f:
        Path(f).write_text(str(path) if path is not None else "")


def is_cached(path: Path, mode: int | None = None) -> bool:
    """A compiled .aimodelc, or a cached specialization of the .aimodel (in ANE bonded compile `mode` when given)."""
    path = Path(path)
    if path.suffix == ".aimodelc":
        return True
    d = cache_dir(path)
    if d is None or not d.is_dir():
        return False
    for mf in d.rglob("manifest.plist"):
        if ".mpsgraphpackage" not in str(mf):
            continue
        if mode is None or re.search(rb"aneBondedCompileMode\W+%d\b" % mode, mf.read_bytes()):
            return True
    return False


def tiles(ctxs, block: int) -> int:
    return sum(math.ceil(int(c) / block) for c in ctxs)


def build_settings(man: dict) -> dict:
    """Graph options and entry counts that drive compile time, from a target manifest."""
    nums = (man.get("chunks") or [{}])[0].get("numerics") or {}
    vb = int(nums.get("ATT_BLOCK", 16384))
    pb = int(nums.get("ATT_BLOCK_PREFILL", vb))
    formats = 2 if man.get("kv_cache", {}).get("format") == "selectable" else 1
    return {"gdn_fast": bool(nums.get("GDN_FAST", False)), "att_block": vb, "att_block_prefill": pb,
            "formats": formats, "ctxs": [int(c) for c in man.get("kv_len", {}).values()] or man.get("ctxs", []),
            "pctxs": [int(c) for c in man.get("pkv_len", {}).values()] or man.get("pctxs", [])}


def estimate_chunk_s(s: dict) -> float:
    """Cold compile seconds of one 4-layer chunk on the M6. Fit to measured chunks: release 82.9 s (both formats) /
    45.9 s (V8), 2048-wide tiles 407.6 s, 2048 / 4096 (prefill) 135.9 s (both) / 90.3 s (V8). Within about 25%."""
    tv, tp = tiles(s["ctxs"], s["att_block"]), tiles(s["pctxs"], s["att_block_prefill"])
    return (40 + 0.25 * tv + 0.024 * tp * tp) * (1.65 if s["formats"] > 1 else 1.0)


def hints(s: dict, total_s: float, build: Path | None = None) -> list[str]:
    out = []
    if s["att_block_prefill"] < 4096 and s["pctxs"]:
        out.append(f"faster: convert with ATT_BLOCK_PREFILL=4096 (prefill speed unchanged; at ATT_BLOCK={s['att_block']}"
                   " about 3x less compile)")
    if s["formats"] > 1:
        out.append("faster: convert with --kv-cache-dtype v8 (one KV format; about 1.5x less compile, no FP16-V option)")
    if total_s > 15 * 60:
        out.append("quick test: convert with fewer contexts, e.g. --ctx 8192,16384 --pctx 8192,16384")
    where = f" --build {build}" if build else " --build <build>"
    out.append(f"compile ahead without serving: python forge.py compile{where} (use the same Python as the server: "
               f"the cache is per macOS build and Python, here '{process_key()}')")
    return out


def fmt(sec: float) -> str:
    sec = max(0, int(round(sec)))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


class CompileGuide:
    """items: (label, package path, estimated cold seconds). Call announce() once, then load(label, fn) per item."""

    def __init__(self, items, log=print, settings: str = "", hint_lines=(), mode: int | None = None):
        self.log, self.settings, self.hint_lines = log, settings, list(hint_lines)
        self.items = [(lbl, Path(p), float(e)) for lbl, p, e in items]
        self.cold = {lbl for lbl, p, _ in self.items if not is_cached(p, mode)}
        self.est = {lbl: e for lbl, _, e in self.items}
        self.paths = {lbl: p for lbl, p, _ in self.items}
        self.done: dict[str, float] = {}

    def remaining_s(self) -> float:
        left = [l for l in self.cold if l not in self.done]
        if not left:
            return 0.0
        return self._scale() * sum(self.est[l] for l in left)

    def announce(self):
        n, k = len(self.items), len(self.cold)
        if not k:
            self.log(f"{TAG} all {n} packages already compiled for this Mac (macOS {os_build()}): loading from cache")
            return
        total = self.remaining_s()
        self.log(f"{TAG} {k} of {n} packages are not compiled yet for this Mac (macOS {os_build()}, Python "
                 f"'{process_key()}'): compiling them once now; later starts load from the cache in seconds")
        self.log(f"{TAG} estimated ~{fmt(total)} in total{f' for {self.settings}' if self.settings else ''}; "
                 "progress and time left are printed below")
        self.log(f"{TAG} safe to stop at any time (Ctrl-C): each package is cached as soon as "
                 "it finishes, and the next start resumes with the rest")
        for h in self.hint_lines:
            self.log(f"{TAG} {h}")

    def load(self, label: str, fn):
        """Run fn(); for a cold package, in a worker thread with a heartbeat and an immediate Ctrl-C exit."""
        mark_loading(self.paths.get(label))
        try:
            return self._load(label, fn)
        finally:
            mark_loading(None)

    def _load(self, label: str, fn):
        if label not in self.cold:
            return fn()
        i = len([l for l in self.done if l in self.cold]) + 1
        k = len(self.cold)
        self.log(f"{TAG} compiling {label} ({i}/{k}): expected ~{fmt(self.est[label] * self._scale())}, "
                 f"~{fmt(self.remaining_s())} left in total")
        box: dict = {}

        def work():
            try:
                box["value"] = fn()
            except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
                box["error"] = e
        t0 = time.time()
        th = threading.Thread(target=work, name=f"ane-compile-{label}", daemon=True)
        th.start()
        beat = t0 + HEARTBEAT_S
        try:
            while th.is_alive():
                th.join(timeout=1.0)
                now = time.time()
                if th.is_alive() and now >= beat:
                    beat = now + HEARTBEAT_S
                    exp = self.est[label] * self._scale()
                    left = max(0.0, exp - (now - t0)) + self.remaining_s() - self.est[label] * self._scale()
                    self.log(f"{TAG} {label} ({i}/{k}): {fmt(now - t0)} so far (expected ~{fmt(exp)}); "
                             f"~{fmt(max(left, 0))} left in total")
        except KeyboardInterrupt:
            cached = len([l for l in self.done if l in self.cold])
            self.log(f"{TAG} interrupted while compiling {label}: {cached} of {k} packages compiled this run are cached;"
                     " run the same command again to resume")
            os._exit(130)
        if "error" in box:
            raise box["error"]
        self.done[label] = time.time() - t0
        verb = "loaded" if self._cache_hit(label) else "compiled"
        self.log(f"{TAG} {verb} {label} in {fmt(self.done[label])} ({i}/{k} done); "
                 f"~{fmt(self.remaining_s())} left in total")
        return box.get("value")

    def _cache_hit(self, label: str) -> bool:
        """A 'cold' load far under its estimate came from a cache the check missed: no compile time to learn from."""
        return self.done[label] < 0.1 * self.est[label]

    def _scale(self) -> float:
        measured = [l for l in self.done if l in self.cold and not self._cache_hit(l)]
        return (sum(self.done[l] for l in measured) / sum(self.est[l] for l in measured)) if measured else 1.0


def target_guide(man: dict, root: Path, log=print, extra=(), mode: int | None = None) -> CompileGuide:
    """Guide for a target build: its chunks and head (plus extra (label, path, est) items, e.g. the drafter)."""
    s = build_settings(man)
    per = estimate_chunk_s(s)
    items = [(c["file"], Path(root) / c["file"], per) for c in man.get("chunks", [])]
    if man.get("head"):
        items.append((man["head"]["file"], Path(root) / man["head"]["file"], HEAD_S))
    items += list(extra)
    settings = (f"GDN_FAST={int(s['gdn_fast'])} ATT_BLOCK={s['att_block']} ATT_BLOCK_PREFILL={s['att_block_prefill']}, "
                f"{s['formats']} KV format{'s' if s['formats'] > 1 else ''}, {len(s['ctxs'])} contexts")
    g = CompileGuide(items, log=log, settings=settings, mode=mode)
    total = sum(e for lbl, _, e in g.items if lbl in g.cold)
    g.hint_lines = hints(s, total, Path(root)) if g.cold else []
    return g
