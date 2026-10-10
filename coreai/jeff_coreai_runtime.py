"""Prefill-only Core AI runner for a Jeff decision package.

Used by jeff-smoke --build on macOS after jeff-convert + forge.py compile. A prompt longer than the prefill entry
(TP rows) runs as consecutive TP-row calls: each chunk's DeltaNet conv / recurrent state is carried, every call's
k / v rows go into the KV caches at its position and the history mask opens up to it, as in CoreAIQwen.prefill_block.
The readout head then runs once on the last prompt row. ``prefill(..., prefix=capture_state())``
resumes from a cached live-last prefix instead of resetting; the server forwards that
when a prefix cache is installed.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from jeff_coreai import (JeffCheckpoint, apply_manifest_temperature, load_decision_config, load_text_config,
                        rms_last, softmax)
from jeff_prefix import resume_at
from system1_env import server_env

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ane_compile_mode as SOC  # noqa: E402
from qwen38_coreai_model import _spec, buffer, pick_package, writable  # noqa: E402
from qwen38_kv_cache import put_rows  # noqa: E402

try:
    from coreai.runtime import AIModel, NDArray
except ImportError:  # Linux / host venv without the Core AI SDK
    AIModel = None

    class NDArray:  # wrap-only stub so outputs() tests run without the SDK
        def __init__(self, tensor=None):
            self._tensor = tensor

        @classmethod
        def _wrap(cls, value):
            return cls(value)

# How prefill splits the fresh rows over the compiled widths (SYSTEM1_ANE_PREFILL, or prefill(mode=...)):
#   chain     consecutive TP-row calls, the path of a single-width build
#   fit       one call in the narrowest entry that holds the rows, else chain
#   cheapest  the chain of compiled widths with the lowest summed per-call cost (see call_ms)
PREFILL_MODES = ("chain", "fit", "cheapest")
# Relative per-call cost by width, measured on chunk L00-03 of a 256..2048 build (ms per call of one chunk).
# Used until calibrate() measures the loaded build; widths in between are interpolated per row.
DEFAULT_CALL_MS = {256: 11.4, 512: 25.4, 1024: 53.8, 1536: 78.3, 2048: 105.3}


def prefill_mode(raw: str | None) -> str:
    mode = (raw or "cheapest").strip().lower()
    if mode not in PREFILL_MODES:
        raise ValueError(f"SYSTEM1_ANE_PREFILL={raw!r}; use one of {', '.join(PREFILL_MODES)}")
    return mode


def default_call_ms(widths) -> dict[int, float]:
    """Per-call cost of each width from DEFAULT_CALL_MS, interpolating the per-row cost between measured widths."""
    pts = sorted((w, ms / w) for w, ms in DEFAULT_CALL_MS.items())
    out = {}
    for w in widths:
        if w <= pts[0][0]:
            per_row = pts[0][1]
        elif w >= pts[-1][0]:
            per_row = pts[-1][1]
        else:
            (a, fa), (b, fb) = next((x, y) for x, y in zip(pts, pts[1:]) if x[0] <= w <= y[0])
            per_row = fa + (fb - fa) * (w - a) / (b - a)
        out[w] = per_row * w
    return out


def cheapest_chain(n: int, call_ms: dict[int, float]) -> list[int]:
    """Widths (widest first) whose calls cover n rows at the lowest summed cost; ties go to fewer calls."""
    widths = sorted(call_ms)
    best = [(0.0, 0, ())] * (n + 1)
    for m in range(1, n + 1):
        best[m] = min((call_ms[w] + best[max(0, m - w)][0], 1 + best[max(0, m - w)][1],
                       (w, *best[max(0, m - w)][2])) for w in widths)
    return sorted(best[n][2], reverse=True)


@contextlib.asynccontextmanager
async def outputs(fn, inputs: dict):
    """Run an InferenceFunction and yield its outputs; they are freed when the block exits.

    The binding (coreai.runtime, macOS 27) has no outputs= argument, and its native call keeps a reference to the
    asyncio Future's set_result, so the Future and the result dict it holds are never collected. Every output
    NDArray in that dict pinned one pooled IOSurface for the life of the process (~1 per output per call) until the
    pool failed to allocate. Copy what you need inside the block; the dict is emptied on exit.
    """
    raw = await fn._function(inputs={k: v._tensor for k, v in inputs.items()}, state={})  # noqa: SLF001
    out = {k: NDArray._wrap(v) for k, v in raw.items()}  # noqa: SLF001
    try:
        yield out
    finally:
        out.clear()
        raw.clear()


# "jeff-decision" is the original manifest kind; "decision" is the neutral alias used by published stock builds.
DECISION_KINDS = ("decision", "jeff-decision")


class JeffCoreAI:
    def __init__(self, build: Path, model: Path, ck: JeffCheckpoint | None = None, log=print):
        if AIModel is None:
            raise RuntimeError("coreai.runtime is not installed; run this in the Core AI SDK Python (COREAI_PYTHON)")
        SOC.apply(strict=True, log=log)
        self.build, self.model, self.log = Path(build), Path(model), log
        self.man = json.loads((self.build / "manifest.json").read_text())
        if self.man.get("kind") not in DECISION_KINDS:
            raise ValueError(f"{build} is not a decision package (manifest kind {self.man.get('kind')!r})")
        c = self.cfg = load_text_config(self.model)
        self.decision = apply_manifest_temperature(load_decision_config(self.model), self.man)
        ck = ck or JeffCheckpoint(self.model)
        self.emb, self.norm, self.readout = ck.embed_table(), ck.norm_weight(), np.asarray(ck.readout, np.float32)
        self.eps = float(c["rms_norm_eps"])
        self.TP, self.P = int(self.man["TP"]), int(self.man["pend"])
        self.ctx = int(self.man["pctxs"][0])
        self.L = int(self.man["pkv_len"][str(self.ctx)])
        self.entry = f"p{self.TP}_{self.ctx // 1024}k"
        self.widths = sorted({self.TP, *(int(w) for w in self.man.get("TPS", ()))})
        self.entries = {w: f"p{w}_{self.ctx // 1024}k" for w in self.widths}
        self.mode = prefill_mode(server_env("PREFILL"))
        self.call_ms = default_call_ms(self.widths)
        self._plans: dict[tuple[str, int], list[int]] = {}
        hid, self.nkv, self.hd = int(c["hidden_size"]), int(c["num_key_value_heads"]), int(c["head_dim"])
        nv, dk, dv = (int(c[k]) for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        cdim = 2 * int(c["linear_num_key_heads"]) * dk + nv * dv
        gshapes = {"conv": (self.P + 3, cdim), "rec": (nv, dk, dv), "pend": (nv, 3 * self.P + 1, dv)}
        rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        self.loop = asyncio.new_event_loop()
        t0 = time.time()

        async def load(file, entries):
            target, _ = pick_package(self.build, file, None, log)
            m = await AIModel.load(target, specialization_options=_spec())
            return m, {e: m.load_function(e) for e in entries}

        self.chunks = []
        for ch in self.man["chunks"]:
            t1 = time.time()
            pkg, fns = self.loop.run_until_complete(load(ch["file"], ch["entries"]))
            state = {f"{n}{j}": buffer(gshapes[n]) for j in ch["gdn_j"] for n in gshapes}
            kv = {f"{s}{j}": buffer((self.nkv, self.L, self.hd)) for j in ch["att_j"] for s in ("k", "v")}
            self.chunks.append({**ch, "pkg": pkg, "fns": fns, "state": state, "kv": kv})
            log(f"loaded {ch['file']} in {time.time() - t1:.1f}s")
        head = self.man["head"]
        self.head_pkg, hf = self.loop.run_until_complete(load(head["file"], [head.get("entry", "h1")]))
        self.head_fn = hf[head.get("entry", "h1")]
        self.load_s = time.time() - t0
        self.pins = {w: {n: buffer(s) for n, s in (
            ("cos", (w, rot)), ("sin", (w, rot)), ("conv_sel", (3, self.P + 3)), ("commit", (1, self.P, 1)),
            ("commit_last", (1, self.P, 1)), ("conv_sel_out", (3, w + 3)), ("valid", (1, w, 1)),
            ("x", (1, hid, 1, w)), ("xb", (1, hid, 1, w)))} for w in self.widths}
        self.hx = buffer((1, hid, 1, 1))
        self.mask = buffer((1, self.L))
        self.pos = 0
        self._token_ids: list[int] = []
        self._last_hidden: np.ndarray | None = None

    def reset(self):
        for ch in self.chunks:
            for _, w in ch["state"].values():
                w[:] = 0
        self.pos = 0
        self._token_ids = []
        self._last_hidden = None

    def capture_state(self) -> dict:
        """Copy position, token ids, last hidden row, GDN/conv state and KV.

        A live-last prefix cache keeps this dict and passes it back as ``prefix``
        to prefill or decide. Copies are independent of the live buffers. The cache
        policy (what prefix to keep, when to reuse it) is not implemented here.
        """
        if self._last_hidden is None or self.pos <= 0:
            raise RuntimeError("capture_state() needs a completed prefill")
        return {
            "pos": self.pos,
            "token_ids": list(self._token_ids),
            "hidden": self._last_hidden.copy(),
            "chunks": [
                {
                    "state": {k: v[1].copy() for k, v in ch["state"].items()},
                    "kv": {k: v[1].copy() for k, v in ch["kv"].items()},
                }
                for ch in self.chunks
            ],
        }

    def _restore(self, prefix: dict) -> None:
        chunks = prefix.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != len(self.chunks):
            raise ValueError(f"prefix has {len(chunks) if isinstance(chunks, list) else 'no'} chunks; "
                             f"this build has {len(self.chunks)}")
        for ch, saved in zip(self.chunks, chunks):
            state, kv = saved.get("state"), saved.get("kv")
            if not isinstance(state, dict) or not isinstance(kv, dict):
                raise ValueError("each prefix chunk needs state and kv dicts")
            if set(state) != set(ch["state"]) or set(kv) != set(ch["kv"]):
                raise ValueError("prefix state names do not match this build")
            for k, arr in state.items():
                ch["state"][k][1][:] = arr
            for k, arr in kv.items():
                ch["kv"][k][1][:] = arr
        self.pos = int(prefix["pos"])
        hidden = prefix.get("hidden")
        self._last_hidden = None if hidden is None else np.asarray(hidden).reshape(self.hx[1].shape)

    def plan(self, n: int, chained: bool = False, mode: str | None = None) -> list[int]:
        """Entry widths of the calls that prefill n rows, in call order (see PREFILL_MODES).
        chained=True is mode="chain"."""
        mode = "chain" if chained else prefill_mode(mode or self.mode)
        key = (mode, n)
        if key not in self._plans:
            fit = [w for w in self.widths if w >= n] if mode == "fit" else []
            if mode == "cheapest" and len(self.widths) > 1:
                self._plans[key] = cheapest_chain(n, self.call_ms)
            else:
                self._plans[key] = [fit[0]] if fit else [self.TP] * -(-n // self.TP)
        return list(self._plans[key])

    def calibrate(self, reps: int = 5) -> dict[int, float]:
        """Measure the ANE time of one full-width backbone call per compiled width (median of reps after a warmup)
        and plan cheapest chains with it. Leaves the runtime reset."""
        if len(self.widths) > 1:
            ms = {}
            for w in self.widths:
                times = []
                for i in range(reps + 1):
                    self.reset()
                    t0 = time.perf_counter()
                    self._block([0] * w, None, w)
                    if i:
                        times.append(1e3 * (time.perf_counter() - t0))
                ms[w] = statistics.median(times)
            self.call_ms = ms
            self._plans.clear()
        self.reset()
        return dict(self.call_ms)

    def _block(self, ids, keep=None, width: int | None = None) -> np.ndarray:
        """Up to width (default TP) prompt tokens at self.pos, all committed. Returns the last row's hidden state
        (1, hid, 1, 1). keep: a list per chunk that receives this call's output rows (n, hid) (diagnostics)."""
        TP = self.TP if width is None else width
        d, n, p0, entry = self.pins[TP], len(ids), self.pos, self.entries[TP]
        if not 0 < n <= TP or p0 + n > self.L:
            raise ValueError(f"{p0 + n} positions exceed the {self.L}-row KV cache of {entry}")
        d["x"][1][:] = 0
        d["x"][1][0, :, 0, :n] = self.emb[np.asarray(ids)].T
        pos = np.minimum(np.arange(p0, p0 + TP), p0 + n - 1)
        ang = np.concatenate([np.outer(pos, self.inv)] * 2, axis=1)
        d["cos"][1][:], d["sin"][1][:] = np.cos(ang), np.sin(ang)
        d["conv_sel"][1][:] = 0
        d["conv_sel"][1][np.arange(3), np.arange(3)] = 1   # nothing pending: the conv rows of the last call come first
        d["commit"][1][:] = 0
        d["commit_last"][1][:] = 0
        d["conv_sel_out"][1][:] = 0
        d["conv_sel_out"][1][np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0
        d["valid"][1][0, :n, 0] = 1
        self.mask[1][:] = -1e4
        self.mask[1][0, :p0] = 0
        shared = {k: v[0] for k, v in d.items() if k not in ("x", "xb")}
        shared["mask"] = self.mask[0]

        async def run():
            x = d["x"][0]
            for ci, ch in enumerate(self.chunks):
                ins = {**shared, "x": x, **{k: v[0] for k, v in ch["state"].items()},
                       **{k: v[0] for k, v in ch["kv"].items()}}
                async with outputs(ch["fns"][entry], ins) as out:
                    for k, (_, w) in ch["state"].items():
                        w[:] = writable(out[f"{k}_out"])
                    for k, (_, w) in ch["kv"].items():
                        put_rows(w, out[f"{k}_new"].numpy()[:, :n], p0, n)
                    d["xb"][1][:] = writable(out["y"])
                x = d["xb"][0]
                if keep is not None:
                    keep[ci].append(d["xb"][1][0, :, 0, :n].T.copy())
        self.loop.run_until_complete(run())
        self.pos = p0 + n
        return d["xb"][1][:, :, :, n - 1:n].copy()

    async def _head(self) -> np.ndarray:
        async with outputs(self.head_fn, {"x": self.hx[0]}) as out:
            return np.array(out["logits"].numpy(), np.float32).reshape(-1)

    def prefill(self, token_ids: list[int], keep_chunks: bool = False, prefix: dict | None = None,
                chained: bool = False, mode: str | None = None) -> dict:
        """Readout logits (ANE head), the last hidden state and timings.

        The fresh rows run in the calls plan() picks for mode (default self.mode, from SYSTEM1_ANE_PREFILL); chained=True
        forces TP-row calls, the path of a single-width build.

        prefix=None resets and prefills every token (one independent decision).
        prefix= a capture_state() dict resumes GDN/KV at prefix["pos"] and prefills
        only the suffix. A prefix that already covers the prompt runs the readout
        on the cached hidden row and does not call the backbone. keep_chunks records
        fresh rows only, and needs at least one.
        """
        ids = [int(t) for t in token_ids]
        if not ids:
            raise ValueError("token_ids must be non-empty")
        start = resume_at(prefix, ids)
        if prefix is None:
            self.reset()
            last = None
        else:
            self._restore(prefix)
            last = self._last_hidden
        self._token_ids = ids
        keep = [[] for _ in self.chunks] if keep_chunks else None
        t0 = time.perf_counter()
        calls, widths = [], self.plan(len(ids) - start, chained, mode) if len(ids) > start else []
        i = start
        for w in widths:
            t1 = time.perf_counter()
            last = self._block(ids[i:i + w], keep, w)
            calls.append(1e3 * (time.perf_counter() - t1))
            i += w
        if last is None:
            raise ValueError("prefix covers the prompt but has no hidden state")
        self._last_hidden = last
        t1 = time.perf_counter()
        self.hx[1][:] = last
        logits = self.loop.run_until_complete(self._head())
        head_ms = 1e3 * (time.perf_counter() - t1)
        r = {"logits": logits, "hidden": last.reshape(-1).astype(np.float32), "calls_ms": calls, "call_widths": widths,
             "head_ms": head_ms, "total_ms": 1e3 * (time.perf_counter() - t0),
             "prefix_tokens": start}
        if keep is not None:
            if any(not rows for rows in keep):
                raise ValueError("keep_chunks needs at least one fresh prefill row")
            r["chunks"] = [np.concatenate(rows, 0) for rows in keep]
        return r

    def decide(self, token_ids: list[int], n_options: int, temperature: float | None = None,
               prefix: dict | None = None, mode: str | None = None) -> dict:
        r = self.prefill(token_ids, prefix=prefix, mode=mode)
        temp = float(self.decision["temperature"] if temperature is None else temperature)
        probs = softmax(r["logits"][:n_options] / temp)
        # the same readout applied on the host to the ANE hidden state: isolates head error from backbone error
        host_logits = self.readout[:n_options] @ rms_last(r["hidden"], self.norm, self.eps)
        codes = self.decision["codes"][:n_options]
        best = int(np.argmax(probs))
        return {
            "probabilities": {codes[i]: float(probs[i]) for i in range(n_options)},
            "answer": codes[best],
            "confidence": float(probs[best]),
            "host_head_probabilities": softmax(host_logits / temp).tolist(),
            "temperature": temp,
            "tokens": len(token_ids),
            "calls": len(r["calls_ms"]),
            "calls_ms": [round(t, 2) for t in r["calls_ms"]],
            "call_widths": list(r["call_widths"]),
            "head_ms": round(r["head_ms"], 2),
            "prefill_ms": round(r["total_ms"], 2),
            "prefix_tokens": int(r["prefix_tokens"]),
            "backend": f"coreai-ane {'+'.join(dict.fromkeys(self.entries[w] for w in r['call_widths'])) or self.entry}",
        }
