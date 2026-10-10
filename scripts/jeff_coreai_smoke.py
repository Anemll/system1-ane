#!/usr/bin/env python3
"""Prefill + readout smoke and parity for a local Jeff checkpoint.

    python forge.py jeff-smoke --model ~/Models/jeff/jeff-base-v1.3 [--build ~/Models/jeff-coreai/coreai]
    python scripts/jeff_coreai_smoke.py --model ... --cases cases.json --build ... [--host] [--bench 5] [--out r.json]

Input, one of: --cases (scripts/jeff_reference.py output: token ids + PyTorch FP32 probabilities), --row (a JSON
decision row {"state": ..., "question": ...}), --state/--options/--instructions (a choice row), or --ids. Rows are
rendered with Jeff's exact prompt (needs transformers); --cases and --ids do not need it.

--host runs the token-at-a-time hybrid DecodeLayer reference (FP32, slow for long prompts). --build runs the
prefill-only Core AI package on the ANE (Core AI SDK Python). With reference probabilities in the case, every backend
reports max |dp|, KL(ref || backend) and argmax agreement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import (JEFF_DEFAULT, JeffCheckpoint, host_decision, prompt_ids,  # noqa: E402
                         question_options, rms_last)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=JEFF_DEFAULT)
    p.add_argument("--build", type=Path, help="coreai/ directory from jeff-convert (compiled with forge.py compile)")
    p.add_argument("--cases", type=Path, help="scripts/jeff_reference.py output")
    p.add_argument("--only", help="comma-separated case names from --cases")
    p.add_argument("--row", type=Path, help="JSON decision row")
    p.add_argument("--state", default="The disk on db-02 is 97 percent full and still growing.")
    p.add_argument("--options", default="page,wait,ignore")
    p.add_argument("--instructions", default="Choose the best next action.")
    p.add_argument("--ids", help="comma-separated prompt token ids (with --n-options)")
    p.add_argument("--n-options", type=int, default=3)
    p.add_argument("--host", action="store_true", help="also run the host DecodeLayer reference")
    p.add_argument("--bench", type=int, default=0, help="time this many extra Core AI prefills per case")
    p.add_argument("--out", type=Path, help="write the results JSON here")
    p.add_argument("--dump", type=Path, help="write every chunk's Core AI output rows to <dir>/<case>.npz "
                                             "(jeff_reference.py --chunk-dump compares them per layer)")
    return p


def load_cases(a, model: Path, ck: JeffCheckpoint) -> list[dict]:
    if a.cases:
        cases = json.loads(a.cases.read_text())["cases"]
        if a.only:
            keep = set(a.only.split(","))
            cases = [c for c in cases if c["name"] in keep]
        return cases
    if a.ids:
        ids = [int(x) for x in a.ids.split(",") if x.strip()]
        return [{"name": "ids", "ids": ids, "tokens": len(ids), "n_options": a.n_options}]
    if a.row:
        row = json.loads(a.row.read_text())
    else:
        options = [x.strip() for x in a.options.split(",") if x.strip()]
        row = {"state": {"latest": a.state},
               "question": {"type": "choice", "instructions": a.instructions, "criteria": {o: None for o in options}}}
    ids = prompt_ids(model, row, ck.decision)
    return [{"name": "row", "row": row, "ids": ids, "tokens": len(ids),
             "n_options": len(question_options(row["question"])[0])}]


def compare(ref: list[float] | None, probs: list[float]) -> dict:
    if ref is None:
        return {}
    r, p = np.asarray(ref, np.float64), np.asarray(probs, np.float64)
    return {"max_abs_dp": float(np.max(np.abs(r - p))),
            "kl_ref_backend": float(np.sum(r * (np.log(r + 1e-12) - np.log(p + 1e-12)))),
            "argmax_match": int(np.argmax(r)) == int(np.argmax(p))}


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    model = a.model.expanduser().resolve()
    ck = JeffCheckpoint(model)
    cases = load_cases(a, model, ck)
    runner = None
    if a.build:
        from jeff_coreai_runtime import JeffCoreAI   # Core AI SDK only (macOS)
        runner = JeffCoreAI(a.build.expanduser().resolve(), model, ck=ck)
        print(f"loaded Core AI build in {runner.load_s:.1f}s ({runner.entry}, KV rows {runner.L})", flush=True)
    results = []
    for case in cases:
        n, ids = case["n_options"], case["ids"]
        ref = (case.get("hf_fp32") or {}).get("probabilities")
        res = {"name": case["name"], "tokens": len(ids), "n_options": n}
        if ref is not None:
            res["hf_fp32"] = {"answer": ck.decision["codes"][int(np.argmax(ref))], "confidence": float(max(ref))}
        if a.host:
            t0 = time.time()
            h = host_decision(ck, ids, n)
            probs = list(h["probabilities"].values())
            res["host"] = {"answer": h["answer"], "confidence": h["confidence"], "s": round(time.time() - t0, 1),
                           **compare(ref, probs)}
        if runner is not None:
            c = runner.decide(ids, n)
            probs = list(c["probabilities"].values())
            res["coreai"] = {k: c[k] for k in ("answer", "confidence", "calls", "calls_ms", "head_ms", "prefill_ms",
                                               "backend")}
            res["coreai"].update(compare(ref, probs))
            res["coreai"]["host_head"] = compare(ref, c["host_head_probabilities"])
            if case.get("hf_fp32", {}).get("hidden_normed") is not None or a.dump:
                r = runner.prefill(ids, keep_chunks=bool(a.dump))
                if case.get("hf_fp32", {}).get("hidden_normed") is not None:
                    normed = rms_last(r["hidden"], runner.norm, runner.eps)
                    res["coreai"]["hidden_cos_vs_hf"] = cosine(normed, case["hf_fp32"]["hidden_normed"])
                if a.dump:
                    a.dump.mkdir(parents=True, exist_ok=True)
                    np.savez(a.dump / f"{case['name']}.npz", ids=np.asarray(ids),
                             layers=np.asarray([ch["layers"][1] for ch in runner.chunks]),
                             **{f"chunk{k}": y for k, y in enumerate(r["chunks"])})
            if a.bench:
                runs = [runner.prefill(ids) for _ in range(a.bench)]
                res["coreai"]["bench"] = {
                    "n": a.bench,
                    "prefill_ms_p50": round(float(np.median([r["total_ms"] for r in runs])), 2),
                    "prefill_ms_min": round(float(np.min([r["total_ms"] for r in runs])), 2),
                    "call_ms_p50": round(float(np.median([t for r in runs for t in r["calls_ms"]])), 2),
                    "head_ms_p50": round(float(np.median([r["head_ms"] for r in runs])), 3),
                    "tok_per_s": round(len(ids) / (np.median([r["total_ms"] for r in runs]) / 1e3), 1)}
        results.append(res)
        print(json.dumps(res), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"model": str(model), "build": str(a.build) if a.build else None,
                                     "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
