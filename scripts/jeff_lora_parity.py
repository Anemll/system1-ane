#!/usr/bin/env python3
"""Compare a merged Core AI build with the PyTorch probabilities from jeff-train-lora.

Run with the Core AI interpreter. ``--model`` is the base checkpoint (embeddings and
temperature). ``--build`` is the adapter's ``coreai/`` directory, whose head already
contains the trained readout.

    $COREAI_PYTHON scripts/jeff_lora_parity.py \\
        --model ~/Models/jeff/jeff-base-v1.3 \\
        --build ~/Models/jeff-coreai/adapters/snake/coreai \\
        --rows ~/Models/jeff-snake/parity_rows.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai_runtime import JeffCoreAI  # noqa: E402


def kl(left: list[float], right: list[float]) -> float:
    total = 0.0
    for p, q in zip(left, right):
        if p > 0:
            total += p * math.log(p / max(q, 1e-12))
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    a = p.parse_args(argv)
    rows = json.loads(a.rows.expanduser().read_text())
    runtime = JeffCoreAI(a.build.expanduser().resolve(), a.model.expanduser().resolve())
    scores = []
    matches = 0
    for row in rows:
        decided = runtime.decide(row["ids"], int(row["n"]))
        ane = [float(value) for value in decided["probabilities"].values()]
        ref = [float(value) for value in row["probabilities"]]
        scores.append(kl(ref, ane))
        if ane.index(max(ane)) == ref.index(max(ref)):
            matches += 1
        print(f"tokens {len(row['ids'])}  kl {scores[-1]:.3e}  "
              f"pt {ref.index(max(ref))} ane {ane.index(max(ane))}", flush=True)
    print(json.dumps({
        "rows": len(rows),
        "kl_mean": sum(scores) / max(1, len(scores)),
        "argmax_matches": matches,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
