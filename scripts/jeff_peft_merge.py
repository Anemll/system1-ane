#!/usr/bin/env python3
"""Merge a published Jeff PEFT adapter into a checkpoint ``jeff-convert`` can compile.

The math matches ``jeff.lora.merge_adapter`` in jeff-src: ``W += (alpha / rank) B A``.
Upstream ``jeff-train`` itself is not used here. It calls ``torch.cuda`` before the first
step, so this Mac trains the Snake sample with ``jeff-train-lora`` and only merges the
published adapters (triage, tools, guard, spam) through this script.

    python scripts/jeff_peft_merge.py \\
        --base ~/Models/jeff/jeff-base-v1.3 \\
        --adapter ~/Models/jeff/adapters/jeff-adapter-triage \\
        --output ~/Models/jeff-published/triage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_lora import merge_peft_adapter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="jeff-base checkpoint directory")
    p.add_argument("--adapter", type=Path, required=True, help="PEFT adapter directory (adapter_model.safetensors)")
    p.add_argument("--output", type=Path, required=True, help="new directory for the merged checkpoint")
    a = p.parse_args(argv)
    info = merge_peft_adapter(a.base.expanduser().resolve(), a.adapter.expanduser().resolve(),
                              a.output.expanduser().resolve())
    print(json.dumps({"output": str(a.output.expanduser().resolve()), **info}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
