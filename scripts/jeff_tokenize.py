#!/usr/bin/env python3
"""Tokenize Jeff decision rows for system1-ane-serve.

Run with the forge interpreter (the one that has transformers). The Core AI
interpreter that loads the ANE packages does not. Protocol, one JSON object
per line: the first stdout line is {"ready": true}. Each later stdin line is
{"row": {state, question}} or {"cmd": "stop"}. Each reply is {"ids": [...]}
or {"error": "..."}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import load_decision_config, prompt_ids  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> int:
    model = Path(sys.argv[1])
    decision = load_decision_config(model)
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as error:
            sys.stdout.write(json.dumps({"error": f"tokenizer request is not JSON: {error}"}) + "\n")
            sys.stdout.flush()
            continue
        if msg.get("cmd") == "stop":
            return 0
        if "clef" in msg:  # Unsloth Clef-head prompt: segment-wise token ids plus the spans the head reads
            try:
                from clef_head import encode_record
                c = msg["clef"]
                sys.stdout.write(json.dumps({"clef": encode_record(tokenizer, c["state"], c["instructions"], c["options"])}) + "\n")
            except (KeyError, TypeError, ValueError) as error:
                sys.stdout.write(json.dumps({"error": str(error)}) + "\n")
            sys.stdout.flush()
            continue
        try:
            ids = prompt_ids(model, msg["row"], decision, tokenizer)
        except (KeyError, TypeError, ValueError) as error:
            sys.stdout.write(json.dumps({"error": str(error)}) + "\n")
        else:
            sys.stdout.write(json.dumps({"ids": ids}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
