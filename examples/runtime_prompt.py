#!/usr/bin/env python3
"""Run one Jeff decision through the Core AI runtime (Apple Silicon + COREAI_PYTHON).

The Core AI venv has no transformers. Token ids come from ``scripts/jeff_tokenize.py``
in a host-Python subprocess, the same path ``system1-ane-serve`` uses (``TOKENIZER_PYTHON``).

    export TOKENIZER_PYTHON=/path/to/host/.venv/bin/python   # has transformers
    $COREAI_PYTHON examples/runtime_prompt.py \
        --model ~/Models/jeff/jeff-base-v1.3 \
        --build ~/Models/jeff-coreai/coreai

If ``TOKENIZER_PYTHON`` is unset, this process is used when it has transformers,
else this repo's ``.venv/bin/python``.

On Linux this imports cleanly and exits with a message that Core AI is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "scripts"))

from jeff_coreai_runtime import AIModel, JeffCoreAI  # noqa: E402

TOKENIZE_SCRIPT = ROOT / "scripts" / "jeff_tokenize.py"


def tokenizer_python() -> str:
    """Host interpreter with transformers (TOKENIZER_PYTHON, this process, or .venv)."""
    env = os.environ.get("TOKENIZER_PYTHON")
    if env:
        return env
    if importlib.util.find_spec("transformers") is not None:
        return sys.executable
    venv = ROOT / ".venv" / "bin" / "python"
    if os.access(venv, os.X_OK):
        return str(venv)
    raise RuntimeError(
        "Set TOKENIZER_PYTHON to a Python with transformers (the host venv that runs forge.py)."
    )


def tokenize_ids(model: Path, row: dict, *, python: str | None = None,
                 script: Path | None = None) -> list[int]:
    """Jeff chat-template token ids via jeff_tokenize.py (no transformers in this process)."""
    python = python or tokenizer_python()
    script = script or TOKENIZE_SCRIPT
    proc = subprocess.Popen(
        [python, str(script), str(model)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None and proc.stdin is not None
    try:
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"Tokenizer worker ({python}) exited before it was ready. "
                "TOKENIZER_PYTHON must be an interpreter with transformers."
                + (f" {err.strip()}" if err.strip() else "")
            )
        ready = json.loads(line)
        if not ready.get("ready"):
            raise RuntimeError(f"Tokenizer worker failed: {ready}")
        proc.stdin.write(json.dumps({"row": row}) + "\n")
        proc.stdin.flush()
        reply = proc.stdout.readline()
        if not reply:
            raise RuntimeError("Tokenizer worker stopped.")
        msg = json.loads(reply)
        if "error" in msg:
            raise RuntimeError(str(msg["error"]))
        return [int(token) for token in msg["ids"]]
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait(timeout=5)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--state", default="The disk on db-02 is 97 percent full and still growing.")
    p.add_argument("--options", default="page,wait,ignore")
    p.add_argument("--instructions", default="Choose the best next action.")
    a = p.parse_args(argv)
    if AIModel is None:
        print("coreai.runtime is not installed; run this with COREAI_PYTHON on macOS.", file=sys.stderr)
        return 2
    keys = [k.strip() for k in a.options.split(",") if k.strip()]
    row = {
        "state": a.state,
        "question": {
            "type": "choice",
            "instructions": a.instructions,
            "criteria": {k: None for k in keys},
        },
    }
    ids = tokenize_ids(a.model.expanduser().resolve(), row)
    rt = JeffCoreAI(a.build.expanduser().resolve(), a.model.expanduser().resolve())
    out = rt.decide(ids, len(keys))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
