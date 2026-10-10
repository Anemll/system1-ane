#!/usr/bin/env python3
"""Call a running system1-ane-serve for one Snake or Tetris decision.

    python examples/decide_http.py --url http://127.0.0.1:8787 --game snake
    python examples/decide_http.py --url http://127.0.0.1:8787 --game tetris --adapter tetris
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_snake import INSTRUCTIONS as SNAKE_INSTRUCTIONS
from jeff_snake import OPTIONS as SNAKE_OPTIONS
from jeff_snake import opening_states, render_state
from jeff_tetris import INSTRUCTIONS as TETRIS_INSTRUCTIONS
from jeff_tetris import generate_tetris_rows


def snake_body() -> dict:
    snake, food = opening_states(1, seed=0)[0]
    return {
        "model": "jeff-latest",
        "state": render_state(snake, food),
        "questions": {
            "move": {
                "type": "choice",
                "instructions": SNAKE_INSTRUCTIONS,
                "criteria": dict(SNAKE_OPTIONS),
            }
        },
    }


def tetris_body() -> dict:
    row = generate_tetris_rows(1, seed=0)[0]
    options = row["options"]
    if not isinstance(options, dict):
        options = {f"o{i + 1}": opt for i, opt in enumerate(options)}
    return {
        "model": "jeff-latest",
        "state": row["state"],
        "questions": {
            "place": {
                "type": "choice",
                "instructions": row.get("instructions") or TETRIS_INSTRUCTIONS,
                "criteria": options,
            }
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8787")
    p.add_argument("--game", choices=("snake", "tetris"), default="snake")
    p.add_argument("--adapter", help="loaded adapter name (snake, tetris, …)")
    a = p.parse_args(argv)
    body = snake_body() if a.game == "snake" else tetris_body()
    if a.adapter:
        body["adapter"] = a.adapter
    req = urllib.request.Request(
        a.url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read().decode())
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
