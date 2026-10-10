#!/usr/bin/env python3
"""Play Snake against a running system1-ane-serve and report food, survival, and latency.

The same opening boards are used for every adapter, including ``base``.

    python scripts/jeff_snake_eval.py --url http://127.0.0.1:8787 --games 8 \\
        --adapter base --adapter snake
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_snake import INSTRUCTIONS, OPTIONS, opening_states, play_game, render_state, summarize_games  # noqa: E402


def post(url: str, adapter: str, state: dict) -> dict:
    body = {
        "model": "jeff-latest",
        "adapter": adapter,
        "state": state,
        "questions": {
            "move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(OPTIONS)},
        },
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8787")
    p.add_argument("--adapter", action="append", default=[], help="repeatable; default base")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    adapters = a.adapter or ["base"]
    states = opening_states(a.games, a.seed)
    report = {}
    for name in adapters:
        latencies = []

        def choose(snake, food, name=name):
            payload = post(a.url, name, render_state(snake, food))
            latencies.append(float(payload["timings"]["total_ms"]))
            return payload["answers"]["move"]["choice"]

        games = []
        for index, (snake, food) in enumerate(states):
            games.append(play_game(choose, snake, food, random.Random(20_000 + index), max_steps=a.max_steps))
        summary = summarize_games(games)
        summary["latency_ms_mean"] = sum(latencies) / max(1, len(latencies))
        summary["decisions"] = len(latencies)
        report[name] = summary
        print(f"{name}: food {summary['food_mean']:.2f}  steps {summary['steps_mean']:.1f}  "
              f"latency {summary['latency_ms_mean']:.1f} ms  decisions {summary['decisions']}", flush=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
