#!/usr/bin/env python3
"""Play Tetris for the El-Tetris heuristic and, when a server is up, for Jeff adapters.

The same piece sequence is used for every player. A game ends when the next piece
has no legal placement, or after ``--max-pieces`` drops.

    python scripts/jeff_tetris_eval.py --games 8 --adapter base --adapter tetris
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_tetris import INSTRUCTIONS, oracle_key, play_game, render_state, tetris_row  # noqa: E402


def post(url: str, adapter: str, board, piece: str) -> dict:
    row = tetris_row(board, piece, oracle_key(board, piece) or "")
    body = {
        "model": "jeff-latest",
        "adapter": adapter,
        "state": render_state(board, piece),
        "questions": {
            "move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": row["options"]},
        },
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def summarize(games: list[dict], latencies: list[float]) -> dict:
    n = max(1, len(games))
    return {
        "games": len(games),
        "lines_mean": sum(game["lines"] for game in games) / n,
        "pieces_mean": sum(game["pieces"] for game in games) / n,
        "lines_total": sum(game["lines"] for game in games),
        "pieces_total": sum(game["pieces"] for game in games),
        "capped": sum(1 for game in games if game["capped"]),
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "decisions": len(latencies),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    parser.add_argument("--adapter", action="append", default=[], help="repeatable Jeff adapter; heuristic always plays")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-pieces", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    seeds = [args.seed + 1000 + index for index in range(args.games)]
    report = {}
    heuristic = [play_game(lambda board, piece: oracle_key(board, piece), seed, args.max_pieces) for seed in seeds]
    report["heuristic"] = summarize(heuristic, [])
    print(f"heuristic: lines {report['heuristic']['lines_mean']:.2f}  "
          f"pieces {report['heuristic']['pieces_mean']:.1f}", flush=True)
    for name in args.adapter:
        latencies = []

        def choose(board, piece, name=name):
            payload = post(args.url, name, board, piece)
            latencies.append(float(payload["timings"]["total_ms"]))
            return payload["answers"]["move"]["choice"]

        games = []
        for index, seed in enumerate(seeds):
            game = play_game(choose, seed, args.max_pieces)
            games.append(game)
            print(f"  {name} game {index + 1}/{len(seeds)}  lines {game['lines']}  pieces {game['pieces']}",
                  flush=True)
        report[name] = summarize(games, latencies)
        latency = report[name]["latency_ms_mean"]
        print(f"{name}: lines {report[name]['lines_mean']:.2f}  pieces {report[name]['pieces_mean']:.1f}  "
              f"latency {latency:.1f} ms  decisions {report[name]['decisions']}", flush=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
