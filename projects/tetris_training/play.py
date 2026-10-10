#!/usr/bin/env python3
"""Evaluate the teacher/served Jeff, or collect teacher corrections on Jeff's boards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

from .dataset import (SPLITS, audit, empty_board, placements, read_jsonl, rollout,
                      sha256, state_key, teacher_pick, tetris_row, write_jsonl)


def request_body(board, piece, options, adapter):
    row = tetris_row(board, piece, teacher_pick(options)["key"], options)
    return {"model": "jeff-latest", "adapter": adapter, "orders": 1, "state": row["state"],
            "questions": {"move": {"type": "choice", "instructions": row["instructions"],
                                    "criteria": row["options"]}}}


class ServedPolicy:
    def __init__(self, url, adapter):
        self.url, self.adapter = url.rstrip("/"), adapter
        self.wall_ms, self.server_ms = [], []

    def __call__(self, board, piece, options):
        request = urllib.request.Request(
            self.url + "/v1/systemone",
            data=json.dumps(request_body(board, piece, options, self.adapter)).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
        self.wall_ms.append((time.perf_counter() - started) * 1000)
        self.server_ms.append(float(payload["timings"]["total_ms"]))
        action = payload["answers"]["move"]["choice"]
        if action not in {item["key"] for item in options}:
            raise ValueError(f"server returned illegal action {action!r}")
        return action


def latency_summary(values):
    if not values:
        return None
    ordered = sorted(values)
    def quantile(q):
        p = (len(ordered) - 1) * q
        lo = int(p)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (p - lo) * (ordered[hi] - ordered[lo])
    return {"n": len(values), "mean": statistics.mean(values), "p50": quantile(.5), "p95": quantile(.95)}


def deduplicate_corrections(rows, existing):
    seen = {state_key(row) for row in existing}
    kept = []
    for row in rows:
        key = state_key(row)
        if key not in seen and len(row["options"]) >= 2:
            kept.append(row)
            seen.add(key)
    return kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="audited dataset directory")
    parser.add_argument("--split", choices=("dev", "test", "train"), default="dev")
    parser.add_argument("--adapter", default="teacher", help="teacher, or an adapter on the running server")
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    parser.add_argument("--games", type=int, help="first N seeds from the selected fixed seed list")
    parser.add_argument("--max-pieces", type=int, default=150)
    parser.add_argument("--collect", action="store_true", help="save oracle corrections; requires --split train")
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    args = parser.parse_args(argv)
    if args.max_pieces < 1 or (args.games is not None and args.games < 1):
        parser.error("max-pieces and games must be positive")
    if args.collect and (args.split != "train" or args.adapter == "teacher"):
        parser.error("--collect requires --split train and a served --adapter")
    if args.split == "train" and not args.collect:
        parser.error("use --split dev/test for evaluation, or --collect for training data")
    dataset, output = args.dataset.expanduser().resolve(), args.output.expanduser().resolve()
    try:
        if output.exists():
            raise ValueError(f"use a new output directory: {output}")
        audit(dataset)
        manifest = json.loads((dataset / "manifest.json").read_text())
        seeds = (manifest["collection_seeds"] if args.collect else manifest["evaluation_seeds"][args.split])
        if args.games is not None and args.games > len(seeds):
            raise ValueError(f"only {len(seeds)} reserved {args.split} seeds")
        seeds = seeds[:args.games] if args.games is not None else seeds
        policy = None if args.adapter == "teacher" else ServedPolicy(args.url, args.adapter)
        health, warmup = None, None
        if policy is not None:
            with urllib.request.urlopen(policy.url + "/health", timeout=10) as response:
                health = json.load(response)
            board = empty_board()
            policy(board, "T", placements(board, "T"))
            warmup = {"wall_ms": policy.wall_ms.pop(), "server_ms": policy.server_ms.pop()}
        games, collected = [], []
        for seed in seeds:
            rows, game = rollout(seed, "teacher" if policy is None else f"jeff:{args.adapter}",
                                 args.max_pieces, 0, f"{args.split}:play:{seed}", choose=policy)
            games.append(game)
            if args.collect:
                collected.extend(rows)
            print(f"{len(games)}/{len(seeds)}: {game['pieces']} pieces, {game['lines']} lines, {game['terminal']}", flush=True)
        report = {
            "dataset": str(dataset), "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
            "adapter": args.adapter, "url": args.url if policy else None, "server_health": health,
            "split": args.split, "max_pieces": args.max_pieces, "greedy": True,
            "terminal_rule": "center-spawn", "games": games,
            "summary": {"games": len(games), "capped": sum(g["capped"] for g in games),
                        "pieces_mean": statistics.mean(g["pieces"] for g in games),
                        "pieces_median": statistics.median(g["pieces"] for g in games),
                        "pieces_min": min(g["pieces"] for g in games),
                        "lines_mean": statistics.mean(g["lines"] for g in games)},
            "first_request": warmup,  # Warm-up excluded below; this does not assert a cold server.
            "warm_wall_ms": latency_summary(policy.wall_ms) if policy else None,
            "warm_server_ms": latency_summary(policy.server_ms) if policy else None,
        }
        output.mkdir(parents=True)
        if args.collect:
            splits = {s: read_jsonl(dataset / f"{s}.jsonl") for s in SPLITS}
            corrections = deduplicate_corrections(collected, [row for rows in splits.values() for row in rows])
            write_jsonl(output / "corrections.jsonl", corrections)
            write_jsonl(output / "train.jsonl", splits["train"] + corrections)
            report["corrections"] = {"visited": len(collected), "unique_new": len(corrections),
                                     "combined_train": len(splits["train"]) + len(corrections),
                                     "validation_dataset": str(dataset / "validation.jsonl"),
                                     "train_sha256": sha256(output / "train.jsonl")}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
    except (ValueError, KeyError, OSError, urllib.error.URLError) as exc:
        parser.exit(1, f"play failed (not counted as a top-out): {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
