#!/usr/bin/env python3
"""Build or audit CPU-only Tetris data for forge.py jeff-train-lora."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "coreai"))
from jeff_snake import as_decision
from jeff_tetris import HEIGHT, WIDTH, PIECES, empty_board, placements, tetris_row

SCHEMA = "jeff-tetris-dataset-v1"
SPLITS = ("train", "validation", "test")
MODES = ("teacher", "noisy", "garbage")


def seed_for(seed: int, scope: str, index: int) -> int:
    raw = f"{SCHEMA}:{seed}:{scope}:{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def topped_out(board, piece: str) -> bool:
    """The browser's centered, unrotated spawn collision rule."""
    shape = PIECES[piece][0]
    col = (WIDTH - max(dc for _, dc in shape) - 1) // 2
    return any(board[dr][col + dc] for dr, dc in shape)


def teacher_pick(options: list[dict]) -> dict:
    return max(options, key=lambda item: (item["rating"], -item["rotation"], -item["column"]))


def board_stats(board) -> tuple[int, int]:
    heights, holes = [], 0
    for col in range(WIDTH):
        first = next((r for r in range(HEIGHT) if board[r][col]), HEIGHT)
        heights.append(HEIGHT - first)
        holes += sum(not board[r][col] for r in range(first, HEIGHT))
    return max(heights), holes


def state_key(row: dict) -> str:
    return json.dumps(row["state"]["latest"], sort_keys=True, separators=(",", ":"))


def board_from_row(row: dict) -> tuple[list[list[int]], str]:
    latest = row["state"]["latest"]
    lines = latest["board"].splitlines()
    if len(lines) != HEIGHT or any(len(line) != WIDTH or set(line) - {".", "#"} for line in lines):
        raise ValueError("board must be a 20 by 10 grid of dots and #")
    piece = latest["piece"]
    if piece not in PIECES:
        raise ValueError(f"unknown piece {piece!r}")
    return [[int(cell == "#") for cell in line] for line in lines], piece


def labeled_row(board, piece, options, *, family, seed, step, mode, action):
    best = teacher_pick(options)
    row = tetris_row(board, piece, best["key"], options)
    height, holes = board_stats(board)
    row.update({
        "id": f"{family}:{step}", "family": family,
        # Side metadata only: the ordinary trainer reads state/options/label/instructions.
        "teacher_scores": {item["key"]: item["rating"] for item in options},
        "source": {"schema": SCHEMA, "mode": mode, "piece_seed": seed, "step": step,
                   "height": height, "holes": holes, "behavior_action": action},
    })
    return row


def rollout(seed: int, mode: str, max_pieces: int, noise: float, family: str, choose=None):
    """Independent piece/action RNGs; all collected labels are teacher labels."""
    piece_rng = random.Random(seed)
    action_rng = random.Random(seed_for(seed, "actions", 0))
    board = empty_board()
    if mode == "garbage":
        garbage_rng = random.Random(seed_for(seed, "garbage", 0))
        for r in range(HEIGHT - garbage_rng.randint(4, 10), HEIGHT):
            board[r] = [1] * WIDTH
            for col in garbage_rng.sample(range(WIDTH), garbage_rng.randint(1, 2)):
                board[r][col] = 0
    rows, lines, reason = [], 0, "cap"
    names = list(PIECES)
    for step in range(max_pieces):
        piece = names[piece_rng.randrange(len(names))]
        if topped_out(board, piece):
            reason = "spawn_blocked"
            break
        options = placements(board, piece)
        if not options:
            reason = "no_legal_placement"
            break
        best = teacher_pick(options)
        if choose is not None:
            action = choose(board, piece, options)
        elif mode == "noisy" and action_rng.random() < noise:
            action = action_rng.choice(options)["key"]
        else:
            action = best["key"]
        picked = next((item for item in options if item["key"] == action), None)
        if picked is None:
            raise ValueError(f"invalid action {action!r} in {family} at step {step}")
        rows.append(labeled_row(board, piece, options, family=family, seed=seed,
                                step=step, mode=mode, action=action))
        lines += picked["features"]["lines"]
        board = picked["board"]
    return rows, {"family": family, "piece_seed": seed, "mode": mode, "pieces": len(rows),
                  "lines": lines, "capped": len(rows) == max_pieces, "terminal": reason}


def stratum(row):
    s = row["source"]
    return (row["piece"], min(s["height"] // 5, 3), min(s["step"] // 25, 3))


def balanced_sample(rows, n: int, seed: int):
    """Round-robin over occupied piece/height/game-stage buckets, without duplication."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[stratum(row)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result = []
    while len(result) < n:
        keys = [key for key in sorted(buckets) if buckets[key]]
        if not keys:
            raise ValueError(f"only {len(result)} distinct candidates for {n} requested rows")
        rng.shuffle(keys)
        for key in keys:
            result.append(buckets[key].pop())
            if len(result) == n:
                break
    rng.shuffle(result)
    return result


def split_summary(rows):
    return {
        "rows": len(rows), "games_represented": len({r["family"] for r in rows}),
        "modes": dict(sorted(Counter(r["source"]["mode"] for r in rows).items())),
        "pieces": dict(sorted(Counter(r["piece"] for r in rows).items())),
        "height_buckets": dict(sorted(Counter(str(r["source"]["height"] // 5 * 5) for r in rows).items())),
        "rows_after_12_pieces": sum(r["source"]["step"] >= 12 for r in rows),
        "rows_with_holes": sum(r["source"]["holes"] > 0 for r in rows),
        "max_step": max(r["source"]["step"] for r in rows),
        "max_height": max(r["source"]["height"] for r in rows),
    }


def build_splits(counts: dict, seed=42, max_pieces=150, noise=0.10, excluded_states=()):
    if set(counts) != set(SPLITS) or any(n < 4 for n in counts.values()):
        raise ValueError("train, validation and test each need at least 4 rows")
    if max_pieces < 1 or not 0 <= noise <= 1:
        raise ValueError("max_pieces must be positive and noise must be in [0, 1]")
    splits, games, used = {}, [], set(excluded_states)
    # Reserve offline held-out states first; all later splits exclude them.
    for split in ("validation", "test", "train"):
        n = counts[split]
        quotas = dict(zip(MODES, (n - 2 * (n // 4), n // 4, n // 4)))
        selected = []
        for mode, quota in quotas.items():
            candidates, seen = [], set()
            for index in range(max(100, quota * 2)):
                game_seed = seed_for(seed, f"{split}:{mode}", index)
                family = f"{split}:{mode}:{game_seed}"
                rows, game = rollout(game_seed, mode, max_pieces, noise, family)
                games.append({"split": split, **game})
                for row in rows:
                    key = state_key(row)
                    if key not in used and key not in seen and len(row["options"]) >= 2:
                        candidates.append(row)
                        seen.add(key)
                if len(candidates) >= quota * 2 and index >= 1:
                    break
            if len(candidates) < quota:
                raise ValueError(f"insufficient distinct candidates for {split}/{mode}; raise --max-pieces")
            sample = balanced_sample(candidates, quota, seed_for(seed, f"sample:{split}:{mode}", 0))
            selected.extend(sample)
            used.update(state_key(row) for row in sample)
        random.Random(seed_for(seed, f"shuffle:{split}", 0)).shuffle(selected)
        splits[split] = selected
    return splits, games


def validate_splits(splits: dict):
    seen, families, seeds = set(), set(), set()
    for split in SPLITS:
        rows = splits[split]
        if not rows:
            raise ValueError(f"empty {split} split")
        local_families, local_seeds = set(), set()
        for row in rows:
            key = state_key(row)
            if key in seen:
                raise ValueError(f"duplicate state in {split}: {row.get('id')}")
            seen.add(key)
            board, piece = board_from_row(row)
            if topped_out(board, piece):
                raise ValueError(f"terminal board in {split}")
            options = placements(board, piece)
            if not options:
                raise ValueError(f"no legal options in {split}")
            expected = tetris_row(board, piece, teacher_pick(options)["key"], options)
            for field in ("state", "instructions", "label", "piece"):
                if row[field] != expected[field]:
                    raise ValueError(f"{field} does not match the teacher/prompt in {row['id']}")
            if list(row["options"].items()) != list(expected["options"].items()):
                raise ValueError(f"option text/order mismatch in {row['id']}")
            if row["teacher_scores"] != {item["key"]: item["rating"] for item in options}:
                raise ValueError(f"teacher score mismatch in {row['id']}")
            _, index = as_decision(row)
            if list(row["options"])[index] != row["label"]:
                raise ValueError("trainer label index mismatch")
            height, holes = board_stats(board)
            if (row["source"]["height"], row["source"]["holes"]) != (height, holes):
                raise ValueError(f"board statistics mismatch in {row['id']}")
            if row["source"]["behavior_action"] not in row["options"]:
                raise ValueError(f"illegal behavior action in {row['id']}")
            local_families.add(row["family"])
            local_seeds.add(row["source"]["piece_seed"])
        if families & local_families or seeds & local_seeds:
            raise ValueError("a game or piece seed crosses split boundaries")
        families.update(local_families)
        seeds.update(local_seeds)
    return {split: split_summary(splits[split]) for split in SPLITS}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported dataset schema")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"checksum mismatch: {name}")
    splits = {split: read_jsonl(directory / f"{split}.jsonl") for split in SPLITS}
    summary = validate_splits(splits)
    if "excluded_states.jsonl" in manifest["files"]:
        excluded = {state_key(r) for r in read_jsonl(directory / "excluded_states.jsonl")}
        if any(state_key(row) in excluded for rows in splits.values() for row in rows):
            raise ValueError("dataset overlaps excluded historical training states")
    if summary != manifest["summary"]:
        raise ValueError("summary does not match dataset")
    games = read_jsonl(directory / "games.jsonl")
    game_seeds = [g["piece_seed"] for g in games]
    by_family = {g["family"]: g for g in games}
    if len(by_family) != len(games) or len(games) != manifest["games_generated"]:
        raise ValueError("game count or family mismatch")
    for split, rows in splits.items():
        if len(rows) != manifest["config"]["counts"][split]:
            raise ValueError(f"row count mismatch: {split}")
        for row in rows:
            game = by_family[row["family"]]
            source = row["source"]
            if (game["split"] != split or game["piece_seed"] != source["piece_seed"]
                    or game["mode"] != source["mode"] or not 0 <= source["step"] < game["pieces"]):
                raise ValueError(f"row provenance mismatch: {row['id']}")
    all_seeds = game_seeds + manifest["collection_seeds"]
    for values in manifest["evaluation_seeds"].values():
        all_seeds.extend(values)
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("generation, collection or evaluation seeds overlap")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="generate and audit a dataset; no model needed")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--train", type=int, default=1024)
    build.add_argument("--validation", type=int, default=256)
    build.add_argument("--test", type=int, default=256)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--max-pieces", type=int, default=150)
    build.add_argument("--noise", type=float, default=0.10)
    build.add_argument("--exclude-dataset", type=Path, action="append", default=[],
                       help="exclude all board/piece states from a prior training JSONL; repeatable")
    check = sub.add_parser("audit", help="recompute teacher labels, prompts, split isolation and checksums")
    check.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            print(json.dumps(audit(args.dataset.expanduser().resolve()), indent=2))
            return 0
        output = args.output.expanduser().resolve()
        if output.exists():
            raise ValueError(f"use a new output directory: {output}")
        counts = {s: getattr(args, s) for s in SPLITS}
        prior_paths = [p.expanduser().resolve() for p in args.exclude_dataset]
        excluded = {state_key(row) for p in prior_paths for row in read_jsonl(p)}
        splits, games = build_splits(counts, args.seed, args.max_pieces, args.noise, excluded)
        summary = validate_splits(splits)
        output.mkdir(parents=True)
        for split in SPLITS:
            write_jsonl(output / f"{split}.jsonl", splits[split])
        write_jsonl(output / "games.jsonl", games)
        write_jsonl(output / "excluded_states.jsonl",
                    [{"state": {"latest": json.loads(key)}} for key in sorted(excluded)])
        files = [f"{s}.jsonl" for s in SPLITS] + ["games.jsonl", "excluded_states.jsonl"]
        manifest = {
            "schema": SCHEMA,
            "config": {"seed": args.seed, "counts": counts, "max_pieces": args.max_pieces,
                       "noise": args.noise, "prompt": "current-demo-prose", "terminal": "center-spawn"},
            "summary": summary, "games_generated": len(games),
            "excluded_states": len(excluded),
            "excluded_dataset_sha256": {str(p): sha256(p) for p in prior_paths},
            "collection_seeds": [seed_for(args.seed, "collect:train", i) for i in range(32)],
            "evaluation_seeds": {s: [seed_for(args.seed, f"evaluate:{s}", i) for i in range(n)]
                                 for s, n in (("dev", 8), ("test", 32))},
            "files": {name: sha256(output / name) for name in files},
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                              (Path(__file__), ROOT / "coreai/jeff_tetris.py", ROOT / "scripts/jeff_demo.html")},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        audit(output)
        print(json.dumps({"output": str(output), "summary": summary, "games_generated": len(games)}, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"dataset: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
