#!/usr/bin/env python3
"""Games exported from the iOS app (Saved games > Export, a .jsonl file) -> adapter-kit rows (same shape as jeff_tetris.generate_tetris_kit_rows).

    python ios/tools/games_to_rows.py GAMES_DIR_OR_JSONL [...] -o rows.jsonl [--label oracle|agree|disagree]

Options are rebuilt from (board, piece) with jeff_tetris.placements(), so the rows match the training prompt exactly.
  --label oracle    (default) every decision, label = El-Tetris choice (the teacher the adapter was trained on)
  --label agree     only decisions where the model matched El-Tetris (confirmed-easy positions)
  --label disagree  only decisions where the model differed (the informative ones)
Each game is one family, so a train/held-out split can keep a game together.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coreai"))
import jeff_tetris as T  # noqa: E402


def games(paths):
    for p in map(Path, paths):
        files = sorted(p.rglob("*.json")) + sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--label", choices=("oracle", "agree", "disagree"), default="oracle")
    a = ap.parse_args()
    seen, n = set(), 0
    with open(a.out, "w") as out:
        for g in games(a.paths):
            if g.get("schema") != "system1-ane-tetris-v1" or g["id"] in seen:
                continue
            seen.add(g["id"])
            for i, d in enumerate(g["decisions"]):
                board = [[1 if c == "#" else 0 for c in row] for row in d["board"].split("\n")]
                options = T.placements(board, d["piece"])
                keys = [o["key"] for o in options]
                if d.get("oracle") not in keys or len(d["probs"]) != len(keys):
                    continue   # corrupt or from a different rules version
                agree = d["pick"] == d["oracle"]
                if (a.label == "agree" and not agree) or (a.label == "disagree" and agree):
                    continue
                row = T.tetris_kit_row(T.tetris_row(board, d["piece"], d["oracle"], options), f"app-{g['id'][:8]}-{i:03d}")
                row["family"] = f"app-{g['id']}"
                row["source"] = {"dataset": "system1-ane-app", "app": g["app"], "model": g["model"],
                                 "model_pick": d["pick"], "model_conf": max(d["probs"])}
                out.write(json.dumps(row) + "\n"); n += 1
    print(f"{n} rows from {len(seen)} games -> {a.out}")


if __name__ == "__main__":
    main()
