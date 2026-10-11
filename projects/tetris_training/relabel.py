"""Relabel games played by the app: stronger teacher, model regret and RL fields, written as trainer rows.

    python3 -m projects.tetris_training.relabel GAMES... -o OUT_DIR [--teacher search2] [--mode distill|awr]

Input: archives made by ``ios/tools/archive_daily.py`` (``games-*.jsonl.gz``), an app export (``.jsonl``) or a folder of either.
Each game is ``{id, decisions: [{board, piece, pick, oracle, probs, ...}], lines, pieces, toppedOut, capped}``; options are rebuilt
from (board, piece) with ``jeff_tetris.placements`` so the rows match the training prompt exactly.

What each decision gets:

* ``teacher_key``   the placement the chosen teacher ranks first. ``el`` is the one-step El-Tetris rating the adapter was
                    trained on; ``search2`` also looks one piece ahead (expectimax over the 7 pieces, leaf = El-Tetris), which
                    lets the student exceed the heuristic it was taught by.
* ``regret``        teacher value of the best placement minus the placement the model played. Near-ties have ~0 regret and teach
                    nothing; real mistakes have large regret.
* ``censored``      true for the last ``--horizon`` moves of a game that did not end in a game over (cap, New game, session
                    end): their return is cut short, so they are excluded from baselines and from ``awr`` rows.
* ``reward``/``return``/``advantage``  lines cleared by the played move, the discounted return over the next ``--horizon``
                    moves (minus a penalty if the game topped out inside it), and the return minus a baseline averaged over
                    similar states (stack height x piece). This is the sequence view used for offline RL.

Modes (what ``label`` and ``weight`` mean in the output rows):

* ``distill`` (default)  label = teacher_key, weight = 1 + k * min(regret, cap) / cap: policy improvement by search, with
                    mistakes weighted up. Safe and needs no exploration in the data.
* ``awr``           advantage-weighted regression: label = the move the model played, weight = exp(advantage / beta) clipped.
                    Only informative when the games contain exploratory moves (the app's Explore mode).

Rows keep the plain ``{state, options, label, instructions}`` shape; ``weight`` is read by ``scripts/jeff_lora_train.py``.
Games, not rows, are split into train / heldout so a game never straddles the split.

Rules: a game ends at the first state whose centered, unrotated spawn is blocked (``dataset.topped_out``, the demo's and the
evaluation's rule). Decisions at or after that point are dropped (the app used to keep playing past it), and the game counts
as topped out for the return penalty.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coreai"))
import jeff_tetris as T  # noqa: E402
from projects.tetris_training.dataset import topped_out  # noqa: E402  (the demo's centered-spawn rule)

GAMMA = 0.9


def parse_board(text: str) -> list[list[int]]:
    return [[1 if c == "#" else 0 for c in row] for row in text.split("\n")]


def read_games(paths: list[str]):
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.name.endswith((".jsonl", ".jsonl.gz", ".json"))) if p.is_dir() else [p]
    seen = set()
    for f in files:
        opener = gzip.open if f.name.endswith(".gz") else open
        with opener(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    g = json.loads(line)
                    if g.get("schema") == "system1-ane-tetris-v1" and g["id"] not in seen:
                        seen.add(g["id"])
                        yield g


def q_values(board, piece: str, options: list[dict], teacher: str, top_k: int, wanted: set[str]) -> dict[str, float]:
    """Teacher value of the placements in ``wanted`` plus its top ``top_k`` (the rest are not searched)."""
    if teacher == "el":
        return {o["key"]: o["rating"] for o in options}
    ranked = sorted(options, key=lambda o: -o["rating"])
    cand = {o["key"] for o in ranked[:top_k]} | wanted
    out: dict[str, float] = {}
    for o in options:
        if o["key"] not in cand:
            continue
        nxt = o["board"]
        total = 0.0
        for p in T.PIECES:                     # uniform piece distribution, as in the app
            follow = T.placements(nxt, p)
            total += max((f["rating"] for f in follow), default=-1e4)   # top-out is a large penalty
        out[o["key"]] = o["rating"] + GAMMA * (total / len(T.PIECES))
    return out


def label_game(args) -> dict:
    g, teacher, top_k, horizon, topout_penalty = args
    decs, rewards = g["decisions"], []
    out = []
    spawn_blocked = False
    for d in decs:
        board = parse_board(d["board"])
        if topped_out(board, d["piece"]):          # real game over: later states are unreachable in the demo / evaluation
            spawn_blocked = True
            break
        options = T.placements(board, d["piece"])
        keys = [o["key"] for o in options]
        if d["pick"] not in keys or len(d.get("probs", [])) != len(keys):
            return {"id": g["id"], "bad": True}
        q = q_values(board, d["piece"], options, teacher, top_k, {d["pick"]})
        best = max(q, key=lambda k: (q[k], k))
        by_key = {o["key"]: o for o in options}
        rewards.append(int(by_key[d["pick"]]["features"]["lines"]))
        el = max(options, key=lambda o: (o["rating"], -o["rotation"], -o["column"]))["key"]   # == T.oracle_key
        out.append({"board": board, "piece": d["piece"], "options": options, "pick": d["pick"], "probs": d["probs"],
                    "teacher_key": best, "el_key": el, "regret": q[best] - q.get(d["pick"], q[best]),
                    "conf": max(d["probs"])})
    n = len(out)
    topped = bool(g.get("toppedOut")) or spawn_blocked
    for i, r in enumerate(out):
        ret = sum((GAMMA ** k) * rewards[i + k] for k in range(min(horizon, n - i)))
        if topped and n - i <= horizon:
            ret -= topout_penalty * (GAMMA ** (n - i))
        r["reward"], r["return"] = rewards[i], ret
        # A game that ended without a game over (120-piece cap, New game, session end) has no future for its last moves, so
        # their return is cut short and biased low. Mark them; they still serve distillation but not advantages or baselines.
        r["censored"] = (not topped) and (n - i < horizon)
    return {"id": g["id"], "game": g, "decisions": out, "spawn_blocked": spawn_blocked, "dropped": len(decs) - len(out)}


def split_of(game_id: str, heldout_frac: float) -> str:
    h = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "heldout" if h < heldout_frac else "train"


def bucket(board, piece: str) -> tuple:
    height = max((T.HEIGHT - r for r in range(T.HEIGHT) if any(board[r])), default=0)
    return (height // 4, piece)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+")
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("--teacher", choices=("el", "search2"), default="search2")
    ap.add_argument("--mode", choices=("distill", "awr"), default="distill")
    ap.add_argument("--top-k", type=int, default=6, help="placements the search teacher expands per state")
    ap.add_argument("--horizon", type=int, default=20, help="moves in the discounted return")
    ap.add_argument("--topout-penalty", type=float, default=10.0)
    ap.add_argument("--regret-cap", type=float, default=8.0)
    ap.add_argument("--hard-weight", type=float, default=3.0, help="distill: extra weight at the regret cap")
    ap.add_argument("--beta", type=float, default=1.0, help="awr: advantage temperature")
    ap.add_argument("--heldout", type=float, default=0.1, help="fraction of games held out")
    ap.add_argument("--min-regret", type=float, default=0.0, help="drop decisions whose model regret is below this (near-ties)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"{a.out} is not empty; use a new directory")
    a.out.mkdir(parents=True, exist_ok=True)

    games = list(read_games(a.games))
    jobs = [(g, a.teacher, a.top_k, a.horizon, a.topout_penalty) for g in games]
    with Pool(a.workers) as pool:
        labelled = [r for r in pool.imap_unordered(label_game, jobs) if not r.get("bad")]
    skipped = len(games) - len(labelled)

    base = defaultdict(list)                            # baseline: mean return of similar states
    for r in labelled:
        for d in r["decisions"]:
            if not d["censored"]:
                base[bucket(d["board"], d["piece"])].append(d["return"])
    mean = {k: sum(v) / len(v) for k, v in base.items()}

    files = {s: open(a.out / f"{s}.jsonl", "w") for s in ("train", "heldout")}
    stats = defaultdict(float)
    seen_state = set()
    dropped = sum(r["dropped"] for r in labelled)
    blocked_games = sum(1 for r in labelled if r["spawn_blocked"])
    for r in labelled:
        split = split_of(r["id"], a.heldout)
        for i, d in enumerate(r["decisions"]):
            adv = 0.0 if d["censored"] else d["return"] - mean.get(bucket(d["board"], d["piece"]), d["return"])
            stats["decisions"] += 1
            stats["censored"] += d["censored"]
            stats["model_eq_el"] += d["pick"] == d["el_key"]
            stats["model_eq_teacher"] += d["pick"] == d["teacher_key"]
            stats["teacher_ne_el"] += d["teacher_key"] != d["el_key"]
            stats["regret_sum"] += d["regret"]
            stats["hard(regret>=%.0f)" % a.regret_cap] += d["regret"] >= a.regret_cap
            state_key = (T.board_text(d["board"]), d["piece"])
            if state_key in seen_state or d["regret"] < a.min_regret or len(d["options"]) < 2:
                continue
            if a.mode == "awr" and d["censored"]:      # no usable advantage for the tail of an unfinished game
                continue
            seen_state.add(state_key)
            if a.mode == "distill":
                label = d["teacher_key"]
                weight = 1.0 + a.hard_weight * min(d["regret"], a.regret_cap) / a.regret_cap
            else:
                label = d["pick"]
                weight = min(5.0, max(0.1, math.exp(adv / a.beta)))
            row = T.tetris_row(d["board"], d["piece"], label, d["options"])
            row = T.tetris_kit_row(row, f"app-{r['id'][:8]}-{i:03d}")
            row["family"] = f"app-{r['id']}"
            row["weight"] = round(weight, 4)
            row["source"] = {"dataset": "system1-ane-app", "teacher": a.teacher, "mode": a.mode, "model_pick": d["pick"],
                             "model_conf": round(d["conf"], 4), "regret": round(d["regret"], 3),
                             "return": round(d["return"], 3), "advantage": round(adv, 3)}
            files[split].write(json.dumps(row) + "\n")
            stats[f"rows_{split}"] += 1
    for f in files.values():
        f.close()
    n = max(1.0, stats["decisions"])
    report = {"games": len(labelled), "skipped_games": skipped, "teacher": a.teacher, "mode": a.mode,
              "decisions": int(stats["decisions"]), "rows_train": int(stats["rows_train"]), "rows_heldout": int(stats["rows_heldout"]),
              "model_agrees_with_el_tetris": round(stats["model_eq_el"] / n, 4),
              "model_agrees_with_teacher": round(stats["model_eq_teacher"] / n, 4),
              "teacher_differs_from_el_tetris": round(stats["teacher_ne_el"] / n, 4),
              "mean_model_regret": round(stats["regret_sum"] / n, 3),
              "hard_share": round(stats["hard(regret>=%.0f)" % a.regret_cap] / n, 4),
              "censored_decisions": int(stats["censored"]),
              "games_reaching_spawn_blocked": blocked_games, "decisions_beyond_game_over_dropped": dropped,
              "lines_per_game_mean": round(sum(r["game"].get("lines", 0) for r in labelled) / max(1, len(labelled)), 2)}
    (a.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
