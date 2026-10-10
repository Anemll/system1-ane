"""Unattended soak of a running system1-ane-serve: plays Tetris (and some Snake) exactly as scripts/jeff_demo.html does.

Requests are sequential. Every --every moves it logs the move count, the server worker's IOSurface count and
phys_footprint (footprint -p), latency, and Tetris lines / pieces. Tetris resets itself on game over (the newly
spawned piece overlaps locked cells, or no legal placement). Exits nonzero on the first error or when the server
dies, after printing the tail of the tmux session that runs the server.

    python scripts/jeff_demo_soak.py --moves 500 --snake-every 4
    python scripts/jeff_demo_soak.py --parity out.json            # fixed boards; answers and probabilities
    python scripts/jeff_demo_soak.py --parity new.json --compare out.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

T_W, T_H = 10, 20
T_TURNS = ("unrotated", "rotated right", "rotated twice", "rotated left")
T_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
T_PIECES = {
    "I": [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (1, 0), (2, 0), (3, 0)]],
    "O": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "T": [[(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 0), (1, 0), (1, 1), (2, 0)], [(0, 0), (0, 1), (0, 2), (1, 1)],
          [(0, 1), (1, 0), (1, 1), (2, 1)]],
    "S": [[(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]],
    "Z": [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)]],
    "J": [[(0, 0), (1, 0), (1, 1), (1, 2)], [(0, 0), (0, 1), (1, 0), (2, 0)], [(0, 0), (0, 1), (0, 2), (1, 2)],
          [(0, 1), (1, 1), (2, 0), (2, 1)]],
    "L": [[(0, 2), (1, 0), (1, 1), (1, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)], [(0, 0), (0, 1), (0, 2), (1, 0)],
          [(0, 0), (0, 1), (1, 1), (2, 1)]],
}
T_RULES = ("10 by 20 board. # is filled, a dot is empty. The top row is printed first. Drop the named piece in one "
           "rotation and column. Do not leave a hole you can avoid.")
T_INSTRUCTIONS = "Choose where to drop this piece."

SIZE = 8
S_RULES = ("8 by 8 grid. H is the head, # is the body, F is food, a dot is empty. Eat F. "
           "Never move onto # or off the board.")
S_CRITERIA = {"up": "move the head one cell up", "down": "move the head one cell down",
              "left": "move the head one cell left", "right": "move the head one cell right"}
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
S_OPENING = [(4, 2), (4, 1), (4, 0)]
S_FOOD = (1, 6)


# ---- Tetris: a line-for-line port of the page's tPlacements / tToppedOut ------------------------------------------
def t_empty():
    return [[0] * T_W for _ in range(T_H)]


def t_fits(board, shape, row, col):
    for dr, dc in shape:
        r, c = row + dr, col + dc
        if c < 0 or c >= T_W or r >= T_H:
            return False
        if r >= 0 and board[r][c]:
            return False
    return True


def t_drop_row(board, shape, col):
    if not t_fits(board, shape, 0, col):
        return None
    row = 0
    while t_fits(board, shape, row + 1, col):
        row += 1
    return row


def t_clear(board):
    kept = [line for line in board if not all(line)]
    lines = T_H - len(kept)
    return [[0] * T_W for _ in range(lines)] + kept, lines


def t_holes(board):
    holes = 0
    for c in range(T_W):
        seen = False
        for r in range(T_H):
            if board[r][c]:
                seen = True
            elif seen:
                holes += 1
    return holes


def t_placements(board, piece):
    found = []
    for rotation, shape in enumerate(T_PIECES[piece]):
        width = max(dc for _, dc in shape) + 1
        for col in range(T_W - width + 1):
            row = t_drop_row(board, shape, col)
            if row is None:
                continue
            painted = [line[:] for line in board]
            for dr, dc in shape:
                if row + dr >= 0:
                    painted[row + dr][col + dc] = piece
            cleared, lines = t_clear(painted)
            holes = t_holes(cleared)
            cols = [col + dc for _, dc in shape]
            lo, hi = min(cols), max(cols)
            span = f"column {lo}" if lo == hi else f"columns {lo}-{hi}"
            bottom = row + max(dr for dr, _ in shape)
            found.append({
                "key": f"r{rotation}c{col}", "board": cleared, "lines": lines,
                "text": (f"{piece} piece, {T_TURNS[rotation]}, {span}, lands on row {bottom}, clears {lines} "
                         f"{'line' if lines == 1 else 'lines'}, leaves {holes} {'hole' if holes == 1 else 'holes'}"),
            })
    return found


def t_topped_out(board, piece):
    shape = T_PIECES[piece][0]
    col = (T_W - (max(dc for _, dc in shape) + 1)) // 2
    return any(board[dr][col + dc] for dr, dc in shape)


def t_text(board):
    return "\n".join("".join("#" if cell else "." for cell in line) for line in board)


def tetris_body(board, piece, options, adapter="tetris"):
    criteria = {item["key"]: item["text"] for item in options}
    return {
        "model": "jeff-latest" if adapter == "base" else adapter,
        "adapter": adapter,
        "state": {"rules": T_RULES, "latest": {"board": t_text(board), "piece": piece}},
        "questions": {"move": {"type": "choice", "instructions": T_INSTRUCTIONS, "criteria": criteria}},
    }


# ---- Snake: the page's snakeRequest ----------------------------------------------------------------------------
def s_board(snake, food):
    rows = []
    for r in range(SIZE):
        line = ""
        for c in range(SIZE):
            if snake[0] == (r, c):
                line += "H"
            elif (r, c) in snake[1:]:
                line += "#"
            elif food == (r, c):
                line += "F"
            else:
                line += "."
        rows.append(line)
    return "\n".join(rows)


def snake_body(snake, food, adapter="snake"):
    return {
        "model": "jeff-latest",
        "adapter": adapter,
        "state": {"rules": S_RULES, "latest": {"board": s_board(snake, food), "head": f"{snake[0][0]},{snake[0][1]}",
                                               "food": f"{food[0]},{food[1]}"}},
        "questions": {"move": {"type": "choice", "instructions": "Pick the snake's next move.",
                               "criteria": S_CRITERIA}},
    }


def snake_step(snake, food, move, rng):
    """(snake, food, died, ate) after one move, with the page's restart on a hit."""
    dr, dc = DIRS[move]
    nxt = (snake[0][0] + dr, snake[0][1] + dc)
    onto_tail = nxt == snake[-1]
    hit = not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) or (nxt in snake and not onto_tail)
    if hit:
        snake = list(S_OPENING)
        return snake, place_food(snake, rng), True, False
    snake = [nxt] + snake
    if nxt == food:
        return snake, place_food(snake, rng), False, True
    return snake[:-1], food, False, False


def place_food(snake, rng):
    open_cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in snake]
    return rng.choice(open_cells)


# ---- server ------------------------------------------------------------------------------------------------------
BUSY = {"retries": 0}


def post(url, body, timeout=60.0, busy_retries=30):
    """One decision. A 529 (another client holds the model, e.g. the live demo page) is retried after Retry-After."""
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    for attempt in range(busy_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as error:
            if error.code != 529 or attempt == busy_retries:
                raise
            BUSY["retries"] += 1
            time.sleep(float(error.headers.get("Retry-After") or 1))


def server_pid(port):
    out = subprocess.run(["pgrep", "-f", f"(system1_ane_serve|jeff_serve)\\.py .*--port {port}"], capture_output=True, text=True).stdout
    pids = [int(p) for p in out.split()]
    return pids[0] if pids else None


def footprint(pid):
    if pid is None:
        return None, None
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    n = re.search(r"\s(\d+)\s+IOSurface\s*$", out, re.M)
    fp = re.search(r"phys_footprint:\s+(\d+(?:\.\d+)?)\s*(\w+)", out)
    mb = None
    if fp:
        mb = float(fp.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}.get(fp.group(2), 1)
    return (int(n.group(1)) if n else None), mb


def tmux_tail(session, lines=40):
    out = subprocess.run(["tmux", "capture-pane", "-p", "-J", "-t", session, "-S", f"-{lines}"],
                         capture_output=True, text=True).stdout
    return "\n".join(line for line in out.splitlines() if not line.startswith('"POST'))


# ---- parity ------------------------------------------------------------------------------------------------------
def parity_boards():
    snake = [((4, 2), (4, 1), (4, 0)), ((2, 5), (3, 5), (4, 5), (4, 4)), ((7, 0), (6, 0), (5, 0), (5, 1))]
    foods = [(1, 6), (6, 2), (0, 7)]
    tetris = []
    board = t_empty()
    rng = random.Random(7)
    for piece in ("T", "I", "S", "L", "O", "Z", "J", "T"):
        tetris.append(([line[:] for line in board], piece))
        options = t_placements(board, piece)
        board = rng.choice(options)["board"]
    bodies = [(f"snake{i}", snake_body(list(s), f)) for i, (s, f) in enumerate(zip(snake, foods))]
    bodies += [(f"snake_base{i}", snake_body(list(s), f, adapter="base")) for i, (s, f) in enumerate(zip(snake, foods))]
    bodies += [(f"tetris{i}", tetris_body(b, p, t_placements(b, p))) for i, (b, p) in enumerate(tetris)]
    return bodies


def run_parity(url, out, compare):
    results = {}
    for name, body in parity_boards():
        answer = post(url, body)["answers"]["move"]
        results[name] = {"choice": answer["choice"], "probabilities": answer["probabilities"]}
    with open(out, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    print(f"wrote {len(results)} answers to {out}")
    if not compare:
        return 0
    with open(compare) as f:
        ref = json.load(f)
    worst, mismatches = 0.0, []
    for name, got in results.items():
        want = ref[name]
        if got["choice"] != want["choice"]:
            mismatches.append(f"{name}: {want['choice']} -> {got['choice']}")
        for key, p in want["probabilities"].items():
            worst = max(worst, abs(p - got["probabilities"][key]))
    print(f"parity vs {compare}: {len(results) - len(mismatches)}/{len(results)} same choice, "
          f"max |dp| {worst:.2e}")
    for line in mismatches:
        print("  " + line)
    return 1 if mismatches else 0


# ---- soak --------------------------------------------------------------------------------------------------------
def run_soak(a):
    rng = random.Random(a.seed)
    pid = server_pid(a.port)
    if pid is None:
        print(f"no system1_ane_serve.py worker on port {a.port}")
        return 2
    board, piece, nxt = t_empty(), rng.choice(T_NAMES), rng.choice(T_NAMES)
    lines = pieces = games = best_lines = 0
    snake, food = list(S_OPENING), S_FOOD
    eaten = deaths = 0
    lat = []
    s0, m0 = footprint(pid)
    warm = None
    print(f"pid {pid}  start: {s0} IOSurfaces, {m0:.0f} MB")
    print(f"{'move':>5} {'surf':>6} {'MB':>7} {'ms(avg)':>8} {'game':>4} {'lines':>5} {'pieces':>6} "
          f"{'best':>4} {'snake eat/die':>13}")
    rows = []
    for move in range(1, a.moves + 1):
        try:
            if a.snake_every and move % a.snake_every == 0:
                t0 = time.perf_counter()
                answer = post(a.url, snake_body(snake, food))["answers"]["move"]
                lat.append(1e3 * (time.perf_counter() - t0))
                snake, food, died, ate = snake_step(snake, food, answer["choice"], rng)
                deaths += died
                eaten += ate
            else:
                if t_topped_out(board, piece) or not t_placements(board, piece):
                    games += 1
                    best_lines = max(best_lines, lines)
                    board, piece, nxt = t_empty(), rng.choice(T_NAMES), rng.choice(T_NAMES)
                    lines = pieces = 0
                options = t_placements(board, piece)
                t0 = time.perf_counter()
                answer = post(a.url, tetris_body(board, piece, options))["answers"]["move"]
                lat.append(1e3 * (time.perf_counter() - t0))
                chosen = next((item for item in options if item["key"] == answer["choice"]), None)
                if chosen is None:
                    raise RuntimeError(f"choice {answer['choice']} is not a legal placement")
                lines += chosen["lines"]
                pieces += 1
                board, piece, nxt = chosen["board"], nxt, rng.choice(T_NAMES)
        except (urllib.error.URLError, OSError, RuntimeError, KeyError, ValueError) as error:
            print(f"\nFAILED at move {move}: {type(error).__name__}: {error}")
            print(f"server alive: {server_pid(a.port) is not None}")
            print("---- tmux tail ----")
            print(tmux_tail(a.tmux))
            return 1
        if move % a.every == 0 or move == a.moves:
            surf, mb = footprint(pid)
            if move == a.warmup:
                warm = surf
            avg = sum(lat[-a.every:]) / len(lat[-a.every:])
            rows.append((move, surf, mb))
            print(f"{move:>5} {surf:>6} {mb:>7.0f} {avg:>8.0f} {games:>4} {lines:>5} {pieces:>6} "
                  f"{max(best_lines, lines):>4} {eaten:>6}/{deaths:<6}", flush=True)
    s1, m1 = footprint(pid)
    after = [r for r in rows if r[0] >= a.warmup]
    span = (max(r[1] for r in after) - min(r[1] for r in after)) if after else 0
    print(f"done: {a.moves} moves, {games} Tetris games ended, IOSurfaces {s0} -> {s1}, footprint {m0:.0f} -> "
          f"{m1:.0f} MB; after move {a.warmup}: {warm} -> {s1}, spread {span}; busy retries {BUSY['retries']}")
    return 0 if span <= a.bound else 3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8787")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--tmux", default="system1-ane-serve")
    p.add_argument("--moves", type=int, default=500)
    p.add_argument("--snake-every", type=int, default=4, help="every Nth move is a Snake move (0: Tetris only)")
    p.add_argument("--every", type=int, default=25, help="log every N moves")
    p.add_argument("--warmup", type=int, default=50, help="flatness is judged from this move on")
    p.add_argument("--bound", type=int, default=200, help="max IOSurface spread after warm-up")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parity", help="write fixed-board answers to this JSON and exit")
    p.add_argument("--compare", help="with --parity: a previous --parity JSON to compare against")
    a = p.parse_args(argv)
    if a.parity:
        return run_parity(a.url, a.parity, a.compare)
    return run_soak(a)


if __name__ == "__main__":
    sys.exit(main())
