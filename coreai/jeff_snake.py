"""Snake decisions in the same shape the Jeff demo sends.

A row other tasks can copy is one JSON object:

    {"state": <text or object>, "options": {"up": "…", "down": "…"}, "label": "up",
     "instructions": "Pick the snake's next move."}

``options`` may be that object or a list of strings. ``label`` is an option key or a
zero-based index. The demo's snake state is an object whose last field is ``latest``
(the board), which is the live-last split Jeff already uses. The strings here match
``scripts/jeff_demo.html``.
"""
from __future__ import annotations

import random
from collections import deque

SIZE = 8
RULES = ("8 by 8 grid. H is the head, # is the body, F is food, a dot is empty. "
         "Eat F. Never move onto # or off the board.")
INSTRUCTIONS = "Pick the snake's next move."
OPTIONS = {
    "up": "move the head one cell up",
    "down": "move the head one cell down",
    "left": "move the head one cell left",
    "right": "move the head one cell right",
}
# Demo order. The first option is answer code A.
DIRECTIONS = (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1))
OPENING_SNAKE = ((4, 2), (4, 1), (4, 0))
OPENING_FOOD = (1, 6)


def board_text(snake, food, size: int = SIZE) -> str:
    """8 lines of 8 characters: H head, # body, F food, . empty. Same glyphs as the demo."""
    occupied = {tuple(cell) for cell in snake[1:]}
    head = tuple(snake[0])
    food = tuple(food)
    rows = []
    for r in range(size):
        line = []
        for c in range(size):
            if (r, c) == head:
                line.append("H")
            elif (r, c) in occupied:
                line.append("#")
            elif (r, c) == food:
                line.append("F")
            else:
                line.append(".")
        rows.append("".join(line))
    return "\n".join(rows)


def _delta(name: str) -> tuple[int, int]:
    for direction, dr, dc in DIRECTIONS:
        if direction == name:
            return dr, dc
    raise ValueError(f"unknown direction {name!r}")


def safe_moves(snake, food, size: int = SIZE) -> list[str]:
    """Moves that stay on the board and off the body. The current tail is free: it vacates unless the move eats,
    and food is never on the snake, so stepping onto the tail does not eat."""
    head = tuple(snake[0])
    tail = tuple(snake[-1])
    blocked = {tuple(cell) for cell in snake}
    blocked.discard(tail)
    moves = []
    for name, dr, dc in DIRECTIONS:
        nxt = (head[0] + dr, head[1] + dc)
        if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in blocked:
            moves.append(name)
    return moves


def step_snake(snake, food, move: str):
    """Apply a safe move. Returns ``(new_snake, ate)``. The caller has already rejected walls and the body."""
    dr, dc = _delta(move)
    head = tuple(snake[0])
    nxt = (head[0] + dr, head[1] + dc)
    grown = [nxt, *[tuple(cell) for cell in snake]]
    ate = nxt == tuple(food)
    if not ate:
        grown = grown[:-1]
    return grown, ate


def place_food(rng: random.Random, snake, size: int = SIZE):
    open_cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in {tuple(cell) for cell in snake}]
    if not open_cells:
        return None
    return open_cells[rng.randrange(len(open_cells))]


def oracle_move(snake, food, size: int = SIZE, search_cap: int = 50000) -> str | None:
    """First step of a shortest path to the food that never hits a wall or the body.

    Equal-length paths prefer the earlier direction in ``DIRECTIONS`` (up, then down, left, right), because that is
    the order the search expands. If no path reaches the food, the result is a safe move that most reduces Manhattan
    distance, or None when every move dies.
    """
    start = tuple(tuple(cell) for cell in snake)
    goal = tuple(food)
    if not start or start[0] == goal:
        return None
    queue = deque([(start, None)])
    seen = {start}
    while queue and len(seen) <= search_cap:
        state, first = queue.popleft()
        for name in safe_moves(state, goal, size):
            nxt, ate = step_snake(state, goal, name)
            chosen = first or name
            if ate:
                return chosen
            key = tuple(tuple(cell) for cell in nxt)
            if key in seen:
                continue
            seen.add(key)
            queue.append((key, chosen))
    return _greedy_safe(start, goal, size)


def _greedy_safe(snake, food, size: int) -> str | None:
    safes = safe_moves(snake, food, size)
    if not safes:
        return None
    head = tuple(snake[0])
    food = tuple(food)

    def rank(name: str):
        dr, dc = _delta(name)
        nxt = (head[0] + dr, head[1] + dc)
        distance = abs(nxt[0] - food[0]) + abs(nxt[1] - food[1])
        return distance, [item[0] for item in DIRECTIONS].index(name)

    return min(safes, key=rank)


def render_state(snake, food, size: int = SIZE, hint: bool = False) -> dict:
    """The demo's state object. ``hint`` adds food direction and the safe-move list in front of ``latest``."""
    snake = [tuple(cell) for cell in snake]
    food = tuple(food)
    latest = {
        "board": board_text(snake, food, size),
        "head": f"{snake[0][0]},{snake[0][1]}",
        "food": f"{food[0]},{food[1]}",
    }
    state = {"rules": RULES, "latest": latest}
    if hint:
        dr, dc = food[0] - snake[0][0], food[1] - snake[0][1]
        vertical = "same row" if dr == 0 else ("up" if dr < 0 else "down")
        horizontal = "same column" if dc == 0 else ("left" if dc < 0 else "right")
        safes = ", ".join(safe_moves(snake, food, size)) or "none"
        state = {
            "rules": RULES,
            "guide": f"Food is {vertical} {abs(dr)} and {horizontal} {abs(dc)}. Safe moves: {safes}.",
            "latest": latest,
        }
    return state


def snake_row(snake, food, label: str, hint: bool = False, size: int = SIZE) -> dict:
    """One training row: state, options, label, instructions.

    ``snake`` and ``food`` are extra fields the trainer ignores when it builds the prompt. They let a hint-prompt
    comparison reuse the same geometry. A hand-written dataset can omit them.
    """
    return {
        "state": render_state(snake, food, size, hint=hint),
        "options": dict(OPTIONS),
        "label": label,
        "instructions": INSTRUCTIONS,
        "snake": [list(cell) for cell in snake],
        "food": [int(food[0]), int(food[1])],
    }


def kit_row(row: dict, row_id: str, family: str) -> dict:
    """One Snake row in the adapter-kit ``rows.jsonl`` shape (``jeff-kit check-rows``).

    ``family`` groups positions from one game so a split can hold the game out together.
    ``label`` and ``target`` are the oracle direction.
    """
    label = str(row["label"])
    return {
        "id": row_id,
        "suite": "snake",
        "family": family,
        "state": row["state"],
        "question": {"type": "choice", "instructions": row["instructions"], "criteria": dict(row["options"])},
        "label": label,
        "target": label,
        "source": {"dataset": "snake-oracle", "round": "oracle-v1"},
    }


def generate_kit_rows(games: int, seed: int, steps: int = 6, size: int = SIZE) -> list[dict]:
    """Official rows: each game is a family, each position is labeled by the oracle."""
    if games < 1 or steps < 1:
        raise ValueError("games and steps must be positive")
    rng = random.Random(seed)
    rows = []
    for game in range(games):
        snake = [tuple(cell) for cell in random_snake(rng, rng.choice((3, 3, 4, 5)), size)]
        food = place_food(rng, snake, size)
        if food is None or oracle_move(snake, food, size) is None:
            continue
        family = f"game-{game:04d}"
        for step in range(steps):
            label = oracle_move(snake, food, size)
            if label is None:
                break
            sample = snake_row(snake, food, label, size=size)
            rows.append(kit_row(sample, f"snake-{family}-{step}", family))
            move = label if rng.random() < 0.7 else safe_moves(snake, food, size)[0]
            snake, ate = step_snake(snake, food, move)
            if ate:
                food = place_food(rng, snake, size)
                if food is None:
                    break
    if not rows:
        raise RuntimeError("the oracle produced no kit rows")
    return rows


def as_decision(row: dict) -> tuple[dict, int]:
    """Generic row -> Jeff ``{state, question}`` plus the label index in option order.

    A hand-written row has ``options``. An adapter-kit row has ``question.criteria`` instead,
    and ``label`` or ``target`` names the key.
    """
    if "options" not in row and isinstance(row.get("question"), dict):
        question = dict(row["question"])
        if question.get("type", "choice") != "choice":
            raise ValueError("only choice questions are trained as option keys")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("question.criteria must be an object of key to description")
        keys = list(criteria)
        label = row["label"] if "label" in row else row["target"]
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise ValueError("label must be an option key or an index")
        index = label if isinstance(label, int) else keys.index(str(label))
        if not 0 <= index < len(keys):
            raise ValueError(f"label {label!r} is outside the {len(keys)} options")
        question["type"] = "choice"
        question["criteria"] = dict(criteria)
        return {"state": row["state"], "question": question}, index
    options = row["options"]
    if isinstance(options, list):
        if not options or not all(isinstance(item, str) and item for item in options):
            raise ValueError("options list must be non-empty strings")
        criteria = {item: item for item in options}
    elif isinstance(options, dict) and options:
        criteria = dict(options)
    else:
        raise ValueError("options must be a non-empty list or an object of key to description")
    keys = list(criteria)
    label = row["label"]
    if isinstance(label, bool) or not isinstance(label, (int, str)):
        raise ValueError("label must be an option key or an index")
    index = label if isinstance(label, int) else keys.index(str(label))
    if not 0 <= index < len(keys):
        raise ValueError(f"label {label!r} is outside the {len(keys)} options")
    question = {"type": "choice", "criteria": criteria}
    instructions = row.get("instructions")
    if instructions is not None:
        question["instructions"] = instructions
    return {"state": row["state"], "question": question}, index


def random_snake(rng: random.Random, length: int | None = None, size: int = SIZE) -> list[tuple[int, int]]:
    """A connected snake, head first, grown by a self-avoiding walk."""
    length = length if length is not None else rng.randint(3, 6)
    for _attempt in range(80):
        snake = [(rng.randrange(size), rng.randrange(size))]
        while len(snake) < length:
            tail = snake[-1]
            nbrs = []
            for _, dr, dc in DIRECTIONS:
                nxt = (tail[0] + dr, tail[1] + dc)
                if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in snake:
                    nbrs.append(nxt)
            if not nbrs:
                break
            snake.append(nbrs[rng.randrange(len(nbrs))])
        food = _far_food(snake, size)
        if len(snake) == length and food is not None and oracle_move(snake, food, size) is not None:
            return snake
    return [tuple(cell) for cell in OPENING_SNAKE]


def _far_food(snake, size: int):
    """A cell used only to reject snakes that are already dead. Not the training food."""
    blocked = {tuple(cell) for cell in snake}
    for r, c in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)):
        if (r, c) not in blocked:
            return (r, c)
    return place_food(random.Random(0), snake, size)


def generate_snake_rows(n: int, seed: int, hint: bool = False, size: int = SIZE) -> list[dict]:
    """``n`` distinct boards labeled by the oracle. Off-policy: the walk mixes oracle steps with random safe steps."""
    rng = random.Random(seed)
    rows = []
    seen = set()
    guard = 0
    while len(rows) < n:
        guard += 1
        if guard > n * 40:
            raise RuntimeError(f"could only build {len(rows)} of {n} snake rows")
        snake = random_snake(rng, rng.choice((3, 3, 4, 5, 6)), size)
        food = place_food(rng, snake, size)
        if food is None:
            continue
        # A short walk so the set is not only freshly spawned snakes.
        for _ in range(rng.randint(0, 8)):
            label = oracle_move(snake, food, size)
            safes = safe_moves(snake, food, size)
            if label is None or not safes:
                break
            move = label if rng.random() < 0.7 else safes[rng.randrange(len(safes))]
            snake, ate = step_snake(snake, food, move)
            if ate:
                food = place_food(rng, snake, size)
                if food is None:
                    break
        if food is None:
            continue
        label = oracle_move(snake, food, size)
        if label is None:
            continue
        key = (tuple(tuple(cell) for cell in snake), tuple(food))
        if key in seen:
            continue
        seen.add(key)
        rows.append(snake_row(snake, food, label, hint=hint, size=size))
    return rows


def opening_states(n: int, seed: int, size: int = SIZE) -> list[tuple[list, tuple]]:
    """Game starts shared by two policies. The first is the demo's opening board."""
    rng = random.Random(seed)
    states = [(list(OPENING_SNAKE), OPENING_FOOD)]
    while len(states) < n:
        snake = random_snake(rng, 3, size)
        food = place_food(rng, snake, size)
        if food is None or oracle_move(snake, food, size) is None:
            continue
        states.append((snake, food))
    return states


def play_game(choose, snake, food, rng: random.Random, max_steps: int = 64, size: int = SIZE) -> dict:
    """One game. ``choose(snake, food) -> direction``. A wall or the body ends it. ``steps`` counts safe moves."""
    snake = [tuple(cell) for cell in snake]
    food = tuple(food)
    steps = 0
    eaten = 0
    for _ in range(max_steps):
        move = choose(snake, food)
        if move not in safe_moves(snake, food, size):
            break
        snake, ate = step_snake(snake, food, move)
        steps += 1
        if ate:
            eaten += 1
            placed = place_food(rng, snake, size)
            if placed is None:
                break
            food = placed
    return {"food": eaten, "steps": steps, "alive": steps == max_steps}


def summarize_games(games: list[dict]) -> dict:
    count = max(1, len(games))
    return {
        "games": len(games),
        "food_mean": sum(game["food"] for game in games) / count,
        "food_total": sum(game["food"] for game in games),
        "steps_mean": sum(game["steps"] for game in games) / count,
        "survived": sum(1 for game in games if game["alive"]),
    }
