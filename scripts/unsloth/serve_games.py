"""N Snake games on one adapter through the server's /v1/systemone; per-decision timings (timings.total_ms).
usage: serve_games.py URL ADAPTER GAMES [OUT.json]    e.g. serve_games.py http://127.0.0.1:8796 snake-stock 8"""
import json, random, sys, statistics as st, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coreai"))
from jeff_snake import INSTRUCTIONS, OPTIONS, opening_states, play_game, render_state, summarize_games
url, adapter, games = sys.argv[1], sys.argv[2], int(sys.argv[3])
out_path = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(f"games_{adapter}.json")
tot, pre, widths, toks = [], [], set(), []
def choose(snake, food):
    body = {"adapter": adapter, "state": render_state(snake, food), "questions": {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(OPTIONS)}}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"}), timeout=120).read())
    q = r["timings"]["questions"][0]; tot.append(r["timings"]["total_ms"]); pre.append(r["timings"]["prefill_ms"]); widths.add(tuple(q["call_widths"])); toks.append(q["tokens"])
    return r["answers"]["move"]["choice"]
res = [play_game(choose, s, f, random.Random(20_000 + i), max_steps=48) for i, (s, f) in enumerate(opening_states(games, 0))]
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
out = {**summarize_games(res), "decisions": len(tot), "total_ms": {"mean": st.mean(tot), "p50": q(tot, .5), "p95": q(tot, .95)}, "prefill_ms_mean": st.mean(pre), "call_widths": sorted(widths), "tokens": [min(toks), round(st.mean(toks), 1), max(toks)]}
print(adapter, json.dumps(out)); out_path.write_text(json.dumps(out))
