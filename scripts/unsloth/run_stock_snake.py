#!/usr/bin/env python3
"""Run the Unsloth-trained STOCK Qwen3.5-0.8B Snake adapter end to end on the ANE (no server, no port).

  state/prompt -> Unsloth Clef prompt shaping (encode_record, tokenizer subprocess) -> ANE backbone prefill
  (merged LoRA, FP16 Core AI build) -> final RMSNorm + Clef joint head on the host (PyTorch) -> softmax(logit / T)
  with the head's own calibration temperature (Colab's FastDecisionModel.calibrate) -> move + probabilities + latency.

Usage (STOCK_BUILD is required: a build directory holding coreai/ and model/, see scripts/unsloth/README.md):
  STOCK_BUILD=<build> python scripts/unsloth/run_stock_snake.py                  # 5 Colab parity rows + 4 sample states + 3 live games
  ... --parity              # only the 5 Colab parity rows vs data/stock_snake_parity_rows.json
  ... --games 8 --seed 0    # live Snake games (the repository's Snake generator, opening_states)
  ... --state-json s.json   # {"state": {...}, "instructions": "...", "options": {...}} or a Snake
                            # {"snake": [[r,c],...], "food": [r,c]}
  ... --heldout 256         # accuracy on the held-out rows (87.1% in Colab, 85.2% measured here, both on the earlier split with 5 of 256 states also in training); needs STOCK_HELDOUT
Env: STOCK_BUILD (required), STOCK_HELDOUT (heldout.jsonl from make_snake_data.py; default ./snake_data/heldout.jsonl),
STOCK_PARITY (default: the rows shipped next to this script), STOCK_REPORT (default ./run_stock_snake_report.json),
CLEF_HEAD_BACKEND=host|ane, COREAI_PYTHON (interpreter with the Core AI runtime; default: the current one),
TOKENIZER_PYTHON (interpreter with transformers; default: the current one).
"""
from __future__ import annotations

import argparse, json, os, random, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COREAI_PY = os.environ.get("COREAI_PYTHON") or sys.executable
TOK_PY = os.environ.get("TOKENIZER_PYTHON") or sys.executable
if Path(sys.executable).resolve() != Path(COREAI_PY).resolve() and os.environ.get("_STOCK_SNAKE_REEXEC") != "1":
    os.environ["_STOCK_SNAKE_REEXEC"] = "1"
    os.execv(COREAI_PY, [COREAI_PY, __file__, *sys.argv[1:]])

import numpy as np  # noqa: E402
sys.path[:0] = [str(REPO / "coreai"), str(REPO / "scripts")]
from clef_engine import ClefEngine  # noqa: E402
from jeff_coreai_runtime import JeffCoreAI  # noqa: E402
from jeff_snake import (INSTRUCTIONS, OPTIONS, opening_states, play_game, render_state, summarize_games)  # noqa: E402

if "STOCK_BUILD" not in os.environ:
    sys.exit("set STOCK_BUILD=<build dir with coreai/ and model/>; see scripts/unsloth/README.md")
BUILD = Path(os.environ["STOCK_BUILD"]) / "coreai"
MODEL = BUILD.parent / "model"
PARITY = Path(os.environ.get("STOCK_PARITY") or HERE / "data/stock_snake_parity_rows.json")
HELDOUT = Path(os.environ.get("STOCK_HELDOUT") or "snake_data/heldout.jsonl")
REPORT = Path(os.environ.get("STOCK_REPORT") or "run_stock_snake_report.json")


class Tokenizer:
    """Clef prompt ids + spans from the forge-interpreter worker (the Core AI interpreter may lack tokenizers/transformers)."""
    def __init__(self):
        self.p = subprocess.Popen([TOK_PY, str(REPO / "scripts/jeff_tokenize.py"), str(MODEL)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1, stderr=subprocess.DEVNULL)
        assert json.loads(self.p.stdout.readline()).get("ready")

    def __call__(self, state, instructions, options):
        self.p.stdin.write(json.dumps({"clef": {"state": state, "instructions": instructions, "options": options}}) + "\n")
        self.p.stdin.flush()
        msg = json.loads(self.p.stdout.readline())
        if "error" in msg:
            raise ValueError(msg["error"])
        return msg["clef"]

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"cmd": "stop"}) + "\n"); self.p.stdin.flush(); self.p.wait(5)
        except Exception:
            self.p.kill()


STATS = {"head_ms": [], "prefill_ms": [], "total_ms": []}   # every timed decide() after the warm-up (printed at the end)


def decide(eng, tok, state, instructions=INSTRUCTIONS, options=None):
    options = dict(options or OPTIONS)
    t0 = time.perf_counter()
    enc = tok(state, instructions, options)
    tok_ms = 1e3 * (time.perf_counter() - t0)
    logits, r = eng.logits_for(enc)
    import torch
    p = torch.softmax(torch.from_numpy(logits) / eng.head.t_choice, -1).numpy()
    k = int(p.argmax())
    STATS["head_ms"].append(float(r["head_ms"])); STATS["prefill_ms"].append(float(sum(r["calls_ms"])))
    STATS["total_ms"].append(tok_ms + float(sum(r["calls_ms"])) + float(r["head_ms"]))
    return {"move": enc["option_keys"][k], "confidence": float(p[k]),
            "probabilities": dict(zip(enc["option_keys"], map(float, p))), "logits": logits.tolist(),
            "tokens": len(enc["input_ids"]), "tokenize_ms": tok_ms, "prefill_ms": float(sum(r["calls_ms"])),
            "head_ms": float(r["head_ms"]), "total_ms": tok_ms + float(sum(r["calls_ms"])) + float(r["head_ms"]), "enc": enc}


def load_heldout(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def fmt(d):
    pr = " ".join(f"{k}={v:.3f}" for k, v in d["probabilities"].items())
    return (f"{d['move']:<5} conf {d['confidence']:.3f} | {pr} | {d['tokens']} tok | tokenize {d['tokenize_ms']:.1f} ms, "
            f"prefill {d['prefill_ms']:.1f} ms, head {d['head_ms']:.1f} ms, total {d['total_ms']:.1f} ms")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--parity", action="store_true"); a.add_argument("--games", type=int, default=None)
    a.add_argument("--seed", type=int, default=0); a.add_argument("--max-steps", type=int, default=48)
    a.add_argument("--state-json", type=Path); a.add_argument("--heldout", type=int, default=0)
    a = a.parse_args()
    default_run = not (a.parity or a.games is not None or a.state_json or a.heldout)
    rt = JeffCoreAI(BUILD, MODEL, log=lambda *_: None)
    tok = Tokenizer(); eng = ClefEngine(rt, BUILD, tok)
    print(f"Clef head: {eng.head_desc}")
    print(f"stock Qwen3.5-0.8B + Unsloth Snake LoRA/Clef head | {rt.entry} | head temperature {eng.head.t_choice:.6f} "
          f"(folded {eng.head.decision.get('folded_temperature')})")
    decide(eng, tok, render_state([(4, 2), (4, 1), (4, 0)], (1, 6)))  # warm-up
    # Only the parity rows and --heldout read validation rows; --games / --state-json do not need the file to exist.
    held = load_heldout(HELDOUT) if (a.parity or default_run or a.heldout) else []
    report = {}
    if a.parity or default_run:
        print("\n== 5 Colab parity rows (stock_snake_parity_rows.json) ==")
        rows = []
        for p in json.load(open(PARITY))["rows"]:
            r = held[p["heldout_row"]]
            d = decide(eng, tok, r["state"], r["instructions"], r["options"])
            same_ids = d["enc"]["input_ids"] == p["input_ids"]
            col = p["calibrated_probabilities"]
            dl = float(np.abs(np.array(d["logits"]) - np.array(p["raw_logits"])).max())
            dp = max(abs(d["probabilities"][k] - col[k]) for k in col)
            rows.append({"row": p["heldout_row"], "answer": d["move"], "colab_answer": p["answer"], "gold": p["gold"],
                         "ids_identical": same_ids, "max_logit_diff": dl, "max_prob_diff": dp, **{k: d[k] for k in ("prefill_ms", "head_ms", "total_ms")}})
            print(f"row {p['heldout_row']}: ANE {d['move']} vs Colab {p['answer']} (gold {p['gold']}) ids_identical={same_ids} "
                  f"max|dlogit|={dl:.3f} max|dprob|={dp:.3f}\n   {fmt(d)}")
        report["parity"] = rows
        print(f"parity: answers match {sum(x['answer'] == x['colab_answer'] for x in rows)}/5, ids identical "
              f"{sum(x['ids_identical'] for x in rows)}/5, max|dlogit| {max(x['max_logit_diff'] for x in rows):.3f}, "
              f"max|dprob| {max(x['max_prob_diff'] for x in rows):.3f}")
    if default_run:
        print("\n== sample Snake states ==")
        for snake, food in opening_states(4, 1):
            print(f"snake {snake} food {food}\n   " + fmt(decide(eng, tok, render_state(snake, food))))
    if a.state_json:
        j = json.load(open(a.state_json))
        st = render_state(j["snake"], j["food"]) if "snake" in j else j["state"]
        print("\n== --state-json ==\n" + fmt(decide(eng, tok, st, j.get("instructions", INSTRUCTIONS), j.get("options"))))
    games = a.games if a.games is not None else (3 if default_run else 0)
    if games:
        print(f"\n== {games} live Snake games (max {a.max_steps} steps) ==")
        lat = []
        def choose(snake, food):
            d = decide(eng, tok, render_state(snake, food)); lat.append(d["total_ms"]); return d["move"]
        results = []
        for i, (snake, food) in enumerate(opening_states(games, a.seed)):
            g = play_game(choose, snake, food, random.Random(20_000 + i), max_steps=a.max_steps); results.append(g)
            print(f"game {i}: food {g['food']} steps {g['steps']} alive {g['alive']}")
        s = summarize_games(results)
        print(f"food_mean {s['food_mean']:.2f} steps_mean {s['steps_mean']:.1f} survived {s['survived']}/{s['games']} "
              f"decisions {len(lat)} latency mean {statistics.mean(lat):.1f} ms median {statistics.median(lat):.1f} ms")
        report["games"] = {**s, "latency_ms_mean": statistics.mean(lat), "decisions": len(lat)}
    if a.heldout:
        n = min(a.heldout, len(held)); ok = 0; lat = []
        for r in held[:n]:
            d = decide(eng, tok, r["state"], r["instructions"], r["options"]); ok += d["move"] == r["label"]; lat.append(d["total_ms"])
        print(f"\n== held-out {n} rows: accuracy {ok / n:.3%} (Colab FP32 87.1%), latency mean {statistics.mean(lat):.1f} ms")
        report["heldout"] = {"n": n, "acc": ok / n}
    st = STATS
    if st["head_ms"]:
        hm = {f"{k}_{f}": float(fn(v)) for k, v in st.items() for f, fn in (("mean", statistics.mean), ("min", min), ("max", max))}
        hm["n"] = len(st["head_ms"]); report["stats"] = hm
        print(f"\nSTATS n={hm['n']} head_ms mean {hm['head_ms_mean']:.2f} min {hm['head_ms_min']:.2f} max {hm['head_ms_max']:.2f} | "
              f"prefill_ms mean {hm['prefill_ms_mean']:.1f} | total_ms mean {hm['total_ms_mean']:.1f}")
    REPORT.write_text(json.dumps(report, indent=1))
    tok.close()


if __name__ == "__main__":
    main()
