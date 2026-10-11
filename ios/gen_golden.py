#!/usr/bin/env python3
"""Golden prompts/token ids from the Python reference, to check the Swift port. usage: gen_golden.py MODEL_DIR OUT.json"""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
import jeff_tetris as T
from jeff_coreai import decision_messages, load_decision_config
from transformers import AutoTokenizer

model = Path(sys.argv[1]); dec = load_decision_config(model)
tok = AutoTokenizer.from_pretrained(str(model))
rng = random.Random(7); board = T.empty_board(); names = list(T.PIECES); out = []
for i in range(12):
    piece = names[rng.randrange(7)]
    opts = T.placements(board, piece)
    if not opts: break
    row = T.tetris_row(board, piece, T.oracle_key(board, piece), opts)
    q = {"type": "choice", "instructions": row["instructions"], "criteria": row["options"]}
    msgs = decision_messages({"state": row["state"], "question": q}, dec["codes"], dec["prompt_layout"])
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out.append({"board": T.board_text(board), "piece": piece, "keys": list(row["options"]), "oracle": row["label"],
                "text": text, "ids": ids})
    board = T.apply_key(board, piece, row["label"])
Path(sys.argv[2]).write_text(json.dumps(out))
print(len(out), "cases; token counts", [len(c["ids"]) for c in out])
