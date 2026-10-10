"""Regenerates the Snake decision dataset (train.jsonl, heldout.jsonl) that the Unsloth notebook and scripts in this folder
train and evaluate on (Qwen3.5-0.8B).

NEW file (not copied from the original run folder). Verified on 2026-10-09 to reproduce
snake_data/train.jsonl and snake_data/heldout.jsonl byte-for-byte:
  train.jsonl   = generate_snake_rows(2048, seed=0)
  heldout.jsonl = generate_snake_rows(256,  seed=1)
Run from the repo root:  python scripts/unsloth/make_snake_data.py OUT_DIR
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coreai.jeff_snake import generate_snake_rows  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
for name, n, seed in (("train.jsonl", 2048, 0), ("heldout.jsonl", 256, 1)):
    with open(out / name, "w") as f:
        for r in generate_snake_rows(n, seed):
            f.write(json.dumps(r) + "\n")
    print("wrote", out / name)
