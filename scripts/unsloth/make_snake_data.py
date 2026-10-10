"""Regenerates the Snake decision dataset (train.jsonl, heldout.jsonl) that the Unsloth notebook and scripts in this folder
train and evaluate on (Qwen3.5-0.8B).

  train.jsonl   = generate_snake_rows(2048, seed=0)
  heldout.jsonl = generate_snake_rows(256,  seed=1), with every board that is also in train.jsonl replaced in place by the
                  next unused board of the same seed-1 stream (so validation shares no state with training, the row
                  count stays 256, and every other row keeps its index; the 5 Colab parity rows are rows 0-4)

The first version of this file kept the 5 of 256 validation boards that also appear in the 2,048 training rows; the
published Colab/ANE numbers were measured on that earlier split. Run from the repo root:
  python scripts/unsloth/make_snake_data.py OUT_DIR
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coreai.jeff_snake import generate_snake_rows, row_key  # noqa: E402

N_TRAIN, N_HELDOUT, SEED_TRAIN, SEED_HELDOUT = 2048, 256, 0, 1


def build_splits():
    """(train_rows, heldout_rows): deterministic, 2048 and 256 rows, no board in both."""
    train = generate_snake_rows(N_TRAIN, SEED_TRAIN)
    banned = {row_key(r) for r in train}
    for spare in (64, 256, 1024):  # the seed-1 stream continues past row 256; draw replacements from there
        pool = generate_snake_rows(N_HELDOUT + spare, SEED_HELDOUT)
        held, extra = pool[:N_HELDOUT], (r for r in pool[N_HELDOUT:] if row_key(r) not in banned)
        try:
            held = [r if row_key(r) not in banned else next(extra) for r in held]
        except StopIteration:
            continue
        assert len(held) == N_HELDOUT and not banned & {row_key(r) for r in held}
        return train, held
    raise RuntimeError("could not find enough validation boards outside the training set")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    train, held = build_splits()
    for name, rows in (("train.jsonl", train), ("heldout.jsonl", held)):
        with open(out / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
