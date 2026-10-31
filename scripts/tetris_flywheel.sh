#!/usr/bin/env bash
# One turn of the Tetris data flywheel: app games -> relabel with a stronger teacher -> continue the adapter -> evaluate.
#
#   BASE_MODEL=~/Models/jeff/jeff-base-v1.3 INIT_ADAPTER=~/Models/jeff-tetris-mps/adapter \
#   PRIOR_TRAIN=~/Models/jeff-tetris-mps/train.jsonl PY=python \
#   scripts/tetris_flywheel.sh ARCHIVE_DIR RUN_DIR [distill|awr]
#
# Needs a training Python (PyTorch). It stops after printing the evaluation command; gameplay evaluation uses the ANE server
# (projects/tetris_training/README.md, "Evaluate on the demo rules"), so run it when the Mac's ANE is free.
set -euo pipefail
ARCHIVE="${1:?folder of exported games (.jsonl or .jsonl.gz)}"
RUN="${2:?new run directory}"
MODE="${3:-distill}"
MIN_GAMES="${MIN_GAMES:-200}"
MIX="${MIX_PRIOR:-1}"          # how many prior-training rows per new row (guards against forgetting); 0 = new rows only
PY="${PY:-python}"
BASE_MODEL="${BASE_MODEL:?base checkpoint (jeff-base)}"
INIT_ADAPTER="${INIT_ADAPTER:?adapter to continue}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -e "$RUN" ] && { echo "$RUN exists; use a new directory"; exit 1; }
mkdir -p "$RUN"

echo "== 1. relabel ($MODE) =="
"$PY" -m projects.tetris_training.relabel "$ARCHIVE" -o "$RUN/data" --mode "$MODE" --teacher search2 --workers "${WORKERS:-8}"
GAMES=$("$PY" -c "import json;print(json.load(open('$RUN/data/report.json'))['games'])")
if [ "$GAMES" -lt "$MIN_GAMES" ]; then echo "only $GAMES games (< MIN_GAMES=$MIN_GAMES): collect more before training"; exit 0; fi

echo "== 2. mix with prior training rows =="
"$PY" - <<PYEOF
import json, random
new = [l for l in open("$RUN/data/train.jsonl")]
prior = [l for l in open("${PRIOR_TRAIN:-/dev/null}")] if "${PRIOR_TRAIN:-}" else []
random.Random(0).shuffle(prior)
rows = new + prior[: len(new) * int("$MIX")]
random.Random(1).shuffle(rows)
open("$RUN/train_mix.jsonl", "w").writelines(rows)
print(f"{len(new)} new + {len(rows) - len(new)} prior rows")
PYEOF

echo "== 3. continue the adapter =="
"$PY" forge.py jeff-train-lora --model "$BASE_MODEL" --task tetris --dataset "$RUN/train_mix.jsonl" \
  --heldout-dataset "$RUN/data/heldout.jsonl" --init-adapter "$INIT_ADAPTER" --output "$RUN/model" \
  --device "${DEVICE:-mps}" --rank 16 --alpha 32 --epochs 1 --batch 4 --lr "${LR:-5e-5}" --readout-lr 5e-6

echo "== 4. evaluate =="
echo "Compile $RUN/model/merged, serve it on a spare port, then compare with the previous adapter on the reserved seeds:"
echo "  python -m projects.tetris_training.play --help"
