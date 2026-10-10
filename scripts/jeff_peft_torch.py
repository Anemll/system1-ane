#!/usr/bin/env python3
"""Score a few published-adapter rows on the merged PyTorch checkpoint.

``jeff-train`` cannot start on this Mac (it calls CUDA before the first step), and
``peft`` is not required here: ``merge_peft_adapter`` already folded
``W += (alpha / rank) B A`` into the checkpoint ``jeff-convert`` reads. These
probabilities are that merged model, with the adapter's fitted temperature applied
only at the softmax, in the same shape ``scripts/jeff_lora_parity.py`` expects.

    python scripts/jeff_peft_torch.py \\
        --adapter triage=~/Models/jeff/adapters/jeff-adapter-triage \\
        --merged ~/Models/jeff-published/triage \\
        --output ~/Models/jeff-published/triage/parity_rows.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "scripts"))

from jeff_coreai import load_decision_config, prompt_ids, question_options  # noqa: E402
from jeff_lora_train import collate, forward_logits  # noqa: E402


def choice_rows(name: str, adapter: Path, n_test: int) -> list[dict]:
    """The published example's choice question, then the first choice rows of test.jsonl."""
    example = json.loads((adapter / "example.json").read_text())
    rows = []
    for key, question in example["questions"].items():
        if question.get("type") != "choice":
            continue
        rows.append({
            "id": f"{name}-example-{key}",
            "state": example["state"],
            "question": question,
        })
        break
    if not rows:
        raise ValueError(f"{adapter} example.json has no choice question")
    taken = 0
    with (adapter / "test.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("question", {}).get("type") != "choice":
                continue
            rows.append(row)
            taken += 1
            if taken >= n_test:
                break
    if taken < n_test:
        raise ValueError(f"{adapter} test.jsonl has {taken} choice rows, wanted {n_test}")
    return rows


def score(merged: Path, rows: list[dict], device: str) -> list[dict]:
    decision = load_decision_config(merged)
    temperature = float(decision["temperature"])
    tokenizer = AutoTokenizer.from_pretrained(str(merged))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    encoded = []
    for row in rows:
        keys, _descriptions = question_options(row["question"])
        ids = prompt_ids(merged, row, decision, tokenizer)
        gold = keys.index(row["label"]) if row.get("label") in keys else None
        # collate stores a label tensor this scorer never reads. The example row has no gold label.
        encoded.append({"id": row.get("id"), "ids": ids, "n": len(keys), "label": 0 if gold is None else gold,
                        "gold": gold, "keys": keys})
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    backbone = AutoModel.from_pretrained(str(merged), dtype=dtype, attn_implementation="sdpa")
    backbone.to(device)
    backbone.eval()
    weight = load_file(str(merged / "readout.safetensors"))["weight"]
    readout = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=False)
    with torch.no_grad():
        readout.weight.copy_(weight.float())
    readout.to(device)
    out = []
    with torch.no_grad():
        for row in encoded:
            ids, mask, index, _labels = collate([row], pad_id, device)
            logits = forward_logits(backbone, readout, ids, mask, index, device)[0, :row["n"]]
            prob = torch.softmax(logits / temperature, dim=-1).detach().float().cpu()
            out.append({
                "id": row["id"],
                "ids": row["ids"],
                "n": row["n"],
                "label": row["gold"],
                "keys": row["keys"],
                "temperature": temperature,
                "probabilities": [float(value) for value in prob],
            })
            best = int(prob.argmax())
            print(f"  {row['id']}  tokens {len(row['ids'])}  "
                  f"pt {row['keys'][best]} {float(prob[best]):.3f}  label {row['gold']}", flush=True)
    del backbone, readout
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", action="append", required=True, metavar="NAME=DIR",
                   help="published PEFT directory that contains example.json and test.jsonl")
    p.add_argument("--merged", type=Path, required=True, help="directory of merged checkpoints, one subdirectory per name")
    p.add_argument("--output", type=Path, required=True, help="directory that receives <name>/parity_rows.json")
    p.add_argument("--rows", type=int, default=2, help="choice rows from test.jsonl, after the example")
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    a = p.parse_args(argv)
    if a.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = a.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested and is not available")
    written = []
    for item in a.adapter:
        if "=" not in item:
            p.error("--adapter must be name=directory")
        name, raw = item.split("=", 1)
        adapter = Path(raw).expanduser().resolve()
        merged = (a.merged.expanduser().resolve() / name)
        dest = a.output.expanduser().resolve() / name / "parity_rows.json"
        print(f"{name}: scoring on {device}", flush=True)
        rows = score(merged, choice_rows(name, adapter, a.rows), device)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rows))
        written.append(str(dest))
    print(json.dumps({"device": device, "wrote": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
