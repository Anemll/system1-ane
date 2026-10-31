#!/usr/bin/env python3
"""Train a sample LoRA on jeff-base and merge it into a new checkpoint.

The default task is Snake, labeled by a shortest-path oracle and rendered with the
same prompt the browser demo sends. Any other task is a JSONL file of rows
``{state, options, label, instructions}``, or ``--task tetris``. Loss matches
upstream ``jeff-train``: cross-entropy on the masked readout logits, with no
temperature inside the loss. ``JeffCoreAI.decide`` still divides those logits by
``decision_config`` temperature before the softmax. Argmax does not change.

LoRA and the readout use different learning rates, as in
``jeff-train --lora-rank 16 --lr 2e-4 --readout-lr 5e-6`` (alpha defaults to
``2 * rank``).

    python forge.py jeff-train-lora \\
        --model ~/Models/jeff/jeff-base-v1.3 \\
        --output ~/Models/jeff-snake

That writes ``adapter/`` (the factors), ``merged/`` (a full checkpoint ``jeff-convert``
accepts), and ``report.json``. Deploy the merged tree with the usual convert and
compile into its own Core AI directory. The ANE package has the LoRA already folded
into the weights.

``--device auto`` is CUDA, then MPS, then CPU. CUDA and MPS train in bfloat16.
An ANE compile does not move this job off the GPU. The first two steps are timed
and then discarded so the run still starts from the base readout.

``--init-adapter DIR`` continues from a saved ``adapter/`` (LoRA factors plus
``readout.safetensors``) instead of a zero LoRA and the base readout. The "base"
numbers in the report are then the starting adapter's. ``--heldout-dataset``
evaluates on a fixed JSONL (every ``--dataset`` row trains). ``--eval-only``
loads, prints held-out accuracy, and exits without writing anything.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import load_decision_config, prompt_ids  # noqa: E402
from jeff_lora import attach_lora, save_lora, save_merged_checkpoint, trainable_parameters  # noqa: E402
from jeff_snake import as_decision, generate_snake_rows, opening_states, play_game, snake_row, summarize_games  # noqa: E402
from jeff_tetris import generate_tetris_rows  # noqa: E402


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested and is not available")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested and is not available")
    return requested


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_init_adapter(directory: Path, layers, readout: nn.Linear) -> None:
    """Copy saved LoRA factors and the readout into freshly attached layers (same rank and targets)."""
    factors = load_file(str(directory / "adapter_model.safetensors"))
    expected = set()
    with torch.no_grad():
        for layer in layers:
            stem = layer.key.removesuffix(".weight")
            a, b = factors[f"{stem}.lora_A"], factors[f"{stem}.lora_B"]
            if a.shape != layer.A.shape or b.shape != layer.B.shape:
                raise ValueError(f"{stem}: adapter shape {tuple(a.shape)} does not match rank {layer.rank}")
            layer.A.copy_(a.float())
            layer.B.copy_(b.float())
            expected.update((f"{stem}.lora_A", f"{stem}.lora_B"))
        extra = set(factors) - expected
        if extra:
            raise ValueError(f"{directory}: {len(extra)} factors match no attached layer, e.g. {sorted(extra)[:2]}")
        readout.weight.copy_(load_file(str(directory / "readout.safetensors"))["weight"].float())


def load_rows(path: Path | None, train_n: int, heldout_n: int, seed: int, task: str) -> tuple[list[dict], list[dict], str]:
    if path is None and task == "tetris":
        train = generate_tetris_rows(train_n, seed)
        held = generate_tetris_rows(heldout_n, seed + 1)
        return train, held, "tetris"
    if path is None:
        train = generate_snake_rows(train_n, seed)
        held = generate_snake_rows(heldout_n, seed + 1)
        train_keys = {json.dumps(row["state"]["latest"], sort_keys=True) for row in train}
        held = [row for row in held if json.dumps(row["state"]["latest"], sort_keys=True) not in train_keys]
        if len(held) < heldout_n:
            extra = generate_snake_rows(heldout_n, seed + 2)
            for row in extra:
                key = json.dumps(row["state"]["latest"], sort_keys=True)
                if key not in train_keys:
                    held.append(row)
                if len(held) >= heldout_n:
                    break
        return train, held[:heldout_n], "snake"
    rows = read_jsonl(path)
    if len(rows) < 2:
        raise ValueError("--dataset needs at least two rows")
    cut = max(1, int(round(len(rows) * 0.8)))
    if cut >= len(rows):
        cut = len(rows) - 1
    return rows[:cut], rows[cut:], path.stem


def encode(model: Path, rows: list[dict], decision: dict, tokenizer) -> list[dict]:
    encoded = []
    for row in rows:
        decision_row, index = as_decision(row)
        ids = prompt_ids(model, decision_row, decision, tokenizer)
        # optional per-row weight (relabel.py: regret weighting for distillation, advantage weighting for AWR); 1.0 when absent
        weight = float(row.get("weight", 1.0)) if isinstance(row, dict) else 1.0
        encoded.append({"ids": ids, "label": index, "n": len(decision_row["question"]["criteria"]), "weight": weight})
    return encoded


def collate(batch: list[dict], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad so content tokens keep positions 0..L-1, matching an unpadded Jeff prompt. The label is read at L-1."""
    width = max(len(row["ids"]) for row in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    index = torch.empty(len(batch), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, row in enumerate(batch):
        length = len(row["ids"])
        ids[i, :length] = torch.tensor(row["ids"], dtype=torch.long)
        mask[i, :length] = 1
        index[i] = length - 1
        labels[i] = row["label"]
    return ids.to(device), mask.to(device), index.to(device), labels.to(device)


def forward_logits(model, readout: nn.Linear, ids, mask, index, device: str) -> torch.Tensor:
    autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16) if device in ("mps", "cuda")
                else contextlib.nullcontext())
    with autocast:
        hidden = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    picked = hidden[torch.arange(hidden.shape[0], device=hidden.device), index]
    return readout(picked.float())


def masked_logits(logits: torch.Tensor, batch: list[dict]) -> torch.Tensor:
    """Jeff's training head: one logit per answer code, unused codes set to -1e9.

    The loss is cross-entropy on these logits with no temperature, matching
    ``jeff-train``. Serving divides by ``decision_config`` temperature afterwards.
    The label index is the option's position, which is also its answer-code index.
    """
    out = logits.clone()
    width = out.shape[1]
    for i, row in enumerate(batch):
        n = row["n"]
        if n < width:
            out[i, n:] = -1e9
    return out


def timed_warmup(backbone, readout, layers, train_enc, batch, pad_id, device, make_optimizer, stop_at=None) -> dict:
    """Two timed optimizer steps (step 1 includes the MPS graph warmup), then put the weights back.

    ``stop_at`` is the ``--train-until`` deadline: past it no optimizer step runs (``run_epoch`` checks it before each step).
    """
    snapshot = ([layer.A.detach().clone() for layer in layers],
                [layer.B.detach().clone() for layer in layers],
                readout.weight.detach().clone())
    timed = run_epoch(backbone, readout, train_enc[:max(batch * 2, batch)], batch, pad_id, device, make_optimizer(),
                      stop_at=stop_at)
    with torch.no_grad():
        for layer, saved_a, saved_b in zip(layers, snapshot[0], snapshot[1]):
            layer.A.copy_(saved_a)
            layer.B.copy_(saved_b)
        readout.weight.copy_(snapshot[2])
    return timed


def run_epoch(model, readout, rows, batch_size, pad_id, device, optimizer=None,
              *, stop_at=None, step_callback=None) -> dict:
    train = optimizer is not None
    model.train(False)
    readout.train(train)
    total_loss = 0.0
    correct = 0
    seen = 0
    steps = 0
    step_seconds = []
    stopped_early = False
    order = torch.randperm(len(rows)).tolist() if train else list(range(len(rows)))
    for start in range(0, len(order), batch_size):
        if train and stop_at is not None and time.time() >= stop_at:
            stopped_early = True
            print("training deadline reached; saving completed work", flush=True)
            break
        batch = [rows[i] for i in order[start:start + batch_size]]
        ids, mask, index, labels = collate(batch, pad_id, device)
        began = time.perf_counter()
        if train:
            # Inference helpers may disable gradients when imported in this process.
            # Scope training's override so the caller's gradient mode is restored.
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                logits = masked_logits(forward_logits(model, readout, ids, mask, index, device), batch)
                per_row = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                w = torch.tensor([float(r.get("weight", 1.0)) for r in batch], dtype=per_row.dtype, device=per_row.device)
                loss = (per_row * w).sum() / w.sum().clamp_min(1e-8)   # weights are 1.0 unless the rows carry them
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 1.0)
                optimizer.step()
                synchronize(device)
        else:
            with torch.no_grad():
                logits = masked_logits(forward_logits(model, readout, ids, mask, index, device), batch)
                loss = torch.nn.functional.cross_entropy(logits, labels)
        total_loss += float(loss.detach()) * len(batch)
        correct += int((logits.detach().argmax(-1) == labels).sum())
        # Scalar reads synchronize MPS/CUDA. Include them in inference timing too.
        elapsed = time.perf_counter() - began
        step_seconds.append(elapsed)
        seen += len(batch)
        steps += 1
        if (train and (steps <= 2 or steps % 8 == 0)) or (not train and (steps == 1 or steps % 16 == 0)):
            print(f"    {'train' if train else 'eval'} step {steps}/{(len(rows) + batch_size - 1) // batch_size}  "
                  f"{elapsed:.2f}s  loss {total_loss / seen:.3f}  acc {correct / seen:.3f}", flush=True)
        if train and step_callback is not None:
            step_callback(steps, seen)
    steady = step_seconds[1:] or step_seconds
    return {"loss": total_loss / max(1, seen), "accuracy": correct / max(1, seen), "n": seen, "steps": steps,
            "seconds_per_step": round(sum(steady) / max(1, len(steady)), 3),
            "stopped_early": stopped_early}


def hint_rows(rows: list[dict]) -> list[dict]:
    """Same labels, with food direction and safe moves added in front of ``latest``."""
    out = []
    for row in rows:
        if "snake" not in row or "food" not in row:
            raise ValueError("hint prompts need the snake and food fields written by the snake generator")
        out.append(snake_row(row["snake"], row["food"], row["label"], hint=True))
    return out


def play_split(model, readout, tokenizer, decision, model_path, device, temperature, states, pad_id, max_steps: int) -> dict:
    def choose(snake, food):
        row = snake_row(snake, food, "up")
        encoded = encode(model_path, [row], decision, tokenizer)
        ids, mask, index, _labels = collate(encoded, pad_id, device)
        with torch.no_grad():
            logits = masked_logits(forward_logits(model, readout, ids, mask, index, device), encoded)
        names = list(row["options"])
        return names[int(logits.argmax(-1)[0])]

    games = []
    for i, (snake, food) in enumerate(states):
        games.append(play_game(choose, snake, food, random.Random(10_000 + i), max_steps=max_steps))
    return summarize_games(games)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def atomic_directory(dest: Path, writer) -> None:
    """Publish a complete checkpoint directory, never a partly written one."""
    temporary = dest.with_name(dest.name + ".partial")
    if dest.exists():
        raise FileExistsError(dest)
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        writer(temporary)
        temporary.replace(dest)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def atomic_report(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="jeff-base checkpoint directory")
    p.add_argument("--output", type=Path, required=True, help="new directory for adapter/, merged/, and report.json")
    p.add_argument("--dataset", type=Path, help="JSONL of {state, options, label}. Default: synthetic snake or tetris")
    p.add_argument("--task", choices=("snake", "tetris"), default="snake")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--train", type=int, default=256, help="synthetic snake rows (ignored with --dataset)")
    p.add_argument("--heldout", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--eval-batch", type=int, help="evaluation batch size; defaults to --batch")
    p.add_argument("--eval-train-rows", type=int, default=0,
                   help="fixed sample for train-accuracy checks only; 0 evaluates all training rows")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="save recovery LoRA/head every N training steps; retain latest two, no optimizer state")
    p.add_argument("--train-until", help="stop optimization at this timezone-aware ISO timestamp, then evaluate/save")
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    p.add_argument("--readout-lr", type=float, default=5e-6, dest="readout_lr")
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--play", type=int, default=0,
                   help="PyTorch self-play games per model (each move is a full forward). 0 skips them; use scripts/jeff_snake_eval.py on the server")
    p.add_argument("--max-steps", type=int, default=48, help="Snake self-play move cap (--play only); does not limit training")
    p.add_argument("--skip-merge", action="store_true")
    p.add_argument("--init-adapter", type=Path, dest="init_adapter",
                   help="continue from this adapter/ directory (LoRA factors and readout.safetensors)")
    p.add_argument("--heldout-dataset", type=Path, dest="heldout_dataset",
                   help="fixed held-out JSONL; with --dataset every --dataset row trains")
    p.add_argument("--eval-only", action="store_true", dest="eval_only",
                   help="load (and --init-adapter), print held-out accuracy, and exit")
    a = p.parse_args(argv)
    if a.rank < 1 or a.alpha <= 0 or a.batch < 1 or a.epochs < 1:
        p.error("--rank, --alpha, --batch, and --epochs must be positive")
    if (a.eval_batch is not None and a.eval_batch < 1) or a.eval_train_rows < 0:
        p.error("--eval-batch must be positive and --eval-train-rows must be nonnegative")
    if a.checkpoint_every < 0:
        p.error("--checkpoint-every must be nonnegative")
    stop_at = None
    if a.train_until:
        try:
            deadline = datetime.fromisoformat(a.train_until)
            if deadline.tzinfo is None:
                raise ValueError("timezone required")
            stop_at = deadline.timestamp()
        except ValueError as exc:
            p.error(f"--train-until needs a timezone-aware ISO timestamp: {exc}")
    output = a.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not a.eval_only:
        p.error(f"Use a new or empty --output directory: {output}")
    model_path = a.model.expanduser().resolve()
    device = pick_device(a.device)
    print(f"jeff-train-lora: device {device}", flush=True)

    torch.manual_seed(a.seed)
    decision = load_decision_config(model_path)
    temperature = float(decision["temperature"])
    train_rows, held_rows, task = load_rows(a.dataset, a.train, a.heldout, a.seed, a.task)
    if a.heldout_dataset is not None:
        if a.dataset is not None:
            train_rows = read_jsonl(a.dataset)
        held_rows = read_jsonl(a.heldout_dataset)
    if a.eval_only:
        train_rows = []
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    print(f"tokenizing {len(train_rows)} train and {len(held_rows)} held-out", flush=True)
    train_enc = encode(model_path, train_rows, decision, tokenizer)
    held_enc = encode(model_path, held_rows, decision, tokenizer)
    hint_enc = encode(model_path, hint_rows(held_rows), decision, tokenizer) if task == "snake" else None
    eval_batch = a.eval_batch or a.batch
    train_eval = (random.Random(a.seed).sample(train_enc, min(a.eval_train_rows, len(train_enc)))
                  if a.eval_train_rows else train_enc)

    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    print(f"loading {model_path} ({dtype})", flush=True)
    backbone = AutoModel.from_pretrained(str(model_path), dtype=dtype, attn_implementation="sdpa")
    backbone.to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    layers = attach_lora(backbone, a.rank, a.alpha)
    readout_weight = load_file(str(model_path / "readout.safetensors"))["weight"]
    hidden = int(readout_weight.shape[1])
    readout = nn.Linear(hidden, int(readout_weight.shape[0]), bias=False)
    with torch.no_grad():
        readout.weight.copy_(readout_weight.float())
    readout.to(device)
    if a.init_adapter is not None:
        init_dir = a.init_adapter.expanduser().resolve()
        load_init_adapter(init_dir, layers, readout)
        print(f"continuing from {init_dir}", flush=True)
    if a.eval_only:
        stats = run_epoch(backbone, readout, held_enc, eval_batch, pad_id, device)
        print(f"  eval heldout: acc {stats['accuracy']:.4f}  loss {stats['loss']:.4f}  n {stats['n']}  "
              f"{stats['seconds_per_step']:.3f} s/step", flush=True)
        return 0
    params = trainable_parameters(layers, readout)
    lora_params = [layer.A for layer in layers] + [layer.B for layer in layers]

    def make_optimizer():
        return torch.optim.AdamW(
            [{"params": lora_params, "lr": a.lr}, {"params": [readout.weight], "lr": a.readout_lr}],
            weight_decay=0.0)

    n_train = sum(parameter.numel() for parameter in params)
    print(f"LoRA layers {len(layers)}  trainable {n_train}  rank {a.rank}  "
          f"lora lr {a.lr:g}  readout lr {a.readout_lr:g}  temperature {temperature:.4f} (inference only)",
          flush=True)
    timed = timed_warmup(backbone, readout, layers, train_enc, a.batch, pad_id, device, make_optimizer, stop_at)
    print(f"timed {device} steps: {timed['seconds_per_step']:.3f} s/step "
          f"(steady, batch {a.batch}; first step is the warmup)", flush=True)
    optimizer = make_optimizer()

    started = time.perf_counter()
    report = {
        "task": task,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "rank": a.rank,
        "alpha": a.alpha,
        "lr": a.lr,
        "readout_lr": a.readout_lr,
        "epochs": a.epochs,
        "batch": a.batch,
        "eval_batch": eval_batch,
        "training_rows": len(train_enc),
        "train_accuracy_rows": len(train_eval),
        "train_until": a.train_until,
        "checkpoint_every": a.checkpoint_every,
        "temperature": temperature,
        "loss": "cross_entropy on masked readout logits, temperature applied only at inference",
        "trainable": n_train,
        "lora_layers": len(layers),
        "timed_seconds_per_step": timed["seconds_per_step"],
        "init_adapter": str(a.init_adapter.expanduser().resolve()) if a.init_adapter else None,
        "base": {},
        "adapter": {},
    }
    print("base accuracy", flush=True)
    report["base"]["train"] = run_epoch(backbone, readout, train_eval, eval_batch, pad_id, device)
    report["base"]["heldout"] = run_epoch(backbone, readout, held_enc, eval_batch, pad_id, device)
    if hint_enc is not None:
        report["base"]["heldout_hint"] = run_epoch(backbone, readout, hint_enc, eval_batch, pad_id, device)
    _print_split("base", report["base"])

    states = opening_states(a.play, a.seed) if task == "snake" and a.play else []
    if states:
        print(f"base self-play x{len(states)}", flush=True)
        report["base"]["games"] = play_split(
            backbone, readout, tokenizer, decision, model_path, device, temperature, states, pad_id, a.max_steps)
        print(f"  {report['base']['games']}", flush=True)

    print("training", flush=True)
    history = []
    def checkpoint(epoch, step, seen):
        if not a.checkpoint_every or step % a.checkpoint_every:
            return
        parent = output / "checkpoints"
        dest = parent / f"epoch-{epoch + 1:03d}-step-{step:06d}"
        atomic_directory(dest, lambda path: save_lora(path, layers, readout, {
            "base_checkpoint": str(model_path), "task": task,
            "epoch": epoch + 1, "step": step, "rows_seen_in_epoch": seen,
            "optimizer_state_saved": False,
        }))
        pointer = output / "latest-recovery.json"
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps({"adapter": str(dest), "epoch": epoch + 1,
                                         "step": step, "rows_seen_in_epoch": seen}) + "\n")
        temporary.replace(pointer)
        for old in sorted(parent.glob("epoch-*-step-*"))[:-2]:
            if old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
        print(f"recovery checkpoint: {dest}", flush=True)
    for epoch in range(a.epochs):
        t0 = time.perf_counter()
        stats = run_epoch(backbone, readout, train_enc, a.batch, pad_id, device, optimizer,
                          stop_at=stop_at, step_callback=lambda step, seen: checkpoint(epoch, step, seen))
        elapsed = time.perf_counter() - t0
        stats["seconds"] = round(elapsed, 1)
        history.append(stats)
        print(f"  epoch {epoch + 1}/{a.epochs}  loss {stats['loss']:.3f}  acc {stats['accuracy']:.3f}  {elapsed:.0f}s",
              flush=True)
        if stats["stopped_early"]:
            break
    report["history"] = history
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "heldout.jsonl", held_rows)
    atomic_directory(output / "adapter", lambda path: save_lora(path, layers, readout, {
        "base_checkpoint": str(model_path),
        "task": task,
        "temperature": temperature,
        "targets": sorted({layer.key.rsplit(".", 2)[-2] for layer in layers}),
    }))
    # A few held-out prompts for the ANE parity check (codes in option order).
    parity = []
    sample = held_enc[:4]
    with torch.no_grad():
        ids, mask, index, _labels = collate(sample, pad_id, device)
        logits = masked_logits(forward_logits(backbone, readout, ids, mask, index, device), sample)
    for i, encoded in enumerate(sample):
        n = encoded["n"]
        prob = torch.softmax(logits[i, :n] / temperature, dim=-1).detach().cpu()
        parity.append({"ids": encoded["ids"], "n": n, "label": encoded["label"],
                       "probabilities": [float(value) for value in prob]})
    atomic_report(output / "parity_rows.json", parity)
    report["save_state"] = "adapter_saved"
    report["seconds"] = round(time.perf_counter() - started, 1)
    atomic_report(output / "report.json", report)
    if not a.skip_merge:
        print(f"writing merged checkpoint to {output / 'merged'}", flush=True)
        atomic_directory(output / "merged", lambda path: save_merged_checkpoint(model_path, path, layers, readout))
    report["save_state"] = "weights_saved" if not a.skip_merge else "adapter_saved"
    report["seconds"] = round(time.perf_counter() - started, 1)
    atomic_report(output / "report.json", report)

    report["adapter"]["train"] = run_epoch(backbone, readout, train_eval, eval_batch, pad_id, device)
    report["adapter"]["heldout"] = run_epoch(backbone, readout, held_enc, eval_batch, pad_id, device)
    _print_split("adapter", report["adapter"])
    if states:
        print(f"adapter self-play x{len(states)}", flush=True)
        report["adapter"]["games"] = play_split(
            backbone, readout, tokenizer, decision, model_path, device, temperature, states, pad_id, a.max_steps)
        print(f"  {report['adapter']['games']}", flush=True)
    report["save_state"] = "complete"
    report["seconds"] = round(time.perf_counter() - started, 1)
    atomic_report(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), "seconds": report["seconds"]}, indent=2))
    return 0


def _print_split(title: str, block: dict) -> None:
    for name in ("train", "heldout", "heldout_hint"):
        if name in block:
            stats = block[name]
            print(f"  {title} {name}: acc {stats['accuracy']:.3f}  loss {stats['loss']:.3f}  n {stats['n']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
