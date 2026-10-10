# This script trains a decision model with Unsloth on Qwen3.5-0.8B using the Snake decision dataset, evaluates it on held-out rows, and saves the LoRA adapter and decision head for later ANE conversion.
# Flow: data -> train -> evaluate/calibrate -> save adapter_head + fp16 LoRA + parity file. Conversion to Core AI is a separate step (see README.md).
# Adapted from Unsloth's "Train your own Decision Model" guide (FastDecisionModel + DecisionTrainer, Clef-style head):
# https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth#train-with-code
# Changes vs the guide: Snake data converted to the decision format, held-out file as eval/calibration set,
# fp16 on T4, 3 epochs, plus ANE-port exports (fp16 LoRA, head, parity).
# A100 variant: base loaded in bf16 (load_in_4bit=False) instead of nf4, with fp32 compute.
# dtype=torch.float16 is requested on purpose: for Qwen3.5 Unsloth then takes its FORCE_FLOAT32 path
# (weights stored bf16, no autocast, fp32 compute) - the same compute path the T4 run uses, minus 4-bit.
# Usage: python run_decision_a100.py   (TAG only names the output folder, default "qwen")
import os, json, time, shutil, tarfile, subprocess
from pathlib import Path

TAG = os.environ.get("TAG", "qwen")
EPOCHS = float(os.environ.get("EPOCHS", "3"))
W = Path("/content/snake"); OUT = W / "out" / TAG; OUT.mkdir(parents=True, exist_ok=True)
DRIVE = Path("/content/drive/MyDrive/snake_decision")
GPU = os.environ.get("GPU", "a100")
LOG = {"tag": TAG, "gpu": GPU, "t0": time.time()}

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{TAG}] {msg}"
    print(line, flush=True)
    with open(W / f"run_{TAG}.log", "a") as f: f.write(line + "\n")

def dump():
    (OUT / "metrics.json").write_text(json.dumps(LOG, indent=2, default=str))

# ---------------------------------------------------------------- 1. base model (stock)
MODEL = "unsloth/Qwen3.5-0.8B"
log(f"base model: {MODEL}")

# ---------------------------------------------------------------- 2. data -> decision format
def convert(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        rows.append({
            "state": json.dumps(r["state"]),
            "questions": json.dumps({"move": {"type": "choice", "instructions": r["instructions"],
                                              "criteria": r["options"]}}),
            "gold": json.dumps({"move": {"type": "choice", "label": r["label"]}}),
        })
    return rows
train_rows, held_rows = convert(W / "train.jsonl"), convert(W / "heldout.jsonl")
log(f"rows: train {len(train_rows)} heldout {len(held_rows)}")

# ---------------------------------------------------------------- 3. Unsloth guide code
from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
from transformers import TrainingArguments
import torch

model, tokenizer = FastDecisionModel.from_pretrained(
    model_name = MODEL, max_seq_length = 1024, load_in_4bit = False,
    dtype = torch.float16 if os.environ.get("COMPUTE", "fp32") == "fp32" else torch.bfloat16)
LOG["loaded"] = True
from collections import Counter
LOG["precision"] = {"forced_float32": bool(getattr(model, "_unsloth_forced_float32", False)),
    "UNSLOTH_FORCE_FLOAT32": os.environ.get("UNSLOTH_FORCE_FLOAT32"),
    "param_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
    "gpu_name": torch.cuda.get_device_name(0)}
log(f"precision: {LOG['precision']}")
model = FastDecisionModel.get_peft_model(
    model, r = 16, lora_alpha = 16, lora_dropout = 0,
    use_gradient_checkpointing = "unsloth", random_state = 3407)

train_items, rep = FastDecisionModel.build_dataset(train_rows, tokenizer, model)
eval_items, rep2 = FastDecisionModel.build_dataset(held_rows, tokenizer, model)
log(f"build_dataset train {rep} heldout {rep2}; max len {max(len(i['input_ids']) for i in train_items)}")
LOG["dataset"] = {"train": rep, "heldout": rep2}

LOG["before"] = FastDecisionModel.evaluate(model, tokenizer, eval_items)
log(f"held-out BEFORE: {LOG['before']}"); dump()

trainer = DecisionTrainer(
    model = model, processing_class = tokenizer,
    train_dataset = train_items, eval_dataset = eval_items,
    args = TrainingArguments(
        per_device_train_batch_size = 8, gradient_accumulation_steps = 4,
        num_train_epochs = EPOCHS, learning_rate = 2e-4, lr_scheduler_type = "cosine",
        warmup_steps = 10, weight_decay = 0.01,
        bf16 = is_bfloat16_supported(), fp16 = not is_bfloat16_supported(),
        eval_strategy = os.environ.get("EVAL_STRATEGY", "no"), logging_steps = 5, output_dir = str(W / "outputs" / TAG),
        save_strategy = "no", report_to = "none", seed = 3407))
t = time.time()
LOG["trainer_precision"] = {"bf16": trainer.args.bf16, "fp16": trainer.args.fp16}
log(f"trainer precision after Clef switch: {LOG['trainer_precision']}"); dump()
trainer.train()
LOG["train_seconds"] = time.time() - t
LOG["s_per_step"] = LOG["train_seconds"] / max(1, trainer.state.global_step)
LOG["log_history"] = trainer.state.log_history
log(f"train time {LOG['train_seconds']/60:.1f} min, {LOG['s_per_step']:.2f} s/step over {trainer.state.global_step} steps")

LOG["after"] = FastDecisionModel.evaluate(model, tokenizer, eval_items)
log(f"held-out AFTER: {LOG['after']}")
LOG["calibrated"] = FastDecisionModel.calibrate(model, tokenizer, eval_items)
LOG["decision_config"] = {k: v for k, v in model.decision_config.items()}
log(f"held-out CALIBRATED: {LOG['calibrated']}"); dump()

# ---------------------------------------------------------------- 4. saves + ANE-port exports
model.save_pretrained(str(OUT / "adapter_head"))
# fp16 copy of the LoRA in standard PEFT layout (unmerged, not 4-bit packed)
from safetensors.torch import load_file, save_file
ad = OUT / "adapter_head" / "adapter_model.safetensors"
if ad.exists():
    sd = load_file(str(ad)); LOG["adapter_dtypes"] = sorted({str(v.dtype) for v in sd.values()})
    (OUT / "lora_fp16").mkdir(exist_ok=True)
    save_file({k: v.half().contiguous() for k, v in sd.items()}, str(OUT / "lora_fp16" / "adapter_model.safetensors"))
    shutil.copy(OUT / "adapter_head" / "adapter_config.json", OUT / "lora_fp16" / "adapter_config.json")
    LOG["adapter_config"] = json.loads((OUT / "adapter_head" / "adapter_config.json").read_text())
# parity: 5 held-out rows, exact token ids + raw head logits + calibrated probabilities
from unsloth.models import decision as D
FastDecisionModel.for_inference(model)
par_items = eval_items[:5]
logits, _ = D._decision_logits(model, tokenizer, par_items, None)
parity = []
tok = getattr(tokenizer, "tokenizer", tokenizer)
for it, z in zip(par_items, logits):
    src = it["source"]; rq = it["record"].questions[0]
    keys = list(rq.option_ids)  # head logit order = Clef option order (choice options sorted by key)
    pred = FastDecisionModel.predict(model, tokenizer, src["state"], src["questions"])["move"]
    ids = [int(x) for x in it["input_ids"]]
    parity.append({"row": it["row"], "input_ids": ids, "n_tokens": len(ids),
                   "rendered_text": tok.decode(ids),
                   "option_keys_in_logit_order": keys, "raw_logits": [float(x) for x in z],
                   "softmax_T1": [float(x) for x in torch.softmax(z.float(), -1)],
                   "calibrated_probabilities": pred.get("probabilities"), "answer": pred.get("answer"),
                   "gold": keys[it["labels"][0]],
                   "question_span": list(rq.question_span),
                   "option_spans": [list(map(int, sp)) for sp in rq.option_spans],
                   "global_token_index": len(ids) - 1})
(OUT / "parity.json").write_text(json.dumps({"base_model": MODEL, "head_temperature": model.decision_config.get("head_temperature"),
    "temperature": model.decision_config.get("temperature"), "rows": parity}, indent=1))
dump()

def pack(name, members):
    tgz = W / "out" / f"{name}.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        for m in members: tf.add(m, arcname=f"{name}/{Path(m).name}")
    if (Path("/content/drive/MyDrive")).is_dir():
        DRIVE.mkdir(parents=True, exist_ok=True); shutil.copy(tgz, DRIVE / tgz.name)
    log(f"packed {tgz} ({tgz.stat().st_size/1e6:.1f} MB)")

shutil.copy(__file__, OUT / "run_decision_a100.py")
pack(f"{TAG}_decision_adapter_head_{GPU}", [OUT / "adapter_head", OUT / "lora_fp16", OUT / "parity.json",
                                       OUT / "metrics.json", OUT / "run_decision_a100.py"])
if os.environ.get("MERGED", "1") == "1":
    try:
        model.save_pretrained_merged(str(OUT / "merged16"))
        pack(f"{TAG}_decision_merged16_{GPU}", [OUT / "merged16"])
    except Exception as e:
        LOG["merged_error"] = repr(e); log(f"merged save failed: {e!r}"); dump()
LOG["done"] = time.time(); dump(); log("DONE")
