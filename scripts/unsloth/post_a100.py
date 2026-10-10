# Post-run step for the Unsloth decision model on Qwen3.5-0.8B (Snake): reload each saved adapter+head from disk (as a consumer would), regenerate the 5-row parity
# file (token ids, spans, raw head logits, calibrated probabilities) and dump the exact prompt template.
import json, sys, torch
from pathlib import Path
W = Path("/content/snake")
from unsloth import FastDecisionModel
from unsloth.models import decision as D
from unsloth._vendor.clef.joint_schema_model import encode_record, SYSTEM_PROMPT

def convert(path, n):
    rows = []
    for line in list(open(path))[:n]:
        r = json.loads(line)
        rows.append({"state": json.dumps(r["state"]),
                     "questions": json.dumps({"move": {"type": "choice", "instructions": r["instructions"], "criteria": r["options"]}}),
                     "gold": json.dumps({"move": {"type": "choice", "label": r["label"]}})})
    return rows

for TAG in sys.argv[1:]:
    OUT = W / "out" / TAG
    if not (OUT / "adapter_head").exists():
        print(TAG, "no adapter_head"); continue
    model, tokenizer = FastDecisionModel.from_pretrained(str(OUT / "adapter_head"), max_seq_length = 1024, load_in_4bit = False, dtype = torch.float16)
    FastDecisionModel.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    rows = convert(W / "heldout.jsonl", 5)
    items, _ = FastDecisionModel.build_dataset(rows, tokenizer, model)
    logits, _ = D._decision_logits(model, tokenizer, items, None)
    old = json.loads((OUT / "parity.json").read_text()) if (OUT / "parity.json").exists() else None
    cfg = model.decision_config
    par = []
    for k, (it, z) in enumerate(zip(items, logits)):
        rq = it["record"].questions[0]; keys = list(rq.option_ids); ids = [int(x) for x in it["input_ids"]]
        pred = FastDecisionModel.predict(model, tokenizer, it["source"]["state"], it["source"]["questions"])["move"]
        e = {"heldout_row": it["row"], "n_tokens": len(ids), "input_ids": ids, "rendered_text": tok.decode(ids),
             "question_span": list(rq.question_span), "option_spans": [list(map(int, s)) for s in rq.option_spans],
             "global_token_index": len(ids) - 1, "option_keys_in_logit_order": keys,
             "raw_logits": [float(x) for x in z], "softmax_T1": [float(x) for x in torch.softmax(z.float(), -1)],
             "calibrated_probabilities": pred.get("probabilities"), "answer": pred.get("answer"), "gold": keys[it["labels"][0]]}
        if old and k < len(old["rows"]):
            e["max_abs_diff_vs_in_training_session_logits"] = float(max(abs(a - b) for a, b in zip(old["rows"][k]["raw_logits"], e["raw_logits"])))
        par.append(e)
    (OUT / "parity_final.json").write_text(json.dumps({"tag": TAG, "base_model": cfg.get("base_model"),
        "loaded_from": "adapter_head (LoRA over bf16 base, A100, no 4-bit; Qwen3.5 forced float32 compute path, no autocast)", "forced_float32": bool(getattr(model, "_unsloth_forced_float32", False)),
        "head_temperature": cfg.get("head_temperature"), "relative_temperature_by_qtype[choice,score,noul]": cfg.get("temperature"),
        "rows": par}, indent=1))
    # template: segment-by-segment token table for row 0
    it = items[0]; ids = [int(x) for x in it["input_ids"]]; rq = it["record"].questions[0]
    table = [{"i": i, "id": t, "piece": tok.decode([t])} for i, t in enumerate(ids)]
    (OUT / "template_example.json").write_text(json.dumps({"system_prompt": SYSTEM_PROMPT, "rendered_text": tok.decode(ids),
        "question_span": list(rq.question_span), "option_spans": [list(map(int, s)) for s in rq.option_spans],
        "option_ids": list(rq.option_ids), "global_token_index": len(ids) - 1, "tokens": table,
        "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id}, indent=1))
    print(TAG, "post ok", [p.get("max_abs_diff_vs_in_training_session_logits") for p in par], flush=True)
    del model; torch.cuda.empty_cache()
print("POST_DONE", flush=True)
