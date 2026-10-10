"""PyTorch FP32 reference: merged checkpoint (prep_ckpt.py output) + trained Clef head vs the 5 Colab parity rows.
usage: torch_reference.py CKPT_DIR ADAPTER_HEAD_DIR [PARITY.json] [OUT.json]   (needs torch + transformers with Qwen3.5 support)
Prints per row: max |logit - Colab logit|, answer, Colab answer, gold."""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coreai"))
from clef_head import ClefHead
from transformers import Qwen3_5ForConditionalGeneration
ckpt, adapter = sys.argv[1], Path(sys.argv[2])
par = json.load(open(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).resolve().parent / "data/stock_snake_parity_rows.json"))
out_path = Path(sys.argv[4] if len(sys.argv) > 4 else "torch_reference.json")
m = Qwen3_5ForConditionalGeneration.from_pretrained(ckpt, dtype=torch.float32).eval()
lm = m.model.language_model; E = m.get_output_embeddings().weight.detach().float()
head = ClefHead(adapter)
out = []
for r in par["rows"]:
    ids = torch.tensor([r["input_ids"]]); t = time.time()
    with torch.no_grad(): H = lm(input_ids=ids).last_hidden_state[0]
    lg = head(H, r["input_ids"], E, r["question_span"], r["option_spans"])
    ref = torch.tensor(r["raw_logits"]); p = head.probs(lg)
    out.append({"row": r["heldout_row"], "torch_logits": lg.tolist(), "colab_logits": r["raw_logits"],
                "max_abs_diff": float((lg - ref).abs().max()), "argmax_match": int(lg.argmax()) == int(ref.argmax()),
                "answer": r["option_keys_in_logit_order"][int(lg.argmax())], "colab_answer": r["answer"], "gold": r["gold"],
                "p_torch": p.tolist(), "p_colab": [r["calibrated_probabilities"][k] for k in r["option_keys_in_logit_order"]]})
    print(out[-1]["row"], "diff", round(out[-1]["max_abs_diff"], 4), out[-1]["answer"], out[-1]["colab_answer"], out[-1]["gold"], round(time.time() - t, 1), flush=True)
out_path.write_text(json.dumps(out, indent=1))
