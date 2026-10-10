#!/usr/bin/env python3
"""Golden Jeff decisions for Core AI parity: exact prompt token ids + PyTorch FP32 option probabilities.

    python scripts/jeff_reference.py --model ~/Models/jeff/jeff-base-v1.3 --out cases.json \
        [--rows rows.json] [--jeff-src ~/Models/jeff/jeff-src/src] [--no-hf]

Runs in an environment with transformers (the forge .venv), not the Core AI SDK one. The prompt is built by the
port in coreai/jeff_coreai.py; with --jeff-src it is also built by Jeff's own decision_messages and the two must
match token for token. The reference forward is jeff.model.DecisionModel's: Qwen3_5Model.from_pretrained(fp32,
sdpa) -> last_hidden_state[0, -1] -> readout[:n] / temperature -> softmax.

Without --rows, four built-in rows of about 180, 250, 1000 and 2000 tokens (support-chat states, live-last) are used.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as Q
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
from jeff_coreai import (JEFF_DEFAULT, decision_messages, load_decision_config, prompt_ids,  # noqa: E402
                         question_options)

INTENTS = {
    "track_refund": "Check the status of a refund they are expecting.",
    "get_refund": "Get their money back for a purchase.",
    "track_order": "Find out where their order is or its current status.",
    "cancel_order": "Cancel an order that has not shipped yet.",
    "change_address": "Change the delivery address of an order.",
    "exchange_item": "Swap a purchased item for another size or color.",
    "report_damage": "Report that an item arrived broken or damaged.",
    "missing_item": "Report that an item is missing from a delivery.",
    "payment_problem": "A card was declined or charged twice.",
    "account_access": "Cannot log in or reset their password.",
    "product_question": "Ask about a product's size, material or availability.",
    "complaint": "Complain about the service without a specific request.",
}
TURNS = [
    "Hello, I ordered a winter jacket in size M three weeks ago.",
    "Agent: Thanks for reaching out. Can you share the order number?",
    "Sure, it is 48213-B. It came quickly but the sleeves were too short, so I packed it back up.",
    "Agent: I see the order. Did you use the prepaid return label that came in the box?",
    "Yes, I dropped it at the post office on the 2nd and the tracking says it was delivered to your warehouse.",
    "Agent: Thank you. Returns are usually processed within five business days of arrival.",
    "It has been more than five business days already, and my bank shows nothing yet.",
    "Agent: Some banks take a few extra days to post the credit. Which card did you pay with?",
    "A Visa debit card ending in 4410. I also had a 10 percent discount code applied at checkout.",
    "Agent: Noted. The discount does not change the refund method; it goes back to the original card.",
]


def support_row(n_turns: int, n_options: int) -> dict:
    keys = list(INTENTS)
    criteria = {k: INTENTS[k] for k in keys[:min(n_options, len(keys))]}
    for i in range(len(criteria), n_options):  # long option lists: numbered order-handling intents (keys never numbers)
        criteria[f"order_topic_{i:03d}"] = f"Order question {i}."
    history = [TURNS[i % len(TURNS)] for i in range(n_turns)]
    return {
        "state": {"service": "Customer support chat of an online shop", "history": history,
                  "message": "I sent the jacket back two weeks ago and still have not seen the money."},
        "question": {"type": "choice",
                     "instructions": "What does the customer want? Choose the request that best matches what the "
                                     "customer is asking for in their message.",
                     "criteria": criteria},
    }


def builtin_rows(model: Path, tok, decision) -> list[dict]:
    """Rows sized just under 256, 1024 and 2048 tokens (one, four and eight 256-row prefill calls), plus the README
    example (3 options)."""
    rows = [{"name": "readme_3opt", **support_row(0, 3)}]
    rows[0]["state"] = {"service": "Customer support chat of an online shop",
                        "message": "I sent the jacket back two weeks ago and still have not seen the money."}
    for name, target, n_options in (("t256_5opt", 256, 5), ("t1024_30opt", 1024, 30), ("t2048_100opt", 2040, 100)):
        best = None
        for turns in range(0, 400):
            row = support_row(turns, n_options)
            n = len(prompt_ids(model, row, decision, tok))
            if n > target:
                break
            best = row
        if best is None:
            raise ValueError(f"{name}: {n_options} options alone exceed {target} tokens")
        rows.append({"name": name, **best})
    return rows


UPSTREAM_FUNCS = ("describe", "options", "decision_messages")
# the fp16 scales built into the Core AI DeltaNet graph (coreai/qwen38_coreai_build.py defaults): the ANE state is
# GDN_SV x the true state and the ANE q . S output GDN_SQ x GDN_SV x the true output
GDN_SQ, GDN_SV = 16.0, 64.0
FP16_MAX, FP16_MIN_NORMAL = 65504.0, 6.1e-5


class Probe:
    """Forward hooks on the decoder layers (outputs) and a wrapper around the DeltaNet chunk kernel (output / final
    state magnitudes), for one forward at a time."""

    def __init__(self, backbone, layers_out: set[int], magnitudes: bool):
        self.layers = backbone.language_model.layers
        self.types = backbone.config.text_config.layer_types
        self.layers_out, self.magnitudes = layers_out, magnitudes
        self.out, self.gdn, self.hooks = {}, [], []
        for i, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(self._hook(i)))
        self._orig = Q.torch_chunk_gated_delta_rule
        if magnitudes:
            Q.torch_chunk_gated_delta_rule = self._gdn

    def _hook(self, i):
        def hook(_mod, _inp, output):
            y = output[0] if isinstance(output, tuple) else output
            if i in self.layers_out:
                self.out[i] = y[0].float().numpy()
            if self.magnitudes:
                self.out.setdefault("max", {})[i] = float(y.abs().max())
        return hook

    def _gdn(self, *args, **kwargs):
        kwargs["output_final_state"] = True
        o, state = self._orig(*args, **kwargs)
        a = o.abs().float()
        self.gdn.append({"out_max": float(a.max()), "out_p50": float(a.median()),
                         "out_p01": float(torch.quantile(a.flatten()[::97], 0.01)), "state_max": float(state.abs().max())})
        return o, None

    def reset(self):
        self.out, self.gdn = {}, []

    def layer_stats(self) -> list[dict]:
        stats, g = [], iter(self.gdn)
        for i, kind in enumerate(self.types):
            s = {"layer": i, "kind": kind, "residual_max": self.out.get("max", {}).get(i)}
            if kind == "linear_attention":
                d = next(g)
                s.update(d)
                s["ane_out_max"] = d["out_max"] * GDN_SQ * GDN_SV
                s["ane_out_p01"] = d["out_p01"] * GDN_SQ * GDN_SV
                s["ane_state_max"] = d["state_max"] * GDN_SV
                s["fp16_headroom_out"] = FP16_MAX / max(s["ane_out_max"], 1e-30)
                s["fp16_headroom_state"] = FP16_MAX / max(s["ane_state_max"], 1e-30)
            stats.append(s)
        return stats

    def close(self):
        for h in self.hooks:
            h.remove()
        Q.torch_chunk_gated_delta_rule = self._orig


def chunk_errors(dump: Path, hf_out: dict) -> list[dict]:
    z = np.load(dump)
    rows = []
    for k, layer in enumerate(z["layers"].tolist()):
        ane, ref = z[f"chunk{k}"].astype(np.float64), hf_out[layer].astype(np.float64)
        diff = ane - ref
        last = float(ane[-1] @ ref[-1] / (np.linalg.norm(ane[-1]) * np.linalg.norm(ref[-1])))
        rows.append({"chunk": k, "after_layer": layer, "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
                     "rel_l2_last_row": float(np.linalg.norm(diff[-1]) / np.linalg.norm(ref[-1])),
                     "cos_last_row": last, "max_abs_ane": float(np.abs(ane).max()), "max_abs_hf": float(np.abs(ref).max())})
    return rows


def upstream_messages(jeff_src: Path):
    """decision_messages from a firelex/jeff checkout's jeff/model.py, run verbatim. Only its prompt functions and
    constants are executed: the package needs Python 3.12 and PIL / the VL processor, which the prompt never uses."""
    path = jeff_src / "jeff" / "model.py"
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in UPSTREAM_FUNCS)
            or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in ("MAX_OPTIONS", "PROMPT_LAYOUTS")
                                                  for t in n.targets))]
    if {n.name for n in keep if isinstance(n, ast.FunctionDef)} != set(UPSTREAM_FUNCS):
        raise ValueError(f"{path} no longer defines {UPSTREAM_FUNCS}")
    for n in keep:
        if isinstance(n, ast.FunctionDef):
            for arg in n.args.args + n.args.kwonlyargs:
                arg.annotation = None
            n.returns = None
    ns = {"json": json}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    return ns["decision_messages"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=JEFF_DEFAULT)
    ap.add_argument("--rows", type=Path, help="JSON list of {name, state, question} rows")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jeff-src", type=Path, help="firelex/jeff src/ directory: check the prompt port against it")
    ap.add_argument("--no-hf", action="store_true", help="token ids only (no reference forward)")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                    help="reference precision; bfloat16 (the GPU / MLX serving precision) measures the noise floor")
    ap.add_argument("--compare", type=Path, help="earlier cases.json: report this run's probabilities against it")
    ap.add_argument("--magnitudes", action="store_true",
                    help="per-layer residual max and DeltaNet output / state magnitudes (FP16 headroom at the "
                         f"ANE graph's GDN_SQ={GDN_SQ:g} / GDN_SV={GDN_SV:g})")
    ap.add_argument("--chunk-dump", type=Path,
                    help="jeff_coreai_smoke.py --dump directory: per-chunk Core AI error against these layers")
    a = ap.parse_args(argv)
    model = a.model.expanduser().resolve()
    decision = load_decision_config(model)
    tok = AutoTokenizer.from_pretrained(str(model))
    rows = json.loads(a.rows.read_text()) if a.rows else builtin_rows(model, tok, decision)
    upstream = upstream_messages(a.jeff_src) if a.jeff_src else None
    backbone = readout = None
    if not a.no_hf:
        torch.set_num_threads(8)
        dtype = getattr(torch, a.dtype)
        backbone = Qwen3_5Model.from_pretrained(str(model), dtype=dtype, attn_implementation="sdpa").eval()
        readout = load_file(str(model / "readout.safetensors"))["weight"].to(dtype)
    key = "hf_fp32" if a.dtype == "float32" else f"hf_{a.dtype}"
    earlier = {c["name"]: c for c in json.loads(a.compare.read_text())["cases"]} if a.compare else {}
    cases = []
    for row in rows:
        ids = prompt_ids(model, row, decision, tok)
        n_options = len(question_options(row["question"])[0])
        case = {"name": row.get("name", f"row{len(cases)}"), "row": row, "ids": ids, "tokens": len(ids),
                "n_options": n_options, "codes": decision["codes"][:n_options]}
        if upstream is not None:
            mine = decision_messages(row, decision["codes"], decision["prompt_layout"])
            theirs = upstream(row, decision["codes"], decision["prompt_layout"])
            text = tok.apply_chat_template(theirs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            case["upstream_prompt_match"] = mine == theirs and ids == list(tok(text, add_special_tokens=False)["input_ids"])
            if not case["upstream_prompt_match"]:
                raise SystemExit(f"{case['name']}: prompt port differs from jeff.model.decision_messages")
        if backbone is not None:
            dump = a.chunk_dump / f"{case['name']}.npz" if a.chunk_dump else None
            dump = dump if dump is not None and dump.is_file() else None
            probe = None
            if a.magnitudes or dump is not None:
                probe = Probe(backbone, set(np.load(dump)["layers"].tolist()) if dump is not None else set(),
                              a.magnitudes)
            t0 = time.time()
            try:
                with torch.inference_mode():
                    hidden = backbone(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0, -1]
                    logits = (readout[:n_options] @ hidden).double() / decision["temperature"]
            finally:
                if probe is not None:
                    probe.close()
            ref = case[key] = {"probabilities": torch.softmax(logits, -1).tolist(),
                               "hidden_normed": hidden.float().tolist(), "s": round(time.time() - t0, 2)}
            if probe is not None and a.magnitudes:
                case["layer_magnitudes"] = probe.layer_stats()
                worst = min((s for s in case["layer_magnitudes"] if "fp16_headroom_out" in s),
                            key=lambda s: min(s["fp16_headroom_out"], s["fp16_headroom_state"]))
                print(f"  {case['name']}: residual max {max(s['residual_max'] for s in case['layer_magnitudes']):.0f}"
                      f"; least GDN fp16 headroom layer {worst['layer']}: out x{worst['fp16_headroom_out']:.1f}, "
                      f"state x{worst['fp16_headroom_state']:.1f}", flush=True)
            if dump is not None:
                case["chunk_vs_coreai"] = chunk_errors(dump, probe.out)
                for r in case["chunk_vs_coreai"]:
                    print(f"  {case['name']} chunk {r['chunk']} (after layer {r['after_layer']}): rel_l2 "
                          f"{r['rel_l2']:.4f} last-row {r['rel_l2_last_row']:.4f} cos {r['cos_last_row']:.6f} "
                          f"max|x| ane {r['max_abs_ane']:.0f} hf {r['max_abs_hf']:.0f}", flush=True)
            old = (earlier.get(case["name"]) or {}).get("hf_fp32")
            if old is not None:
                r, p = np.asarray(old["probabilities"]), np.asarray(ref["probabilities"])
                ref["vs_fp32"] = {"max_abs_dp": float(np.max(np.abs(r - p))),
                                  "kl_ref_backend": float(np.sum(r * (np.log(r + 1e-12) - np.log(p + 1e-12)))),
                                  "argmax_match": int(np.argmax(r)) == int(np.argmax(p))}
        cases.append(case)
        print(f"{case['name']}: {len(ids)} tokens, {n_options} options"
              + (f", hf {a.dtype} argmax {case['codes'][int(np.argmax(ref['probabilities']))]}"
                 f" p={max(ref['probabilities']):.4f} ({ref['s']}s)" if backbone is not None else "")
              + (f", vs fp32 {ref['vs_fp32']}" if backbone is not None and "vs_fp32" in ref else "")
              + (f", upstream prompt match {case['upstream_prompt_match']}" if upstream is not None else ""), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": str(model), "temperature": decision["temperature"],
                                 "prompt_layout": decision["prompt_layout"], "cases": cases}))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
