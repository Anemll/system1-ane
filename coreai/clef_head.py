"""Unsloth FastDecisionModel ("Clef" joint-schema) head, host-side PyTorch port, plus its prompt encoder.

Reconstructed from the Unsloth training run's HEAD_NOTES (joint_head.safetensors, 122 tensors). The backbone (merged LoRA)
runs on the ANE; this head reads the backbone's post-final-norm hidden rows of the *whole prompt* and scores the
options on the host. Needs torch only (the backbone hidden rows come from the Core AI runtime or from PyTorch).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

LN100 = math.log(100.0)
P = ("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly "
     "one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n")
F0 = "\n\nSCHEMA FIELDS:\n"
F2 = "\nALLOWED OPTIONS:\n"
END = "END FIELD\n"
X = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"


def render(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def encode_record(tok, state, instructions: str, options: dict, field_id: str = "move") -> dict:
    """Token ids + spans exactly as Unsloth's Clef encode_record builds them (segments tokenized separately)."""
    ids, spans = [], {}

    def add(text, name=None):
        t = tok(text, add_special_tokens=False)["input_ids"]
        if name is not None:
            spans[name] = [len(ids), len(ids) + len(t)]
        ids.extend(t)

    add(P); add(render(state)); add(F0)
    add(f"\nFIELD 1\nID: {field_id}\nTYPE: choice\nINSTRUCTION: ")
    add(instructions, "question"); add(F2)
    keys = sorted(options)
    for i, key in enumerate(keys, 1):
        add(f"OPTION {i}: ")
        add(render({"option_id": key, "description": options[key]}), f"opt{i}")
        add("\n")
    add(END); add(X)
    return {"input_ids": ids, "question_span": spans["question"], "option_keys": keys,
            "option_spans": [spans[f"opt{i}"] for i in range(1, len(keys) + 1)]}


def head_dtype(name: str | None = None) -> torch.dtype:
    """CLEF_HEAD_DTYPE=fp32 (default, old behavior) | fp16 | bf16."""
    n = (name or os.environ.get("CLEF_HEAD_DTYPE") or "fp32").lower()
    return {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16,
            "bf16": torch.bfloat16}[n]


class ClefHead(nn.Module):
    def __init__(self, path: Path, dtype: torch.dtype | str | None = None, device: str | None = None):
        """dtype/device default to CLEF_HEAD_DTYPE (fp32) / CLEF_HEAD_DEVICE (cpu). fp16 loads joint_head_fp16.safetensors
        when the build ships it (written by convert_head_fp16.py), else casts the original bf16 checkpoint."""
        super().__init__()
        self.dtype = dtype if isinstance(dtype, torch.dtype) else head_dtype(dtype)
        self.device = torch.device(device or os.environ.get("CLEF_HEAD_DEVICE") or "cpu")
        path = Path(path)
        cfg = json.loads((path / "joint_head_config.json").read_text())
        self.cfg = cfg
        self.decision = json.loads((path / "unsloth_decision_config.json").read_text())
        h, w, ff, nh = cfg["hidden_size"], cfg["width"], cfg["feedforward"], cfg["heads"]
        self.w = w
        lin = lambda i, o, b=False: nn.Linear(i, o, bias=b)  # noqa: E731
        self.hidden_norm = nn.LayerNorm(h)
        self.memory_projection = lin(h, w)
        self.option_context_projection = lin(h, w)
        self.option_lexical_projection = lin(h, w)
        self.option_question_projection = lin(h, w)
        self.question_projection = lin(h, w)
        self.global_projection = lin(h, w)
        self.option_norm = nn.LayerNorm(w)
        self.option_summary_norm = nn.LayerNorm(w)
        self.field_norm = nn.LayerNorm(w)
        self.type_embedding = nn.Embedding(3, w)

        class Evidence(nn.Module):
            def __init__(s):
                super().__init__()
                s.query_norm, s.memory_norm, s.feedforward_norm = nn.LayerNorm(w), nn.LayerNorm(w), nn.LayerNorm(w)
                s.attention = nn.MultiheadAttention(w, nh, batch_first=True)
                s.feedforward = nn.Sequential(nn.Linear(w, ff), nn.GELU(), nn.Dropout(0.0), nn.Linear(ff, w))

        self.evidence_layers = nn.ModuleList(Evidence() for _ in range(cfg["routing_layers"]))
        self.layers = nn.ModuleList(nn.TransformerDecoderLayer(w, nh, ff, 0.0, "gelu", batch_first=True, norm_first=True)
                                    for _ in range(cfg["layers"]))
        self.residual_scorer = nn.Sequential(nn.Linear(4 * w, w), nn.GELU(), nn.Dropout(0.0), nn.Linear(w, 1))
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))
        f16 = path / "joint_head_fp16.safetensors"
        src = f16 if (self.dtype == torch.float16 and f16.is_file()) else path / "joint_head.safetensors"
        self.weights_file = src.name
        sd = {k: v.float() for k, v in load_file(str(src)).items()}  # fp16/bf16 -> fp32 is exact; cast back below
        self.load_state_dict(sd, strict=True)
        self.eval().requires_grad_(False)
        # the three logit scales stay fp32 python floats (exp/sigmoid applied in fp32), whatever the head dtype
        self.prior_scale = math.exp(min(float(self.prior_logit_scale), LN100))
        self.joint_scale = math.exp(min(float(self.joint_logit_scale), LN100))
        self.gate = 1.0 / (1.0 + math.exp(-float(self.residual_gate)))
        self.to(device=self.device, dtype=self.dtype)
        # remaining relative temperature for choice questions (head temperature is already folded into the head)
        self.t_choice = float(self.decision["temperature"][0])

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor, ids: list[int], embed: torch.Tensor, qspan, ospans) -> torch.Tensor:
        """hidden: (L, 1024) post-final-RMSNorm rows; embed: (vocab, 1024) tied output embedding. Returns fp32 option logits
        (cpu). The network runs in self.dtype on self.device (CLEF_HEAD_DTYPE / CLEF_HEAD_DEVICE); fp32/cpu = original."""
        dt, dev = self.dtype, self.device
        L = hidden.shape[0]
        idt = torch.as_tensor(ids)
        lex = torch.stack([embed[idt[a:b]].float().mean(0) for a, b in ospans]).to(dev, dt)  # host gather, then cast
        H = self.hidden_norm(hidden.to(dev, dt))
        M = self.memory_projection(H)
        q = H[qspan[0]:qspan[1]].mean(0)
        g = H[L - 1]
        c = torch.stack([H[a:b].mean(0) for a, b in ospans])
        r = self.option_context_projection(c) + self.option_lexical_projection(lex) + self.option_question_projection(q)
        for ev in self.evidence_layers:
            mq, mk = ev.query_norm(r)[None], ev.memory_norm(M)[None]
            r = r + ev.attention(mq, mk, mk, need_weights=False)[0][0]
            r = r + ev.feedforward(ev.feedforward_norm(r))
        f0 = self.question_projection(q)
        wgt = torch.softmax(r @ f0 / math.sqrt(self.w), dim=0)
        s = (wgt[:, None] * r).sum(0)
        field = f0 + self.option_summary_norm(s) + self.global_projection(g) + self.type_embedding.weight[1]
        f = field[None, None]
        for layer in self.layers:
            f = layer(f, M[None])
        field = self.field_norm(f[0, 0])
        anchor = F.normalize(q + g, dim=-1)
        prior = self.prior_scale * F.cosine_similarity(lex, anchor[None], dim=-1).float()
        n = self.option_norm(r)
        cos = F.cosine_similarity(field[None].expand_as(n), n, dim=-1).float()
        fe = field[None].expand_as(n)
        res = self.residual_scorer(torch.cat([fe, n, fe * n, (fe - n).abs()], -1)).squeeze(-1).float()
        return (prior + self.gate * (self.joint_scale * cos + res)).cpu()

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float() / self.t_choice, -1)
