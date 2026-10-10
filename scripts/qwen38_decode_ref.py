"""Host hybrid DecodeLayer (Gated DeltaNet + gated attention).

Origin: Anemll/anemll-forge. Jeff host smoke and ``jeff_coreai.host_prefill_hidden`` step this
config-driven layer. It matches Jeff's PyTorch FP32 readout to about 2e-6.

Single-token decode reference for Qwen3.5 decoder layers, with explicit state:
Gated DeltaNet layers keep a conv state (conv_dim, 3) and a recurrent state (v_heads, k_dim, v_dim);
full-attention layers keep a KV cache.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(os.path.expanduser(os.environ.get("TEST_DIR", "~/Models/Qwen3.8-27B-test")))
TENSORS = [ROOT / "test_tensors_L30-33.safetensors", ROOT / "test_tensors_L30-33_attn_head.safetensors"]
torch.set_grad_enabled(False)


def text_config():
    return json.loads((ROOT / "config.json").read_text())["text_config"]


def load_layer_weights(layers):
    w = {}
    for f in TENSORS:
        for k, v in load_file(f).items():
            for l in layers:
                p = f"model.language_model.layers.{l}."
                if k.startswith(p):
                    w.setdefault(l, {})[k[len(p):]] = v.float()
    return w


def rms_zc(x, w, eps):  # Qwen3_5RMSNorm: zero-centered weight
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + w)


class DecodeLayer:
    """One decoder layer, one token at a time. Weights: dict with the checkpoint's layer-relative names."""

    def __init__(self, cfg, idx, w, ctx=4096):
        self.cfg, self.w, self.eps = cfg, w, cfg["rms_norm_eps"]
        self.kind = cfg["layer_types"][idx]
        if self.kind == "linear_attention":
            self.nk, self.nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
            self.dk, self.dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
            conv_dim = 2 * self.nk * self.dk + self.nv * self.dv
            self.conv_state = torch.zeros(conv_dim, cfg["linear_conv_kernel_dim"] - 1)
            self.state = torch.zeros(self.nv, self.dk, self.dv)
        else:
            self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
            rope = cfg["rope_parameters"]
            self.rot = int(self.hd * rope["partial_rotary_factor"])
            self.inv_freq = 1.0 / rope["rope_theta"] ** (torch.arange(0, self.rot, 2).float() / self.rot)
            self.k_cache = torch.zeros(self.nkv, ctx, self.hd)
            self.v_cache = torch.zeros(self.nkv, ctx, self.hd)
        self.pos = 0

    def gdn(self, x):
        w = self.w
        qkv = w["linear_attn.in_proj_qkv.weight"] @ x
        z = (w["linear_attn.in_proj_z.weight"] @ x).view(self.nv, self.dv)
        b = w["linear_attn.in_proj_b.weight"] @ x
        a = w["linear_attn.in_proj_a.weight"] @ x
        seq = torch.cat([self.conv_state, qkv[:, None]], 1)                   # (conv_dim, 4)
        conv = F.silu((seq * w["linear_attn.conv1d.weight"][:, 0]).sum(1))
        self.conv_state = seq[:, 1:]
        kd = self.nk * self.dk
        q, k, v = conv[:kd].view(self.nk, self.dk), conv[kd:2 * kd].view(self.nk, self.dk), conv[2 * kd:].view(self.nv, self.dv)
        rep = self.nv // self.nk
        q, k = q.repeat_interleave(rep, 0), k.repeat_interleave(rep, 0)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / self.dk ** 0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(b)
        g = -w["linear_attn.A_log"].exp() * F.softplus(a + w["linear_attn.dt_bias"])
        s = self.state * g.exp()[:, None, None]
        kv_mem = (s * k[:, :, None]).sum(1)                                     # (nv, dv)
        delta = (v - kv_mem) * beta[:, None]
        s = s + k[:, :, None] * delta[:, None, :]
        self.state = s
        o = (s * q[:, :, None]).sum(1)                                          # (nv, dv)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps) * w["linear_attn.norm.weight"] * F.silu(z)
        return w["linear_attn.out_proj.weight"] @ o.reshape(-1)

    def attn(self, x):
        w, p = self.w, self.pos
        qg = (w["self_attn.q_proj.weight"] @ x).view(self.nh, 2 * self.hd)
        q, gate = qg[:, :self.hd], qg[:, self.hd:].reshape(-1)
        q = rms_zc(q, w["self_attn.q_norm.weight"], self.eps)
        k = rms_zc((w["self_attn.k_proj.weight"] @ x).view(self.nkv, self.hd), w["self_attn.k_norm.weight"], self.eps)
        v = (w["self_attn.v_proj.weight"] @ x).view(self.nkv, self.hd)
        f = p * self.inv_freq
        cos, sin = torch.cat([f, f]).cos(), torch.cat([f, f]).sin()

        def rope(t):
            r, rest = t[:, :self.rot], t[:, self.rot:]
            half = self.rot // 2
            return torch.cat([r * cos + torch.cat([-r[:, half:], r[:, :half]], 1) * sin, rest], 1)

        q, k = rope(q), rope(k)
        self.k_cache[:, p], self.v_cache[:, p] = k, v
        kk = self.k_cache[:, :p + 1].repeat_interleave(self.nh // self.nkv, 0)
        vv = self.v_cache[:, :p + 1].repeat_interleave(self.nh // self.nkv, 0)
        att = torch.softmax((kk @ q[:, :, None])[..., 0] / self.hd ** 0.5, -1)  # (nh, p+1)
        o = (att[:, :, None] * vv).sum(1).reshape(-1) * torch.sigmoid(gate)
        return w["self_attn.o_proj.weight"] @ o

    def step(self, x):
        w = self.w
        h = rms_zc(x, w["input_layernorm.weight"], self.eps)
        x = x + (self.gdn(h) if self.kind == "linear_attention" else self.attn(h))
        h = rms_zc(x, w["post_attention_layernorm.weight"], self.eps)
        y = w["mlp.down_proj.weight"] @ (F.silu(w["mlp.gate_proj.weight"] @ h) * (w["mlp.up_proj.weight"] @ h))
        self.pos += 1
        return x + y


def validate(layers=(30, 31, 32, 33), t=24):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    cfg = text_config()
    tcfg = Qwen3_5TextConfig(**cfg)
    tcfg._attn_implementation = "eager"
    weights = load_layer_weights(layers)
    x = torch.randn(1, t, cfg["hidden_size"], generator=torch.Generator().manual_seed(0)) * 0.5
    # transformers: prefill all t tokens through the layers (no cache)
    rot = Qwen3_5TextRotaryEmbedding(tcfg)
    pos = torch.arange(t).view(1, 1, -1).expand(4, 1, -1)
    kw = dict(config=tcfg, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=pos[0])
    masks = {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}
    pe, h_hf = rot(x, pos[1:]), x
    for l in layers:
        layer = Qwen3_5DecoderLayer(tcfg, l).float()
        layer.load_state_dict({k: v for k, v in weights[l].items()}, strict=True)
        h_hf = layer(h_hf, position_embeddings=pe, attention_mask=masks[cfg["layer_types"][l]],
                     position_ids=pos[0], past_key_values=None, use_cache=False)
    # reference: token by token with explicit state
    ref = [DecodeLayer(cfg, l, weights[l], ctx=t) for l in layers]
    outs = []
    for i in range(t):
        h = x[0, i]
        for d in ref:
            h = d.step(h)
        outs.append(h)
    h_ref = torch.stack(outs)
    err = (h_ref - h_hf[0]).norm(dim=-1) / h_hf[0].norm(dim=-1)
    print(f"layers {layers} ({[cfg['layer_types'][l][:4] for l in layers]}), {t} tokens: "
          f"relative error step-decode vs transformers prefill: max {err.max():.2e}, mean {err.mean():.2e}")


if __name__ == "__main__":
    validate()
