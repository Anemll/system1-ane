"""Prefill-only Core AI export of a Jeff decision model (FP16 or light INT8).

Reuses the Qwen3.5 hybrid GDN / gated-attention / SwiGLU graphs in qwen38_coreai_build.py
after rebinding their module-level widths to the Jeff text config. The vocab LUT lm_head and
DFlash2 taps are not built. Palettization / GPTQ / VQ are skipped.

The 27B builder reads MODEL/config.json at import time. load_builder() patches that hook
before import so a Jeff (or fixture) config is used. Documented lazy import: pulling
qwen38_coreai_build at module import would require a 27B checkpoint and the Core AI SDK.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from jeff_coreai import JeffCheckpoint, convert_plan, layer_arrays

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_builder(cfg: dict):
    """Import the 27B Core AI graph modules with Jeff (or fixture) widths bound in."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    os.environ.setdefault("EXPORT_DIR", str(ROOT / ".jeff-export-unused"))
    os.environ.setdefault("KV_CACHE_DTYPE", "fp16")
    os.environ.setdefault("TPS", "256")
    import qwen38_ane_model as M
    M.cfg = lambda: cfg
    import qwen38_coreai_build as B
    bind_builder_cfg(B, cfg)
    return B


def bind_builder_cfg(B, cfg: dict) -> None:
    B.CFG = cfg
    B.nk = int(cfg["linear_num_key_heads"])
    B.nv = int(cfg["linear_num_value_heads"])
    B.dk = int(cfg["linear_key_head_dim"])
    B.dv = int(cfg["linear_value_head_dim"])
    B.kd = B.nk * B.dk
    B.vd = B.nv * B.dv
    B.cdim = 2 * B.kd + B.vd
    B.nh = int(cfg["num_attention_heads"])
    B.nkv = int(cfg["num_key_value_heads"])
    B.hd = int(cfg["head_dim"])
    B.hid = int(cfg["hidden_size"])
    B.grp = B.nh // B.nkv
    B.rot = int(B.hd * cfg["rope_parameters"]["partial_rotary_factor"])
    B.EPS = float(cfg["rms_norm_eps"])
    B.TAPS = []
    B.KV_CACHE_DTYPE = "fp16"
    B.STABLE_ATTN = False
    B.ATT_INT8MM = ""
    B.ATT_INT8MM_M5 = None
    B.ATT_INT8MM_BY_LAYER = {}
    B.KNOWN_LUTS.clear()


def save_dense_program(B, entries, out: Path) -> float:
    """Core AI package of dense (or INT8 QConv) graphs. No 4-bit palettization."""
    import torch
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision

    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for name, mod, ins, outs in entries:
        example = mod.example() if hasattr(mod, "example") else (
            torch.zeros(1, B.hid, 1, mod.T, dtype=torch.float16),)
        ep = torch.export.export(mod, example, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=ins, output_names=outs, entrypoint_name=name)
    prog = conv.to_coreai()
    prog.optimize()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    del prog, conv
    gc.collect()
    return sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6


class ReadoutHead:
    """Final RMSNorm + 255x hidden readout. Built after load_builder so nn / rms_hidden exist."""

    @staticmethod
    def make(B, ck: JeffCheckpoint, T: int = 1):
        import torch
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.T = T
                hid = int(ck.cfg["hidden_size"])
                norm = (1.0 + ck.norm_weight()).astype(np.float16)
                self.register_buffer("normw", torch.from_numpy(norm.reshape(1, -1, 1, 1)))
                w = np.asarray(ck.readout, np.float16)
                self.proj = nn.Conv2d(hid, w.shape[0], 1, bias=False)
                self.proj.weight = nn.Parameter(
                    torch.from_numpy(w.copy()).view(w.shape[0], hid, 1, 1), requires_grad=False)
                self.n_codes = int(w.shape[0])

            def forward(self, x):
                h = B.rms_hidden(x, self.normw)
                return self.proj(h).reshape(self.n_codes, self.T).transpose(0, 1)

            def example(self):
                return (torch.zeros(1, int(ck.cfg["hidden_size"]), 1, self.T, dtype=torch.float16),)

        return Head().eval().to(torch.float16)


def prefill_entry(width: int, ctx: int) -> str:
    return f"p{width}_{ctx // 1024}k"


def build_prefill_chunk(B, ck: JeffCheckpoint, layers: list[int], ctx: int, prefill: int | list[int],
                        quant: str, out: Path) -> dict:
    """One chunk package. A list of prefill widths gives one entry per width over one weight copy; every entry
    reads and writes the same KV rows (kv_len does not depend on the width below 64K)."""
    import torch.nn as nn
    widths = sorted({int(prefill)} if isinstance(prefill, int) else {int(w) for w in prefill})
    W = {}
    for i in layers:
        W.update(layer_arrays(ck, i, quant))
    mods = nn.ModuleList(B.LayerW(W, i) for i in layers).eval()
    import torch
    mods = mods.to(torch.float16)
    del W
    gc.collect()
    entries = [(prefill_entry(w, ctx), B.Entry(mods, ctx, w, kv_cache_dtype="fp16")) for w in widths]
    e = entries[0][1]
    mb = save_dense_program(B, [(name, m, m.input_names(), m.output_names()) for name, m in entries], out)
    return {
        "file": out.name,
        "layers": [layers[0], layers[-1]],
        "entries": [name for name, _ in entries],
        "gdn_j": e.gdn_j,
        "att_j": e.att_j,
        "taps": [],
        "mb": round(mb, 1),
        "entries_ctx": [[], [ctx]],
        "numerics": {"SILU": B.SILU, "MLP_SILU": B.MLP_SILU, "GDN_FAST": B.GDN_FAST,
                     "ATT_BLOCK": B.ATT_BLOCK, "ATT_BLOCK_PREFILL": B.ATT_BLOCK_PREFILL},
    }


def build_readout_head(B, ck: JeffCheckpoint, out: Path) -> dict:
    head = ReadoutHead.make(B, ck, T=1)
    mb = save_dense_program(B, [("h1", head, ["x"], ["logits"])], out)
    return {"file": out.name, "mb": round(mb, 1), "entry": "h1", "entries": ["h1"],
            "shape": list(ck.readout.shape)}


def export_jeff(ck: JeffCheckpoint, out_dir: Path, ctx: int, prefill: int,
                quant: str = "fp16", chunk: int = 4, extra_prefills: tuple[int, ...] = ()) -> dict:
    """extra_prefills: more prefill widths in every chunk (manifest TPS). TP stays ``prefill``, the entry a runtime
    without TPS support uses."""
    plan = convert_plan(ck, ctx, prefill, quant, chunk)
    widths = sorted({int(prefill), *(int(w) for w in extra_prefills)})
    for w in widths:
        if w % 8 or w <= 8:
            raise ValueError(f"prefill width {w}: use a multiple of 8 above 8")
    plan["prefill_widths"] = widths
    B = load_builder(ck.cfg)
    B.TPS = widths
    B.OUT = out_dir
    coreai = out_dir / "coreai"
    model_dir = out_dir / "model"
    coreai.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ck.write_embedding(model_dir / "embed_tokens_fp16.npy")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "decision_config.json",
                 "readout.safetensors", "chat_template.jinja"):
        src = ck.model / name
        if src.is_file():
            shutil.copy2(src, model_dir / name)
    (model_dir / "decision_config.json").write_text(json.dumps(ck.decision, indent=2))

    man = {
        "version": "jeff-coreai1",
        "kind": "jeff-decision",
        "T": 8,
        "TP": prefill,
        "TPS": widths,
        "pend": B.P,
        "taps": [],
        "ctxs": [ctx],
        "pctxs": [ctx],
        "kv_len": {str(ctx): B.kv_len(ctx, 8)},
        "pkv_len": {str(ctx): B.kv_len(ctx, prefill)},
        "kv_cache": {"format": "fp16", "keys": "float16", "values": "float16", "scales": None},
        "quant": quant,
        "dflash2": False,
        "head": {},
        "chunks": [],
        "convert": plan,
        "numerics": {"SILU": B.SILU, "MLP_SILU": B.MLP_SILU, "GDN_FAST": B.GDN_FAST},
    }
    man_path = coreai / "manifest.json"
    chunks = []
    for layers in chunk_plan_from(plan):
        dest = coreai / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        info = build_prefill_chunk(B, ck, layers, ctx, widths, quant, dest)
        chunks.append(info)
        man["chunks"] = chunks
        man_path.write_text(json.dumps(man, indent=1))
    head_path = coreai / "head_readout.aimodel"
    man["head"] = build_readout_head(B, ck, head_path)
    man_path.write_text(json.dumps(man, indent=1))
    plan["build"] = str(coreai)
    plan["manifest"] = str(man_path)
    return plan


def chunk_plan_from(plan: dict) -> list[list[int]]:
    return [list(range(int(a), int(b) + 1)) for a, b in (r.split("-") for r in plan["chunk_plan"])]
