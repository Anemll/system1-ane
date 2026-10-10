"""Hybrid Qwen3.5 Core AI graphs (GDN / gated attention / SwiGLU).

Origin: Anemll/anemll-forge. Jeff convert rebinds module-level widths via
``jeff_coreai_build.load_builder`` / ``bind_builder_cfg`` and does not build the
27B vocab head or DFlash2 taps. Import is safe without a 27B checkpoint: missing
``MODEL/config.json`` leaves placeholder widths until bind_builder_cfg runs.

Core AI build of the Qwen3.8-27B target (export full_mix25_mixer4_head4): one .aimodel per chunk whose entry points
share ONE weight copy on the ANE (Core ML multifunction models wire a copy per function):
    v8_<ctx>k   8 rows (decode / DFlash2 verify), lazy-commit DeltaNet, KV history of <ctx> positions
    p64_<ctx>k  64 rows (prefill: all rows committed, padding masked by `valid`)
plus head.aimodel (final norm + LUT4 lm_head, 8 rows). A torch mirror of qwen38_ane_chunk.py (v4 form) with the
Core AI compiler fixes (COREAI_PORT_NOTES.md): forward substitution instead of the doubling inverse, scale-free
RMSNorm, overflow-safe softplus, per-channel scales as a mul after the conv, exact exported LUT / indices injected
before optimize(). Weights come straight from the checkpoint + export (qwen38_ane_model.Checkpoint / layer_quant).

    .venv/bin/python qwen38_coreai_build.py chunk 0-3 [--ctx 2048,8192,16384] [--pctx 2048] [--name NAME]
    .venv/bin/python qwen38_coreai_build.py head
    .venv/bin/python qwen38_coreai_build.py all  [--plan 0-3,4-7,...] [--ctx ...] [--pctx ...]
Env: EXPORT_DIR (default ~/Models/vq27b/export/full_mix25_mixer4_head4), OUT (default ~/Models/vq27b/coreai);
SILU, MLP_SILU (tanh | native), GDN_SQ, GDN_SV, MLP_DS_TABLE: ANE fp16 numerics fixes (see silu()); GDN_FAST (default 1):
faster exact DeltaNet core; ATT_BLOCK / ATT_BLOCK_PREFILL (default 2048 / 4096): attention history tile width of the
verify / prefill entries; KV_CACHE_DTYPE / --kv-cache-dtype (default v8; kv8 also stores keys as INT8). The release graph: GDN_FAST=0
ATT_BLOCK=16384. DBG_O=1 debug outputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"  # scripts
# KMeans is imported lazily by qwen3_lut_common; no sklearn module stubs.
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("EXPORT_DIR", os.path.expanduser("~/Models/vq27b/export/full_mix25_mixer4_head4"))
import qwen38_ane_model as M  # noqa: E402

OUT = Path(os.path.expanduser(os.environ.get("OUT", "~/Models/vq27b/coreai"))) / M.EXPORT_DIR.name
CFG = {}  # filled below when MODEL/config.json exists; Jeff convert overwrites via bind_builder_cfg
P, C_SUB = 8, 8                                    # pending rows (lazy commit), prefill sub-chunk
TPS = [int(x) for x in os.environ.get("TPS", "64").split(",")]  # prefill entry sizes (rows)
# ATT_BLOCK: history slice width of the 8-row verify entries (32K / 64K single-softmax graphs fail ANEC; 2048 is the
# fastest on the M6). ATT_BLOCK_PREFILL: the same for the 64-row prefill entries; they gain nothing below 4096 and
# narrow slices cost ANE compile time, so it defaults to max(ATT_BLOCK, 4096). Release graph: ATT_BLOCK=16384.
ATT_BLOCK = int(os.environ.get("ATT_BLOCK", "2048"))
ATT_BLOCK_PREFILL = int(os.environ.get("ATT_BLOCK_PREFILL", str(max(ATT_BLOCK, 4096))))
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "v8")  # v8 | kv8 | fp16 | both (selectable, about 1.5x the compile)
# cache inputs per attention layer: v8 = FP16 keys + INT8 values; kv8 = INT8 keys and values (scales per token/head)
KV_INPUTS = {"fp16": ("k", "v"), "v8": ("k", "v", "vs"), "kv8": ("k", "v", "ks", "vs")}
STABLE_ATTN = os.environ.get("ATT_STABLE", "0") == "1"
# research switch for the history softmax: two_pass (global max over all tiles first, the default), online (running max
# from tile to tile, flash attention), split (each tile with its own max, sum and output, combined at the end: flash
# decoding), recompute (timing control: the second pass recomputes the scores). ONLINE_SOFTMAX=1 is the older spelling of online.
ATT_SOFTMAX = os.environ.get("ATT_SOFTMAX", "online" if os.environ.get("ONLINE_SOFTMAX", "0") == "1" else "two_pass")
ATT_SOFTMAX_PREFILL = os.environ.get("ATT_SOFTMAX_PREFILL", ATT_SOFTMAX)  # the prefill entries' form (hybrid: split)
# INT8 caches: dequantize each history tile next to its matmul instead of the whole history once per call (no FP16
# copy of the cache inside the program; the same values, dequantization is elementwise)
ATT_TILE_DEQUANT = os.environ.get("ATT_TILE_DEQUANT", "0") == "1"
# key cache layout: KV_KEYS_T=1 stores keys as (KV head, head dim, token), the (256, token) operand QK reads, so no key
# tile is transposed inside the program (otherwise a pass per tile before every QK); the new key rows are still
# returned as (KV head, T, head dim) and the runtime writes them transposed
KV_KEYS_T = os.environ.get("KV_KEYS_T", "0") == "1"
# timing research only (wrong numerics): INT8 x INT8 history matmuls on kv8 caches. qk quantizes the queries, pv the
# exp weights, each with a fixed scale, as quantize -> dequantize next to the matmul so the ANE compiler can fuse them
# variants: qk | qkt (keys untransposed) | pv | both | botht | pvdq (control: V dequantized per tile, FP16 weights); the
# *o forms (qkto | pvo | botho) also requantize the matmul output, as W8A8 graphs do (activation scales in and out);
# nomm (timing only) replaces the history QK and PV matmuls by reductions that still read every key / value of a tile;
# pvn (bothn: plus qk) quantizes the exp weights correctly: per row and tile divided by their maximum (values in [0, 1],
# step 1/127), the partial output multiplied back, so only INT8 rounding of the weights changes the result; pvt gets
# weights in [0, 1] without that extra pass over the scores: exp against each tile's own row maximum (already computed
# for the global max) and the value scales divided by their tile maximum per head, both corrections on the PV output;
# pvta is pvt with asymmetric codes (zero point -128, step 1/255): the weights are never negative; pvtu uses UINT8
# codes (step 1/255, zero point 0) instead; qkn quantizes the queries per row to [-1, 1] (step 1/127), the scores
# rescaled per row; qkf puts that per-row scale inside the quantize / dequantize pair (per-axis scale, no rescale of
# the scores; pvtm is pvt with minval mode (offset by qmin instead of a zero point); pvf8 / pvf5 quantize the pvt
# weights to FP8 e4m3 (scale 1/256) / e5m2 (scale 1/32768); s8 puts an INT8 pair (step ATT_S8_UNIT) on the raw QK
# output (before the key scales), s8b an INT8 pair (step ATT_S8B_UNIT) on the scores after the key scales and the mask
# (the tensor the max and exp passes read; on v8, without key scales, s8 and s8b sit on the same true scores),
# t8 an INT8 pair (step 1/8) on s - m_t before the exp of the pvt forms;
# sm8 (FP8, M6 only) quantizes the exp output of the pvt forms to FP8 e4m3 (scale ATT_PF8_UNIT) and takes the softmax sum
# from it, an 8-bit sum without the UINT8 underflow bias; the PV probabilities are then those FP8 probabilities times the value
# scales; s8r subtracts each row's block maximum plus ATT_S8R_SHIFT from all scores before the pairs (softmax is
# shift-invariant), so the fixed INT8 range covers [m_b - 32 + shift, m_b + 32 + shift] per row instead of +-32
# absolute (long contexts reach scores above 32). Forms combine as a comma-separated list (e.g. s8r,s8,s8b,sm8,pvf8)
ATT_INT8MM = os.environ.get("ATT_INT8MM", "")
# per-layer override of ATT_INT8MM, "layer:forms;layer:forms" (forms may be empty: production attention), e.g.
# "63:s8,s8b" keeps INT8 scores but FP16 PV in layer 63
ATT_INT8MM_BY_LAYER = {int(k): v for k, v in (x.split(":", 1) for x in os.environ.get("ATT_INT8MM_BY_LAYER", "").split(";") if x)}
# M5 functions in the same package: when set (e.g. "s8,s8b"; "none" for production attention), every entry is also
# exported as <entry>_m5 traced with these forms instead of ATT_INT8MM (the M5 ANE compiler rejects FP8, sm8 / pvf8);
# the weights are shared, the manifest maps them (entries_by_soc) and the runtime picks the chip's set
ATT_INT8MM_M5 = os.environ.get("ATT_INT8MM_M5")
_FORMS_OVERRIDE = None  # set by Entry.forward while an alternative function (M5) is traced
P8_STATS = None  # host research only: a list collects (rounded-to-zero, masked) fractions of the pvn weight codes
S8_STATS = None  # host research only: a list collects max |raw QK score| per tile (s8 step choice)
# score steps: 1/4 covers real Qwen3.8 scores (up to about 32) at 1.2% attention error (v8: true scores); kv8 raw
# scores before the key scales need a finer step (set ATT_S8_UNIT)
ATT_S8_UNIT = float(os.environ.get("ATT_S8_UNIT", "0.25"))
ATT_S8B_UNIT = float(os.environ.get("ATT_S8B_UNIT", "0.25"))
ATT_S8R_SHIFT = float(os.environ.get("ATT_S8R_SHIFT", "8"))
S8B_STATS = None  # host research only: max |score| per tile after the key scales, masked entries excluded
ATT_INT8MM_UNITS = [float(u) for u in os.environ.get("ATT_INT8MM_UNITS", "0.0625,0.0078125,0.25").split(",")]  # act, cache, out


def quant8(x, unit, zero, dtype=torch.int8, axis=0, minval=None):
    return torch.ops.coreai.quantize(x, unit, dtype, zero_point=zero, minval=minval, axis=axis)
TAPS = M.TAPS
try:
    CFG = M.cfg()
    nk, nv = CFG["linear_num_key_heads"], CFG["linear_num_value_heads"]
    dk, dv = CFG["linear_key_head_dim"], CFG["linear_value_head_dim"]
    kd, vd = nk * dk, nv * dv
    cdim = 2 * kd + vd
    nh, nkv, hd, hid = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"], CFG["hidden_size"]
    grp, rot = nh // nkv, int(hd * CFG["rope_parameters"]["partial_rotary_factor"])
    EPS = CFG["rms_norm_eps"]
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    # Jeff convert calls bind_builder_cfg before using these. A missing 27B checkpoint must not fail import.
    CFG = {}
    nk = nv = dk = dv = kd = vd = cdim = nh = nkv = hd = hid = grp = rot = 0
    EPS = 1e-6
SMALL = ("input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.conv1d.weight",
         "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
         "linear_attn.norm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight")
KNOWN_LUTS: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # sha1(fp16 dense weight) -> (lut, idx)


def wkey(a: np.ndarray) -> str:
    """Content key of a weight (shape-free: the program hands conv weights over as (Cout, Cin, 1, 1))."""
    a = np.ascontiguousarray(a, np.float16)
    return f"{a.size}:{hashlib.sha1(a.data).hexdigest()}"


# ---- weights -----------------------------------------------------------------------------------------------------
def layer_arrays(ck, i: int) -> dict:
    """The layer's weights exactly as the v4 build consumes them (checkpoint small tensors + export matrices)."""
    w = ck.layer(i)
    q = M.layer_quant(ck, i, w)
    arrs = {f"{i}/{k}": w[k].float().numpy() for k in SMALL if k in w}
    for k, v in q.items():
        if k == "mlp.rotation":
            arrs[f"{i}/mlp.rotation"] = np.array(v, np.int64)
            continue
        lr = None
        if isinstance(v[0], str) and v[0] == "int8":
            arrs[f"{i}/{k}/int8"], arrs[f"{i}/{k}/scale"] = v[1], np.asarray(v[2], np.float16)
            lr = v[3] if len(v) > 3 else None
        elif isinstance(v[0], str) and v[0] == "dense":
            arrs[f"{i}/{k}/dense"] = np.asarray(v[1], np.float16)
        else:
            lut, idx, s = v[:3]
            arrs[f"{i}/{k}/lut"], arrs[f"{i}/{k}/idx"] = np.asarray(lut, np.float16), np.asarray(idx, np.uint8)
            if s is not None:
                arrs[f"{i}/{k}/scale"] = np.asarray(s, np.float16).reshape(-1)
            lr = v[4] if len(v) > 4 else None
        if lr is not None:  # trained low-rank factors a (Cout, r), b (r, Cin), in the conv input's basis
            arrs[f"{i}/{k}/lr_a"], arrs[f"{i}/{k}/lr_b"] = np.asarray(lr[0], np.float16), np.asarray(lr[1], np.float16)
    return arrs


class QConv(nn.Module):
    """1x1 conv with an exported weight: LUT (dense lut[idx], registered for exact palettization) + per-channel scale as
    a mul after the conv, int8 (a compile-time INT8 constant with its per-channel scales; QCONV_INT8=0: dequantized to
    dense fp16) or dense."""

    def __init__(self, W: dict, key: str) -> None:
        super().__init__()
        self.register_buffer("scale", None)
        if f"{key}/lut" in W:
            lut, idx = W[f"{key}/lut"], W[f"{key}/idx"]
            cd = lut.shape[1]
            w = lut[idx].transpose(0, 2, 1).reshape(idx.shape[0] * cd, idx.shape[1]).astype(np.float16)
            KNOWN_LUTS[wkey(w)] = (lut, idx)
            if f"{key}/scale" in W:
                self.register_buffer("scale", torch.from_numpy(W[f"{key}/scale"].astype(np.float16)).view(1, -1, 1, 1))
        elif f"{key}/int8" in W and QCONV_INT8:
            import coreai_torch._compression.custom_layers  # noqa: F401  registers coreai::constexpr_blockwise_shift_scale
            codes = np.ascontiguousarray(W[f"{key}/int8"])
            self.register_buffer("w8", torch.from_numpy(codes).view(codes.shape[0], codes.shape[1], 1, 1))
            self.register_buffer("w8_scale", torch.from_numpy(np.asarray(W[f"{key}/scale"], np.float16)).view(-1, 1, 1, 1))
            w = None
        elif f"{key}/int8" in W:
            w = (W[f"{key}/int8"].astype(np.float32) * W[f"{key}/scale"].astype(np.float32)[:, None]).astype(np.float16)
        else:
            w = W[f"{key}/dense"].astype(np.float16)
        self.conv = None
        if w is not None:
            self.conv = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
            self.conv.weight = nn.Parameter(torch.from_numpy(w).view(w.shape[0], w.shape[1], 1, 1), requires_grad=False)
        self.lr_b = self.lr_a = None
        if f"{key}/lr_a" in W:  # + a @ (b @ x): fp16 low-rank error correction (two 1x1 convs, not palettized)
            a, b = W[f"{key}/lr_a"].astype(np.float16), W[f"{key}/lr_b"].astype(np.float16)
            self.lr_b = nn.Conv2d(b.shape[1], b.shape[0], 1, bias=False)
            self.lr_b.weight = nn.Parameter(torch.from_numpy(b.copy()).view(b.shape[0], b.shape[1], 1, 1), requires_grad=False)
            self.lr_a = nn.Conv2d(a.shape[1], a.shape[0], 1, bias=False)
            self.lr_a.weight = nn.Parameter(torch.from_numpy(a.copy()).view(a.shape[0], a.shape[1], 1, 1), requires_grad=False)

    def forward(self, x):
        if self.conv is None:  # INT8 constant: never expanded to an FP16 weight in the program
            y = F.conv2d(x, torch.ops.coreai.constexpr_blockwise_shift_scale(self.w8, self.w8_scale, None, None, torch.int8))
        else:
            y = self.conv(x)
        y = y if self.scale is None else y * self.scale
        return y if self.lr_a is None else y + self.lr_a(self.lr_b(x))


class Hadamard(nn.Module):
    """x (1, n, 1, T) -> x M, M = blockdiag(diag(signs) H_1024) / 32 (the export's online rotation, same seeds)."""

    def __init__(self, n: int, seed: int, block: int = 1024) -> None:
        super().__init__()
        from scipy.linalg import hadamard
        h = hadamard(block)
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], n)
        wt = np.concatenate([(signs[b * block:(b + 1) * block, None] * h).T for b in range(n // block)])
        self.conv = nn.Conv2d(n, n, 1, groups=n // block, bias=False)
        self.conv.weight = nn.Parameter(torch.from_numpy((wt / np.sqrt(block)).astype(np.float16)).view(n, block, 1, 1),
                                        requires_grad=False)

    def forward(self, x):
        return self.conv(x)


def rms_hidden(x, w_plus):
    """Scale-free RMSNorm over channels: xs = x / max|x| keeps the squares in [0, 1] (tiny layer-0 embeddings made
    Core AI's lowering of the /64-prescaled form mis-normalize; massive activations overflow the plain form)."""
    m = x.abs().amax(1, keepdim=True).clamp_min(1e-3)
    xs = x / m
    return xs * torch.rsqrt((xs * xs).mean(1, keepdim=True) + EPS / (m * m)) * w_plus


def rms_last(x, w, eps=EPS):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def softplus(x):  # the ANE's fp16 softplus returns 0 above ~11 (exp overflow)
    return F.relu(x) + torch.log(1 + torch.exp(-torch.abs(x)))


# ANE fp16 DeltaNet / MLP numerics (ANE_DELTANET_NUMERICS.md), same env switches as qwen38_ane_chunk.py.
# SILU: the ANE's native silu has ~1e-3 ABSOLUTE error near 0 and >99% of the DeltaNet conv outputs lie in [-0.5, 0.5]
#   (12% error on q / k / v); "tanh" (default) = 0.5 x (1 + tanh(x / 2)) for the conv and the gate silu(z), "native" =
#   F.silu. MLP_SILU: the same for the MLP gate (default tanh here; 5-6% error on silu(gate) * up with native).
# GDN_SQ / GDN_SV: q . S is fp16-subnormal (median 4e-5) and the ANE flushes it; q is scaled by GDN_SQ and v (hence u,
#   the pending rows and the recurrent state) by GDN_SV, and the gated RMSNorm absorbs it exactly with
#   eps * (GDN_SQ * GDN_SV)^2. The state is host-owned I/O starting at 0, so the host code does not change.
# MLP_DS_TABLE=<json {"ds": {layer: scale}}> (else MLP_DS): the down projection's input is scaled by ds and its output
#   by 1 / ds. Core ML needs it (per-channel scale folded into the weight -> fp16-subnormal products); here the scale
#   is a mul after the conv (|LUT| ~0.9), and ds = 64 changed nothing measurable at layers 9 / 11: leave it unset.
SILU, MLP_SILU = os.environ.get("SILU", "tanh"), os.environ.get("MLP_SILU", "tanh")
GDN_SQ, GDN_SV = float(os.environ.get("GDN_SQ", "16")), float(os.environ.get("GDN_SV", "64"))
MLP_DS = float(os.environ.get("MLP_DS", "1"))
# GDN_FAST (default 1): exact rewrites of the DeltaNet core that the M6 ANE runs faster (anemll-forge m6_gdn_bench): the
#   triangular solve as one matmul with a Neumann-product inverse (tri_solve), the causal conv1d as one native
#   depthwise conv, and one state matmul per prefill sub-chunk instead of two. GDN_FAST=0: the release graph.
GDN_FAST = os.environ.get("GDN_FAST", "1") == "1"
# INT8 export weights (full-attention k / v projections) stay compile-time INT8 constants with their per-channel
# scales; QCONV_INT8=0 expands them to dense FP16 as builds before 4 October 2026 did (byte-for-byte reproduction)
QCONV_INT8 = os.environ.get("QCONV_INT8", "1") == "1"
MLP_DS_TABLE = json.loads(Path(os.path.expanduser(os.environ["MLP_DS_TABLE"])).read_text())["ds"] \
    if os.environ.get("MLP_DS_TABLE") else None
DBG_O = os.environ.get("DBG_O") == "1"   # debug: every layer also outputs the tensor entering out_proj / o_proj (o<j>_dbg)
_DBG: list = []


def silu(x, mode: str = SILU):
    if mode == "native":
        return F.silu(x)
    half = x * 0.5
    return half * (1 + torch.tanh(half))


def tri(n: int, strict: bool):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return torch.from_numpy(((i > j) if strict else (i >= j)).astype(np.float16))


def fwd_sub(n, rhs, rows: int):
    """Solve (I + N) X = rhs for strictly lower-triangular N over the last-but-one axis (rows), row by row (the
    doubling-inverse chain of computed matmuls keeps MPSGraph off the ANE). This is the UT transform of the chunkwise
    delta rule (WY representation, T = (I - A)^-1 with A = -N, by forward substitution) as explained in Songlin Yang,
    "DeltaNet Explained (Part II)", https://sustcsonglin.github.io/blog/2024/deltanet-2/"""
    xs = [rhs[..., 0:1, :]]
    for t in range(1, rows):
        xs.append(rhs[..., t:t + 1, :] - (n[..., t, 0:t].unsqueeze(-1) * torch.cat(xs, -2)).sum(-2, keepdim=True))
    return torch.cat(xs, -2)


def inv_unit_lower(n, rows: int):
    """(I + N)^-1 for strictly lower-triangular (nilpotent) N: (I - N)(I + N^2)(I + N^4)... exactly. With A = -N this is
    the path sum I + A + ... + A^(rows-1) of the UT transform's graph view (entry [i, j] sums the weights of all paths
    from j to i; no path in a block is longer than rows - 1; Yang, "DeltaNet Explained (Part II)", see fwd_sub),
    grouped by binary path length so it takes log2(rows) steps instead of rows - 1 dependent row updates. The small
    products are broadcast multiply + reduce: ANEC fails on matmul chains between computed tensors (an internal-error
    compile that sends the program to the GPU), but takes these."""
    eye = torch.eye(rows, dtype=n.dtype).expand(n.shape)
    t, p, k = eye - n, n, 2
    while k < rows:
        p = (p.unsqueeze(-1) * p.unsqueeze(-3)).sum(-2)
        t = (t.unsqueeze(-1) * (eye + p).unsqueeze(-3)).sum(-2)
        k *= 2
    return t


def tri_solve(n, rhs, rows: int):
    """(I + N) X = rhs: GDN_FAST, one matmul with the Neumann-product inverse; else the row-by-row substitution."""
    if GDN_FAST:
        return inv_unit_lower(n, rows) @ rhs
    return fwd_sub(n, rhs, rows)


# ---- layers ------------------------------------------------------------------------------------------------------
# Gated DeltaNet in the chunkwise form (8-row blocks: WY representation, UT transform, one state update per block);
# background: Songlin Yang, "DeltaNet Explained (Part II)", https://sustcsonglin.github.io/blog/2024/deltanet-2/
class GDNW(nn.Module):
    def __init__(self, W: dict, i: int) -> None:
        super().__init__()
        p = f"{i}/linear_attn."
        self.qkv, self.z, self.out = QConv(W, p + "in_proj_qkv.weight"), QConv(W, p + "in_proj_z.weight"), QConv(W, p + "out_proj.weight")
        self.a = QConv({f"{p}in_proj_a.weight/dense": W[p + "in_proj_a.weight"]}, p + "in_proj_a.weight")
        self.b = QConv({f"{p}in_proj_b.weight/dense": W[p + "in_proj_b.weight"]}, p + "in_proj_b.weight")
        self.register_buffer("cw", torch.from_numpy(W[p + "conv1d.weight"][:, 0].T.astype(np.float16).copy()))
        self.register_buffer("neg_a", torch.from_numpy((-np.exp(W[p + "A_log"])).reshape(nv, 1, 1).astype(np.float16)))
        self.register_buffer("dt", torch.from_numpy(W[p + "dt_bias"].reshape(nv, 1, 1).astype(np.float16)))
        self.register_buffer("normw", torch.from_numpy(W[p + "norm.weight"].astype(np.float16)))

    def proj(self, h, T: int):
        qkv = self.qkv(h).reshape(cdim, T)
        z = self.z(h).reshape(nv, dv, T).permute(0, 2, 1)
        return qkv, z, self.b(h).reshape(nv, T, 1), self.a(h).reshape(nv, T, 1)

    def qkv_heads(self, rows, T: int):
        if GDN_FAST:  # the causal conv1d as one depthwise conv over the channel-major (1, cdim, 1, T + 3) rows
            conv = F.conv2d(rows.transpose(0, 1).reshape(1, cdim, 1, T + 3),
                            self.cw.transpose(0, 1).reshape(cdim, 1, 1, 4), groups=cdim)
            conv = silu(conv).reshape(cdim, T)
        else:
            conv = rows[0:T] * self.cw[0:1] + rows[1:T + 1] * self.cw[1:2] + rows[2:T + 2] * self.cw[2:3] + rows[3:T + 3] * self.cw[3:4]
            conv = silu(conv).transpose(0, 1)
        qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

        def heads(t):
            t = t.reshape(nk, dk, T).permute(0, 2, 1)
            return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

        def l2n(t, s):
            return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s
        vh = vv.reshape(nv, dv, T).permute(0, 2, 1)
        vh = vh * GDN_SV if GDN_SV != 1 else vh    # q . S out of fp16 subnormals (see GDN_SQ / GDN_SV)
        return l2n(heads(qq), dk ** -0.5 * GDN_SQ), l2n(heads(kk), 1.0), vh

    def commit_pending(self, rec, pend, commit, commit_last):
        kp, up, wkp = pend[:, 0:P, 0:dk], pend[:, P:2 * P, 0:dv], pend[:, 2 * P:3 * P, 0:dk]
        cum_p = pend[:, 3 * P:3 * P + 1, 0:P].reshape(nv, P, 1)
        total = (cum_p * commit_last).sum(1, keepdim=True)
        kd_ = kp * commit * torch.exp(torch.clamp(total - cum_p, max=0))
        return rec * torch.exp(total) + kd_.transpose(1, 2) @ (up - wkp @ rec)

    def finish(self, o, z, T: int):
        # o = q . S carries the GDN_SQ * GDN_SV scale: eps * scale^2 makes the gated RMSNorm exactly the unscaled one
        o = rms_last(o, self.normw, EPS * (GDN_SQ * GDN_SV) ** 2) * silu(z)
        o = o.permute(0, 2, 1).reshape(1, vd, 1, T)
        if DBG_O:
            _DBG.append(o)
        return self.out(o)

    def verify(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T: int):
        """T = P rows, lazy commit: returns (y, conv rows (T + 3), committed state S', this call's pending rows)."""
        qkv, z, b, a = self.proj(h, T)
        rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)                  # (T + 3, cdim)
        qh, kh, vh = self.qkv_heads(rows, T)
        beta, g = torch.sigmoid(b), softplus(a + self.dt) * self.neg_a
        s1 = self.commit_pending(rec, pend, commit, commit_last)
        l_inc, l_str = tri(T, False), tri(T, True)
        cum = (g.reshape(nv, 1, T) @ l_inc.T).reshape(nv, T, 1)
        pair = torch.exp(torch.clamp(cum - cum.reshape(nv, 1, T), max=0)) * l_inc
        kb, vb = kh * beta, vh * beta
        n = (kb @ kh.transpose(1, 2)) * (pair * l_str)
        x = tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), T)
        u, wk = x[..., :dv], x[..., dv:]
        crow = torch.cat([cum.reshape(nv, 1, T), torch.zeros(nv, 1, dv - T, dtype=cum.dtype)], 2)
        pend_out = torch.cat([kh, u, wk, crow], 1)
        vn = u - wk @ s1
        o = (qh * torch.exp(cum)) @ s1 + ((qh @ kh.transpose(1, 2)) * pair) @ vn
        return self.finish(o, z, T), rows, s1, pend_out

    def prefill(self, h, conv_rows, conv_sel, conv_sel_out, rec, pend, commit, commit_last, valid, T: int):
        """T > P rows, all committed (padding rows: valid = 0 -> no state change). Returns (y, conv rows in the P-row
        layout (the 3 rows ending at the last valid token, then zeros), state, zero pending rows)."""
        qkv, z, b, a = self.proj(h, T)
        rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)                  # (T + 3, cdim)
        qh, kh, vh = self.qkv_heads(rows, T)
        v1 = valid.reshape(1, T, 1)
        beta, g = torch.sigmoid(b) * v1, softplus(a + self.dt) * self.neg_a * v1
        s = self.commit_pending(rec, pend, commit, commit_last)
        NB, C = T // C_SUB, C_SUB
        l_inc, l_str = tri(C, False), tri(C, True)
        q4, k4, v4 = qh.reshape(nv, NB, C, dk), kh.reshape(nv, NB, C, dk), vh.reshape(nv, NB, C, dv)
        b4, g4 = beta.reshape(nv, NB, C, 1), g.reshape(nv, NB, C, 1)
        cum = (g4.reshape(nv, NB, 1, C) @ l_inc.T).reshape(nv, NB, C, 1)                   # within-sub-chunk cumsum
        pair = torch.exp(torch.clamp(cum - cum.reshape(nv, NB, 1, C), max=0)) * l_inc
        kb, vb = k4 * b4, v4 * b4
        n = (kb @ k4.transpose(-1, -2)) * (pair * l_str)
        x = tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), C)                         # all sub-chunks at once
        u, wk = x[..., :dv], x[..., dv:]
        total = cum[:, :, C - 1:C, :]                                                       # (nv, NB, 1, 1)
        qd, kdc = q4 * torch.exp(cum), k4 * torch.exp(total - cum)
        intra = (q4 @ k4.transpose(-1, -2)) * pair
        outs = []
        wq = torch.cat([wk, qd], 2) if GDN_FAST else None                                  # (nv, NB, 2C, dk)
        for bi in range(NB):
            if GDN_FAST:  # wk @ S and qd @ S in one matmul
                ws = wq[:, bi] @ s
                vn = u[:, bi] - ws[:, :C]
                outs.append(ws[:, C:] + intra[:, bi] @ vn)
            else:
                vn = u[:, bi] - wk[:, bi] @ s
                outs.append(qd[:, bi] @ s + intra[:, bi] @ vn)
            s = s * torch.exp(total[:, bi]) + kdc[:, bi].transpose(1, 2) @ vn
        conv_out = torch.cat([conv_sel_out @ rows, torch.zeros(P, cdim, dtype=rows.dtype)], 0)  # (P + 3, cdim)
        return self.finish(torch.cat(outs, 1), z, T), conv_out, s, pend * 0


def dequant8(codes, unit, zero, axis=0, minval=None, input_dtype=None):
    """INT8 cache codes to FP16 codes * unit with the native op (host tests substitute a torch version)."""
    return torch.ops.coreai.dequantize(codes, unit, zero_point=zero, minval=minval, axis=axis, input_dtype=input_dtype,
                                       output_dtype=torch.float16)


class AttnW(nn.Module):
    def __init__(self, W: dict, i: int) -> None:
        super().__init__()
        p = f"{i}/self_attn."
        self.layer_index = i
        self.q, self.k, self.v, self.o = (QConv(W, p + f"{m}_proj.weight") for m in "qkvo")
        self.register_buffer("qn", torch.from_numpy((1 + W[p + "q_norm.weight"]).astype(np.float16)))
        self.register_buffer("kn", torch.from_numpy((1 + W[p + "k_norm.weight"]).astype(np.float16)))
        self.cache_v8, self.cache_k8 = KV_CACHE_DTYPE in ("v8", "kv8"), KV_CACHE_DTYPE == "kv8"
        if KV_CACHE_DTYPE in ("v8", "kv8", "both"):
            import coreai_torch._compression.custom_layers  # registers native quantize/dequantize
            self.register_buffer("v8_unit", torch.tensor(1 / 128, dtype=torch.float16))
            self.register_buffer("v8_zero", torch.tensor(0, dtype=torch.int8))
            self.register_buffer("mm8_unit", torch.tensor(ATT_INT8MM_UNITS[0], dtype=torch.float16))  # ATT_INT8MM operands
            self.register_buffer("mm8_cache_unit", torch.tensor(ATT_INT8MM_UNITS[1], dtype=torch.float16))
            self.register_buffer("mm8_out_unit", torch.tensor(ATT_INT8MM_UNITS[2], dtype=torch.float16))
            self.register_buffer("p8_unit", torch.tensor(1 / 127, dtype=torch.float16))  # pvn: weights in [0, 1]
            self.register_buffer("p8a_unit", torch.tensor(1 / 255, dtype=torch.float16))  # pvta: [0, 1] on -128..127
            self.register_buffer("p8a_zero", torch.tensor(-128, dtype=torch.int8))
            self.register_buffer("p8u_zero", torch.tensor(0, dtype=torch.uint8))
            self.register_buffer("p8_minval", torch.tensor(0.0, dtype=torch.float16))  # pvtm
            self.register_buffer("s8_unit", torch.tensor(ATT_S8_UNIT, dtype=torch.float16))  # s8: raw scores
            self.register_buffer("t8_unit", torch.tensor(1 / 8, dtype=torch.float16))  # t8: s - m_t in [-16, 0]
            self.register_buffer("s8b_unit", torch.tensor(ATT_S8B_UNIT, dtype=torch.float16))  # s8b: true scores
            # pvf8 / sm8 FP8 scale: values in [0, 1] become at most 64. The M6 ANE computed INT8 x FP8 PV wrongly for a
            # head whose FP8 operand reached about 200 (scale 1/256), although e4m3 holds 448; at 100 and below it matches
            self.register_buffer("pf8_unit", torch.tensor(float(os.environ.get("ATT_PF8_UNIT", 1 / 64)),
                                                          dtype=torch.float16))
            self.register_buffer("pf5_unit", torch.tensor(1 / 32768, dtype=torch.float16))  # pvf5: e5m2 max 57344

    def forward(self, h, cos, sin, mask, k_st, v_st, ctx: int, T: int, vscale=None, cache_v8=None, kscale=None,
                cache_k8=None):
        cache_v8 = self.cache_v8 if cache_v8 is None else cache_v8
        cache_k8 = self.cache_k8 if cache_k8 is None else cache_k8
        def tmajor(x, c):
            return x.reshape(c, T).transpose(0, 1)
        qg = tmajor(self.q(h), 2 * nh * hd).reshape(T, nh, 2 * hd)
        qh, gate = rms_last(qg[:, :, :hd], self.qn), qg[:, :, hd:].reshape(T, nh * hd)
        kh = rms_last(tmajor(self.k(h), nkv * hd).reshape(T, nkv, hd), self.kn)
        vh = tmajor(self.v(h), nkv * hd).reshape(T, nkv, hd)
        c3, s3 = cos.reshape(T, 1, rot), sin.reshape(T, 1, rot)

        def rope(t):
            r, rest = t[..., :rot], t[..., rot:]
            return torch.cat([r * c3 + torch.cat([-r[..., rot // 2:], r[..., :rot // 2]], -1) * s3, rest], -1)
        qh = rope(qh)
        kt, vt = rope(kh).permute(1, 0, 2), vh.permute(1, 0, 2)                              # (nkv, T, hd)
        qg4 = qh.reshape(T, nkv, grp, hd).permute(1, 2, 0, 3).reshape(nkv, grp * T, hd)
        causal = (1 - tri(T, False)) * -1e4
        sc_b = ((qg4 @ kt.transpose(1, 2)) * hd ** -0.5).reshape(nkv, grp * T, T) + causal.repeat(grp, 1)  # rows (g, t)
        forms = _FORMS_OVERRIDE if _FORMS_OVERRIDE is not None else \
            ATT_INT8MM_BY_LAYER.get(getattr(self, "layer_index", -1), ATT_INT8MM)
        mm8 = set(filter(None, forms.split(","))) if cache_v8 else set()  # INT8 values: v8 and kv8

        def has(*forms):
            return any(f in mm8 for f in forms)
        if not cache_k8 and has("qk", "both", "qkt", "botht", "qkto", "botho", "bothn"):
            raise ValueError(f"ATT_INT8MM {sorted(mm8)}: these forms read INT8 key codes (kv8 only)")
        tile_v = cache_v8 and ATT_TILE_DEQUANT and not mm8  # per-tile dequantize (see ATT_TILE_DEQUANT)
        tile_k = cache_k8 and ATT_TILE_DEQUANT and not mm8
        if cache_v8 and not tile_v and not has("pv", "both", "botht", "pvdq", "pvo", "botho", "pvn", "bothn", "pvt", "pvta", "pvtu", "pvtm",
                                              "pvf8", "pvf5"):
            v_st = dequant8(v_st, self.v8_unit, self.v8_zero)
        if cache_k8 and not tile_k and not has("qk", "both", "qkt", "botht", "qkto", "botho", "bothn"):
            k_st = dequant8(k_st, self.v8_unit, self.v8_zero)

        if KV_KEYS_T and has("qk", "both", "qkt", "botht", "qkto", "botho", "bothn"):
            raise ValueError("KV_KEYS_T: the INT8 key-code forms read the (token, head dim) layout")

        def kT(a, b):
            """History keys a:b as the QK operand (KV head, head dim, b - a)."""
            if KV_KEYS_T:
                return dequant8(k_st[:, :, a:b], self.v8_unit, self.v8_zero) if tile_k else k_st[:, :, a:b]
            return (dequant8(k_st[:, a:b], self.v8_unit, self.v8_zero) if tile_k else k_st[:, a:b]).transpose(1, 2)

        def vtile(a, b):
            return dequant8(v_st[:, a:b], self.v8_unit, self.v8_zero) if tile_v else v_st[:, a:b]
        blk = ATT_BLOCK_PREFILL if T > P else ATT_BLOCK
        form = ATT_SOFTMAX_PREFILL if T > P else ATT_SOFTMAX
        if ctx <= blk and not cache_v8 and not cache_k8 and not STABLE_ATTN:
            sc_h = ((qg4 @ (k_st if KV_KEYS_T else k_st.transpose(1, 2))) * hd ** -0.5) + mask.reshape(1, 1, ctx)
            pr = torch.softmax(torch.cat([sc_h, sc_b], -1), -1)
            o = pr[:, :, :ctx] @ v_st + pr[:, :, ctx:] @ vt
        else:
            # history in ATT_BLOCK slices (ANEC fails on the 32K / 64K single-softmax graph): one global max over all
            # blocks, then exp(s - m) per block - exactly softmax over [history | block], no tensor wider than a block
            edges = list(range(0, ctx, blk)) + [ctx]
            spans = list(zip(edges[:-1], edges[1:]))
            if has("s8r"):  # scores relative to the row's block maximum (exact: softmax is shift-invariant)
                shift = sc_b.amax(-1, keepdim=True) + ATT_S8R_SHIFT
                sc_b = sc_b - shift
            if has("qkn"):  # INT8 queries per row in [-1, 1] x INT8 key codes; scores rescaled per row
                rq = torch.clamp_min(qg4.abs().amax(-1, keepdim=True), 1e-4)
                qn8 = dequant8(quant8(qg4 * (1 / rq), self.p8_unit, self.v8_zero), self.p8_unit, self.v8_zero)
                qscale = rq * hd ** -0.5
            if has("qkf"):  # the same per-row scale as the pair's per-axis scale (queries flattened to rows x hd)
                q2 = qg4.reshape(nkv * grp * T, hd)
                sq = torch.clamp_min(q2.abs().amax(-1, keepdim=True), 1e-4) * (1 / 127)
                qf8 = dequant8(quant8(q2, sq, None, axis=0), sq, None, axis=0).reshape(nkv, grp * T, hd)

            def hist(a, b):
                if has("qkn"):
                    s_ = (qn8 @ kT(a, b)) * qscale
                elif has("qkf"):
                    s_ = (qf8 @ kT(a, b)) * hd ** -0.5
                elif has("qk", "both", "bothn"):  # INT8 queries x INT8 key codes (timing research)
                    qd = dequant8(quant8(qg4, self.mm8_unit, self.v8_zero), self.mm8_unit, self.v8_zero)
                    s_ = (qd @ dequant8(k_st[:, a:b], self.mm8_cache_unit, self.v8_zero).transpose(1, 2)) * hd ** -0.5
                elif has("nomm"):  # timing only: scores of the same shape without the QK multiply-adds
                    s_ = qg4.sum(-1, keepdim=True) * kT(a, b).sum(1).reshape(nkv, 1, b - a) * hd ** -0.5
                elif has("qkt", "botht", "qkto", "botho"):  # keys as the untransposed operand: (K Q^T)^T
                    qd = dequant8(quant8(qg4, self.mm8_unit, self.v8_zero), self.mm8_unit, self.v8_zero)
                    raw = dequant8(k_st[:, a:b], self.mm8_cache_unit, self.v8_zero) @ qd.transpose(1, 2)
                    if has("qkto", "botho"):  # INT8 output boundary
                        raw = dequant8(quant8(raw, self.mm8_out_unit, self.v8_zero), self.mm8_out_unit, self.v8_zero)
                    s_ = raw.transpose(1, 2) * hd ** -0.5
                else:
                    s_ = (qg4 @ kT(a, b)) * hd ** -0.5
                if has("s8r") and not cache_k8:  # v8: QK gives the true scores
                    s_ = s_ - shift
                if S8_STATS is not None:
                    S8_STATS.append(float(s_.abs().max()))
                if has("s8"):  # INT8 raw scores: QK writes 8-bit, the softmax passes read 8-bit
                    s_ = dequant8(quant8(s_, self.s8_unit, self.v8_zero), self.s8_unit, self.v8_zero)
                if cache_k8:  # key scales per token on the scores: q . (codes / 128) * (scale * 128) = q . k
                    s_ = s_ * (kscale[:, a:b].reshape(nkv, 1, b - a) * 128)
                    if has("s8r"):
                        s_ = s_ - shift
                mk = mask[:, a:b].reshape(1, 1, b - a)
                if S8B_STATS is not None:
                    S8B_STATS.append(float((s_ * (mk > -1)).abs().max()))
                if has("s8b"):  # INT8 scores after the key scales and the mask (masked entries clip to -128 steps)
                    return dequant8(quant8(s_ + mk, self.s8b_unit, self.v8_zero), self.s8b_unit, self.v8_zero)
                return s_ + mk
            if form == "online":
                # running max over [block | tiles]: partial sums rescale when it grows, so each tile's scores are
                # used as soon as they exist instead of staying live until the global max is known
                m = sc_b.amax(-1, keepdim=True)
                e_b = torch.exp(sc_b - m)
                den, num = e_b.sum(-1, keepdim=True), e_b @ vt
                for a, b in spans:
                    s_ = hist(a, b)
                    m_new = torch.maximum(m, s_.amax(-1, keepdim=True))
                    alpha = torch.exp(m - m_new)
                    e_ = torch.exp(s_ - m_new)
                    den = den * alpha + e_.sum(-1, keepdim=True)
                    if cache_v8:
                        e_ = e_ * (vscale[:, a:b].reshape(nkv, 1, b - a) * 128)
                    num = num * alpha + e_ @ vtile(a, b)
                    m = m_new
            elif form == "split":
                # every tile independent (its own max, sum and output); only these small partials are combined, so
                # no tile waits for another and no tile's scores outlive it
                m_b = sc_b.amax(-1, keepdim=True)
                e_b = torch.exp(sc_b - m_b)
                parts = [(m_b, e_b.sum(-1, keepdim=True), e_b @ vt)]
                for a, b in spans:
                    s_ = hist(a, b)
                    m_i = s_.amax(-1, keepdim=True)
                    e_ = torch.exp(s_ - m_i)
                    d_i = e_.sum(-1, keepdim=True)
                    if cache_v8:
                        e_ = e_ * (vscale[:, a:b].reshape(nkv, 1, b - a) * 128)
                    parts.append((m_i, d_i, e_ @ vtile(a, b)))
                m = parts[0][0]
                for m_i, _, _ in parts[1:]:
                    m = torch.maximum(m, m_i)
                den = num = None
                for m_i, d_i, n_i in parts:
                    w = torch.exp(m_i - m)
                    den = d_i * w if den is None else den + d_i * w
                    num = n_i * w if num is None else num + n_i * w
            else:
                scs = [hist(a, b) for a, b in spans]
                mts = [s_.amax(-1, keepdim=True) for s_ in scs]
                m = sc_b.amax(-1, keepdim=True)
                for m_t in mts:
                    m = torch.maximum(m, m_t)
                e_b = torch.exp(sc_b - m)
                den, num = e_b.sum(-1, keepdim=True), e_b @ vt
                for s_, m_t, (a, b) in zip(scs, mts, spans):
                    if form == "recompute":  # timing control only: a true second pass that recomputes the scores
                        s_ = hist(a, b)
                    if has("pvt", "pvta", "pvtu", "pvtm", "pvf8", "pvf5"):  # 8-bit PV, weights in [0, 1] per tile
                        t_ = s_ - m_t
                        if has("t8"):
                            t_ = dequant8(quant8(t_, self.t8_unit, self.v8_zero), self.t8_unit, self.v8_zero)
                        e_ = torch.exp(t_)
                        if has("sm8"):  # FP8 softmax probabilities; the sum below reads them
                            e_ = dequant8(quant8(e_, self.pf8_unit, None, torch.float8_e4m3fn), self.pf8_unit, None)
                        w_t = torch.exp(m_t - m)
                        den = den + e_.sum(-1, keepdim=True) * w_t
                        vs_t = vscale[:, a:b].reshape(nkv, 1, b - a) * 128
                        vmax = torch.clamp_min(vs_t.amax(-1, keepdim=True), 1e-4)
                        pn = e_ * (vs_t * (1 / vmax))
                        unit, zero, dt, mv = ((self.p8a_unit, self.p8a_zero, torch.int8, None) if has("pvta") else
                                              (self.p8a_unit, self.p8u_zero, torch.uint8, None) if has("pvtu") else
                                              (self.p8a_unit, None, torch.int8, self.p8_minval) if has("pvtm") else
                                              (self.pf8_unit, None, torch.float8_e4m3fn, None) if has("pvf8") else
                                              (self.pf5_unit, None, torch.float8_e5m2, None) if has("pvf5") else
                                              (self.p8_unit, self.v8_zero, torch.int8, None))
                        if P8_STATS is not None:  # weights that become zero codes (host research only)
                            z = ((pn / unit.float()).to(dt).float() == 0) if dt.is_floating_point else (
                                (pn / unit.float()).round() == 0)
                            P8_STATS.append((float(z.logical_and(pn > 0).float().mean()), float((pn == 0).float().mean())))
                        ed = dequant8(quant8(pn, unit, zero, dt, minval=mv), unit, zero, minval=mv,
                                      input_dtype=dt if mv is not None else None)
                        # both matmul operands as quantize -> dequantize, the INT8 value codes too (their quantize is
                        # exact: codes / 128 * 128); this compiler gives the same program without it, the pair is the
                        # form Apple's W8A8 export emits for every quantized operand
                        vd = dequant8(v_st[:, a:b], self.v8_unit, self.v8_zero)
                        vd = dequant8(quant8(vd, self.v8_unit, self.v8_zero), self.v8_unit, self.v8_zero)
                        num = num + (ed @ vd) * (w_t * vmax)
                        continue
                    e_ = torch.exp(s_ - m)
                    den = den + e_.sum(-1, keepdim=True)
                    if cache_v8:
                        # Move dynamic V scales onto exp scores, leaving the global denominator unchanged.
                        e_ = e_ * (vscale[:, a:b].reshape(nkv, 1, b - a) * 128)
                    if has("pvn", "bothn"):  # INT8 weights scaled to [0, 1] per row and tile x INT8 value codes
                        r = torch.clamp_min(e_.amax(-1, keepdim=True), 1e-4)
                        pn = e_ * (1 / r)
                        if P8_STATS is not None:
                            P8_STATS.append((float(((pn * 127).round() == 0).logical_and(pn > 0).float().mean()),
                                             float((pn == 0).float().mean())))
                        ed = dequant8(quant8(pn, self.p8_unit, self.v8_zero), self.p8_unit, self.v8_zero)
                        num = num + (ed @ dequant8(v_st[:, a:b], self.v8_unit, self.v8_zero)) * r
                    elif has("pv", "both", "botht", "pvo", "botho"):  # INT8 exp weights x INT8 value codes
                        ed = dequant8(quant8(e_, self.mm8_unit, self.v8_zero), self.mm8_unit, self.v8_zero)
                        pv_ = ed @ dequant8(v_st[:, a:b], self.mm8_cache_unit, self.v8_zero)
                        if has("pvo", "botho"):  # INT8 output boundary
                            pv_ = dequant8(quant8(pv_, self.mm8_out_unit, self.v8_zero), self.mm8_out_unit, self.v8_zero)
                        num = num + pv_
                    elif has("pvdq"):  # control: the same per-tile V dequantize, FP16 exp weights
                        num = num + e_ @ dequant8(v_st[:, a:b], self.v8_unit, self.v8_zero)
                    elif has("nomm"):  # timing only: a partial output of the same shape without the PV multiply-adds
                        num = num + e_.sum(-1, keepdim=True) * v_st[:, a:b].sum(1, keepdim=True)
                    else:
                        num = num + e_ @ vtile(a, b)
            o = num / den
        o = o.reshape(nkv, grp, T, hd).permute(2, 0, 1, 3).reshape(T, nh * hd) * torch.sigmoid(gate)
        o = o.transpose(0, 1).reshape(1, nh * hd, 1, T)
        if DBG_O:
            _DBG.append(o)
        return self.o(o), kt, vt


class LayerW(nn.Module):
    def __init__(self, W: dict, i: int) -> None:
        super().__init__()
        self.i, self.kind = i, CFG["layer_types"][i]
        self.mix = GDNW(W, i) if self.kind == "linear_attention" else AttnW(W, i)
        self.register_buffer("ln1", torch.from_numpy((1 + W[f"{i}/input_layernorm.weight"]).reshape(1, -1, 1, 1).astype(np.float16)))
        self.register_buffer("ln2", torch.from_numpy((1 + W[f"{i}/post_attention_layernorm.weight"]).reshape(1, -1, 1, 1).astype(np.float16)))
        self.gate, self.up, self.down = (QConv(W, f"{i}/mlp.{m}_proj.weight") for m in ("gate", "up", "down"))
        seeds = W.get(f"{i}/mlp.rotation")
        self.rin = Hadamard(hid, int(seeds[0])) if seeds is not None else None
        self.rmid = Hadamard(CFG["intermediate_size"], int(seeds[1])) if seeds is not None else None
        self.ds = float(MLP_DS_TABLE[str(i)]) if MLP_DS_TABLE is not None else MLP_DS

    def mlp(self, x):
        h = rms_hidden(x, self.ln2)
        h = self.rin(h) if self.rin is not None else h
        a = silu(self.gate(h), MLP_SILU) * self.up(h)
        a = self.rmid(a) if self.rmid is not None else a
        if self.ds != 1:  # keep the down projection's products out of fp16 subnormals (see MLP_DS_TABLE)
            return x + self.down(a * self.ds) * (1 / self.ds)
        return x + self.down(a)


ANE_MAX_DIM = 65536


def kv_len(ctx: int, T: int) -> int:
    """KV history rows of every entry at context ctx. The single-softmax graph concatenated [history | block] along one
    axis, which the ANE caps at 65536 (64K + 8 failed ANEC for the whole package), so entries up to 64K keep
    65536 - the largest block (the runtime binds the verify and prefill entries of a context to the same KV buffers;
    T is only checked). Longer entries always take the tiled history path, which never forms that tensor: 80K and
    100K attention cores compile fully onto the ANE (anemll-forge long-ctx notes), so they hold their whole context."""
    assert T == P or T in TPS, T
    if ctx > ANE_MAX_DIM:
        return ctx
    return min(ctx, ANE_MAX_DIM - max([P] + TPS))


class Entry(nn.Module):
    """One entry point of a chunk: T rows, KV history kv_len(ctx, T), verify (T = P, lazy commit) or prefill (T > P)."""

    def __init__(self, layers: nn.ModuleList, ctx: int, T: int, kv_cache_dtype=None, att_forms=None) -> None:
        super().__init__()
        self.layers, self.T, self.prefill = layers, T, T > P
        self.att_forms = att_forms  # attention forms of this function (ATT_INT8MM_M5); None: ATT_INT8MM
        mode = kv_cache_dtype or ("fp16" if KV_CACHE_DTYPE == "both" else KV_CACHE_DTYPE)
        if mode not in KV_INPUTS:
            raise ValueError("Entry KV format must be fp16, v8 or kv8")
        self.kv_mode, self.cache_v8, self.cache_k8 = mode, mode in ("v8", "kv8"), mode == "kv8"
        self.ctx = kv_len(ctx, T)
        self.gdn_j = [j for j, l in enumerate(layers) if l.kind == "linear_attention"]
        self.att_j = [j for j, l in enumerate(layers) if l.kind != "linear_attention"]
        last = layers[-1].i
        self.taps = [l.i for l in layers if l.i in TAPS and l.i != last]

    def input_names(self):
        return (["x", "cos", "sin", "mask", "conv_sel", "commit", "commit_last"]
                + (["conv_sel_out", "valid"] if self.prefill else [])
                + [f"{s}{j}" for j in self.gdn_j for s in ("conv", "rec", "pend")]
                + [f"{s}{j}" for j in self.att_j for s in KV_INPUTS[self.kv_mode]])

    def output_names(self):
        return (["y"] + [f"tap{l}" for l in self.taps]
                + [f"{s}{j}_out" for j in self.gdn_j for s in ("conv", "rec", "pend")]
                + [f"{s}{j}_new" for j in self.att_j for s in ("k", "v")]
                + ([f"o{j}_dbg" for j in range(len(self.layers))] if DBG_O else []))

    def forward(self, x, cos, sin, mask, conv_sel, commit, commit_last, *rest):
        global _FORMS_OVERRIDE
        prev, _FORMS_OVERRIDE = _FORMS_OVERRIDE, self.att_forms
        try:
            return self._forward(x, cos, sin, mask, conv_sel, commit, commit_last, *rest)
        finally:
            _FORMS_OVERRIDE = prev

    def _forward(self, x, cos, sin, mask, conv_sel, commit, commit_last, *rest):
        _DBG.clear()
        it = iter(rest)
        if self.prefill:
            conv_sel_out, valid = next(it), next(it)
        gdn_in = {j: (next(it), next(it), next(it)) for j in self.gdn_j}
        att_in = {j: {s: next(it) for s in KV_INPUTS[self.kv_mode]} for j in self.att_j}
        taps, gdn_out, att_out = [], [], []
        for j, layer in enumerate(self.layers):
            h = rms_hidden(x, layer.ln1)
            if j in gdn_in:
                cr, rc, pd = gdn_in[j]
                if self.prefill:
                    y, rows, s1, pend_out = layer.mix.prefill(h, cr, conv_sel, conv_sel_out, rc, pd, commit, commit_last,
                                                              valid, self.T)
                else:
                    y, rows, s1, pend_out = layer.mix.verify(h, cr, conv_sel, rc, pd, commit, commit_last, self.T)
                gdn_out += [rows, s1, pend_out]
            else:
                c = att_in[j]
                y, kt, vt = layer.mix(h, cos, sin, mask, c["k"], c["v"], self.ctx, self.T, c.get("vs"), self.cache_v8,
                                      c.get("ks"), self.cache_k8)
                att_out += [kt, vt]
            x = layer.mlp(x + y)
            if layer.i in self.taps:
                taps.append(x)
        return (x, *taps, *gdn_out, *att_out, *_DBG)

    def example(self):
        T, f = self.T, torch.float16
        ex = [torch.randn(1, hid, 1, T, dtype=f) * 0.02, torch.ones(T, rot, dtype=f), torch.zeros(T, rot, dtype=f),
              torch.zeros(1, self.ctx, dtype=f), torch.zeros(3, P + 3, dtype=f), torch.zeros(1, P, 1, dtype=f),
              torch.zeros(1, P, 1, dtype=f)]
        if self.prefill:
            ex += [torch.zeros(3, T + 3, dtype=f), torch.ones(1, T, 1, dtype=f)]
        for _ in self.gdn_j:
            ex += [torch.zeros(P + 3, cdim, dtype=f), torch.zeros(nv, dk, dv, dtype=f), torch.zeros(nv, 3 * P + 1, dv, dtype=f)]
        int8 = {"k": self.cache_k8, "v": self.cache_v8}
        for _ in self.att_j:
            for s in KV_INPUTS[self.kv_mode]:
                shape = (nkv, hd, self.ctx) if s == "k" and KV_KEYS_T else (nkv, self.ctx, hd)
                ex.append(torch.ones(nkv, self.ctx, dtype=f) / 128 if s in ("ks", "vs") else
                          torch.zeros(*shape, dtype=torch.int8 if int8[s] else f))
        return tuple(ex)


class Head(nn.Module):
    """Final RMSNorm + LUT4 lm_head in HEAD_PARTS row parts: x (1, hid, 1, T) -> logits (T, vocab)."""

    def __init__(self, ck, T: int = 8, parts: int = 8) -> None:
        super().__init__()
        from safetensors.torch import load_file
        self.T = T
        self.register_buffer("normw", (1 + ck.get("model.language_model.norm.weight").float()).to(torch.float16).view(1, -1, 1, 1))
        t = load_file(M.EXPORT_DIR / "lm_head.safetensors")
        q = M.as_quant(t, "lm_head")
        lut, idx, s = np.asarray(q[0], np.float16), np.asarray(q[1], np.uint8), q[2]
        v = CFG["vocab_size"]
        step = -(-v // parts)
        mods = []
        for a in range(0, v, step):
            b = min(a + step, v)
            W = {"h/lut": lut, "h/idx": idx[a:b]}
            if s is not None:
                W["h/scale"] = np.asarray(s, np.float16).reshape(-1)[a:b]
            mods.append(QConv(W, "h"))
        self.parts = nn.ModuleList(mods)

    def forward(self, x):
        h = rms_hidden(x, self.normw)
        return torch.cat([p(h).reshape(-1, self.T).transpose(0, 1) for p in self.parts], 1)


# ---- export ------------------------------------------------------------------------------------------------------
def patch_palettizer():
    from coreai_opt.coreai_utils._utils.palettize_utils import LutParams
    from coreai_opt.coreai_utils.passes import weight_palettization as wp
    if getattr(wp, "_qwen38_patched", False):
        return
    orig = wp._blockwise_compress

    def compress(original_data, mode, *args, **kwargs):
        known = KNOWN_LUTS.get(wkey(original_data))
        if known is not None:
            lut, idx = known
            cd = lut.shape[1]
            extra = (1,) * (original_data.ndim - 2)
            return LutParams(indices=idx.reshape(*idx.shape, *extra).astype(np.uint8),
                             lut=lut.reshape(*(1,) * original_data.ndim, *lut.shape), vector_axis=0 if cd > 1 else None)
        return orig(original_data, "UNIQUE", *args, **kwargs)   # Hadamard (+-1/32): exact unique values
    wp._blockwise_compress = compress
    wp._is_cluster_dim_valid = lambda op, cluster_dim, channel_axis: list(op.result.type.shape)[channel_axis] % cluster_dim == 0
    wp._qwen38_patched = True


def save_program(entries: list[tuple[str, nn.Module, list, list]], out: Path, lut_dtype=None) -> float:
    """entries: (entrypoint name, module, input names, output names). Returns MB on disk. lut_dtype (research): a
    coreai_opt DType for the LUT values, e.g. DType.INT8 (quantized LUT: the same indices, INT8 entries and a scale)."""
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    from coreai_opt.coreai_utils.common import CompressionGranularity
    from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for name, mod, ins, outs in entries:
        ep = torch.export.export(mod, mod.example() if hasattr(mod, "example") else (torch.zeros(1, hid, 1, mod.T, dtype=torch.float16),),
                                 strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=ins, output_names=outs, entrypoint_name=name)
    prog = conv.to_coreai()
    patch_palettizer()
    # palettize before optimize(): optimize folds the per-channel-scale mul into the weight (no longer a per-tensor LUT)
    from coreai_opt.coreai_utils.passes import weight_palettization as wp
    exact, memo = wp._blockwise_compress, {}
    def cached(data, mode, *args, **kwargs):
        # Identical weights recur across contexts and cache formats. Preserve the
        # exact compressor result, including None; never refit the exported LUT.
        key = (wkey(data), data.shape, mode, repr(args), repr(sorted(kwargs.items())))
        if key not in memo:
            memo[key] = exact(data, mode, *args, **kwargs)
        return memo[key]
    wp._blockwise_compress = cached
    try:
        prog = palettize_weights(prog, lut_dtype=lut_dtype, n_bits=4, granularity=CompressionGranularity.PER_TENSOR,
                                 cluster_dim=2, weight_num_threshold=1024, enable_fast_kmeans_mode=False)
    finally:
        wp._blockwise_compress = exact
    prog.optimize()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    del prog, conv
    gc.collect()
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"   saved {out.name}: {mb:.0f} MB", flush=True)
    return mb


_ARCH = None


def precompile(src: Path, drop_src: bool) -> Path:
    """xcrun coreai-build compile -> <stem>.aimodelc next to src (neural-engine preferred, this Mac's architecture);
    optionally delete the source .aimodel. The runtime loads the .aimodelc directly (no compile cache entry)."""
    import subprocess
    global _ARCH
    if _ARCH is None:  # the M6 is h18g (deviceDescriptor of its Core AI / AFM packages); COREAI_ARCH overrides
        _ARCH = os.environ.get("COREAI_ARCH", "h18g")
    dst = src.with_suffix(".aimodelc")
    shutil.rmtree(dst, ignore_errors=True)
    t0 = time.time()
    cmd = ["xcrun", "coreai-build", "compile", str(src), "--output", str(dst), "--platform", "macOS",
           "--preferred-compute", "neural-engine", "--architecture", _ARCH]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        noise = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if not l.startswith("objc["))
        raise RuntimeError(f"coreai-build compile failed rc={proc.returncode}: {noise[-2000:]}")
    mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    # the ANE programs are built / cached at first load (placement shows in call time, not in the package)
    print(f"   compiled {dst.name}: {mb:.0f} MB in {time.time() - t0:.0f}s ({_ARCH})", flush=True)
    if drop_src:
        shutil.rmtree(src, ignore_errors=True)
    return dst


def build_chunk(ck, layers: list[int], ctxs: list[int], pctxs: list[int], name: str | None = None) -> dict:
    t0 = time.time()
    KNOWN_LUTS.clear()
    W = {}
    for i in layers:
        W.update(layer_arrays(ck, i))
    mods = nn.ModuleList(LayerW(W, i) for i in layers).eval().to(torch.float16)
    del W
    gc.collect()
    entries, aliases, soc_aliases = [], {}, {}
    modes = ("fp16", "v8") if KV_CACHE_DTYPE == "both" else (KV_CACHE_DTYPE,)
    m5_forms = None if ATT_INT8MM_M5 is None else ("" if ATT_INT8MM_M5 == "none" else ATT_INT8MM_M5)
    if m5_forms is not None:
        if len(modes) > 1:
            raise ValueError("ATT_INT8MM_M5 needs a single KV cache format, not --kv-cache-dtype both")
        extra = set(filter(None, m5_forms.split(","))) - set(filter(None, ATT_INT8MM.split(",")))
        if extra:
            raise ValueError(f"ATT_INT8MM_M5 forms {sorted(extra)} must be a subset of ATT_INT8MM (their buffers)")
    for mode in modes:
        aliases[mode] = {}
        shapes = [(f"v8_{ctx // 1024}k", ctx, 8) for ctx in ctxs]
        shapes += [(f"p{tp}_{ctx // 1024}k", ctx, tp) for ctx in pctxs for tp in TPS]
        for canonical, ctx, rows in shapes:
            physical = canonical + ("_kvv8" if len(modes) > 1 and mode == "v8" else "")
            e = Entry(mods, ctx, rows, kv_cache_dtype=mode)
            entries.append((physical, e, e.input_names(), e.output_names()))
            aliases[mode][canonical] = physical
            if m5_forms is not None:
                e5 = Entry(mods, ctx, rows, kv_cache_dtype=mode, att_forms=m5_forms)
                entries.append((physical + "_m5", e5, e5.input_names(), e5.output_names()))
                soc_aliases[canonical] = physical + "_m5"
    name = name or f"chunk_L{layers[0]:02d}-{layers[-1]:02d}"
    out = OUT / f"{name}.aimodel"
    mb = save_program(entries, out)
    e0 = entries[0][1]
    info = {"file": out.name, "layers": [layers[0], layers[-1]], "entries": [x[0] for x in entries],
            "gdn_j": e0.gdn_j, "att_j": e0.att_j, "taps": e0.taps, "mb": round(mb),
            "numerics": {"SILU": SILU, "MLP_SILU": MLP_SILU, "GDN_SQ": GDN_SQ, "GDN_SV": GDN_SV,   # fp16 fixes built in
                         "MLP_DS_TABLE": os.environ.get("MLP_DS_TABLE"), "MLP_DS": MLP_DS, "GDN_FAST": GDN_FAST,
                         "ATT_BLOCK": ATT_BLOCK, "ATT_BLOCK_PREFILL": ATT_BLOCK_PREFILL,
                         **({"ATT_SOFTMAX": ATT_SOFTMAX} if ATT_SOFTMAX != "two_pass" else {}),
                         **({"ATT_SOFTMAX_PREFILL": ATT_SOFTMAX_PREFILL} if ATT_SOFTMAX_PREFILL != ATT_SOFTMAX else {}),
                         **({"ATT_INT8MM": ATT_INT8MM, "ATT_S8_UNIT": ATT_S8_UNIT, "ATT_S8B_UNIT": ATT_S8B_UNIT,
                             **({"ATT_S8R_SHIFT": ATT_S8R_SHIFT} if "s8r" in ATT_INT8MM else {}),
                             **({"ATT_PF8_UNIT": float(os.environ.get("ATT_PF8_UNIT", 1 / 64))}
                                if "sm8" in ATT_INT8MM or "pvf8" in ATT_INT8MM else {})}
                            if ATT_INT8MM else {}),
                         **({"ATT_INT8MM_BY_LAYER": {str(k): v for k, v in ATT_INT8MM_BY_LAYER.items() if k in layers}}
                            if any(k in layers for k in ATT_INT8MM_BY_LAYER) else {}),
                         **({"QCONV_INT8": QCONV_INT8} if not QCONV_INT8 else {}),
                         **({"ATT_TILE_DEQUANT": True} if ATT_TILE_DEQUANT else {}),
                         **({"KV_KEYS_T": True} if KV_KEYS_T else {})}}
    if len(modes) > 1:
        info["entries_by_kv"] = aliases
    if m5_forms is not None:
        info["entries_by_soc"] = {"m5": soc_aliases}
        info["numerics"]["ATT_INT8MM_M5"] = m5_forms
    print(f"chunk {layers[0]}-{layers[-1]}: {len(entries)} entries in {time.time() - t0:.0f}s", flush=True)
    return info


def build_head(ck) -> dict:
    t0 = time.time()
    KNOWN_LUTS.clear()
    h = Head(ck).eval().to(torch.float16)
    out = OUT / "head_T8.aimodel"
    mb = save_program([("h8", h, ["x"], ["logits"])], out)
    print(f"head in {time.time() - t0:.0f}s", flush=True)
    return {"file": out.name, "mb": round(mb)}


def parse_plan(s: str) -> list[list[int]]:
    return [list(range(int(a), int(b) + 1)) for a, b in (r.split("-") for r in s.split(","))]


def main():
    global KV_CACHE_DTYPE, STABLE_ATTN, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("chunk", "head", "all"))
    ap.add_argument("layers", nargs="?", default="0-3")
    ap.add_argument("--ctx", default="2048,8192,16384")
    ap.add_argument("--pctx", default="2048")
    ap.add_argument("--plan", default=",".join(f"{i}-{i + 3}" for i in range(0, 64, 4)))
    ap.add_argument("--name", default=None)
    ap.add_argument("--kv-cache-dtype", choices=("fp16", "v8", "kv8", "both"), default=KV_CACHE_DTYPE)
    ap.add_argument("--kv-cache-default", choices=("fp16", "v8"), default="v8",
                    help="startup default for a shared-weight --kv-cache-dtype both export (default: v8)")
    ap.add_argument("--stable-attention", action="store_true", default=STABLE_ATTN,
                    help="global exp/sum at every context (always used with V8); matched FP16 research control")
    ap.add_argument("--compile", action="store_true", help="precompile each package to .aimodelc")
    ap.add_argument("--drop-src", action="store_true", help="with --compile: delete the source .aimodel")
    a = ap.parse_args()
    if a.kv_cache_dtype not in ("fp16", "v8", "kv8", "both"):
        ap.error("KV_CACHE_DTYPE for conversion must be fp16, v8, kv8 or both (auto is a serving option)")
    KV_CACHE_DTYPE, STABLE_ATTN = a.kv_cache_dtype, a.stable_attention or a.kv_cache_dtype in ("v8", "kv8")
    if KV_CACHE_DTYPE in ("v8", "kv8"):
        OUT = OUT.with_name(OUT.name + ("_kvv8" if KV_CACHE_DTYPE == "v8" else "_kv8"))
    elif KV_CACHE_DTYPE == "both":
        OUT = OUT.with_name(OUT.name + "_kvselect" + ("_stable" if STABLE_ATTN else ""))
    elif STABLE_ATTN:
        OUT = OUT.with_name(OUT.name + "_stable")
    ctxs = [int(x) for x in a.ctx.split(",") if x]
    pctxs = [int(x) for x in a.pctx.split(",") if x]
    ck = M.Checkpoint()
    if a.what == "chunk":
        print(json.dumps(build_chunk(ck, parse_plan(a.layers)[0], ctxs, pctxs, a.name)))
    elif a.what == "head":
        print(json.dumps(build_head(ck)))
    else:
        man_path = OUT / "manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        prior_format = man.get("kv_cache", {}).get("format", "fp16")
        wanted_format = "selectable" if KV_CACHE_DTYPE == "both" else KV_CACHE_DTYPE
        if man_path.exists() and prior_format != wanted_format:
            raise ValueError("Existing build has a different KV format; choose a new OUT directory")
        man.update({"version": "coreai1", "T": 8, "TP": 64 if pctxs else 0, "pend": P, "taps": TAPS, "ctxs": ctxs,
                    "pctxs": pctxs, "kv_len": {str(c): kv_len(c, 8) for c in ctxs},
                    "pkv_len": {str(c): kv_len(c, 64) for c in pctxs}, "export": str(M.EXPORT_DIR)})
        def layout(mode):
            q8 = mode in ("v8", "kv8")
            return {"format": mode, "keys": "int8" if mode == "kv8" else "float16",
                    "values": "int8" if q8 else "float16", "scales": "float16" if q8 else None,
                    "scale_granularity": "token_head" if q8 else None, "stable_attention": STABLE_ATTN or q8,
                    **({"key_layout": "dim_token"} if KV_KEYS_T else {})}
        man["kv_cache"] = ({"format": "selectable", "default": a.kv_cache_default,
                            "formats": {mode: layout(mode) for mode in ("fp16", "v8")}}
                           if KV_CACHE_DTYPE == "both" else layout(KV_CACHE_DTYPE))
        chunks = {c["file"]: c for c in man.get("chunks", [])}
        for layers in parse_plan(a.plan):
            f = f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
            done = (OUT / f).exists() or (OUT / f).with_suffix(".aimodelc").exists()
            if f in chunks and done and chunks[f].get("entries_ctx") == [ctxs, pctxs]:
                continue
            info = build_chunk(ck, layers, ctxs, pctxs)
            info["entries_ctx"] = [ctxs, pctxs]
            if a.compile:
                info["compiled"] = precompile(OUT / f, a.drop_src).name
            chunks[f] = info
            man["chunks"] = sorted(chunks.values(), key=lambda c: c["layers"][0])
            man_path.write_text(json.dumps(man, indent=1))
            free = shutil.disk_usage(OUT).free / 2**30
            print(f"   disk free {free:.1f} GiB", flush=True)
            if free < 15:
                raise SystemExit(f"stopping: disk free {free:.1f} GiB < 15")
        if not ((OUT / "head_T8.aimodel").exists() or (OUT / "head_T8.aimodelc").exists()):
            man["head"] = build_head(ck)
            if a.compile:
                man["head"]["compiled"] = precompile(OUT / "head_T8.aimodel", a.drop_src).name
        else:
            man.setdefault("head", {"file": "head_T8.aimodel"})
        man_path.write_text(json.dumps(man, indent=1))
        print(f"manifest -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
