"""ANE hybrid-layer MIL graphs (Gated DeltaNet + gated full attention).

Origin: Anemll/anemll-forge. Imported by ``qwen38_ane_model``. coremltools is optional at
import time so Linux can load the module.

ANE single-token decode chunk for Qwen3.8-27B: layers 30-33 (3 Gated DeltaNet + 1 gated full attention =
one repeat of the model's 64-layer pattern) with quantized weights and Core ML states.

States: per DeltaNet layer a recurrent state (48, 128, 128) and a conv ring buffer (4, 10240) (a shift
register state fails to load in Core ML: "MIL->EIR ... bad_cast"); per full attention layer a KV cache
(4, CTX, 256) x 2. All state writes are masked arithmetic (no slice_update / gather, which run on the CPU):
  cache = cache * (1 - onehot(pos)) + k * onehot(pos);  ring = ring * (1 - onehot(slot)) + qkv * onehot(slot)
Inputs: x (1, 5120, 1, 1), cos / sin (1, 64), mask (1, CTX) additive, kv_onehot (1, CTX, 1) at pos,
slot_onehot (4, 1) at pos % 4, conv_perm (4, 4): row j selects the ring slot of token pos - 3 + j.
Weights: MLP in MLP_FMT, big attention / DeltaNet projections in ATT_FMT (round-to-nearest here; GPTQ
tensors from the M3U pipeline later), in_proj_a / in_proj_b / norms / conv in fp16.

    CTX=2048 python qwen38_ane_chunk.py            # build + check vs qwen38_decode_ref (dequantized weights)
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import Function, Program, types
except ImportError:  # Linux import check
    ct = mb = Function = Program = types = None
from qwen3_lut_common import FORMATS, kmeans_vector_codebook
from qwen38_decode_ref import DecodeLayer, load_layer_weights, text_config

LAYERS = [int(x) for x in os.environ.get("LAYERS", "30,31,32,33").split(",")]
REPEAT = int(os.environ.get("REPEAT", "1"))   # stack the layer group REPEAT times (bigger chunks, same weights)
CTX = int(os.environ.get("CTX", "2048"))
MLP_FMT = os.environ.get("MLP_FMT", "vector 2x16 + pcs")
ATT_FMT = os.environ.get("ATT_FMT", "LUT4 per-tensor + pcs")
CHECK = int(os.environ.get("CHECK", "8"))     # decode steps compared with the reference (0 = skip)
GDN_CHUNK = int(os.environ.get("GDN_CHUNK", "8"))  # DeltaNet prefill sub-chunk (tokens)
# ANE state rule: a value derived from read_state may only reach the outputs through that state's update. Prefill
# outputs of the recurrence therefore leave through a state write: "concat" = extra rows in the recurrent state,
# "scratch" = a separate per-layer state. SCR_ROWS = the largest prefill block.
REC_OUT = os.environ.get("REC_OUT", "concat")
# GDN_IO: DeltaNet conv / recurrent states are plain inputs and outputs (host-owned buffers) instead of MLState;
# the conv input is always the last 3 tokens, so blocks may start anywhere. KV caches stay MLState.
GDN_IO = os.environ.get("GDN_IO", "0") == "1"
KV_IO = os.environ.get("KV_IO", "0") == "1"
# KV_IN (T-row functions): the KV caches k<j>, v<j> (nkv, CTX, hd) are READ-ONLY inputs holding the committed
# history (positions < p0, mask (1, CTX)); the block's own k / v rows are outputs k<j>_new, v<j>_new (nkv, T, hd)
# that the host copies into the caches for the rows it commits. Attention = history + causal block, one softmax
# softmax over [history | block]. No KV writes in the graph, no KV rollback.
KV_IN = os.environ.get("KV_IN", "0") == "1"
# LAZY (with GDN_IO): one T-row function for decode / prefill / DFlash verify. DeltaNet rows are committed one call
# late: a call first applies the first `commit` pending rows of the previous call (host one-hot commit / commit_last),
# then leaves its own rows pending (pend<j>_out). conv<j> in = the previous call's T + 3 raw rows, conv_sel picks the
# 3 rows ending at the committed length. Math from dflash2_gdn_lazy.py.
LAZY = os.environ.get("LAZY", "0") == "1"
PEND = 8  # pending capacity (rows) = the verify block  # KV caches as plain inputs / outputs (k<j>, v<j> -> k<j>_out, v<j>_out)
CONV_OUT_ALL = os.environ.get("CONV_OUT_ALL", "0") == "1"  # I/O mode: output all T + 3 conv rows (verify)
JOFF = 0  # index of the first layer within its chunk: state / I/O names (conv<j>, rec<j>, k<j>, v<j>) use chunk-level j
TAPS = []  # layers whose output hidden state is also a model output ("tap<l>"), e.g. DFlash drafter features
# DBG_MIXER_IN=1 (T > 1 builds): every mixer matrix's input becomes an extra output "dbg<n>" (one per distinct input
# tensor: in_proj_qkv / in_proj_z share h, q / k / v share h); DBG_MAP (json next to the model) maps
# "<layer>:<matrix>" -> output name. For ANE-in-the-loop refits of the low-rank factors.
MIXER_KEYS = ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.out_proj.weight",
              "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
              "self_attn.o_proj.weight")
DBG_MIXER_IN = os.environ.get("DBG_MIXER_IN", "0") == "1"
DBG_GDN = os.environ.get("DBG_GDN", "0") == "1"  # with DBG_MIXER_IN: also tap the DeltaNet core intermediates
# with DBG_MIXER_IN: groups of intermediates exposed as "dbg_<name>" outputs (gdn = DeltaNet core, att, mlp)
DBG_TAPS = set(filter(None, os.environ.get("DBG_TAPS", "gdn" if DBG_GDN else "").split(",")))


def _tap(name, t, grp="gdn"):
    """Expose an intermediate tensor as output "dbg_<name>" when its group is in DBG_TAPS (one output per tensor)."""
    if _dbg is not None and grp in DBG_TAPS and f"tap:{name}" not in _dbg["vars"]:
        _dbg["vars"][f"tap:{name}"] = mb.identity(x=t, name=f"dbg_{name}")
    return t


_dbg = None  # while building: {"names": {id(q tuple): "<layer>:<matrix>"}, "vars": {var name: output}, "map": {}}
SCR_ROWS = int(os.environ.get("SCR_ROWS", "64"))
OUT = Path(__file__).parent / "qwen38_chunk"
# coremltools types are optional at import; IDX is only used when converting.
IDX = {} if types is None else {
    2: types.np_uint1_dtype, 4: types.np_uint2_dtype, 16: types.np_uint4_dtype, 64: types.np_uint6_dtype,
    256: np.uint8,
}
QUANT = {"mlp.gate_proj.weight": "mlp", "mlp.up_proj.weight": "mlp", "mlp.down_proj.weight": "mlp",
         "linear_attn.in_proj_qkv.weight": "att", "linear_attn.in_proj_z.weight": "att",
         "linear_attn.out_proj.weight": "att", "self_attn.q_proj.weight": "att", "self_attn.k_proj.weight": "att",
         "self_attn.v_proj.weight": "att", "self_attn.o_proj.weight": "att"}
torch.set_grad_enabled(False)


def quantize(w, fmt):
    """Per-tensor (vector) LUT + optional per-output-channel scale: (lut, idx, scale or None, dequantized)."""
    spec = FORMATS[fmt][1]
    pcs = spec[-1] == "pcs"
    cd, nb = (1, spec[2]) if spec[0] == "group" else (spec[1], spec[2])
    s = w.pow(2).mean(1, keepdim=True).sqrt().half().float() if pcs else torch.ones(w.shape[0], 1)
    wn = w / s
    c = kmeans_vector_codebook(wn, cd, nb, None).half().float()
    cout, cin = w.shape
    v = wn.reshape(cout // cd, cd, cin).permute(0, 2, 1).reshape(-1, cd)
    lab = torch.cat([torch.cdist(v[i:i + (1 << 18)], c).argmin(1) for i in range(0, len(v), 1 << 18)])
    idx = lab.reshape(cout // cd, cin)
    deq = c[idx].permute(0, 2, 1).reshape(cout, cin) * s
    return c.numpy().astype(np.float16), idx.numpy().astype(np.uint8), s.numpy().astype(np.float16) if pcs else None, deq


def lowrank(x, lr):
    """x -> a @ (b @ x) as two 1x1 convs in fp16 (a (Cout, r), b (r, Cin)): the low-rank error correction."""
    a, b = lr
    h = mb.conv(x=x, weight=b.astype(np.float16).reshape(*b.shape, 1, 1))
    return mb.conv(x=h, weight=a.astype(np.float16).reshape(*a.shape, 1, 1))


def lut_linear(x, q):
    """1x1 conv with a quantized weight: (lut, idx, scale[, deq[, lowrank]]) LUT, ("int8", codes, scale[, lowrank])
    or ("dense", w). lowrank = (a, b) adds a @ (b @ x) (qwen38_blockrecon.py factors, in x's basis)."""
    if _dbg is not None and id(q) in _dbg["names"]:
        out = _dbg["vars"].get(x.name)
        if out is None:
            out = _dbg["vars"][x.name] = mb.identity(x=x, name=f"dbg{len(_dbg['vars'])}")
        _dbg["map"][_dbg["names"][id(q)]] = out.name
    if isinstance(q[0], str) and q[0] == "int8":
        codes, sc = q[1], q[2]
        w = mb.constexpr_blockwise_shift_scale(data=codes.reshape(*codes.shape, 1, 1),
                                               scale=sc.reshape(-1, 1, 1, 1).astype(np.float16))
        y = mb.conv(x=x, weight=w)
        return mb.add(x=y, y=lowrank(x, q[3])) if len(q) > 3 and q[3] is not None else y
    if isinstance(q[0], str) and q[0] == "dense":
        return mb.conv(x=x, weight=q[1].astype(np.float16).reshape(*q[1].shape, 1, 1))
    lut, idx, s = q[:3]
    k, cd = lut.shape
    w = mb.constexpr_lut_to_dense(indices=idx.reshape(*idx.shape, 1, 1).astype(IDX[k]),
                                  lut=lut.reshape(1, 1, 1, 1, k, cd), vector_axis=0 if cd > 1 else None)
    if s is not None:
        w = mb.constexpr_blockwise_shift_scale(data=w, scale=s.reshape(-1, 1, 1, 1))
    y = mb.conv(x=x, weight=w)
    return mb.add(x=y, y=lowrank(x, q[4])) if len(q) > 4 and q[4] is not None else y


def rot_conv(x, n, seed, block=1024):
    """x (1, n, 1, T) -> x M, M = blockdiag(diag(signs) H_block) / sqrt(block), as a grouped conv whose
    +-1/sqrt(block) weights are a 1-bit LUT (the pipeline's online Hadamard, same seeds)."""
    from scipy.linalg import hadamard
    h = hadamard(block)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], n)
    # conv weight W[o, i] = M[i, o] within each block = signs[i] * H[i, o]; H is symmetric
    wt = np.concatenate([(signs[b * block:(b + 1) * block, None] * h).T for b in range(n // block)])
    w = mb.constexpr_lut_to_dense(indices=(wt > 0).reshape(n, block, 1, 1).astype(types.np_uint1_dtype),
                                  lut=(np.array([-1, 1]) / np.sqrt(block)).astype(np.float16).reshape(1, 1, 1, 1, 2, 1))
    return mb.conv(x=x, weight=w, groups=n // block)


def dense_linear(x, w):
    return mb.conv(x=x, weight=w.numpy().astype(np.float16).reshape(*w.shape, 1, 1))


# DeltaNet fp16 range: q . S is tiny (median |o| ~4e-5, 61% of it fp16-subnormal on the ANE -> 20% error, 46% after
# the gated RMSNorm). v (hence u and the recurrent state) is scaled by GDN_SV and q by GDN_SQ; the gated RMSNorm
# absorbs it exactly with eps * (GDN_SQ * GDN_SV)^2. The state is host-owned I/O, so the scale is internal.
GDN_SQ, GDN_SV = float(os.environ.get("GDN_SQ", "16")), float(os.environ.get("GDN_SV", "64"))
SILU = os.environ.get("SILU", "tanh")  # "tanh": 0.5 x (1 + tanh(x / 2)); "native": mb.silu


def silu(x):
    """silu for the DeltaNet conv / gate. The ANE's native silu has ~1e-3 ABSOLUTE error near 0 (measured on the
    (8, 10240) conv of layer 9: 12% relative error on the |x| < 0.5 values that make up >99% of q / k / v, 51% at the
    DeltaNet output). 0.5 x (1 + tanh(x / 2)) is the same function with relative precision near 0; x * sigmoid(x)
    does not help: mil_backend::fuse_activation_silu turns it back into silu."""
    if SILU == "native":
        return mb.silu(x=x)
    half = mb.mul(x=x, y=np.float16(0.5))
    return mb.mul(x=half, y=mb.add(x=mb.tanh(x=half), y=np.float16(1)))


MLP_DS = float(os.environ.get("MLP_DS", "1"))  # MLP down-projection input scale (output scaled back by 1 / MLP_DS)
# MLP_DS_DYN=1: per-token scale min(MLP_DS, MLP_DS_C / max|a|) (a static scale overflows fp16 in layers whose down
# input has large channels; measured on ane7g: inf from layer 32 on)
MLP_DS_DYN = os.environ.get("MLP_DS_DYN", "0") == "1"
MLP_DS_C = float(os.environ.get("MLP_DS_C", "64"))
# MLP_DS_TABLE=<json with {"ds": {layer: scale}}>: per-layer scales (small where the layer output is large: the
# down conv's partial sums overflow fp16 at scale x output > ~20000); overrides MLP_DS
MLP_DS_TABLE = json.loads(Path(os.path.expanduser(os.environ["MLP_DS_TABLE"])).read_text())["ds"] \
    if os.environ.get("MLP_DS_TABLE") else None
MLP_SILU = os.environ.get("MLP_SILU", "tanh")  # "tanh": 0.5 x (1 + tanh(x / 2)) for the MLP gate too


def mlp_silu(x):
    if MLP_SILU == "native":
        return mb.silu(x=x)
    half = mb.mul(x=x, y=np.float16(0.5))
    return mb.mul(x=half, y=mb.add(x=mb.tanh(x=half), y=np.float16(1)))


def softplus(x):
    """softplus(x) = relu(x) + log(1 + exp(-|x|)): the ANE's fp16 softplus returns 0 for x >~ 11 (exp overflow),
    which zeroed the DeltaNet decay gate of heads with large a + dt (no decay instead of a near reset)."""
    return mb.add(x=mb.relu(x=x), y=mb.log(x=mb.add(x=mb.exp(x=mb.mul(x=mb.abs(x=x), y=np.float16(-1))), y=np.float16(1))))


def rms_hidden(x, w_plus, eps):
    """RMSNorm over channels of (1, C, 1, 1), computed on x/64 so the squares stay inside fp16 range."""
    xs = mb.mul(x=x, y=np.float16(1 / 64))
    ms = mb.reduce_mean(x=mb.mul(x=xs, y=xs), axes=[1], keep_dims=True)
    return mb.mul(x=mb.mul(x=xs, y=mb.rsqrt(x=ms, epsilon=eps / 4096)), y=w_plus.reshape(1, -1, 1, 1))


def rms_last(x, w, eps):
    ms = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
    return mb.mul(x=mb.mul(x=x, y=mb.rsqrt(x=ms, epsilon=eps)), y=w)


def gdn_block(cfg, w, q, h, conv_st, rec_st, slot_oh, perm):
    nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim"))
    kd, vd = nk * dk, nv * dv
    eps = cfg["rms_norm_eps"]
    cdim = 2 * kd + vd
    qkv = mb.reshape(x=lut_linear(h, q["linear_attn.in_proj_qkv.weight"]), shape=(1, cdim))
    z = mb.reshape(x=lut_linear(h, q["linear_attn.in_proj_z.weight"]), shape=(nv, 1, dv))
    b = mb.reshape(x=dense_linear(h, w["linear_attn.in_proj_b.weight"]), shape=(nv, 1, 1))
    a = mb.reshape(x=dense_linear(h, w["linear_attn.in_proj_a.weight"]), shape=(nv, 1, 1))
    keep = mb.sub(x=np.float16(1), y=slot_oh)                                       # (8, 1)
    buf = mb.add(x=mb.mul(x=mb.read_state(input=conv_st), y=keep), y=mb.mul(x=qkv, y=slot_oh))
    buf = mb.coreml_update_state(state=conv_st, value=buf)                          # (8, conv_dim): 2 rings
    ordered = mb.matmul(x=perm, y=buf)                                              # oldest .. current token
    cw = np.ascontiguousarray(w["linear_attn.conv1d.weight"][:, 0].numpy().astype(np.float16).T)  # (4, conv_dim)
    conv = mb.reshape(x=silu(mb.reduce_sum(x=mb.mul(x=ordered, y=cw), axes=[0])), shape=(1, cdim))
    qq, kk, vv = mb.split(x=conv, split_sizes=[kd, kd, vd], axis=1)
    rep = nv // nk

    def heads(t):  # (1, nk*dk) -> (nv, 1, dk): each key head serves `rep` consecutive value heads
        t = mb.tile(x=mb.reshape(x=t, shape=(nk, 1, dk)), reps=[1, rep, 1])
        return mb.reshape(x=t, shape=(nv, 1, dk))

    def l2n(t, scale):
        ss = mb.reduce_sum(x=mb.mul(x=t, y=t), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.mul(x=t, y=mb.rsqrt(x=ss, epsilon=1e-6)), y=np.float16(scale))

    qh, kh = l2n(heads(qq), dk ** -0.5), l2n(heads(kk), 1.0)
    vh = mb.reshape(x=vv, shape=(nv, 1, dv))
    beta = mb.sigmoid(x=b)
    neg_a = (-w["linear_attn.A_log"].exp()).numpy().astype(np.float16).reshape(nv, 1, 1)
    dt = w["linear_attn.dt_bias"].numpy().astype(np.float16).reshape(nv, 1, 1)
    decay = mb.exp(x=mb.mul(x=softplus(mb.add(x=a, y=dt)), y=neg_a))
    rd = mb.read_state(input=rec_st)
    if REC_OUT == "concat":  # rows dk.. are the prefill output rows: carried through untouched
        scr = mb.slice_by_index(x=rd, begin=[0, dk, 0], end=[nv, dk + SCR_ROWS, dv])
        rd = mb.slice_by_index(x=rd, begin=[0, 0, 0], end=[nv, dk, dv])
    s = mb.mul(x=rd, y=decay)                                                       # (nv, dk, dv)
    kv_mem = mb.matmul(x=kh, y=s)                                                   # (nv, 1, dv)
    delta = mb.mul(x=mb.sub(x=vh, y=kv_mem), y=beta)
    s = mb.add(x=s, y=mb.matmul(x=kh, y=delta, transpose_x=True))                   # + k^T delta
    if REC_OUT == "concat":
        s = mb.slice_by_index(x=mb.coreml_update_state(state=rec_st, value=mb.concat(values=[s, scr], axis=1)),
                              begin=[0, 0, 0], end=[nv, dk, dv])
    else:
        s = mb.coreml_update_state(state=rec_st, value=s)
    o = mb.matmul(x=qh, y=s)                                                        # (nv, 1, dv)
    o = mb.mul(x=rms_last(o, w["linear_attn.norm.weight"].numpy().astype(np.float16), eps), y=silu(z))
    return lut_linear(mb.reshape(x=o, shape=(1, vd, 1, 1)), q["linear_attn.out_proj.weight"])


def kv_read(st):
    """KV cache value: the input tensor (KV_IO) or the state read."""
    return st if KV_IO else mb.read_state(input=st)


def kv_write(st, value):
    """Updated KV cache: the value itself (KV_IO; returned as an output by the builder) or the state write."""
    return value if KV_IO else mb.coreml_update_state(state=st, value=value)


def attn_block(cfg, w, q, h, k_st, v_st, cos, sin, mask, kv_oh):
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    rot = int(hd * cfg["rope_parameters"]["partial_rotary_factor"])
    eps = cfg["rms_norm_eps"]
    qg = mb.reshape(x=lut_linear(h, q["self_attn.q_proj.weight"]), shape=(nh, 2 * hd))
    qh = mb.slice_by_index(x=qg, begin=[0, 0], end=[nh, hd])
    gate = mb.reshape(x=mb.slice_by_index(x=qg, begin=[0, hd], end=[nh, 2 * hd]), shape=(1, nh * hd))
    qh = rms_last(qh, (1 + w["self_attn.q_norm.weight"]).numpy().astype(np.float16), eps)
    kh = rms_last(mb.reshape(x=lut_linear(h, q["self_attn.k_proj.weight"]), shape=(nkv, hd)),
                  (1 + w["self_attn.k_norm.weight"]).numpy().astype(np.float16), eps)
    vh = mb.reshape(x=lut_linear(h, q["self_attn.v_proj.weight"]), shape=(nkv, 1, hd))

    def rope(t, n):
        r = mb.slice_by_index(x=t, begin=[0, 0], end=[n, rot])
        rest = mb.slice_by_index(x=t, begin=[0, rot], end=[n, hd])
        r1 = mb.slice_by_index(x=r, begin=[0, 0], end=[n, rot // 2])
        r2 = mb.slice_by_index(x=r, begin=[0, rot // 2], end=[n, rot])
        rh = mb.concat(values=[mb.mul(x=r2, y=np.float16(-1)), r1], axis=1)
        return mb.concat(values=[mb.add(x=mb.mul(x=r, y=cos), y=mb.mul(x=rh, y=sin)), rest], axis=1)

    qh, kh = rope(qh, nh), mb.reshape(x=rope(kh, nkv), shape=(nkv, 1, hd))
    keep = mb.sub(x=np.float16(1), y=kv_oh)                                         # (1, CTX, 1)
    kc = kv_write(k_st, mb.add(x=mb.mul(x=kv_read(k_st), y=keep),
                                                           y=mb.mul(x=kh, y=kv_oh)))
    vc = kv_write(v_st, mb.add(x=mb.mul(x=kv_read(v_st), y=keep),
                                                           y=mb.mul(x=vh, y=kv_oh)))
    qg4 = mb.reshape(x=qh, shape=(nkv, nh // nkv, hd))                             # query heads grouped by kv head
    sc = mb.mul(x=mb.matmul(x=qg4, y=kc, transpose_y=True), y=np.float16(hd ** -0.5))  # (nkv, grp, CTX)
    p = mb.softmax(x=mb.add(x=sc, y=mb.reshape(x=mask, shape=(1, 1, CTX))), axis=-1)
    o = mb.reshape(x=mb.matmul(x=p, y=vc), shape=(1, nh * hd))
    o = mb.mul(x=o, y=mb.sigmoid(x=gate))
    y = lut_linear(mb.reshape(x=o, shape=(1, nh * hd, 1, 1)), q["self_attn.o_proj.weight"])
    return (y, kc, vc) if KV_IO else y


def chunked_delta(qh, kh, vh, beta, g, s, rec_st, nv, T, dk, dv, scr_st=None):
    """Gated delta rule over T tokens in sub-chunks of GDN_CHUNK (the UT-transform form of the transformers
    chunked kernel). Within a sub-chunk the unit lower-triangular system (I + N) X = B is solved with the
    doubling product (I - N)(I + N^2)(I + N^4)..., all matmuls; every exp argument is <= 0.
    fp16 error vs the per-token recurrence is ~1e-3 for sub-chunks of 8 (16 is marginal with correlated keys).
    Returns the per-sub-chunk outputs (nv, C, dv); writes the final state."""
    C = min(GDN_CHUNK, T)
    f16 = np.float16
    blk = np.arange(T) // C
    same = blk[:, None] == blk[None, :]
    i, j = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    l_inc, l_str = (same & (i >= j)).astype(f16), (same & (i > j)).astype(f16)
    g_row = mb.reshape(x=g, shape=(nv, 1, T))
    cum = mb.reshape(x=mb.matmul(x=g_row, y=np.ascontiguousarray(l_inc.T)), shape=(nv, T, 1))   # within-chunk cumsum
    bsum = mb.reshape(x=mb.matmul(x=g_row, y=same.astype(f16)), shape=(nv, T, 1))            # chunk total per row
    pair = mb.mul(x=mb.exp(x=mb.minimum(x=mb.sub(x=cum, y=mb.reshape(x=cum, shape=(nv, 1, T))), y=f16(0))), y=l_inc)
    kb, vb = mb.mul(x=kh, y=beta), mb.mul(x=vh, y=beta)
    n = mb.mul(x=mb.matmul(x=kb, y=kh, transpose_y=True), y=mb.mul(x=pair, y=l_str))
    ec = mb.exp(x=cum)
    # (I + N) X = B for B = [v*beta | k*beta*exp(cum)]. (I + N)^-1 = (I - N)(I + N^2)(I + N^4)... (N^C = 0).
    # In a stateful graph the ANE plan builder rejects (-14) matmul(N, N) and chains of 3+ `B - N @ X` steps;
    # N^2 = I - (I - N)(I + N) (a product of two different tensors) loads.
    rhs = mb.concat(values=[vb, mb.mul(x=kb, y=ec)], axis=2)                          # (nv, T, dv + dk)
    eye = np.eye(T, dtype=f16)
    inv, npow, m = mb.sub(x=eye, y=n), n, 1
    while 2 * m < C:
        npow = mb.sub(x=eye, y=mb.matmul(x=mb.sub(x=eye, y=npow), y=mb.add(x=eye, y=npow)))
        inv = mb.matmul(x=inv, y=mb.add(x=eye, y=npow))
        m *= 2
    xs = mb.matmul(x=inv, y=rhs)
    u = mb.slice_by_index(x=xs, begin=[0, 0, 0], end=[nv, T, dv])
    wk = mb.slice_by_index(x=xs, begin=[0, 0, dv], end=[nv, T, dv + dk])
    intra = mb.mul(x=mb.matmul(x=qh, y=kh, transpose_y=True), y=pair)
    qd, kd = mb.mul(x=qh, y=ec), mb.mul(x=kh, y=mb.exp(x=mb.sub(x=bsum, y=cum)))
    if s is None:  # read the state only here, right before the sub-chunk scan
        s = mb.read_state(input=rec_st)
        if REC_OUT == "concat":
            s = mb.slice_by_index(x=s, begin=[0, 0, 0], end=[nv, dk, dv])
    inter, vns = [], []
    for b in range(T // C):
        r0, r1 = b * C, (b + 1) * C

        def rows(t, d):  # sub-chunk rows; slicing the last (width) axis at an offset gives wrong values on the ANE
            return mb.slice_by_index(x=t, begin=[0, r0, 0], end=[nv, r1, d])
        vn = mb.sub(x=rows(u, dv), y=mb.matmul(x=rows(wk, dk), y=s))
        inter.append(mb.matmul(x=rows(qd, dk), y=s))
        vns.append(vn)
        cd = mb.exp(x=mb.slice_by_index(x=bsum, begin=[0, r0, 0], end=[nv, r0 + 1, 1]))
        s = mb.add(x=mb.mul(x=s, y=cd), y=mb.matmul(x=rows(kd, dk), y=vn, transpose_x=True))
    cat = (lambda v: mb.concat(values=v, axis=1)) if len(inter) > 1 else (lambda v: v[0])
    o = mb.add(x=cat(inter), y=mb.matmul(x=intra, y=cat(vns)))                          # intra is block-diagonal
    if rec_st is None:  # I/O mode: no state write
        return [o], s
    if T < SCR_ROWS:
        o = mb.concat(values=[o, np.zeros((nv, SCR_ROWS - T, dv), np.float16)], axis=1)
    if REC_OUT == "concat":
        ret = mb.coreml_update_state(state=rec_st, value=mb.concat(values=[s, o], axis=1))
        return [mb.slice_by_index(x=ret, begin=[0, dk, 0], end=[nv, dk + T, dv])]
    mb.coreml_update_state(state=rec_st, value=s)
    ret = mb.coreml_update_state(state=scr_st, value=o)
    return [mb.slice_by_index(x=ret, begin=[0, 0, 0], end=[nv, T, dv])]


def gdn_proj(cfg, w, q, h, T):
    """Projections of a DeltaNet layer for T tokens: qkv (cdim, T), z (nv, T, dv), b / a (nv, T, 1)."""
    nv, dv = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
    cdim = 2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] + nv * dv
    qkv = mb.reshape(x=lut_linear(h, q["linear_attn.in_proj_qkv.weight"]), shape=(cdim, T))
    z = mb.transpose(x=mb.reshape(x=lut_linear(h, q["linear_attn.in_proj_z.weight"]), shape=(nv, dv, T)), perm=[0, 2, 1])
    b = mb.reshape(x=dense_linear(h, w["linear_attn.in_proj_b.weight"]), shape=(nv, T, 1))
    a = mb.reshape(x=dense_linear(h, w["linear_attn.in_proj_a.weight"]), shape=(nv, T, 1))
    return qkv, z, b, a


def gdn_io(cfg, w, q, h, conv_in, rec_in, T, valid=None, conv_sel=None):
    """DeltaNet layer over T >= 1 tokens with host-owned states: conv_in (3, cdim) = the 3 tokens before this
    block (oldest first), rec_in (nv, dk, dv). Returns (y, conv_out, rec_out). valid (1, T, 1): 0 marks padding
    rows, which leave the recurrent state unchanged (beta = 0, log decay = 0). conv_out: conv_sel (3, T + 3) @ the
    rows [3 previous tokens, this block] (the 3 rows ending at the last valid token), or all T + 3 rows when
    conv_sel is None (verify: the host picks the rows ending at the accepted length)."""
    nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim"))
    kd, vd, eps = nk * dk, nv * dv, cfg["rms_norm_eps"]
    cdim = 2 * kd + vd
    qkv, z, b, a = gdn_proj(cfg, w, q, h, T)
    rows = mb.concat(values=[conv_in, mb.transpose(x=qkv, perm=[1, 0])], axis=0)        # (T + 3, cdim)
    cw = w["linear_attn.conv1d.weight"][:, 0].numpy().astype(np.float16).T              # (4, cdim), oldest first
    conv = None
    for j in range(4):  # row-axis (height) slices only
        term = mb.mul(x=mb.slice_by_index(x=rows, begin=[j, 0], end=[j + T, cdim]), y=np.ascontiguousarray(cw[j:j + 1]))
        conv = term if conv is None else mb.add(x=conv, y=term)
    conv = mb.transpose(x=silu(conv), perm=[1, 0])                                 # (cdim, T)
    qq, kk, vv = mb.split(x=conv, split_sizes=[kd, kd, vd], axis=0)
    rep = nv // nk

    def heads(t):  # (nk*dk, T) -> (nv, T, dk)
        t = mb.transpose(x=mb.reshape(x=t, shape=(nk, dk, T)), perm=[0, 2, 1])
        return mb.reshape(x=mb.tile(x=mb.reshape(x=t, shape=(nk, 1, T, dk)), reps=[1, rep, 1, 1]), shape=(nv, T, dk))

    def l2n(t, scale):
        ss = mb.reduce_sum(x=mb.mul(x=t, y=t), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.mul(x=t, y=mb.rsqrt(x=ss, epsilon=1e-6)), y=np.float16(scale))

    qh, kh = l2n(heads(qq), dk ** -0.5), l2n(heads(kk), 1.0)
    vh = mb.transpose(x=mb.reshape(x=vv, shape=(nv, dv, T)), perm=[0, 2, 1])            # (nv, T, dv)
    beta = mb.sigmoid(x=b)
    neg_a = (-w["linear_attn.A_log"].exp()).numpy().astype(np.float16).reshape(nv, 1, 1)
    dt = w["linear_attn.dt_bias"].numpy().astype(np.float16).reshape(nv, 1, 1)
    g = mb.mul(x=softplus(mb.add(x=a, y=dt)), y=neg_a)                             # log decay (nv, T, 1)
    if valid is not None:
        beta, g = mb.mul(x=beta, y=valid), mb.mul(x=g, y=valid)
    if T == 1:
        s = mb.mul(x=rec_in, y=mb.exp(x=g))
        delta = mb.mul(x=mb.sub(x=vh, y=mb.matmul(x=kh, y=s)), y=beta)
        s = mb.add(x=s, y=mb.matmul(x=kh, y=delta, transpose_x=True))
        o = mb.matmul(x=qh, y=s)
    else:
        outs, s = chunked_delta(qh, kh, vh, beta, g, rec_in, None, nv, T, dk, dv)
        o = outs[0]
    o = mb.mul(x=rms_last(o, w["linear_attn.norm.weight"].numpy().astype(np.float16), eps), y=silu(z))
    o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1]), shape=(1, vd, 1, T))
    conv_out = rows if conv_sel is None else mb.matmul(x=conv_sel, y=rows)
    return lut_linear(o, q["linear_attn.out_proj.weight"]), conv_out, s


def gdn_lazy_block(cfg, w, q, h, conv_rows, conv_sel, rec_in, pend_in, commit, commit_last, T):
    """DeltaNet layer, lazy commit (see LAZY). conv_rows (T + 3, cdim) = the previous call's conv rows, rec_in
    (nv, dk, dv) = committed state, pend_in (nv, 3P + 1, dv) = the previous call's pending [k | u | wk | cum].
    Returns (y, this call's conv rows (T + 3, cdim), committed state S', pending rows of this call)."""
    raise ImportError(
        "dflash2_gdn_lazy was removed from this Jeff-only tree. "
        "Jeff convert uses the torch graphs in coreai/qwen38_coreai_build.py, not this MIL helper."
    )
    P = PEND
    nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim"))
    kd, vd, eps = nk * dk, nv * dv, cfg["rms_norm_eps"]
    cdim = 2 * kd + vd
    f16 = np.float16
    qkv, z, b, a = gdn_proj(cfg, w, q, h, T)
    prev = mb.matmul(x=conv_sel, y=conv_rows)                                            # (3, cdim)
    rows = mb.concat(values=[prev, mb.transpose(x=qkv, perm=[1, 0])], axis=0)           # (T + 3, cdim)
    cw = w["linear_attn.conv1d.weight"][:, 0].numpy().astype(f16).T
    conv = None
    for j in range(4):
        term = mb.mul(x=mb.slice_by_index(x=rows, begin=[j, 0], end=[j + T, cdim]), y=np.ascontiguousarray(cw[j:j + 1]))
        conv = term if conv is None else mb.add(x=conv, y=term)
    _tap("conv_pre", conv)
    conv = _tap("conv", mb.transpose(x=silu(conv), perm=[1, 0]))
    qq, kk, vv = mb.split(x=conv, split_sizes=[kd, kd, vd], axis=0)
    rep = nv // nk

    def heads(t):
        t = mb.transpose(x=mb.reshape(x=t, shape=(nk, dk, T)), perm=[0, 2, 1])
        return mb.reshape(x=mb.tile(x=mb.reshape(x=t, shape=(nk, 1, T, dk)), reps=[1, rep, 1, 1]), shape=(nv, T, dk))

    def l2n(t, scale):
        ss = mb.reduce_sum(x=mb.mul(x=t, y=t), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.mul(x=t, y=mb.rsqrt(x=ss, epsilon=1e-6)), y=f16(scale))

    _tap("qh_raw", heads(qq))
    qh, kh = _tap("qh", l2n(heads(qq), dk ** -0.5 * GDN_SQ)), _tap("kh", l2n(heads(kk), 1.0))
    vh = _tap("vh", mb.mul(x=mb.transpose(x=mb.reshape(x=vv, shape=(nv, dv, T)), perm=[0, 2, 1]), y=f16(GDN_SV)))
    beta = _tap("beta", mb.sigmoid(x=b))
    neg_a = (-w["linear_attn.A_log"].exp()).numpy().astype(f16).reshape(nv, 1, 1)
    dt = w["linear_attn.dt_bias"].numpy().astype(f16).reshape(nv, 1, 1)
    g = _tap("g", mb.mul(x=softplus(mb.add(x=a, y=dt)), y=neg_a))
    # commit the first k pending rows of the previous call
    kp, up = rows_slice(pend_in, 0, P, nv, dk), rows_slice(pend_in, P, 2 * P, nv, dv)
    wkp = rows_slice(pend_in, 2 * P, 3 * P, nv, dk)
    cum_p = mb.reshape(x=mb.slice_by_index(x=pend_in, begin=[0, 3 * P, 0], end=[nv, 3 * P + 1, P]), shape=(nv, P, 1))
    total = mb.reduce_sum(x=mb.mul(x=cum_p, y=commit_last), axes=[1], keep_dims=True)
    kd_ = mb.mul(x=mb.mul(x=kp, y=commit), y=mb.exp(x=mb.minimum(x=mb.sub(x=total, y=cum_p), y=f16(0))))
    s1 = mb.add(x=mb.mul(x=rec_in, y=mb.exp(x=total)),
                y=mb.matmul(x=kd_, y=mb.sub(x=up, y=mb.matmul(x=wkp, y=rec_in)), transpose_x=True))
    _tap("s1", s1)
    core = delta_core(kh, vh, beta, g, T, nv)
    cum, _, u, wk = core
    _tap("cum", cum), _tap("pair", core[1]), _tap("u", u), _tap("wk", wk)
    crow = mb.concat(values=[t for t in (mb.reshape(x=cum, shape=(nv, 1, T)), np.zeros((nv, 1, dv - T), f16))
                             if not (isinstance(t, np.ndarray) and t.size == 0)], axis=2)
    pend_out = mb.concat(values=[pad_rows(kh, T, P, nv, dk), pad_rows(u, T, P, nv, dv), pad_rows(wk, T, P, nv, dk),
                                 crow], axis=1)
    o = _tap("o_raw", delta_out(s1, qh, kh, core))
    o = mb.mul(x=rms_last(o, w["linear_attn.norm.weight"].numpy().astype(f16), eps * (GDN_SQ * GDN_SV) ** 2), y=silu(z))
    o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1]), shape=(1, vd, 1, T))
    return lut_linear(o, q["linear_attn.out_proj.weight"]), rows, s1, pend_out


def gdn_lazy_prefill_block(cfg, w, q, h, conv_rows, conv_sel, conv_sel_out, rec_in, pend_in, commit, commit_last,
                           valid, T):
    """Prefill companion of gdn_lazy_block (T > PEND rows, all committed in this call): applies the previous call's
    pending rows, runs the chunked recurrence over this block (padding rows: valid = 0 -> beta = 0, log decay = 0,
    no state change) and returns the committed state, no pending rows (zeros) and the conv rows in the T=PEND
    layout: rows 0..2 = the 3 rows ending at the last valid token (conv_sel_out picks them), the rest zero, so the
    next call (either function) takes them with commit = 0."""
    raise ImportError(
        "dflash2_gdn_lazy was removed from this Jeff-only tree. "
        "Jeff convert uses the torch graphs in coreai/qwen38_coreai_build.py, not this MIL helper."
    )
    P = PEND
    nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim"))
    kd, vd, eps = nk * dk, nv * dv, cfg["rms_norm_eps"]
    cdim = 2 * kd + vd
    f16 = np.float16
    qkv, z, b, a = gdn_proj(cfg, w, q, h, T)
    prev = mb.matmul(x=conv_sel, y=conv_rows)                                            # (3, cdim)
    rows = mb.concat(values=[prev, mb.transpose(x=qkv, perm=[1, 0])], axis=0)           # (T + 3, cdim)
    cw = w["linear_attn.conv1d.weight"][:, 0].numpy().astype(f16).T
    conv = None
    for j in range(4):
        term = mb.mul(x=mb.slice_by_index(x=rows, begin=[j, 0], end=[j + T, cdim]), y=np.ascontiguousarray(cw[j:j + 1]))
        conv = term if conv is None else mb.add(x=conv, y=term)
    conv = mb.transpose(x=silu(conv), perm=[1, 0])
    qq, kk, vv = mb.split(x=conv, split_sizes=[kd, kd, vd], axis=0)
    rep = nv // nk

    def heads(t):
        t = mb.transpose(x=mb.reshape(x=t, shape=(nk, dk, T)), perm=[0, 2, 1])
        return mb.reshape(x=mb.tile(x=mb.reshape(x=t, shape=(nk, 1, T, dk)), reps=[1, rep, 1, 1]), shape=(nv, T, dk))

    def l2n(t, scale):
        ss = mb.reduce_sum(x=mb.mul(x=t, y=t), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.mul(x=t, y=mb.rsqrt(x=ss, epsilon=1e-6)), y=f16(scale))

    qh, kh = l2n(heads(qq), dk ** -0.5 * GDN_SQ), l2n(heads(kk), 1.0)
    vh = mb.mul(x=mb.transpose(x=mb.reshape(x=vv, shape=(nv, dv, T)), perm=[0, 2, 1]), y=f16(GDN_SV))
    beta = mb.mul(x=mb.sigmoid(x=b), y=valid)
    neg_a = (-w["linear_attn.A_log"].exp()).numpy().astype(f16).reshape(nv, 1, 1)
    dt = w["linear_attn.dt_bias"].numpy().astype(f16).reshape(nv, 1, 1)
    g = mb.mul(x=mb.mul(x=softplus(mb.add(x=a, y=dt)), y=neg_a), y=valid)
    kp, up = rows_slice(pend_in, 0, P, nv, dk), rows_slice(pend_in, P, 2 * P, nv, dv)
    wkp = rows_slice(pend_in, 2 * P, 3 * P, nv, dk)
    cum_p = mb.reshape(x=mb.slice_by_index(x=pend_in, begin=[0, 3 * P, 0], end=[nv, 3 * P + 1, P]), shape=(nv, P, 1))
    total = mb.reduce_sum(x=mb.mul(x=cum_p, y=commit_last), axes=[1], keep_dims=True)
    kd_ = mb.mul(x=mb.mul(x=kp, y=commit), y=mb.exp(x=mb.minimum(x=mb.sub(x=total, y=cum_p), y=f16(0))))
    s1 = mb.add(x=mb.mul(x=rec_in, y=mb.exp(x=total)),
                y=mb.matmul(x=kd_, y=mb.sub(x=up, y=mb.matmul(x=wkp, y=rec_in)), transpose_x=True))
    outs, s_out = chunked_delta(qh, kh, vh, beta, g, s1, None, nv, T, dk, dv)
    o = mb.mul(x=rms_last(outs[0], w["linear_attn.norm.weight"].numpy().astype(f16), eps * (GDN_SQ * GDN_SV) ** 2), y=silu(z))
    o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1]), shape=(1, vd, 1, T))
    conv_out = mb.concat(values=[mb.matmul(x=conv_sel_out, y=rows), np.zeros((P, cdim), f16)], axis=0)  # (P + 3, cdim)
    return lut_linear(o, q["linear_attn.out_proj.weight"]), conv_out, s_out, mb.mul(x=pend_in, y=f16(0))


def gdn_prefill(cfg, w, q, h, conv_st, rec_st, T, ring_keep, ring_sel, prev_sel, scr_st=None):
    """Gated DeltaNet over T tokens (h: (1, 5120, 1, T)); block starts at a position divisible by 4 and
    T % 4 == 0, so the conv ring's slots 1..3 hold the 3 previous tokens and after the block slots 0..3 hold
    the last 4 tokens in order. The recurrence runs in sub-chunks (chunked_delta)."""
    nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim"))
    kd, vd, eps = nk * dk, nv * dv, cfg["rms_norm_eps"]
    cdim = 2 * kd + vd
    qkv = mb.reshape(x=lut_linear(h, q["linear_attn.in_proj_qkv.weight"]), shape=(cdim, T))
    z = mb.transpose(x=mb.reshape(x=lut_linear(h, q["linear_attn.in_proj_z.weight"]), shape=(nv, dv, T)), perm=[0, 2, 1])
    b = mb.reshape(x=dense_linear(h, w["linear_attn.in_proj_b.weight"]), shape=(nv, T, 1))
    a = mb.reshape(x=dense_linear(h, w["linear_attn.in_proj_a.weight"]), shape=(nv, T, 1))
    # One write per state, and only its returned value is used downstream (a graph that also uses the raw read
    # value, or writes a state twice, fails to load: MIL->EIR bad_cast). The conv state holds two 4-token
    # rings: keep the active one (it has the 3 previous tokens), write this block's last 4 tokens into the
    # other (ring_keep / ring_sel), then read the previous tokens back from the written value (prev_sel).
    ring = mb.coreml_update_state(state=conv_st, value=mb.add(
        x=mb.mul(x=mb.read_state(input=conv_st), y=ring_keep), y=mb.matmul(x=ring_sel, y=qkv, transpose_y=True)))
    prev = mb.transpose(x=mb.matmul(x=prev_sel, y=ring), perm=[1, 0])                # (cdim, 3)
    seq = mb.concat(values=[prev, qkv], axis=1)                                       # (cdim, T + 3)
    cw = w["linear_attn.conv1d.weight"][:, 0].numpy().astype(np.float16)             # (cdim, 4), oldest first
    conv = None
    for j in range(4):
        term = mb.mul(x=mb.slice_by_index(x=seq, begin=[0, j], end=[cdim, j + T]), y=cw[:, j:j + 1])
        conv = term if conv is None else mb.add(x=conv, y=term)
    conv = silu(conv)                                                            # (cdim, T)
    qq, kk, vv = mb.split(x=conv, split_sizes=[kd, kd, vd], axis=0)
    rep = nv // nk

    def heads(t):  # (nk*dk, T) -> (nv, T, dk)
        t = mb.transpose(x=mb.reshape(x=t, shape=(nk, dk, T)), perm=[0, 2, 1])
        return mb.reshape(x=mb.tile(x=mb.reshape(x=t, shape=(nk, 1, T, dk)), reps=[1, rep, 1, 1]), shape=(nv, T, dk))

    def l2n(t, scale):
        ss = mb.reduce_sum(x=mb.mul(x=t, y=t), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.mul(x=t, y=mb.rsqrt(x=ss, epsilon=1e-6)), y=np.float16(scale))

    qh, kh = l2n(heads(qq), dk ** -0.5), l2n(heads(kk), 1.0)
    vh = mb.transpose(x=mb.reshape(x=vv, shape=(nv, dv, T)), perm=[0, 2, 1])          # (nv, T, dv)
    beta = mb.sigmoid(x=b)
    neg_a = (-w["linear_attn.A_log"].exp()).numpy().astype(np.float16).reshape(nv, 1, 1)
    dt = w["linear_attn.dt_bias"].numpy().astype(np.float16).reshape(nv, 1, 1)
    g = mb.mul(x=softplus(mb.add(x=a, y=dt)), y=neg_a)                           # log decay (nv, T, 1)
    outs = chunked_delta(qh, kh, vh, beta, g, None, rec_st, nv, T, dk, dv, scr_st)
    o = mb.concat(values=outs, axis=1) if len(outs) > 1 else outs[0]                   # (nv, T, dv)
    o = mb.mul(x=rms_last(o, w["linear_attn.norm.weight"].numpy().astype(np.float16), eps), y=silu(z))
    o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1]), shape=(1, vd, 1, T))
    return lut_linear(o, q["linear_attn.out_proj.weight"])


def attn_prefill(cfg, w, q, h, k_st, v_st, cos, sin, mask, kvw, T):
    """Gated full attention over T tokens: cos / sin (T, 64), mask (T, CTX) additive causal, kvw (T, CTX)
    one-hot rows = cache positions written."""
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    rot, eps, grp = int(hd * cfg["rope_parameters"]["partial_rotary_factor"]), cfg["rms_norm_eps"], nh // nkv

    def tmajor(x, c):  # (1, c, 1, T) -> (T, c)
        return mb.transpose(x=mb.reshape(x=x, shape=(c, T)), perm=[1, 0])

    qg = mb.reshape(x=tmajor(lut_linear(h, q["self_attn.q_proj.weight"]), 2 * nh * hd), shape=(T, nh, 2 * hd))
    qh = mb.slice_by_index(x=qg, begin=[0, 0, 0], end=[T, nh, hd])
    gate = mb.reshape(x=mb.slice_by_index(x=qg, begin=[0, 0, hd], end=[T, nh, 2 * hd]), shape=(T, nh * hd))
    qh = _tap("att_qn", rms_last(qh, (1 + w["self_attn.q_norm.weight"]).numpy().astype(np.float16), eps), "att")
    kh = rms_last(mb.reshape(x=tmajor(lut_linear(h, q["self_attn.k_proj.weight"]), nkv * hd), shape=(T, nkv, hd)),
                  (1 + w["self_attn.k_norm.weight"]).numpy().astype(np.float16), eps)
    vh = mb.reshape(x=tmajor(lut_linear(h, q["self_attn.v_proj.weight"]), nkv * hd), shape=(T, nkv, hd))
    c3, s3 = mb.reshape(x=cos, shape=(T, 1, rot)), mb.reshape(x=sin, shape=(T, 1, rot))

    def rope(t, n):
        r = mb.slice_by_index(x=t, begin=[0, 0, 0], end=[T, n, rot])
        rest = mb.slice_by_index(x=t, begin=[0, 0, rot], end=[T, n, hd])
        r1 = mb.slice_by_index(x=r, begin=[0, 0, 0], end=[T, n, rot // 2])
        r2 = mb.slice_by_index(x=r, begin=[0, 0, rot // 2], end=[T, n, rot])
        rh = mb.concat(values=[mb.mul(x=r2, y=np.float16(-1)), r1], axis=2)
        return mb.concat(values=[mb.add(x=mb.mul(x=r, y=c3), y=mb.mul(x=rh, y=s3)), rest], axis=2)

    qh = _tap("att_q", rope(qh, nh), "att")
    kt = _tap("att_k", mb.transpose(x=rope(kh, nkv), perm=[1, 0, 2]), "att")          # (nkv, T, hd)
    vt = _tap("att_v", mb.transpose(x=vh, perm=[1, 0, 2]), "att")
    if KV_IN:
        qg4 = mb.reshape(x=mb.transpose(x=mb.reshape(x=qh, shape=(T, nkv, grp, hd)), perm=[1, 2, 0, 3]),
                         shape=(nkv, grp * T, hd))
        sc_h = mb.add(x=mb.reshape(x=mb.mul(x=mb.matmul(x=qg4, y=k_st, transpose_y=True), y=np.float16(hd ** -0.5)),
                                   shape=(nkv, grp, T, CTX)), y=mb.reshape(x=mask, shape=(1, 1, 1, CTX)))
        causal = np.where(np.arange(T)[None, :] <= np.arange(T)[:, None], 0, -1e4).astype(np.float16)
        sc_b = mb.add(x=mb.reshape(x=mb.mul(x=mb.matmul(x=qg4, y=kt, transpose_y=True), y=np.float16(hd ** -0.5)),
                                   shape=(nkv, grp, T, T)), y=causal)
        # one softmax over [history | block] (a hand-written two-part softmax sent the whole graph to the CPU)
        pr = _tap("att_p", mb.reshape(x=mb.softmax(x=mb.concat(values=[sc_h, sc_b], axis=-1), axis=-1),
                                      shape=(nkv, grp * T, CTX + T)), "att")
        o = mb.add(x=mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, 0], end=[nkv, grp * T, CTX]), y=v_st),
                   y=mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, CTX], end=[nkv, grp * T, CTX + T]), y=vt))
        o = _tap("att_o", mb.reshape(x=mb.transpose(x=mb.reshape(x=o, shape=(nkv, grp, T, hd)), perm=[2, 0, 1, 3]),
                                     shape=(T, nh * hd)), "att")
        o = mb.mul(x=o, y=_tap("att_gate", mb.sigmoid(x=gate), "att"))
        y = lut_linear(mb.reshape(x=mb.transpose(x=o, perm=[1, 0]), shape=(1, nh * hd, 1, T)),
                       q["self_attn.o_proj.weight"])
        return y, kt, vt
    keep = mb.reshape(x=mb.sub(x=np.float16(1), y=mb.reduce_sum(x=kvw, axes=[0], keep_dims=True)), shape=(1, CTX, 1))
    pt = mb.reshape(x=mb.transpose(x=kvw, perm=[1, 0]), shape=(1, CTX, T))
    kc = kv_write(k_st, mb.add(x=mb.mul(x=kv_read(k_st), y=keep),
                                                           y=mb.matmul(x=pt, y=kt)))
    vc = kv_write(v_st, mb.add(x=mb.mul(x=kv_read(v_st), y=keep),
                                                           y=mb.matmul(x=pt, y=vt)))
    qg4 = mb.reshape(x=mb.transpose(x=mb.reshape(x=qh, shape=(T, nkv, grp, hd)), perm=[1, 2, 0, 3]),
                     shape=(nkv, grp * T, hd))
    sc = mb.reshape(x=mb.mul(x=mb.matmul(x=qg4, y=kc, transpose_y=True), y=np.float16(hd ** -0.5)),
                    shape=(nkv, grp, T, CTX))
    p = mb.reshape(x=mb.softmax(x=mb.add(x=sc, y=mask), axis=-1), shape=(nkv, grp * T, CTX))
    o = mb.reshape(x=mb.matmul(x=p, y=vc), shape=(nkv, grp, T, hd))
    o = mb.reshape(x=mb.transpose(x=o, perm=[2, 0, 1, 3]), shape=(T, nh * hd))
    o = mb.mul(x=o, y=mb.sigmoid(x=gate))
    y = lut_linear(mb.reshape(x=mb.transpose(x=o, perm=[1, 0]), shape=(1, nh * hd, 1, T)),
                   q["self_attn.o_proj.weight"])
    return (y, kc, vc) if KV_IO else y


def build(cfg, weights, quant, T=1, compile_model=True):
    """Decode graph (T = 1) or T-token prefill graph with the same states. Returns the compiled model, or the
    .mlpackage when compile_model is False (for multifunction packaging)."""
    if T > 1:
        return build_prefill(cfg, weights, quant, T, compile_model)
    specs = {"x": mb.TensorSpec((1, cfg["hidden_size"], 1, 1), types.fp16),
             "cos": mb.TensorSpec((1, 64), types.fp16), "sin": mb.TensorSpec((1, 64), types.fp16),
             "mask": mb.TensorSpec((1, CTX), types.fp16), "kv_onehot": mb.TensorSpec((1, CTX, 1), types.fp16),
             "slot_onehot": mb.TensorSpec((8, 1), types.fp16), "conv_perm": mb.TensorSpec((4, 8), types.fp16)}
    order = [l for _ in range(REPEAT) for l in LAYERS]
    if GDN_IO:
        for k in ("slot_onehot", "conv_perm"):
            del specs[k]
        specs.update(gdn_io_specs(cfg, order))
    specs.update(state_specs(cfg, order))
    extra, states_out = [], []
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        x, cos, sin, mask = (fn.inputs[k] for k in ("x", "cos", "sin", "mask"))
        kv_oh = fn.inputs["kv_onehot"]
        slot_oh, perm = fn.inputs.get("slot_onehot"), fn.inputs.get("conv_perm")
        eps = cfg["rms_norm_eps"]
        T = 1
        for j, l in enumerate(order, JOFF):
            w, q = weights[l], quant[l]
            h = rms_hidden(x, (1 + w["input_layernorm.weight"]).numpy().astype(np.float16), eps)
            if cfg["layer_types"][l] == "linear_attention" and GDN_IO:
                y, rows, rec_o = gdn_io(cfg, w, q, h, fn.inputs[f"conv{j}"], fn.inputs[f"rec{j}"], T)
                states_out.append(mb.identity(x=mb.slice_by_index(x=rows, begin=[1, 0], end=[4, rows.shape[1]]),
                                              name=f"conv{j}_out"))
                states_out.append(mb.identity(x=rec_o, name=f"rec{j}_out"))
            elif cfg["layer_types"][l] == "linear_attention":
                y = gdn_block(cfg, w, q, h, fn.inputs[f"conv{j}"], fn.inputs[f"rec{j}"], slot_oh, perm)
            else:
                y = attn_block(cfg, w, q, h, fn.inputs[f"k{j}"], fn.inputs[f"v{j}"], cos, sin, mask, kv_oh)
                if KV_IO:
                    y, kc, vc = y
                    states_out += [mb.identity(x=kc, name=f"k{j}_out"), mb.identity(x=vc, name=f"v{j}_out")]
            x = mb.add(x=x, y=y)
            h = rms_hidden(x, (1 + w["post_attention_layernorm.weight"]).numpy().astype(np.float16), eps)
            rot = q.get("mlp.rotation")  # (seed_in, seed_mid) when the MLP was quantized in the online basis
            h = rot_conv(h, h.shape[1], rot[0]) if rot else h
            g, u = lut_linear(h, q["mlp.gate_proj.weight"]), lut_linear(h, q["mlp.up_proj.weight"])
            a = mb.mul(x=mb.silu(x=g), y=u)
            a = rot_conv(a, a.shape[1], rot[1]) if rot else a
            x = mb.add(x=x, y=lut_linear(a, q["mlp.down_proj.weight"]))
            if l in TAPS and l != order[-1]:  # a tap on the last layer is "y" (a second identity output of the same
                extra.append(mb.identity(x=x, name=f"tap{l}"))  # tensor comes back as zeros / fails to load)
        fn.set_outputs([mb.identity(x=x, name="y")] + extra + states_out)
    prog = Program()
    prog.add_function("main", fn)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"L{LAYERS[0]}-{LAYERS[-1]}x{REPEAT}_ctx{CTX}_{MLP_FMT.replace(' ', '')}_{ATT_FMT.replace(' ', '')}"
    pkg, mlc = OUT / f"{tag}.mlpackage", OUT / f"{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
               pass_pipeline=pipeline).save(str(pkg))
    if not compile_model:
        return pkg
    ct.models.utils.compile_model(str(pkg), str(mlc))
    return mlc


def gdn_io_specs(cfg, order, T=1):
    """I/O mode inputs per DeltaNet layer: conv{j} (3, cdim) [LAZY: (T + 3, cdim), the previous call's rows],
    rec{j} (nv, dk, dv) [LAZY: + pend{j} (nv, 3P + 1, dv)]."""
    specs = {}
    for j, l in enumerate(order, JOFF):
        if cfg["layer_types"][l] == "linear_attention":
            nv, dk, dv = (cfg[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
            cdim = 2 * cfg["linear_num_key_heads"] * dk + nv * dv
            specs[f"conv{j}"] = mb.TensorSpec((PEND + 3 if LAZY else 3, cdim), types.fp16)
            specs[f"rec{j}"] = mb.TensorSpec((nv, dk, dv), types.fp16)
            if LAZY:
                specs[f"pend{j}"] = mb.TensorSpec((nv, 3 * PEND + 1, dv), types.fp16)
        elif KV_IO or KV_IN:
            specs[f"k{j}"] = mb.TensorSpec((cfg["num_key_value_heads"], CTX, cfg["head_dim"]), types.fp16)
            specs[f"v{j}"] = mb.TensorSpec((cfg["num_key_value_heads"], CTX, cfg["head_dim"]), types.fp16)
    return specs


def state_specs(cfg, order):
    specs = {}
    for j, l in enumerate(order, JOFF):
        if cfg["layer_types"][l] == "linear_attention":
            conv_dim = 2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] + \
                cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            if GDN_IO:
                continue
            specs[f"conv{j}"] = mb.StateTensorSpec((8, conv_dim), types.fp16)
            nv, dk, dv = (cfg[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
            specs[f"rec{j}"] = mb.StateTensorSpec((nv, dk + (SCR_ROWS if REC_OUT == "concat" else 0), dv), types.fp16)
            if REC_OUT == "scratch":
                specs[f"scr{j}"] = mb.StateTensorSpec((nv, SCR_ROWS, dv), types.fp16)
        elif not (KV_IO or KV_IN):
            specs[f"k{j}"] = mb.StateTensorSpec((cfg["num_key_value_heads"], CTX, cfg["head_dim"]), types.fp16)
            specs[f"v{j}"] = mb.StateTensorSpec((cfg["num_key_value_heads"], CTX, cfg["head_dim"]), types.fp16)
    return specs


def build_prefill(cfg, weights, quant, T, compile_model=True):
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    specs = {"x": mb.TensorSpec((1, cfg["hidden_size"], 1, T), types.fp16),
             "cos": mb.TensorSpec((T, rot), types.fp16), "sin": mb.TensorSpec((T, rot), types.fp16),
             "mask": mb.TensorSpec((T, CTX), types.fp16), "kv_write": mb.TensorSpec((T, CTX), types.fp16),
             "ring_keep": mb.TensorSpec((8, 1), types.fp16), "ring_sel": mb.TensorSpec((8, T), types.fp16),
             "prev_sel": mb.TensorSpec((3, 8), types.fp16)}
    order = [l for _ in range(REPEAT) for l in LAYERS]
    if GDN_IO:
        for k in ("ring_keep", "ring_sel", "prev_sel"):
            del specs[k]
        if KV_IN:
            del specs["kv_write"]
            specs["mask"] = mb.TensorSpec((1, CTX), types.fp16)
        if LAZY:
            specs["conv_sel"] = mb.TensorSpec((3, PEND + 3), types.fp16)
            specs["commit"] = mb.TensorSpec((1, PEND, 1), types.fp16)
            specs["commit_last"] = mb.TensorSpec((1, PEND, 1), types.fp16)
            if T > PEND:  # prefill companion: all rows committed, padding masked
                specs["valid"] = mb.TensorSpec((1, T, 1), types.fp16)
                specs["conv_sel_out"] = mb.TensorSpec((3, T + 3), types.fp16)
        else:
            specs["valid"] = mb.TensorSpec((1, T, 1), types.fp16)
            if not CONV_OUT_ALL:
                specs["conv_sel"] = mb.TensorSpec((3, T + 3), types.fp16)
        specs.update(gdn_io_specs(cfg, order, T))
    specs.update(state_specs(cfg, order))
    extra, states_out = [], []
    global _dbg
    _dbg = {"names": {id(quant[l][k]): f"{l}:{k[:-len('.weight')]}" for l in order for k in MIXER_KEYS
                      if k in quant[l]}, "vars": {}, "map": {}} if DBG_MIXER_IN else None
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        x, cos, sin, mask = (fn.inputs[k] for k in ("x", "cos", "sin", "mask"))
        kvw = fn.inputs.get("kv_write")
        ring_keep, ring_sel, prev_sel = (fn.inputs.get(k) for k in ("ring_keep", "ring_sel", "prev_sel"))
        eps = cfg["rms_norm_eps"]
        for j, l in enumerate(order, JOFF):
            w, q = weights[l], quant[l]
            h = rms_hidden(x, (1 + w["input_layernorm.weight"]).numpy().astype(np.float16), eps)
            if cfg["layer_types"][l] == "linear_attention" and GDN_IO and LAZY and T > PEND:
                y, conv_o, rec_o, pend_o = gdn_lazy_prefill_block(
                    cfg, w, q, h, fn.inputs[f"conv{j}"], fn.inputs["conv_sel"], fn.inputs["conv_sel_out"],
                    fn.inputs[f"rec{j}"], fn.inputs[f"pend{j}"], fn.inputs["commit"], fn.inputs["commit_last"],
                    fn.inputs["valid"], T)
                states_out += [mb.identity(x=conv_o, name=f"conv{j}_out"), mb.identity(x=rec_o, name=f"rec{j}_out"),
                               mb.identity(x=pend_o, name=f"pend{j}_out")]
            elif cfg["layer_types"][l] == "linear_attention" and GDN_IO and LAZY:
                y, conv_o, rec_o, pend_o = gdn_lazy_block(
                    cfg, w, q, h, fn.inputs[f"conv{j}"], fn.inputs["conv_sel"], fn.inputs[f"rec{j}"],
                    fn.inputs[f"pend{j}"], fn.inputs["commit"], fn.inputs["commit_last"], T)
                states_out += [mb.identity(x=conv_o, name=f"conv{j}_out"), mb.identity(x=rec_o, name=f"rec{j}_out"),
                               mb.identity(x=pend_o, name=f"pend{j}_out")]
            elif cfg["layer_types"][l] == "linear_attention" and GDN_IO:
                y, conv_o, rec_o = gdn_io(cfg, w, q, h, fn.inputs[f"conv{j}"], fn.inputs[f"rec{j}"], T,
                                          fn.inputs["valid"], None if CONV_OUT_ALL else fn.inputs["conv_sel"])
                states_out.append(mb.identity(x=conv_o, name=f"conv{j}_out"))
                states_out.append(mb.identity(x=rec_o, name=f"rec{j}_out"))
            elif cfg["layer_types"][l] == "linear_attention":
                y = gdn_prefill(cfg, w, q, h, fn.inputs[f"conv{j}"], fn.inputs[f"rec{j}"], T, ring_keep, ring_sel,
                                prev_sel, fn.inputs.get(f"scr{j}"))
            else:
                y = attn_prefill(cfg, w, q, h, fn.inputs[f"k{j}"], fn.inputs[f"v{j}"], cos, sin, mask, kvw, T)
                if KV_IN:
                    y, kn, vn = y
                    states_out += [mb.identity(x=kn, name=f"k{j}_new"), mb.identity(x=vn, name=f"v{j}_new")]
                elif KV_IO:
                    y, kc, vc = y
                    states_out += [mb.identity(x=kc, name=f"k{j}_out"), mb.identity(x=vc, name=f"v{j}_out")]
            x = mb.add(x=x, y=y)
            h = rms_hidden(x, (1 + w["post_attention_layernorm.weight"]).numpy().astype(np.float16), eps)
            rot_s = q.get("mlp.rotation")
            h = _tap("mlp_h", rot_conv(h, h.shape[1], rot_s[0]) if rot_s else h, "mlp")
            g, u = lut_linear(h, q["mlp.gate_proj.weight"]), lut_linear(h, q["mlp.up_proj.weight"])
            _tap("mlp_g", g, "mlp"), _tap("mlp_u", u, "mlp")
            a = _tap("mlp_act", mb.mul(x=mlp_silu(g), y=u), "mlp")
            a = _tap("mlp_a", rot_conv(a, a.shape[1], rot_s[1]) if rot_s else a, "mlp")
            ds = float(MLP_DS_TABLE[str(l)]) if MLP_DS_TABLE is not None else MLP_DS
            if MLP_DS_DYN:  # per token f = min(ds, C / max|a|): small inputs scaled up, large ones not overflowed
                m_a = mb.reduce_max(x=mb.abs(x=a), axes=[1], keep_dims=True)               # (1, 1, 1, T)
                f = mb.minimum(x=mb.real_div(x=np.float16(MLP_DS_C), y=mb.maximum(x=m_a, y=np.float16(1e-3))),
                               y=np.float16(ds))
                dn = mb.real_div(x=lut_linear(mb.mul(x=a, y=f), q["mlp.down_proj.weight"]), y=f)
            elif ds != 1:  # keep the down projection's products out of fp16 subnormals (see MLP_DS)
                dn = mb.mul(x=lut_linear(mb.mul(x=a, y=np.float16(ds)), q["mlp.down_proj.weight"]),
                            y=np.float16(1 / ds))
            else:
                dn = lut_linear(a, q["mlp.down_proj.weight"])
            x = mb.add(x=x, y=_tap("mlp_down", dn, "mlp"))
            if l in TAPS and l != order[-1]:  # a tap on the last layer is "y" (a second identity output of the same
                extra.append(mb.identity(x=x, name=f"tap{l}"))  # tensor comes back as zeros / fails to load)
        dbg_outs = list(_dbg["vars"].values()) if _dbg is not None else []
        fn.set_outputs([mb.identity(x=x, name="y")] + extra + states_out + dbg_outs)
    prog = Program()
    prog.add_function("main", fn)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"L{LAYERS[0]}-{LAYERS[-1]}x{REPEAT}_ctx{CTX}_prefill{T}"
    if _dbg is not None:
        (OUT / f"{tag}.dbg.json").write_text(json.dumps(_dbg["map"], indent=1))
        _dbg = None
    pkg, mlc = OUT / f"{tag}.mlpackage", OUT / f"{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
               pass_pipeline=pipeline).save(str(pkg))
    if not compile_model:
        return pkg
    ct.models.utils.compile_model(str(pkg), str(mlc))
    return mlc


def prefill_inputs(p0, T, cfg, half=0):
    """Host-side inputs for a T-token prefill block starting at position p0 (p0 % 4 == 0, T % 4 == 0) with conv
    ring half `half` active; afterwards the other half (1 - half) is active."""
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    f = np.outer(np.arange(p0, p0 + T), inv)
    ang = np.concatenate([f, f], axis=1)
    pos = np.arange(p0, p0 + T)[:, None]
    mask = np.where(np.arange(CTX)[None, :] <= pos, 0, -1e4).astype(np.float16)
    kvw = np.zeros((T, CTX), np.float16)
    kvw[np.arange(T), np.arange(p0, p0 + T)] = 1
    keep = np.zeros((8, 1), np.float16)
    keep[4 * half:4 * half + 4] = 1
    sel = np.zeros((8, T), np.float16)
    sel[4 * (1 - half) + np.arange(4), np.arange(T - 4, T)] = 1         # slot s of the other ring = token p0 + T - 4 + s
    prev = np.zeros((3, 8), np.float16)
    prev[np.arange(3), 4 * half + np.arange(1, 4)] = 1                   # tokens p0 - 3 .. p0 - 1 sit in slots 1..3
    return {"cos": np.cos(ang).astype(np.float16), "sin": np.sin(ang).astype(np.float16), "mask": mask, "kv_write": kvw,
            "ring_keep": keep, "ring_sel": sel, "prev_sel": prev}


def step_inputs(t, half=0, rows=8):
    """Host-side per-token inputs for the masked state writes (conv ring half `half` active). rows = 4 for builds
    made before the two-ring conv state (no batched prefill)."""
    kv = np.zeros((1, CTX, 1), np.float16)
    kv[0, t, 0] = 1
    slot = np.zeros((rows, 1), np.float16)
    slot[4 * half + t % 4, 0] = 1
    perm = np.zeros((4, rows), np.float16)
    for j in range(4):
        perm[j, 4 * half + (t - 3 + j) % 4] = 1
    return {"kv_onehot": kv, "slot_onehot": slot, "conv_perm": perm}


def main():
    cfg = text_config()
    weights = load_layer_weights(LAYERS)
    quant, deq = {}, {}
    for l in LAYERS:
        quant[l], deq[l] = {}, dict(weights[l])
        for name, role in QUANT.items():
            if name in weights[l]:
                quant[l][name] = quantize(weights[l][name], MLP_FMT if role == "mlp" else ATT_FMT)
                deq[l][name] = quant[l][name][3]
        print(f"layer {l} quantized", flush=True)
    mlc = build(cfg, weights, quant)
    print(f"built {mlc.name}", flush=True)
    if not CHECK:
        return
    units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_AND_NE"))
    model = ct.models.CompiledMLModel(str(mlc), compute_units=units)
    state = model.make_state()
    ref = [DecodeLayer(cfg, l, deq[l], ctx=CTX) for _ in range(REPEAT) for l in LAYERS]
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    rng = np.random.default_rng(0)
    for t in range(CHECK):
        x = (rng.standard_normal(cfg["hidden_size"]) * 2).astype(np.float16)
        f = t * inv
        cos, sin = np.cos(np.concatenate([f, f])), np.sin(np.concatenate([f, f]))
        mask = np.where(np.arange(CTX) <= t, 0, -1e4).astype(np.float16)
        y = model.predict({"x": x.reshape(1, -1, 1, 1), "cos": cos[None].astype(np.float16),
                           "sin": sin[None].astype(np.float16), "mask": mask[None],
                           **step_inputs(t)}, state=state)["y"].astype(np.float32).ravel()
        h = torch.from_numpy(x.astype(np.float32))
        for d in ref:
            h = d.step(h)
        h = h.numpy()
        print(f"step {t}: cos(ANE, reference) {float(y @ h / np.linalg.norm(y) / np.linalg.norm(h)):.5f}  "
              f"rel err {np.linalg.norm(y - h) / np.linalg.norm(h):.4f}", flush=True)


if __name__ == "__main__":
    main()
