"""Shared LUT / Hadamard helpers.

Origin: Anemll/anemll-forge. Pulled in by ``qwen38_ane_chunk`` (Jeff convert's graph stack).

Shared code for the Qwen3 MLP LUT experiments: an fp32 Qwen3 reference forward (no transformers
needed), WikiText-2 token chunks, Hadamard rotations, LUT / FP8 quantizers and GPTQ.

Environment: MODEL (Qwen3 checkpoint dir, default ~/Models/Qwen3-0.6B), WIKI (dir with wiki2_train.txt
and wiki2_test.txt, default ~/Models/wikitext). Token ids are cached there as qwen3_{split}_ids.npy;
reading Qwen3's tokenizer.json needs tokenizers >= 0.19.
"""
import json
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from scipy.linalg import hadamard

torch.set_grad_enabled(False)
MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3-0.6B")))
WIKI = Path(os.path.expanduser(os.environ.get("WIKI", "~/Models/wikitext")))
SEQ = 512


# ---------------------------------------------------------------- minimal Qwen3 forward (fp32)
class Qwen3:
    """MLP inputs are u = the RMS-normalized residual *without* the norm weight."""

    def __init__(self, path=MODEL):
        cfg = json.loads((path / "config.json").read_text())
        self.w = {k: v.float() for k, v in load_file(path / "model.safetensors").items()}
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.nl, self.eps = cfg["num_hidden_layers"], cfg["rms_norm_eps"]
        self.emb = self.w["model.embed_tokens.weight"]  # tied with lm_head
        inv = 1.0 / cfg["rope_theta"] ** (torch.arange(0, self.hd, 2).float() / self.hd)
        ang = torch.outer(torch.arange(SEQ).float(), inv)
        self.cos, self.sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()

    def rms(self, x, w=None):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x if w is None else x * w

    def rope(self, x):  # (heads, T, head_dim)
        x1, x2 = x.chunk(2, -1)
        return x * self.cos[: x.shape[1]] + torch.cat([-x2, x1], -1) * self.sin[: x.shape[1]]

    def attention(self, l, x):
        """Residual stream after layer l's attention block."""
        w, p, t = self.w, f"model.layers.{l}.self_attn.", x.shape[0]
        h = self.rms(x, w[f"model.layers.{l}.input_layernorm.weight"])
        q = self.rms((h @ w[p + "q_proj.weight"].T).view(t, self.nh, self.hd), w[p + "q_norm.weight"])
        k = self.rms((h @ w[p + "k_proj.weight"].T).view(t, self.nkv, self.hd), w[p + "k_norm.weight"])
        v = (h @ w[p + "v_proj.weight"].T).view(t, self.nkv, self.hd).transpose(0, 1)
        r = self.nh // self.nkv
        o = F.scaled_dot_product_attention(
            self.rope(q.transpose(0, 1))[None], self.rope(k.transpose(0, 1)).repeat_interleave(r, 0)[None],
            v.repeat_interleave(r, 0)[None], is_causal=True)[0]
        return x + o.transpose(0, 1).reshape(t, self.nh * self.hd) @ w[p + "o_proj.weight"].T

    def mlp_weights(self, l):
        """(RMSNorm weight, gate, up, down) of layer l."""
        p = f"model.layers.{l}."
        return (self.w[p + "post_attention_layernorm.weight"],
                *(self.w[p + f"mlp.{n}_proj.weight"] for n in ("gate", "up", "down")))

    def mlp(self, l, u):
        gamma, wg, wu, wd = self.mlp_weights(l)
        h = u * gamma
        return (F.silu(h @ wg.T) * (h @ wu.T)) @ wd.T

    def nll(self, ids, x):
        """Summed next-token NLL from the final residual stream."""
        logits = self.rms(x, self.w["model.norm.weight"]) @ self.emb.T
        return F.cross_entropy(logits[:-1], ids[1:], reduction="sum").item()


def wikitext_chunks(split, n):
    """First n SEQ-token chunks of WikiText-2 `split` ("train" / "test")."""
    cached = WIKI / f"qwen3_{split}_ids.npy"
    if not cached.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        np.save(cached, np.array(tok.encode((WIKI / f"wiki2_{split}.txt").read_text()[:4_000_000]).ids))
    ids = np.load(cached)
    return [torch.from_numpy(ids[i * SEQ:(i + 1) * SEQ].astype(np.int64)) for i in range(n)]


# ---------------------------------------------------------------- Hadamard rotations
def paley12():
    qr = {(i * i) % 11 for i in range(1, 11)}
    q = np.array([[0 if i == j else (1 if (j - i) % 11 in qr else -1) for j in range(11)] for i in range(11)])
    s = np.zeros((12, 12))
    s[0, 1:], s[1:, 0], s[1:, 1:] = 1, -1, q
    h = np.eye(12) + s
    assert np.allclose(h @ h.T, 12 * np.eye(12))
    return h


def rht(n, rng):
    """Randomized Hadamard, orthogonal: H diag(signs) / sqrt(n); n = 2^k or 12 * 2^k."""
    base = hadamard(n) if n & (n - 1) == 0 else np.kron(paley12(), hadamard(n // 12))
    return torch.tensor(base * rng.choice([-1.0, 1.0], n) / np.sqrt(n), dtype=torch.float32)


# fold: RMSNorm weight folded into gate/up; rin: Hadamard on the gate/up input; r4: Hadamard on the
# down input; rout: Hadamard on the down output (the rotated residual stream).
BASES = {
    "plain": dict(fold=False, rin=False, r4=False, rout=False),   # as converted today
    "had": dict(fold=True, rin=True, r4=True, rout=True),         # QuaRot-style R1 (offline) + R4
    "had-R1": dict(fold=True, rin=True, r4=False, rout=True),     # R1 only
    "online": dict(fold=False, rin=True, r4=True, rout=False),    # online Hadamards after the norm weight
}


class MLPBasis:
    """One MLP expressed in a (possibly rotated) basis. The float MLP output is unchanged."""

    def __init__(self, basis, gamma, wg, wu, wd, r1, r4):
        self.b, self.gamma, self.r1, self.r4 = BASES[basis], gamma, r1, r4
        b = self.b
        gu = [w * gamma if b["fold"] else w for w in (wg, wu)]
        gu = [w @ r1.T if b["rin"] else w for w in gu]
        down = r1 @ wd if b["rout"] else wd
        self.w = {"gate": gu[0], "up": gu[1], "down": down @ r4.T if b["r4"] else down}

    def gate_up_input(self, u):
        z = u if self.b["fold"] else u * self.gamma
        return z @ self.r1.T if self.b["rin"] else z

    def down_input(self, z, q):
        a = F.silu(z @ q["gate"].T) * (z @ q["up"].T)
        return a @ self.r4.T if self.b["r4"] else a

    def forward(self, q, u):
        y = self.down_input(self.gate_up_input(u), q) @ q["down"].T
        return y @ self.r1 if self.b["rout"] else y


# ---------------------------------------------------------------- quantizers
# Each format fits its codebook (or scales) on the full weight, then returns a rounder that maps any
# block of columns W[:, j:k] to the nearest representable values. Round-to-nearest (RTN) applies it to
# the whole matrix; GPTQ applies it one input column at a time. Vectors run along Cout, so every vector
# lies inside one column and GPTQ's column-by-column error feedback needs no change.
FORMATS = {  # name: (bits/weight of indices, spec)
    "FP8 E4M3 per-channel": (8, ("fp8",)),
    "LUT4 per-group-8 (anemll)": (4, ("group", 8, 4)),
    "LUT4 per-tensor": (4, ("group", 0, 4)),
    "vector 2x64": (3, ("vector", 2, 6, False)),
    "LUT2 per-group-8": (2, ("group", 8, 2)),
    "vector 2x16": (2, ("vector", 2, 4, False)),
    "vector 4x64": (1.5, ("vector", 4, 6, False)),
    "vector 4x64 FP8 LUT": (1.5, ("vector", 4, 6, True)),
    "vector 4x16": (1, ("vector", 4, 4, False)),
    "vector 4x16 FP8 LUT": (1, ("vector", 4, 4, True)),
    # + a per-output-channel scale after the per-tensor LUT (constexpr_blockwise_shift_scale): ANE speed
    # unchanged (qwen38_mlp_ane_probe.py); the scale adds 16 bits per output channel.
    "LUT4 per-tensor + pcs": (4, ("group", 0, 4, "pcs")),
    "vector 2x16 + pcs": (2, ("vector", 2, 4, False, "pcs")),
    "vector 2x64 + pcs": (3, ("vector", 2, 6, False, "pcs")),  # 6-bit indices (read like 8-bit on the ANE)
    "vector 4x64 + pcs": (1.5, ("vector", 4, 6, False, "pcs")),
    "vector 4x16 + pcs": (1, ("vector", 4, 4, False, "pcs")),
    "ternary + pcs": (2, ("tern",)),  # 1.58 bits of information, stored as 2-bit LUT indices on the ANE
    "INT8 per-channel": (8, ("int8",)),  # symmetric, one fp16 scale per output channel
}


def _nearest_grouped(x, c, budget=1 << 25):
    """Per-row nearest codebook entry for x (G, N) and codebooks c (G, K), in memory-bounded chunks."""
    g, n = x.shape
    rows = max(1, budget // (n * c.shape[1]))
    cols = n if rows > 1 else max(1, budget // c.shape[1])
    lab = torch.empty(g, n, dtype=torch.long, device=x.device)
    for r in range(0, g, rows):
        for j in range(0, n, cols):
            lab[r:r + rows, j:j + cols] = (x[r:r + rows, j:j + cols, None] - c[r:r + rows, None]).abs().argmin(-1)
    return lab


def kmeans_grouped_codebook(w, group, nb, cw, iters=40, fit_cols=1 << 16, seed=0):
    """Scalar LUT per group of `group` output channels (group = Cout: per-tensor). Weighted 1-D Lloyd,
    fitted on at most fit_cols values per group (all of them for groups up to 65536 values)."""
    cout, cin = w.shape
    x = w.reshape(cout // group, group * cin)
    wt = (torch.ones(cin) if cw is None else cw).repeat(group)
    if x.shape[1] > fit_cols:
        pick = torch.from_numpy(np.random.default_rng(seed).choice(x.shape[1], fit_cols, replace=False))
        x, wt = x[:, pick], wt[pick]
    wt = wt[None].expand_as(x)
    k = 1 << nb
    c = torch.quantile(x, torch.linspace(0.5 / k, 1 - 0.5 / k, k), dim=1).T.contiguous()
    for _ in range(iters):
        lab = _nearest_grouped(x, c)
        num = torch.zeros_like(c).scatter_add_(1, lab, x * wt)
        den = torch.zeros_like(c).scatter_add_(1, lab, wt)
        c = torch.where(den > 0, num / den.clamp_min(1e-30), c)
    return c.half().float()  # (groups, entries), fp16-exact


def kmeans_vector_codebook(w, cd, nb, cw, fp8=False, seed=0):
    """Per-tensor vector LUT: one index -> cd consecutive output channels at one input index."""
    cout, cin = w.shape
    v = w.reshape(cout // cd, cd, cin).permute(0, 2, 1).reshape(-1, cd).numpy()
    sw = None if cw is None else np.tile(cw.numpy(), cout // cd)
    pick = np.random.default_rng(seed).choice(len(v), min(300_000, len(v)), replace=False)
    from sklearn.cluster import KMeans  # lazy: only quantization needs it (the serving venvs may lack sklearn)
    km = KMeans(1 << nb, n_init=3, max_iter=200, random_state=seed)
    c = km.fit(v[pick], sample_weight=None if sw is None else sw[pick]).cluster_centers_.astype(np.float32)
    if fp8:  # FP8 E4M3 LUT values with one fp16 per-tensor scale
        s = np.float16(np.abs(c).max() / 240)
        c = (c / np.float32(s)).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * np.float32(s)
    return torch.from_numpy(c).half().float()  # (entries, cd), fp16-exact like the ANE's LUT


def make_rounder(w, spec, cw=None, device=None):
    """Rounder for format `spec` fitted on w (CPU); it runs on `device` (default: w's device)."""
    kind, device = spec[0], device or w.device
    if kind == "fp8":  # FP8 E4M3 per output channel; the ANE reads FP8 weight codes <= 240
        s = (w.abs().amax(1, keepdim=True) / 240).half().float()
        return lambda x: torch.from_numpy(  # CPU only (ml_dtypes)  # saturate: GPTQ updates can push codes past 240
            (x / s).clamp(-240, 240).numpy().astype(ml_dtypes.float8_e4m3fn).astype(np.float32)) * s  # noqa
    if kind == "int8":
        s = (w.abs().amax(1, keepdim=True) / 127).clamp_min(1e-12).half().float().to(device)
        out = lambda x: (x / s).round().clamp(-127, 127) * s  # noqa: E731
        out.codebook, out.cd, out.row_scale, out.int8 = None, 1, s, True
        return out
    if spec[-1] == "pcs":  # per-output-channel scale after the LUT (free on the ANE): fit on w / row RMS
        s = w.pow(2).mean(1, keepdim=True).sqrt().half().float()
        s[s == 0] = 1
        rnd = make_rounder(w / s, spec[:-1], cw, device)
        s = s.to(device)
        out = lambda x: rnd(x / s) * s  # noqa: E731
        out.codebook, out.cd, out.row_scale = rnd.codebook, rnd.cd, s
        return out
    if kind == "tern":  # ternary {-a, 0, +a} per output channel (TWN threshold): 2-bit LUT + per-channel scale
        m = w.abs().mean(1, keepdim=True)
        delta = 0.7 * m
        keep = w.abs() > delta
        a = ((w.abs() * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp_min(1)).half().float()
        delta, a = delta.to(device), a.to(device)
        return lambda x: torch.sign(x) * (x.abs() > delta) * a
    if kind == "group":
        _, group, nb = spec
        c = kmeans_grouped_codebook(w, group or w.shape[0], nb, cw).to(device)

        def rnd(x):
            xx = x.reshape(c.shape[0], -1)
            return torch.gather(c, 1, _nearest_grouped(xx, c)).reshape(x.shape)
        rnd.codebook, rnd.cd, rnd.row_scale = c, 1, None
        return rnd
    _, cd, nb, fp8 = spec
    c = kmeans_vector_codebook(w, cd, nb, cw, fp8).to(device)

    def rnd(x):
        cout, n = x.shape
        v = x.reshape(cout // cd, cd, n).permute(0, 2, 1).reshape(-1, cd)
        step = 1 << 18
        lab = torch.cat([torch.cdist(v[i:i + step], c).argmin(1) for i in range(0, len(v), step)])
        return c[lab].reshape(cout // cd, n, cd).permute(0, 2, 1).reshape(cout, n)
    rnd.codebook, rnd.cd, rnd.row_scale = c, cd, None
    return rnd


def encode(rnd, q):
    """Indices of a quantized matrix q (output of rnd / GPTQ with rnd) for a per-tensor (vector) LUT
    format: returns (lut (K, cd) fp16, idx (Cout / cd, Cin) uint8, per-output-channel scale fp16 or None)."""
    c, cd, s = rnd.codebook, rnd.cd, rnd.row_scale
    if getattr(rnd, "int8", False):  # (None, int8 codes, scale)
        return None, (q / s).round().clamp(-127, 127).to(torch.int8).cpu(), s.reshape(-1).half().cpu()
    if c.dim() == 2 and cd == 1 and c.shape[0] > 1:
        raise ValueError("per-group LUTs are not exported")
    qn = q / s if s is not None else q
    cout, cin = q.shape
    lut = c.reshape(-1, cd)
    v = qn.reshape(cout // cd, cd, cin).permute(0, 2, 1).reshape(-1, cd)
    step = 1 << 18
    idx = torch.cat([torch.cdist(v[i:i + step].float(), lut.float()).argmin(1) for i in range(0, len(v), step)])
    return (lut.half().cpu(), idx.reshape(cout // cd, cin).to(torch.uint8).cpu(),
            None if s is None else s.reshape(-1).half().cpu())


def _chol_lower(a, block=128):
    """Blocked Cholesky in plain torch. On macOS 27 (26A428) with torch 2.8, torch.linalg.cholesky
    (upper=True) returned wrong factors and its Accelerate DPOTRF call raised SIGILL intermittently."""
    a, n = a.clone(), len(a)
    for j in range(0, n, block):
        k = min(j + block, n)
        for i in range(j, k):
            col = a[i:, i] - a[i:, j:i] @ a[i, j:i]
            a[i, i] = col[0].sqrt()
            a[i + 1:, i] = col[1:] / a[i, i]
        a[k:, k:] -= a[k:, j:k] @ a[k:, j:k].T
    return a.tril()


def _inv_upper(u):
    n = len(u)
    if n <= 32:
        inv = torch.zeros_like(u)
        for i in reversed(range(n)):
            inv[i, i] = 1 / u[i, i]
            inv[i, i + 1:] = -(u[i, i + 1:] @ inv[i + 1:, i + 1:]) / u[i, i]
        return inv
    m = n // 2
    a_inv, d_inv = _inv_upper(u[:m, :m]), _inv_upper(u[m:, m:])
    out = torch.zeros_like(u)
    out[:m, :m], out[m:, m:], out[:m, m:] = a_inv, d_inv, -a_inv @ u[:m, m:] @ d_inv
    return out


def chol_inv_upper(h):
    """Upper U with H^-1 = U^T U (GPTQ's Hinv): H = V V^T with V upper, then U = V^-1."""
    return _inv_upper(_chol_lower(h.flip(0, 1)).flip(0, 1))


def gptq(w, h, rnd, block=128, damp=0.01):
    """GPTQ with activation order: minimize ||(W - Q) X||^2 given H = X^T X / N, column by column.
    The Hessian factorization runs on the CPU in float64; the column loop runs on w's device."""
    dev = w.device
    h = h.cpu().double().clone()
    dead = h.diag() == 0
    h[dead, dead] = 1
    perm = h.diag().argsort(descending=True)
    w, h = w[:, perm.to(dev)].clone(), h[perm][:, perm]
    h += damp * h.diag().mean() * torch.eye(len(h), dtype=h.dtype)
    hinv = chol_inv_upper(h).float().to(dev)
    q = torch.zeros_like(w)
    for i1 in range(0, w.shape[1], block):
        i2 = min(i1 + block, w.shape[1])
        w1, err1, hinv1 = w[:, i1:i2].clone(), torch.zeros(w.shape[0], i2 - i1, device=dev), hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            qc = rnd(w1[:, i:i + 1])[:, 0]
            q[:, i1 + i] = qc
            err = (w1[:, i] - qc) / hinv1[i, i]
            w1[:, i:] -= err[:, None] @ hinv1[i, i:][None]
            err1[:, i] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q[:, perm.argsort().to(dev)]


def quantize_mlp(mb, u, spec, method, aw=False):
    """Quantized gate/up/down of MLPBasis `mb`, calibrated on normalized inputs u. `spec` is one format
    spec for all three, or {"gate"/"up"/"down": spec or None}; None keeps that matrix in float.
    GPTQ is true-sequential: down's Hessian uses the quantized gate/up outputs."""
    specs = spec if isinstance(spec, dict) else dict.fromkeys(("gate", "up", "down"), spec)

    def quant(m, h):
        if specs.get(m) is None:
            return mb.w[m]
        rnd = make_rounder(mb.w[m], specs[m], h.diag() if aw else None)
        return gptq(mb.w[m], h, rnd) if method == "gptq" else rnd(mb.w[m])

    z = mb.gate_up_input(u)
    hz = z.T @ z / len(z)
    q = {m: quant(m, hz) for m in ("gate", "up")}
    a = mb.down_input(z, q if method == "gptq" else mb.w)
    q["down"] = quant("down", a.T @ a / len(a))
    return q


def snr(ref, q):
    return 10 * np.log10(ref.pow(2).sum().item() / (ref - q).pow(2).sum().clamp_min(1e-30).item())
