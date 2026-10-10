"""Fixed-shape fp16 ANE-layout port of the Unsloth Clef JointSchemaHead (27.3M params), T prompt rows x K options.

Layout (1, C, 1, T) with 1x1 convs for every Linear. Dynamic parts are host-built inputs:
  x     (1,1024,1,T)  post-final-RMSNorm hidden rows (rows >= n are zero)
  pool  (T, 2+K)      columns: question mean (1/len baked in), last-token one-hot, K option mean pools
  lex   (1,1024,1,K)  mean tied-embedding of each option span (host gather) ; lexu = lex / |lex|
  mmask (1,1,1,T)     additive memory-validity mask (0 / -1e4) ; omask (1,1,1,K) additive option mask
Attention is explicit broadcast-multiply + reduce (no matmul between computed tensors); LayerNorm uses the max-scaled
RMS form of qwen38_coreai_build.rms_hidden (centred first). GELU: exact erf (GELU_MODE=erf) or tanh form (=tanh).
Output logits (1,1,1,K) = the host ClefHead logits (before / t_choice)."""
import math, os
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

K_OPT = 8
NH = 8


def ln_c(x, w, b, eps=1e-5):
    xc = x - x.mean(1, keepdim=True)
    m = xc.abs().amax(1, keepdim=True).clamp_min(1e-3)
    xs = xc / m
    return xs * torch.rsqrt((xs * xs).mean(1, keepdim=True) + eps / (m * m)) * w + b


def l2n(x, dim=1):
    m = x.abs().amax(dim, keepdim=True).clamp_min(1e-4)
    xs = x / m
    return xs * torch.rsqrt((xs * xs).sum(dim, keepdim=True) + 1e-12)


def gelu(x):
    if os.environ.get("GELU_MODE", "erf") == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


class AneClefHead(nn.Module):
    def __init__(self, sd: dict, cfg: dict, T: int = 304, K: int = K_OPT):
        super().__init__()
        self.T, self.K = T, K
        w = cfg["width"]; self.w = w; self.nh = cfg["heads"]; self.hd = w // self.nh
        self.nev, self.nl = cfg["routing_layers"], cfg["layers"]
        sd = {k: v.float() for k, v in sd.items()}
        self.prior_scale = math.exp(min(float(sd["prior_logit_scale"]), math.log(100.0)))
        self.joint_scale = math.exp(min(float(sd["joint_logit_scale"]), math.log(100.0)))
        self.gate = 1.0 / (1.0 + math.exp(-float(sd["residual_gate"])))
        qs = 1.0 / math.sqrt(self.hd)

        def conv(wt, bias=None, scale=1.0):
            c = nn.Conv2d(wt.shape[1], wt.shape[0], 1, bias=bias is not None)
            c.weight = nn.Parameter((wt * scale).reshape(wt.shape[0], wt.shape[1], 1, 1).clone(), requires_grad=False)
            if bias is not None:
                c.bias = nn.Parameter((bias * scale).clone(), requires_grad=False)
            return c

        def ln(p):
            self.register_buffer(p.replace(".", "_") + "_w", sd[p + ".weight"].reshape(1, -1, 1, 1).clone())
            self.register_buffer(p.replace(".", "_") + "_b", sd[p + ".bias"].reshape(1, -1, 1, 1).clone())
            return p.replace(".", "_")

        self.names = {}
        for n in ("hidden_norm", "option_norm", "option_summary_norm", "field_norm"):
            self.names[n] = ln(n)
        self.memory_projection = conv(sd["memory_projection.weight"])
        self.ocp = conv(sd["option_context_projection.weight"])
        self.olp = conv(sd["option_lexical_projection.weight"])
        self.oqp = conv(sd["option_question_projection.weight"])
        self.qp = conv(sd["question_projection.weight"])
        self.gp = conv(sd["global_projection.weight"])
        self.register_buffer("type1", sd["type_embedding.weight"][1].reshape(1, -1, 1, 1).clone())

        def attn(prefix):
            W, B = sd[prefix + ".in_proj_weight"], sd[prefix + ".in_proj_bias"]
            return nn.ModuleDict({
                "q": conv(W[:w], B[:w], qs), "k": conv(W[w:2 * w], B[w:2 * w]), "v": conv(W[2 * w:], B[2 * w:]),
                "o": conv(sd[prefix + ".out_proj.weight"], sd[prefix + ".out_proj.bias"])})

        self.ev = nn.ModuleList(); self.ev_ln = []
        for i in range(self.nev):
            p = f"evidence_layers.{i}"
            self.ev.append(nn.ModuleDict({"att": attn(p + ".attention"),
                                          "f1": conv(sd[p + ".feedforward.0.weight"], sd[p + ".feedforward.0.bias"]),
                                          "f2": conv(sd[p + ".feedforward.3.weight"], sd[p + ".feedforward.3.bias"])}))
            self.ev_ln.append([ln(p + ".query_norm"), ln(p + ".memory_norm"), ln(p + ".feedforward_norm")])
        self.dl = nn.ModuleList(); self.dl_ln = []
        for i in range(self.nl):
            p = f"layers.{i}"
            sa = attn(p + ".self_attn")  # one query token: softmax over a single key = 1, so sa = out(v(x))
            self.dl.append(nn.ModuleDict({"sa_v": sa["v"], "sa_o": sa["o"], "ca": attn(p + ".multihead_attn"),
                                          "f1": conv(sd[p + ".linear1.weight"], sd[p + ".linear1.bias"]),
                                          "f2": conv(sd[p + ".linear2.weight"], sd[p + ".linear2.bias"])}))
            self.dl_ln.append([ln(p + ".norm1"), ln(p + ".norm2"), ln(p + ".norm3")])
        self.rs1 = conv(sd["residual_scorer.0.weight"], sd["residual_scorer.0.bias"])
        self.rs2 = conv(sd["residual_scorer.3.weight"], sd["residual_scorer.3.bias"])
        self.eval()

    def LN(self, x, name):
        return ln_c(x, getattr(self, name + "_w"), getattr(self, name + "_b"))

    def mha(self, a, xq, mem, mmask):
        Kq, T = xq.shape[-1], mem.shape[-1]
        q = a["q"](xq).reshape(self.nh, self.hd, Kq, 1)
        k = a["k"](mem).reshape(self.nh, self.hd, 1, T)
        v = a["v"](mem).reshape(self.nh, self.hd, 1, T)
        s = (q * k).sum(1) + mmask.reshape(1, 1, T)             # (nh, Kq, T)
        s = s - s.amax(-1, keepdim=True)
        e = torch.exp(s)
        p = e / e.sum(-1, keepdim=True)
        # Keep the reduction rank-4 (nh, hd, Kq, 1) and run the output conv in (1, C, Kq, 1) layout, then permute back:
        # reshape((nh, hd, Kq) -> (1, C, 1, Kq)) straight into the conv compiles on the ANE but returns WRONG values.
        o = (p.reshape(self.nh, 1, Kq, T) * v).sum(-1, keepdim=True)
        return a["o"](o.reshape(1, self.w, Kq, 1)).permute(0, 1, 3, 2)

    def forward(self, x, pool, lex, lexu, mmask, omask):
        T, K = self.T, self.K
        H = self.LN(x, self.names["hidden_norm"])                                  # (1,1024,1,T)
        M = self.memory_projection(H)                                              # (1,512,1,T)
        P = (H.reshape(H.shape[1], T) @ pool).reshape(1, H.shape[1], 1, K + 2)     # q | g | options
        q, g, c = P[..., 0:1], P[..., 1:2], P[..., 2:]
        r = self.ocp(c) + self.olp(lex) + self.oqp(q)                              # (1,512,1,K)
        for ev, (nq, nm, nf) in zip(self.ev, self.ev_ln):
            r = r + self.mha(ev["att"], self.LN(r, nq), self.LN(M, nm), mmask)
            r = r + ev["f2"](gelu(ev["f1"](self.LN(r, nf))))
        f0 = self.qp(q)
        wl = (r * f0).sum(1, keepdim=True) * (1.0 / math.sqrt(self.w)) + omask     # (1,1,1,K)
        wl = wl - wl.amax(-1, keepdim=True)
        wgt = torch.exp(wl); wgt = wgt / wgt.sum(-1, keepdim=True)
        s = (wgt * r).sum(-1, keepdim=True)                                        # (1,512,1,1)
        field = f0 + self.LN(s, self.names["option_summary_norm"]) + self.gp(g) + self.type1
        for lay, (n1, n2, n3) in zip(self.dl, self.dl_ln):
            field = field + lay["sa_o"](lay["sa_v"](self.LN(field, n1)))
            field = field + self.mha(lay["ca"], self.LN(field, n2), M, mmask)
            field = field + lay["f2"](gelu(lay["f1"](self.LN(field, n3))))
        field = self.LN(field, self.names["field_norm"])
        prior = self.prior_scale * (lexu * l2n(q + g)).sum(1, keepdim=True)        # (1,1,1,K)
        n = self.LN(r, self.names["option_norm"])
        cos = (l2n(field) * l2n(n)).sum(1, keepdim=True)
        fe = field.expand(-1, -1, -1, K)
        res = self.rs2(gelu(self.rs1(torch.cat([fe, n, fe * n, (fe - n).abs()], 1))))
        return (prior + self.gate * (self.joint_scale * cos + res)).reshape(1, K)

    def example(self):
        T, K = self.T, self.K
        z = lambda *s: torch.zeros(*s, dtype=torch.float16)  # noqa: E731
        return (z(1, 1024, 1, T), z(T, K + 2), z(1, 1024, 1, K), z(1, 1024, 1, K), z(1, 1, 1, T), z(1, 1, 1, K))


INPUT_NAMES = ["x", "pool", "lex", "lexu", "mmask", "omask"]


def build_inputs(Hrows, ids, embed, qspan, ospans, T=304, K=K_OPT):
    """Host-side dynamic inputs (numpy fp32). Hrows (n,1024) post-norm; embed (vocab,1024) float32 array/tensor."""
    n = Hrows.shape[0]
    assert n <= T and len(ospans) <= K
    x = np.zeros((1, 1024, 1, T), np.float32); x[0, :, 0, :n] = Hrows.T
    pool = np.zeros((T, K + 2), np.float32)
    pool[qspan[0]:qspan[1], 0] = 1.0 / (qspan[1] - qspan[0])
    pool[n - 1, 1] = 1.0
    lex = np.zeros((1, 1024, 1, K), np.float32); lexu = np.zeros_like(lex)
    ids = np.asarray(ids)
    for i, (a, b) in enumerate(ospans):
        pool[a:b, 2 + i] = 1.0 / (b - a)
        e = np.asarray(embed[ids[a:b]], np.float32).mean(0)
        lex[0, :, 0, i] = e; lexu[0, :, 0, i] = e / max(float(np.linalg.norm(e)), 1e-8)
    mmask = np.full((1, 1, 1, T), -1e4, np.float32); mmask[..., :n] = 0
    omask = np.full((1, 1, 1, K), -1e4, np.float32); omask[..., :len(ospans)] = 0
    return {"x": x, "pool": pool, "lex": lex, "lexu": lexu, "mmask": mmask, "omask": omask}


def example_inputs(T: int = 304, K: int = K_OPT):
    return AneClefHead.example(type("S", (), {"T": T, "K": K})())


def export(clef_dir, out, T: int = 304):
    """Write the Core AI package of the head (same path as the repo's Core AI build helper save_dense_program). Needs the Core AI SDK python."""
    import json, shutil
    from pathlib import Path
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    clef_dir, out = Path(clef_dir), Path(out)
    f16 = clef_dir / "joint_head_fp16.safetensors"
    sd = load_file(str(f16 if f16.is_file() else clef_dir / "joint_head.safetensors"))
    cfg = json.loads((clef_dir / "joint_head_config.json").read_text())
    m = AneClefHead(sd, cfg, T=T).to(torch.float16)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    ep = torch.export.export(m, m.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv.add_exported_program(ep, input_names=INPUT_NAMES, output_names=["logits"], entrypoint_name="clef")
    prog = conv.to_coreai(); prog.optimize()
    out.parent.mkdir(parents=True, exist_ok=True); shutil.rmtree(out, ignore_errors=True); prog.save_asset(out)
    (out.parent / "ane_head_config.json").write_text(json.dumps({"file": out.name, "entry": "clef", "T": T, "K": K_OPT, "gelu": "erf"}))
    return out


class AneHeadRunner:
    """Runs the exported head package on the ANE (CLEF_HEAD_BACKEND=ane). Host inputs: build_inputs(); call -> (n_options,) fp32 logits."""

    def __init__(self, clef_dir, loop):
        import json
        from pathlib import Path
        from coreai.runtime import AIModel
        from qwen38_coreai_model import _spec, buffer
        self.cfg = json.loads((Path(clef_dir) / "ane_head_config.json").read_text())
        self.T, self.K, self.loop = int(self.cfg["T"]), int(self.cfg["K"]), loop
        self.model = loop.run_until_complete(AIModel.load(Path(clef_dir) / self.cfg["file"], specialization_options=_spec()))
        self.fn = self.model.load_function(self.cfg["entry"])
        shapes = {"x": (1, 1024, 1, self.T), "pool": (self.T, self.K + 2), "lex": (1, 1024, 1, self.K),
                  "lexu": (1, 1024, 1, self.K), "mmask": (1, 1, 1, self.T), "omask": (1, 1, 1, self.K)}
        self.bufs = {k: buffer(s) for k, s in shapes.items()}

    def fits(self, n_rows: int, n_options: int) -> bool:
        return n_rows <= self.T and n_options <= self.K

    def __call__(self, hidden, ids, embed, qspan, ospans):
        from jeff_coreai_runtime import outputs
        ins = build_inputs(np.asarray(hidden, np.float32), ids, embed, qspan, ospans, self.T, self.K)
        for k, v in ins.items():
            self.bufs[k][1][:] = v

        async def run():
            async with outputs(self.fn, {k: b[0] for k, b in self.bufs.items()}) as out:
                return np.array(out["logits"].numpy(), np.float32).reshape(-1)

        return self.loop.run_until_complete(run())[:len(ospans)]


if __name__ == "__main__":  # python clef_ane_head.py <clef_head dir> [T]  -> <clef_head dir>/clef_head_ane.aimodel
    import sys
    from pathlib import Path
    d = Path(sys.argv[1]); print("exported", export(d, d / "clef_head_ane.aimodel", int(sys.argv[2]) if len(sys.argv) > 2 else 304))
