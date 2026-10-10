"""Qwen3.5 hybrid checkpoint helpers and Core ML chunk builder.

Origin: Anemll/anemll-forge. Jeff convert imports this module to rebind ``cfg`` and to
reuse ``Checkpoint`` / layer layout. Core ML / coremltools imports are optional so the
module loads on Linux; conversion still needs them at runtime.

Qwen3.8-27B on the ANE end to end: build decode chunks (CHUNK layers each, default 4 = 3 Gated DeltaNet +
1 attention) and a head model (final RMSNorm + lm_head) from the quantized export of qwen38_gptq_27b.py, then
generate token by token with Core ML states (embedding lookup on the CPU).

    EXPORT_DIR=~/Models/vq27b/export/<tag> python qwen38_ane_model.py build
    python qwen38_ane_model.py generate "The capital of France is"
v2 builds (build_v2): each chunk is a multifunction model, "infer" (1 token) and "prefill0" (PREFILL_T tokens) over
the chunk's layers, sharing the (deduplicated) weights and the KV-cache MLState (same state set in both functions); Gated DeltaNet conv / recurrent states are host-owned inputs / outputs (ct.models.SharedArray,
swapped between calls, no copies); DFlash drafter taps are extra outputs.
    CTX=8192 python qwen38_ane_model.py build_v2
v3 builds (build_v3): ONE T=8 function per chunk for decode (1 row), prefill (8-token blocks) and DFlash verify
(8 rows): DeltaNet rows are committed one call late (lazy commit; see qwen38_ane_chunk.LAZY), so any number of a
block's rows can be accepted; KV caches in MLState; one ANE program per chunk (a second function per chunk doubles
the ANE-resident weights and thrashes a 32 GB M6). Head: final norm + lm_head on 8 rows.
    CTX=8192 python qwen38_ane_model.py build_v3      # -> *_v4 (read-only KV inputs; V3_KV_IN=0: *_v3, KV MLState)
Tensors missing from the export (unquantized projections) are taken from the bf16 checkpoint (MODEL).
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
except ImportError:  # Linux import check; jeff-convert needs coremltools on the Core AI SDK
    ct = mb = types = None
import qwen38_ane_chunk as C

MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))
EXPORT_DIR = Path(os.path.expanduser(os.environ.get("EXPORT_DIR", "~/Models/vq27b/export/full_mix25_mixer4_head4")))
OUT = Path(os.path.expanduser(os.environ.get("ANE_OUT", "~/Models/vq27b/ane"))) / EXPORT_DIR.name
CHUNK = int(os.environ.get("CHUNK", "4"))          # layers per group (one 3 DeltaNet + 1 attention repeat)
MAX_GB = float(os.environ.get("MAX_GB", "1.9"))     # pack consecutive groups into chunks up to this size
CTX = int(os.environ.get("CTX", "2048"))
ONLY = [int(c) for c in os.environ["ONLY"].split(",")] if os.environ.get("ONLY") else None  # build only these chunks
V3_T = 8                                            # v3: rows per call (= DFlash verify block)
V3_KV_IN = os.environ.get("V3_KV_IN", "1") == "1"    # v3 -> "v4": KV caches as read-only inputs (host commits rows)
V3_PREFILL = int(os.environ.get("V3_PREFILL", "0"))  # >0: second function "prefill" with this many rows (-> "v5")
V3_TAG = "v5" if V3_PREFILL else ("v4" if V3_KV_IN else "v3")
PREFILL_T = int(os.environ.get("PREFILL_T", "16"))  # v2: tokens per prefill call (T=32 fails to compile at 12 layers)
TAPS = [5, 19, 33, 47, 61]                          # v2: hidden states after these layers (DFlash2 drafter features)
STEP_MAX = 3                                        # v2: prompt remainders up to this many tokens use decode steps
PREFILL_SPLIT = int(os.environ.get("PREFILL_SPLIT", "64"))  # v2: layers per prefill function (default: the whole chunk)
KV_IO = os.environ.get("KV_IO", "0") == "1"                # v2: KV caches as I/O (slower decode) instead of MLState
MIXERS = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
          "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
torch.set_grad_enabled(False)


def cfg():
    return json.loads((MODEL / "config.json").read_text())["text_config"]


class Checkpoint:
    def __init__(self):
        self._wmap = None

    @property
    def wmap(self):
        # Prebuilt inference only needs config/tokenizer and the prepared embedding table.
        # Original shards are required when conversion asks for checkpoint tensors.
        if self._wmap is None:
            self._wmap = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
        return self._wmap

    def embed_table(self):
        """The (vocab, hidden) fp16 embedding table, memory-mapped: written once as a raw .npy (EMBED_NPY, default
        ~/Models/vq27b/embed_tokens_fp16.npy) and then np.load(mmap_mode="r"). Only the rows a conversation uses get
        paged in, and the pages are clean and file-backed (dropped under memory pressure instead of swapped), where a
        loaded table is 2.5 GB of anonymous memory."""
        path = Path(os.path.expanduser(os.environ.get("EMBED_NPY", "~/Models/vq27b/embed_tokens_fp16.npy")))
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, self.get("model.language_model.embed_tokens.weight").to(torch.float16).numpy())
            tmp.rename(path)
        table = np.load(path, mmap_mode="r", allow_pickle=False)
        c = cfg()
        expected = (c["vocab_size"], c["hidden_size"])
        if table.shape != expected or table.dtype != np.float16 or not table.flags.c_contiguous:
            raise ValueError(f"Embedding table must be C-contiguous float16 {expected}; got {table.dtype} {table.shape}: {path}")
        return table

    def get(self, name):
        with safe_open(MODEL / self.wmap[name], framework="pt") as f:
            return f.get_tensor(name)

    def layer(self, i):
        pre = f"model.language_model.layers.{i}."
        return {k[len(pre):]: self.get(k).float() for k in self.wmap if k.startswith(pre)}


def as_quant(t, key, rot_seed=None):
    """Export tensors of one matrix -> the chunk builder's weight tuple. Trained low-rank factors ({key}.lr_a / lr_b,
    qwen38_blockrecon.py, original basis) ride along as the last element; with rot_seed (online MLP basis: the
    builder feeds the conv M^T x) b is moved to that basis (b @ M, M = qwen38_kl.rotation(n, seed))."""
    lr = None
    if f"{key}.lr_a" in t:
        a, b = t[f"{key}.lr_a"].float().numpy(), t[f"{key}.lr_b"].float().numpy()
        if rot_seed is not None:
            from scipy.linalg import hadamard
            n, blk = b.shape[1], 1024
            h = hadamard(blk) / np.sqrt(blk)
            sgn = np.random.default_rng(rot_seed).choice([-1.0, 1.0], n)
            b = ((b * sgn).reshape(b.shape[0], n // blk, blk) @ h).reshape(b.shape)
        lr = (a, b)
    if f"{key}.int8" in t:
        return ("int8", t[f"{key}.int8"].numpy(), t[f"{key}.scale"].numpy(), lr)
    if f"{key}.lut" in t:
        return (t[f"{key}.lut"].numpy(), t[f"{key}.idx"].numpy(),
                t[f"{key}.scale"].numpy() if f"{key}.scale" in t else None, None, lr)
    if f"{key}.weight" in t:
        return ("dense", t[f"{key}.weight"].float().numpy())
    return None


def layer_quant(ck, i, w):
    q = {}
    path = EXPORT_DIR / f"layer_{i:02d}.safetensors"
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    t = load_file(path)
    online = meta.get("basis") == "online"
    for m in ("gate", "up", "down"):
        seed = (int(meta["seed_in"]) if m != "down" else int(meta["seed_mid"])) if online else None
        q[f"mlp.{m}_proj.weight"] = as_quant(t, m, seed)
    if meta.get("basis") == "online":
        q["mlp.rotation"] = (int(meta["seed_in"]), int(meta["seed_mid"]))
    mixer = EXPORT_DIR / f"layer_{i:02d}_mixer.safetensors"
    tm = load_file(mixer) if mixer.exists() else {}
    for name in MIXERS:  # the export's quantized matrix; the checkpoint's dense one only where the export has none
        quant = as_quant(tm, name)
        if quant is not None:
            q[f"{name}.weight"] = quant
        elif f"{name}.weight" in w:
            q[f"{name}.weight"] = ("dense", w[f"{name}.weight"].numpy())
    return q


def packed_bytes(i):
    """Bytes of layer i's weights on the ANE (packed indices + LUTs + scales, int8 codes, fp16 rest)."""
    total = 0
    for f in (EXPORT_DIR / f"layer_{i:02d}.safetensors", EXPORT_DIR / f"layer_{i:02d}_mixer.safetensors"):
        if not f.exists():
            continue
        t = load_file(f)
        for k, v in t.items():
            if k.endswith(".idx"):
                entries = t[k[:-4] + ".lut"].shape[0]
                total += v.numel() * max(1, int(np.ceil(np.log2(entries)))) / 8
            elif k.endswith((".int8",)):
                total += v.numel()
            else:
                total += v.numel() * 2
    return total + 30e6  # norms, conv taps, in_proj_a / b and small tensors (fp16)


def plan_chunks(c):
    """Consecutive groups of CHUNK layers packed into chunks of at most MAX_GB (the estimate is conservative), or
    the explicit CHUNK_PLAN, e.g. "0-11,12-23,24-35,36-47,48-59,60-63"."""
    if os.environ.get("CHUNK_PLAN"):
        return [list(range(int(a), int(b) + 1)) for a, b in (r.split("-") for r in os.environ["CHUNK_PLAN"].split(","))]
    groups = [list(range(g, g + CHUNK)) for g in range(0, c["num_hidden_layers"], CHUNK)]
    chunks, cur, size = [], [], 0.0
    for g in groups:
        gsize = sum(packed_bytes(i) for i in g) / 1e9
        if cur and size + gsize > MAX_GB:
            chunks.append(cur)
            cur, size = [], 0.0
        cur, size = cur + g, size + gsize
    chunks.append(cur)
    return chunks


def build():
    c, ck = cfg(), Checkpoint()
    OUT.mkdir(parents=True, exist_ok=True)
    C.CTX, C.REPEAT, C.OUT = CTX, 1, OUT / "tmp"
    chunks = plan_chunks(c)
    manifest = {"ctx": CTX, "ring_rows": 8,
                "chunks": [f"chunk_L{l[0]:02d}-{l[-1]:02d}_ctx{CTX}.mlmodelc" for l in chunks]}
    (OUT / f"manifest_ctx{CTX}.json").write_text(json.dumps(manifest, indent=1))
    print("chunk plan:", [f"{l[0]}-{l[-1]} ({sum(packed_bytes(i) for i in l) / 1e9:.2f} GB)" for l in chunks], flush=True)
    for k, layers in enumerate(chunks):
        dst = OUT / manifest["chunks"][k]
        if (ONLY is not None and k not in ONLY) or dst.exists():
            continue
        t = time.time()
        weights = {i: ck.layer(i) for i in layers}
        quant = {i: layer_quant(ck, i, weights[i]) for i in layers}
        C.LAYERS = layers
        mlc = C.build(c, weights, quant)
        shutil.move(str(mlc), str(dst))
        shutil.rmtree(C.OUT, ignore_errors=True)
        print(f"chunk {k:02d} layers {layers[0]}-{layers[-1]} built ({time.time() - t:.0f}s)", flush=True)
    dst = OUT / "head.mlmodelc"
    if not dst.exists() and (ONLY is None or -1 in ONLY):
        build_head(c, ck, dst)


def build_v2():
    """Multifunction chunks (infer + prefill) with DeltaNet states as I/O and drafter taps; see the module doc."""
    c, ck = cfg(), Checkpoint()
    OUT.mkdir(parents=True, exist_ok=True)
    C.CTX, C.REPEAT, C.OUT = CTX, 1, OUT / "tmp"
    C.GDN_IO, C.KV_IO, C.CONV_OUT_ALL, C.TAPS = True, KV_IO, False, TAPS
    # With the KV caches in MLState, every function must declare the chunk's full state set: sharing one MLState
    # between functions that declare different subsets led to ANE resets and a watchdog panic.
    assert KV_IO or PREFILL_SPLIT >= max(len(l) for l in plan_chunks(c)), "PREFILL_SPLIT < chunk size needs KV_IO=1"
    chunks = plan_chunks(c)
    files = [f"chunk_L{l[0]:02d}-{l[-1]:02d}_ctx{CTX}_v2.mlmodelc" for l in chunks]
    manifest = {"version": 2, "kv_io": KV_IO, "ctx": CTX, "prefill_t": PREFILL_T, "gdn_chunk": C.GDN_CHUNK, "taps": TAPS,
                "chunks": [{"file": f, "layers": [l[0], l[-1]],
                            "prefill": [[a, min(a + PREFILL_SPLIT, len(l)) - 1] for a in range(0, len(l), PREFILL_SPLIT)]}
                           for f, l in zip(files, chunks)], "head": "head.mlmodelc"}
    (OUT / f"manifest_ctx{CTX}_v2.json").write_text(json.dumps(manifest, indent=1))
    print("chunk plan:", [f"{l[0]}-{l[-1]} ({sum(packed_bytes(i) for i in l) / 1e9:.2f} GB)" for l in chunks], flush=True)
    for k, layers in enumerate(chunks):
        dst = OUT / files[k]
        if (ONLY is not None and k not in ONLY) or dst.exists():
            continue
        t = time.time()
        weights = {i: ck.layer(i) for i in layers}
        quant = {i: layer_quant(ck, i, weights[i]) for i in layers}
        desc = ct.utils.MultiFunctionDescriptor()
        C.LAYERS, C.JOFF = layers, 0
        desc.add_function(str(C.build(c, weights, quant, T=1, compile_model=False)), "main", "infer")
        for i, (a, b) in enumerate(manifest["chunks"][k]["prefill"]):  # j-ranges within the chunk
            C.LAYERS, C.JOFF = layers[a:b + 1], a
            desc.add_function(str(C.build(c, weights, quant, T=PREFILL_T, compile_model=False)), "main", f"prefill{i}")
        C.JOFF = 0
        del weights, quant
        desc.default_function_name = "infer"
        mf = dst.with_suffix(".mlpackage")
        shutil.rmtree(mf, ignore_errors=True)
        ct.utils.save_multifunction(desc, str(mf))
        ct.models.utils.compile_model(str(mf), str(dst))
        shutil.rmtree(mf, ignore_errors=True)
        shutil.rmtree(C.OUT, ignore_errors=True)
        print(f"chunk {k:02d} layers {layers[0]}-{layers[-1]} built ({time.time() - t:.0f}s)", flush=True)
    dst = OUT / "head.mlmodelc"
    if not dst.exists() and (ONLY is None or -1 in ONLY):
        build_head(c, ck, dst)


def build_v3():
    """One T=V3_T lazy-commit function per chunk (+ drafter taps) and an 8-row head; see the module doc."""
    c, ck = cfg(), Checkpoint()
    OUT.mkdir(parents=True, exist_ok=True)
    C.CTX, C.REPEAT, C.OUT = CTX, 1, OUT / "tmp"
    C.GDN_IO, C.KV_IO, C.KV_IN, C.LAZY, C.CONV_OUT_ALL, C.TAPS, C.JOFF = True, False, V3_KV_IN, True, False, TAPS, 0
    chunks = plan_chunks(c)
    files = [f"chunk_L{l[0]:02d}-{l[-1]:02d}_ctx{CTX}_{V3_TAG}.mlmodelc" for l in chunks]
    manifest = {"version": 5 if V3_PREFILL else (4 if V3_KV_IN else 3), "kv_in": V3_KV_IN, "ctx": CTX, "T": V3_T,
                "prefill_t": V3_PREFILL, "pend": C.PEND, "taps": TAPS,
                "chunks": [{"file": f, "layers": [l[0], l[-1]]} for f, l in zip(files, chunks)],
                "head": f"head_T{V3_T}.mlmodelc"}
    (OUT / f"manifest_ctx{CTX}_{V3_TAG}.json").write_text(json.dumps(manifest, indent=1))
    for k, layers in enumerate(chunks):
        dst = OUT / files[k]
        if (ONLY is not None and k not in ONLY) or dst.exists():
            continue
        t = time.time()
        weights = {i: ck.layer(i) for i in layers}
        quant = {i: layer_quant(ck, i, weights[i]) for i in layers}
        C.LAYERS = layers
        if V3_PREFILL:  # multifunction: "verify" (T=8, default) + "prefill" (T=V3_PREFILL), shared weights
            desc = ct.utils.MultiFunctionDescriptor()
            desc.add_function(str(C.build(c, weights, quant, T=V3_T, compile_model=False)), "main", "verify")
            desc.add_function(str(C.build(c, weights, quant, T=V3_PREFILL, compile_model=False)), "main", "prefill")
            desc.default_function_name = "verify"
            mf = dst.with_suffix(".mlpackage")
            shutil.rmtree(mf, ignore_errors=True)
            ct.utils.save_multifunction(desc, str(mf))
            ct.models.utils.compile_model(str(mf), str(dst))
            shutil.rmtree(mf, ignore_errors=True)
        else:
            mlc = C.build(c, weights, quant, T=V3_T, compile_model=True)
            shutil.move(str(mlc), str(dst))
            dbg = mlc.parent / f"{mlc.stem}.dbg.json"  # DBG_MIXER_IN=1: mixer-input output names
            if dbg.exists():
                shutil.move(str(dbg), str(dst.with_suffix(".dbg.json")))
        shutil.rmtree(C.OUT, ignore_errors=True)
        print(f"chunk {k:02d} layers {layers[0]}-{layers[-1]} built ({time.time() - t:.0f}s)", flush=True)
    dst = OUT / manifest["head"]
    if not dst.exists() and (ONLY is None or -1 in ONLY):
        build_head(c, ck, dst, T=V3_T)


def build_head(c, ck, dst, T=1):
    """Final RMSNorm + lm_head. Output logits (1, vocab) fp16, or (T, vocab) for T rows."""
    t = load_file(EXPORT_DIR / "lm_head.safetensors") if (EXPORT_DIR / "lm_head.safetensors").exists() else {}
    q = as_quant(t, "lm_head") or ("dense", ck.get("lm_head.weight").float().numpy())
    norm = (1 + ck.get("model.language_model.norm.weight").float()).numpy().astype(np.float16)
    v, parts = c["vocab_size"], int(os.environ.get("HEAD_PARTS", "8"))
    step = -(-v // parts)

    def part(qq, a, b):  # rows [a, b) of the weight
        if isinstance(qq[0], str):  # ("dense", w) or ("int8", codes, scale)
            return (qq[0], qq[1][a:b]) + ((qq[2][a:b],) if len(qq) > 2 else ())
        lut, idx, sc = qq[:3]  # as_quant returns (lut, idx, scale, deq, lowrank); the head has no low-rank part
        cd = lut.shape[1]
        return (lut, idx[a // cd:b // cd], None if sc is None else sc[a:b], None)

    @mb.program(input_specs=[mb.TensorSpec((1, c["hidden_size"], 1, T), types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        h = C.rms_hidden(x, norm, c["rms_norm_eps"])
        if T == 1:
            outs = [mb.reshape(x=C.lut_linear(h, part(q, a, min(a + step, v))), shape=(1, -1)) for a in range(0, v, step)]
            return mb.concat(values=outs, axis=1, name="logits")
        outs = [mb.transpose(x=mb.reshape(x=C.lut_linear(h, part(q, a, min(a + step, v))), shape=(-1, T)), perm=[1, 0])
                for a in range(0, v, step)]                                                     # (T, part) each
        return mb.concat(values=outs, axis=1, name="logits")

    pkg = dst.with_suffix(".mlpackage")
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True, pass_pipeline=pipeline).save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(dst))
    shutil.rmtree(pkg, ignore_errors=True)
    print("head built", flush=True)


class AneQwen:
    def __init__(self):
        self.c, ck = cfg(), Checkpoint()
        units = ct.ComputeUnit.CPU_AND_NE
        manifest = json.loads((OUT / f"manifest_ctx{CTX}.json").read_text())
        names = manifest["chunks"]
        self.ring_rows = manifest.get("ring_rows", 4)  # 8 = two conv rings (batched-prefill builds)
        self.half = 0
        self.chunks = [ct.models.CompiledMLModel(str(OUT / n), compute_units=units) for n in names]
        self.head = ct.models.CompiledMLModel(str(OUT / "head.mlmodelc"), compute_units=units)
        self.states = [m.make_state() for m in self.chunks]
        self.emb = ck.embed_table()
        rot = int(self.c["head_dim"] * self.c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / self.c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        self.pos = 0

    def reset(self):
        """Fresh Core ML states (DeltaNet / conv / KV) and position 0."""
        self.states = [m.make_state() for m in self.chunks]
        self.pos, self.half = 0, 0

    def step(self, token):
        t = self.pos
        f = t * self.inv
        feed = {"x": self.emb[token].reshape(1, -1, 1, 1),
                "cos": np.cos(np.concatenate([f, f]))[None].astype(np.float16),
                "sin": np.sin(np.concatenate([f, f]))[None].astype(np.float16),
                "mask": np.where(np.arange(CTX) <= t, 0, -1e4).astype(np.float16)[None],
                **C.step_inputs(t, self.half, self.ring_rows)}
        for m, s in zip(self.chunks, self.states):
            feed["x"] = m.predict(feed, state=s)["y"]
        self.pos += 1
        return self.head.predict({"x": feed["x"]})["logits"][0]


class AneQwen2:
    """Runtime for v2 builds. DeltaNet states live in two sets of IOSurface-backed SharedArrays per chunk: each call
    reads `cur` and writes `nxt` (output backings), then the two swap; hidden states pass from chunk to chunk in
    SharedArrays too. Only small per-token inputs are written (fp16 memcpy) and only the logits are read."""

    def __init__(self):
        c = self.c = cfg()
        ck = Checkpoint()
        man = json.loads((OUT / f"manifest_ctx{CTX}_v2.json").read_text())
        self.T, self.taps, self.kv_io = man["prefill_t"], man["taps"], man.get("kv_io", False)
        SA, units = ct.models.SharedArray, ct.ComputeUnit.CPU_AND_NE
        hid, T = c["hidden_size"], self.T
        nv, dk, dv = (c[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        cdim = 2 * c["linear_num_key_heads"] * dk + nv * dv
        self.chunks = []
        for ch in man["chunks"]:
            path = str(OUT / ch["file"])
            t0 = time.time()
            inf = ct.models.CompiledMLModel(path, compute_units=units, function_name="infer")
            layers = list(range(ch["layers"][0], ch["layers"][1] + 1))
            gdn = [j for j, l in enumerate(layers) if c["layer_types"][l] == "linear_attention"]
            att = [j for j in range(len(layers)) if j not in gdn]
            kvs = (c["num_key_value_heads"], CTX, c["head_dim"])
            shapes = {f"conv{j}": (3, cdim) for j in gdn} | {f"rec{j}": (nv, dk, dv) for j in gdn} | \
                     ({f"{s_}{j}": kvs for j in att for s_ in ("k", "v")} if self.kv_io else {})
            taps = [l for l in self.taps if l in layers]
            pre = []
            for i, (a, b) in enumerate([] if os.environ.get("V2_NO_PREFILL") else ch["prefill"]):
                js = range(a, b + 1)
                pre.append({"m": ct.models.CompiledMLModel(path, compute_units=units, function_name=f"prefill{i}"),
                            "names": [f"{s_}{j}" for j in js for s_ in (("conv", "rec") if j in gdn else
                                                                         (("k", "v") if self.kv_io else ()))],
                            "taps": [l for l in taps if layers[a] <= l <= layers[b]], "last": layers[b]})
            self.chunks.append({
                "inf": inf, "pre": pre, "names": list(shapes), "last": layers[-1],
                "state": None if self.kv_io else inf.make_state(),
                "gdn_names": [n for n in shapes if n.startswith(("conv", "rec"))],
                "cur": {n: SA(v) for n, v in shapes.items()}, "nxt": {n: SA(v) for n, v in shapes.items()},
                "y1": SA((1, hid, 1, 1)), "taps": taps,
                "tap1": {l: SA((1, hid, 1, 1)) for l in taps}, "tapT": {l: SA((1, hid, 1, T)) for l in taps}})
            print(f"loaded {ch['file']} ({time.time() - t0:.0f}s)", flush=True)
        self.head = ct.models.CompiledMLModel(str(OUT / man["head"]), compute_units=units)
        self.head_x, self.logits = SA((1, hid, 1, 1)), SA((1, c["vocab_size"]))
        self.yT = [SA((1, hid, 1, T)), SA((1, hid, 1, T))]  # prefill hidden states, alternating between calls
        # A tap on a function's last layer is not a separate output (two identity outputs of one tensor break on the
        # ANE: zeros or a load failure); that tap is the function's "y".
        self.emb = ck.embed_table()
        rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        self.d = {"x": SA((1, hid, 1, 1)), "cos": SA((1, rot)), "sin": SA((1, rot)), "mask": SA((1, CTX)),
                  "kv_onehot": SA((1, CTX, 1))}
        self.p = {"x": SA((1, hid, 1, T)), "cos": SA((T, rot)), "sin": SA((T, rot)), "mask": SA((T, CTX)),
                  "kv_write": SA((T, CTX)), "valid": SA((1, T, 1)), "conv_sel": SA((3, T + 3))}
        self.pos = 0
        self.stats = {"prefill_calls": 0, "decode_calls": 0}

    def reset(self):
        """Zero DeltaNet states, position 0 (KV rows at or past the position are masked, so the caches need none)."""
        for ch in self.chunks:
            for n in ch["gdn_names"]:
                ch["cur"][n].zero()
            if not self.kv_io:
                ch["state"] = ch["inf"].make_state()
        self.pos = 0

    def snapshot(self):
        """Copy of the DeltaNet states and the position (the KV cache is masked by position, so it needs none)."""
        snap = []
        for ch in self.chunks:
            saved = {}
            for n in ch["gdn_names"]:
                saved[n] = ct.models.SharedArray(ch["cur"][n].shape)
                saved[n].copy_from(ch["cur"][n])
            snap.append(saved)
        return {"pos": self.pos, "states": snap}

    def restore(self, snap):
        """Back to a snapshot. KV rows before its position are untouched by later tokens, so only DeltaNet state
        is restored."""
        for ch, saved in zip(self.chunks, snap["states"]):
            for n in ch["gdn_names"]:
                ch["cur"][n].copy_from(saved[n])
        self.pos = snap["pos"]

    def _outputs(self, y, taps, tapbufs, last):
        """Output backings for y and the taps; a tap on the function's last layer is y itself."""
        if last in taps:
            tapbufs[last] = y
        return {"y": y} | {f"tap{l}": tapbufs[l] for l in taps if l != last}

    def _run(self, fn, small, x, y_key, tap_key):
        for ch in self.chunks:
            back = self._outputs(ch[y_key], ch["taps"], ch[tap_key], ch["last"]) | \
                   {f"{n}_out": ch["nxt"][n] for n in ch["names"]}
            ch[fn].predict({**small, "x": x, **ch["cur"]}, state=ch["state"], output_backings=back)
            ch["cur"], ch["nxt"] = ch["nxt"], ch["cur"]
            x = ch[y_key]
        return x

    def _run_prefill(self, small, x):
        k = 0
        for ch in self.chunks:
            for sub in ch["pre"]:
                y = self.yT[k % 2]
                back = self._outputs(y, sub["taps"], ch["tapT"], sub["last"]) | \
                       {f"{n}_out": ch["nxt"][n] for n in sub["names"]}
                sub["m"].predict({**small, "x": x, **{n: ch["cur"][n] for n in sub["names"]}}, state=ch["state"],
                                 output_backings=back)
                x, k = y, k + 1
            ch["cur"], ch["nxt"] = ch["nxt"], ch["cur"]
        return x

    def _head(self, x):
        self.head.predict({"x": x}, output_backings={"logits": self.logits})
        return self.logits.to_numpy()[0]

    def step(self, token):
        """One token through the decode functions; returns the next-token logits (fp16)."""
        t = self.pos
        f = t * self.inv
        ang = np.concatenate([f, f])[None]
        self.d["x"].write(self.emb[token].reshape(1, -1, 1, 1))
        self.d["cos"].write(np.cos(ang).astype(np.float16))
        self.d["sin"].write(np.sin(ang).astype(np.float16))
        self.d["mask"].write(np.where(np.arange(CTX) <= t, 0, -1e4).astype(np.float16)[None])
        kv = np.zeros((1, CTX, 1), np.float16)
        kv[0, t, 0] = 1
        self.d["kv_onehot"].write(kv)
        small = {k: v for k, v in self.d.items() if k != "x"}
        y = self._run("inf", small, self.d["x"], "y1", "tap1")
        self.pos += 1
        self.stats["decode_calls"] += 1
        return self._head(y)

    def prefill_block(self, ids):
        """Up to T tokens through the prefill functions (padding rows are masked out); returns the last logits."""
        T, k, p0 = self.T, len(ids), self.pos
        assert 0 < k <= T and p0 + k <= CTX
        x = np.zeros((1, self.c["hidden_size"], 1, T), np.float16)
        x[0, :, 0, :k] = self.emb[ids].T
        pos = np.minimum(np.arange(p0, p0 + T), p0 + k - 1)            # padding rows repeat the last position
        f = np.outer(pos, self.inv)
        ang = np.concatenate([f, f], axis=1)
        mask = np.where(np.arange(CTX)[None, :] <= pos[:, None], 0, -1e4).astype(np.float16)
        kvw = np.zeros((T, CTX), np.float16)
        kvw[np.arange(k), np.arange(p0, p0 + k)] = 1                     # padding rows write nothing
        valid = np.zeros((1, T, 1), np.float16)
        valid[0, :k, 0] = 1
        sel = np.zeros((3, T + 3), np.float16)
        sel[np.arange(3), k + np.arange(3)] = 1                          # the 3 rows ending at the last valid token
        for name, v in (("x", x), ("cos", np.cos(ang).astype(np.float16)), ("sin", np.sin(ang).astype(np.float16)),
                        ("mask", mask), ("kv_write", kvw), ("valid", valid), ("conv_sel", sel)):
            self.p[name].write(v)
        small = {n: v for n, v in self.p.items() if n != "x"}
        y = self._run_prefill(small, self.p["x"])
        self.pos += k
        self.stats["prefill_calls"] += 1
        self.head_x.write(np.ascontiguousarray(y.to_numpy()[:, :, :, k - 1:k]))
        return self._head(self.head_x)

    def feed(self, ids):
        """Feed prompt tokens (any count, any start position); returns the logits after the last one."""
        logits, i = None, 0
        while i < len(ids):
            r = len(ids) - i
            if r <= STEP_MAX:
                logits = self.step(ids[i])
                i += 1
            else:
                k = min(self.T, r)
                logits = self.prefill_block(ids[i:i + k])
                i += k
        return logits


class AneQwen3:
    """Runtime for v3 builds: one T-row function per chunk. A call runs up to T rows at `pos`; accept(k) then commits
    the first k (decode / prefill: all of them; DFlash verify: 1 + accepted drafts). DeltaNet buffers ping-pong in
    SharedArrays (no host copies); the host keeps only `pos` and the pending count.
    Growing context (KV as inputs, v4+): with a ladder of context lengths built with the same tag, the runtime starts
    at `ctx` and switches to the next length when a call would not fit (resize: release this length's chunk models,
    load the other's, move the KV rows); the DeltaNet buffers do not depend on the context length."""

    def __init__(self, tag=None, ctx=None, ladder=None):
        c = self.c = cfg()
        ck = Checkpoint()
        self.ctx = ctx or CTX
        tag = tag or next(t for t in ("v5", "v4", "v3") if (OUT / f"manifest_ctx{self.ctx}_{t}.json").exists())
        man = json.loads((OUT / f"manifest_ctx{self.ctx}_{tag}.json").read_text())
        self.tag, self.ladder = tag, sorted(set(ladder or []) | {self.ctx})
        self.T, self.P, self.taps, self.kv_in = man["T"], man["pend"], man["taps"], man.get("kv_in", False)
        self.TP = man.get("prefill_t", 0)  # rows of the prefill function (0: none)
        self.layer_ranges = [ch["layers"] for ch in man["chunks"]]
        SA = ct.models.SharedArray
        L = self.ctx
        hid, T = c["hidden_size"], self.T
        nv, dk, dv = (c[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        cdim = 2 * c["linear_num_key_heads"] * dk + nv * dv
        self.chunks = []
        for ch in man["chunks"]:
            t0 = time.time()
            m, m_pre = self._load(ch["file"])
            layers = list(range(ch["layers"][0], ch["layers"][1] + 1))
            gdn = [j for j, l in enumerate(layers) if c["layer_types"][l] == "linear_attention"]
            shapes = {}
            for j in gdn:
                shapes |= {f"conv{j}": (T + 3, cdim), f"rec{j}": (nv, dk, dv), f"pend{j}": (nv, 3 * self.P + 1, dv)}
            taps = [l for l in self.taps if l in layers]
            kvs = (c["num_key_value_heads"], L, c["head_dim"])
            att = [j for j in range(len(layers)) if j not in gdn]
            kv = {f"{s_}{j}": SA(kvs) for j in att for s_ in ("k", "v")} if self.kv_in else {}
            TP = self.TP or 1
            self.chunks.append({"m": m, "m_pre": m_pre, "state": None if self.kv_in else m.make_state(), "last": layers[-1],
                                "taps": taps, "kv": kv, "kv_new": {n: SA((kvs[0], T, kvs[2])) for n in kv},
                                "kv_newP": {n: SA((kvs[0], TP, kvs[2])) for n in kv}, "yP": SA((1, hid, 1, TP)),
                                "tapP": {l: SA((1, hid, 1, TP)) for l in taps},
                                "cur": {n: SA(v) for n, v in shapes.items()}, "nxt": {n: SA(v) for n, v in shapes.items()},
                                "y": SA((1, hid, 1, T)), "tap": {l: SA((1, hid, 1, T)) for l in taps}})
            print(f"loaded {ch['file']} ({time.time() - t0:.0f}s)", flush=True)
        self.head = ct.models.CompiledMLModel(str(OUT / man["head"]), compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.logits = SA((T, c["vocab_size"]))
        self.emb = ck.embed_table()
        rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        self.inp = {"x": SA((1, hid, 1, T)), "cos": SA((T, rot)), "sin": SA((T, rot)),
                    "mask": SA((1, L) if self.kv_in else (T, L)), "conv_sel": SA((3, T + 3)),
                    "commit": SA((1, self.P, 1)), "commit_last": SA((1, self.P, 1))}
        if not self.kv_in:
            self.inp["kv_write"] = SA((T, L))
        if self.TP:
            TP = self.TP
            self.pin = {"x": SA((1, hid, 1, TP)), "cos": SA((TP, rot)), "sin": SA((TP, rot)), "mask": SA((1, L)),
                        "conv_sel": SA((3, T + 3)), "conv_sel_out": SA((3, TP + 3)), "valid": SA((1, TP, 1)),
                        "commit": SA((1, self.P, 1)), "commit_last": SA((1, self.P, 1))}
            self.head_x = SA((1, hid, 1, T))
        self.nkv = c["num_key_value_heads"]
        self.pos, self.pending, self.hi = 0, 0, 0   # hi: KV rows written since reset (snapshots may point below it)
        self.stats = {"calls": 0, "resize": []}

    def _load(self, file):
        units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_AND_NE"))  # UNITS=CPU_ONLY: numerics checks
        if self.TP:
            return (ct.models.CompiledMLModel(str(OUT / file), compute_units=units, function_name="verify"),
                    ct.models.CompiledMLModel(str(OUT / file), compute_units=units, function_name="prefill"))
        return ct.models.CompiledMLModel(str(OUT / file), compute_units=units), None

    def resize(self, ctx):
        """Switch to this tag's build for another context length (grow, or shrink when the rows in use fit):
        release the current length's chunk models first (two program sets do not fit in memory), move the KV rows
        [0, hi) into caches of the new length, load the new length's chunk models. DeltaNet buffers, pending rows,
        position and head do not depend on the context length and carry over."""
        assert self.kv_in, "resize needs KV cache inputs (v4+ builds)"
        assert self.pos <= ctx, f"position {self.pos} does not fit in {ctx}"
        man = json.loads((OUT / f"manifest_ctx{ctx}_{self.tag}.json").read_text())
        assert man["T"] == self.T and man["pend"] == self.P and man.get("prefill_t", 0) == self.TP and \
            [ch["layers"] for ch in man["chunks"]] == self.layer_ranges, f"ctx {ctx} {self.tag} build differs"
        old, t0 = self.ctx, time.time()
        for ch in self.chunks:
            ch["m"] = ch["m_pre"] = None
        import gc
        gc.collect()
        t1 = time.time()
        keep = min(self.hi, ctx)
        kvs = (self.nkv, ctx, self.c["head_dim"])
        src, dst = [h * old for h in range(self.nkv)], [h * ctx for h in range(self.nkv)]
        for ch in self.chunks:
            for n_ in list(ch["kv"]):
                new = ct.models.SharedArray(kvs)
                if keep:
                    new.copy_rows_from(ch["kv"][n_], src, dst, keep)
                ch["kv"][n_] = new
        t2 = time.time()
        for ch, mc in zip(self.chunks, man["chunks"]):
            ch["m"], ch["m_pre"] = self._load(mc["file"])
        t3 = time.time()
        self.inp["mask"] = ct.models.SharedArray((1, ctx))
        if self.TP:
            self.pin["mask"] = ct.models.SharedArray((1, ctx))
        self.ctx, self.hi = ctx, keep
        ev = {"from": old, "to": ctx, "pos": self.pos, "release_s": t1 - t0, "kv_ms": 1e3 * (t2 - t1),
              "load_s": t3 - t2}
        self.stats["resize"].append(ev)
        print(f"[ctx] {old} -> {ctx} at pos {self.pos}: released in {ev['release_s']:.1f}s, {keep} KV rows moved in "
              f"{ev['kv_ms']:.0f} ms, loaded in {ev['load_s']:.1f}s", flush=True)
        return ev

    def fit(self, need):
        """Grow to the smallest ladder length that holds `need` positions; False if none does."""
        if need <= self.ctx:
            return True
        nxt = next((s for s in self.ladder if s >= need), None)
        if nxt is None:
            return False
        self.resize(nxt)
        return True

    def reset(self):
        """Fresh KV states, zero DeltaNet buffers, position 0."""
        for ch in self.chunks:
            if not self.kv_in:
                ch["state"] = ch["m"].make_state()
            for a in ch["cur"].values():
                a.zero()
        self.pos, self.pending, self.hi = 0, 0, 0

    def snapshot(self):
        """DeltaNet buffers (committed state, pending rows, conv rows), position and pending count; the KV cache is
        masked by position and needs no snapshot."""
        snap = []
        for ch in self.chunks:
            saved = {}
            for n, a in ch["cur"].items():
                saved[n] = ct.models.SharedArray(a.shape)
                saved[n].copy_from(a)
            snap.append(saved)
        return {"pos": self.pos, "pending": self.pending, "states": snap}

    def restore(self, snap):
        for ch, saved in zip(self.chunks, snap["states"]):
            for n, a in ch["cur"].items():
                a.copy_from(saved[n])
        self.pos, self.pending = snap["pos"], snap["pending"]

    def call(self, ids):
        """Run len(ids) <= T tokens at `pos` (not committed until accept). Returns logits (n, vocab) fp16."""
        T, n, p0, k = self.T, len(ids), self.pos, self.pending
        assert 0 < n <= T
        if not self.fit(p0 + n):  # grows to the next ladder length when the block does not fit
            raise ValueError(f"{p0 + n} positions exceed the largest context {self.ladder[-1]}")
        CTX = self.ctx
        x = np.zeros((1, self.c["hidden_size"], 1, T), np.float16)
        x[0, :, 0, :n] = self.emb[ids].T
        pos = np.minimum(np.arange(p0, p0 + T), p0 + n - 1)          # padding rows repeat the last position
        f = np.outer(pos, self.inv)
        ang = np.concatenate([f, f], axis=1)
        kvw = np.zeros((T, CTX), np.float16)
        kvw[np.arange(n), np.arange(p0, p0 + n)] = 1                  # padding rows write no KV
        sel = np.zeros((3, T + 3), np.float16)
        sel[np.arange(3), k + np.arange(3)] = 1                       # 3 conv rows ending at the last committed token
        com, last = np.zeros((1, self.P, 1), np.float16), np.zeros((1, self.P, 1), np.float16)
        com[0, :k] = 1
        if k:
            last[0, k - 1] = 1
        if self.kv_in:  # history = committed positions < p0; the block attends to itself causally in the graph
            mask = np.where(np.arange(CTX)[None, :] < p0, 0, -1e4).astype(np.float16)
        else:
            mask = np.where(np.arange(CTX)[None, :] <= pos[:, None], 0, -1e4).astype(np.float16)
        for name, v in (("x", x), ("cos", np.cos(ang).astype(np.float16)), ("sin", np.sin(ang).astype(np.float16)),
                        ("mask", mask), ("conv_sel", sel), ("commit", com), ("commit_last", last)) + \
                (() if self.kv_in else (("kv_write", kvw),)):
            self.inp[name].write(v)
        small = {n_: v for n_, v in self.inp.items() if n_ != "x"}
        xin = self.inp["x"]
        for ch in self.chunks:
            back = {"y": ch["y"]} | {f"tap{l}": ch["tap"][l] for l in ch["taps"] if l != ch["last"]} | \
                   {f"{n_}_out": ch["nxt"][n_] for n_ in ch["cur"]} | {f"{n_}_new": v for n_, v in ch["kv_new"].items()}
            ch["m"].predict({**small, "x": xin, **ch["cur"], **ch["kv"]}, state=ch["state"], output_backings=back)
            ch["cur"], ch["nxt"] = ch["nxt"], ch["cur"]
            if ch["last"] in ch["taps"]:
                ch["tap"][ch["last"]] = ch["y"]
            xin = ch["y"]
        self.head.predict({"x": xin}, output_backings={"logits": self.logits})
        self.stats["calls"] += 1
        self._n = n
        return self.logits.to_numpy()[:n]

    def call_chunk(self, c, ids, x):
        """Chunk c alone on the rows of `ids` at `pos`, fed x (1, hidden, 1, T) fp16 instead of the previous chunk's
        output (divergence tests: a chunk's own error with a reference input); same position / commit inputs as
        call(), then accept(k) as usual. Returns the chunk's output rows (T, hidden) float32."""
        assert self.kv_in, "call_chunk needs a v4+ build (KV as inputs)"
        T, n, p0, k = self.T, len(ids), self.pos, self.pending
        assert 0 < n <= T and p0 + n <= self.ctx
        pos = np.minimum(np.arange(p0, p0 + T), p0 + n - 1)
        f = np.outer(pos, self.inv)
        ang = np.concatenate([f, f], axis=1)
        sel, com, last = self._commit_inputs(k)
        mask = np.where(np.arange(self.ctx)[None, :] < p0, 0, -1e4).astype(np.float16)
        for name, v in (("x", x), ("cos", np.cos(ang).astype(np.float16)), ("sin", np.sin(ang).astype(np.float16)),
                        ("mask", mask), ("conv_sel", sel), ("commit", com), ("commit_last", last)):
            self.inp[name].write(v)
        ch = self.chunks[c]
        back = {"y": ch["y"]} | {f"tap{l}": ch["tap"][l] for l in ch["taps"] if l != ch["last"]} | \
               {f"{n_}_out": ch["nxt"][n_] for n_ in ch["cur"]} | {f"{n_}_new": v for n_, v in ch["kv_new"].items()}
        small = {n_: v for n_, v in self.inp.items() if n_ != "x"}
        res = ch["m"].predict({**small, "x": self.inp["x"], **ch["cur"], **ch["kv"]}, state=ch["state"],
                              output_backings=back)
        self.last_extra = {k_: v for k_, v in (res or {}).items() if k_.startswith("dbg")}  # DBG_MIXER_IN builds
        ch["cur"], ch["nxt"] = ch["nxt"], ch["cur"]
        self._n = n
        return ch["y"].to_numpy()[0, :, 0, :].T.astype(np.float32)

    def _commit_inputs(self, k):
        sel = np.zeros((3, self.T + 3), np.float16)
        sel[np.arange(3), k + np.arange(3)] = 1                       # 3 conv rows ending at the last committed token
        com, last = np.zeros((1, self.P, 1), np.float16), np.zeros((1, self.P, 1), np.float16)
        com[0, :k] = 1
        if k:
            last[0, k - 1] = 1
        return sel, com, last

    def prefill_block(self, ids):
        """Up to TP tokens through the prefill functions, all committed (padding masked). Returns the last token's
        logits; the rows' tap features stay readable through features_prefill(n) until the next call."""
        T, TP, n, p0 = self.T, self.TP, len(ids), self.pos
        assert 0 < n <= TP
        if not self.fit(p0 + n):  # grows to the next ladder length when the block does not fit
            raise ValueError(f"{p0 + n} positions exceed the largest context {self.ladder[-1]}")
        CTX = self.ctx
        x = np.zeros((1, self.c["hidden_size"], 1, TP), np.float16)
        x[0, :, 0, :n] = self.emb[ids].T
        pos = np.minimum(np.arange(p0, p0 + TP), p0 + n - 1)
        f = np.outer(pos, self.inv)
        ang = np.concatenate([f, f], axis=1)
        sel, com, last = self._commit_inputs(self.pending)
        sel_out = np.zeros((3, TP + 3), np.float16)
        sel_out[np.arange(3), n + np.arange(3)] = 1                   # 3 rows ending at the last valid token
        valid = np.zeros((1, TP, 1), np.float16)
        valid[0, :n, 0] = 1
        for name, v in (("x", x), ("cos", np.cos(ang).astype(np.float16)), ("sin", np.sin(ang).astype(np.float16)),
                        ("mask", np.where(np.arange(CTX)[None, :] < p0, 0, -1e4).astype(np.float16)),
                        ("conv_sel", sel), ("conv_sel_out", sel_out), ("valid", valid), ("commit", com),
                        ("commit_last", last)):
            self.pin[name].write(v)
        small = {n_: v for n_, v in self.pin.items() if n_ != "x"}
        xin = self.pin["x"]
        for ch in self.chunks:
            back = {"y": ch["yP"]} | {f"tap{l}": ch["tapP"][l] for l in ch["taps"] if l != ch["last"]} | \
                   {f"{n_}_out": ch["nxt"][n_] for n_ in ch["cur"]} | {f"{n_}_new": v for n_, v in ch["kv_newP"].items()}
            ch["m_pre"].predict({**small, "x": xin, **ch["cur"], **ch["kv"]}, output_backings=back)
            ch["cur"], ch["nxt"] = ch["nxt"], ch["cur"]
            if ch["last"] in ch["taps"]:
                ch["tapP"][ch["last"]] = ch["yP"]
            xin = ch["yP"]
        src = [h * TP for h in range(self.nkv)]
        dst = [h * CTX + p0 for h in range(self.nkv)]
        for ch in self.chunks:
            for n_, buf in ch["kv"].items():
                buf.copy_rows_from(ch["kv_newP"][n_], src, dst, n)
        self.pending, self.pos, self._nP = 0, p0 + n, n
        self.hi = max(self.hi, self.pos)
        self.stats["calls"] += 1
        hx = np.zeros((1, self.c["hidden_size"], 1, T), np.float16)
        hx[0, :, 0, 0] = xin.to_numpy()[0, :, 0, n - 1]
        self.head_x.write(hx)
        self.head.predict({"x": self.head_x}, output_backings={"logits": self.logits})
        return self.logits.to_numpy()[0]

    def features_prefill(self, n):
        taps = {l: t for ch in self.chunks for l, t in ch["tapP"].items()}
        return np.concatenate([taps[l].to_numpy()[0, :, 0, :n].T for l in self.taps], axis=1)

    def features(self, n):
        """Drafter features of the last call's first n rows: the tap hidden states concatenated in TAPS order,
        (n, len(TAPS) * hidden) fp16. Read before the next call (the buffers are reused)."""
        taps = {l: t for ch in self.chunks for l, t in ch["tap"].items()}
        return np.concatenate([taps[l].to_numpy()[0, :, 0, :n].T for l in self.taps], axis=1)

    def accept(self, k):
        """Commit the first k rows of the last call (they become the next call's committed prefix); with read-only
        KV inputs their k / v rows are copied into the caches here (rejected rows never reach them)."""
        assert 0 <= k <= self._n
        if self.kv_in and k:
            src = [h * self.T for h in range(self.nkv)]
            dst = [h * self.ctx + self.pos for h in range(self.nkv)]
            for ch in self.chunks:
                for n_, buf in ch["kv"].items():
                    buf.copy_rows_from(ch["kv_new"][n_], src, dst, k)
        self.pending, self.pos = k, self.pos + k
        self.hi = max(self.hi, self.pos)

    def step(self, token):
        logits = self.call([token])[0]
        self.accept(1)
        return logits

    def feed(self, ids, on_features=None):
        """Feed prompt tokens from any position (TP-token prefill calls when there is a prefill function, T-token
        calls for the rest); returns the logits after the last one. on_features(features, positions) receives the
        drafter features of every block (DFlash context)."""
        logits, i = None, 0
        while i < len(ids):
            r = len(ids) - i
            if self.TP and r > self.T:
                blk = ids[i:i + min(r, self.TP)]
                p0 = self.pos
                logits = self.prefill_block(blk)
                if on_features:
                    on_features(self.features_prefill(len(blk)), np.arange(p0, p0 + len(blk)))
            else:
                blk = ids[i:i + self.T]
                logits = self.call(blk)[len(blk) - 1]
                if on_features:
                    on_features(self.features(len(blk)), np.arange(self.pos, self.pos + len(blk)))
                self.accept(len(blk))
            i += len(blk)
        return logits


def load_model():
    """The newest runtime with a build for this context: v3 (one T=8 function), v2, else v1 (decode only).
    CTX_LADDER="2048,8192,16384,65536" (lengths <= CTX with builds of one tag): growing context, starting at the
    smallest length."""
    lad = [int(x) for x in os.environ.get("CTX_LADDER", "").replace(",", " ").split() if int(x) <= CTX]
    if lad:
        return AneQwen3(ctx=min(lad), ladder=lad)
    if any((OUT / f"manifest_ctx{CTX}_{t}.json").exists() for t in ("v5", "v4", "v3")):
        return AneQwen3()
    return AneQwen2() if (OUT / f"manifest_ctx{CTX}_v2.json").exists() else AneQwen()


def generate(prompt, max_new=int(os.environ.get("MAX_NEW", "64"))):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    ids = tok.encode(prompt).ids
    model = load_model()
    t0 = time.time()
    if hasattr(model, "feed"):
        logits = model.feed(ids)
    else:
        for i in ids[:-1]:
            model.step(i)
        logits = model.step(ids[-1])
    t_prefill = time.time() - t0
    out, t1 = [], time.time()
    for _ in range(max_new):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        if nxt in (248044, 248046):
            break
        logits = model.step(nxt)
    dt = time.time() - t1
    print(prompt + tok.decode(out))
    print(f"\n[{len(ids)} prompt tokens in {t_prefill:.1f}s; {len(out)} generated in {dt:.1f}s = "
          f"{len(out) / dt:.2f} tok/s]", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "build_v2":
        build_v2()
    elif sys.argv[1] == "build_v3":
        build_v3()
    else:
        generate(" ".join(sys.argv[2:]))
