#!/usr/bin/env python3
"""Jeff on the Apple Neural Engine: train LoRA, export Core AI, compile, serve the demo.

Compatible entry point for the live server:

    python forge.py system1-ane-serve --model ... --build ... --adapter snake=... --adapter tetris=...
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import ane_compile_mode as SOC
import coreai_compile_guide as G


def path(value):
    return Path(value).expanduser().resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report environment without loading a model")
    c = sub.add_parser("compile", help="compile a Jeff Core AI build for this Mac's ANE ahead of serving")
    c.add_argument("--build", type=path, required=True, help="coreai/ directory from jeff-convert")
    c.add_argument("--force", action="store_true", help="drop this Python's cached ANE specializations first")
    c.add_argument("--dry-run", action="store_true", help="print command only")
    jc = sub.add_parser("jeff-convert", help="prefill-only Core AI export of a Jeff decision checkpoint")
    jc.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    jc.add_argument("--output", type=path, required=True, help="new empty directory (model/ + coreai/)")
    jc.add_argument("--ctx", type=int, default=2048, help="KV history of the prefill entry")
    jc.add_argument("--prefill", type=int, default=256, help="prefill rows (multiple of 8, > 8)")
    jc.add_argument("--prefill-extra", default="",
                    help="more prefill widths in the same chunks, e.g. 512,1024,1536,2048")
    jc.add_argument("--quant", choices=("fp16", "int8"), default="fp16")
    jc.add_argument("--chunk-layers", type=int, default=4)
    jc.add_argument("--dry-run", action="store_true")
    js = sub.add_parser("jeff-smoke", help="Jeff prefill+readout smoke (host DecodeLayer; optional Core AI --build)")
    js.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    js.add_argument("--build", type=path, help="coreai/ directory from jeff-convert")
    js.add_argument("--state", default="The disk on db-02 is 97 percent full and still growing.")
    js.add_argument("--options", default="page,wait,ignore")
    js.add_argument("--instructions", default="Choose the best next action.")
    js.add_argument("--ids", help="comma-separated token ids; skips tokenizer + prompt render")
    js.add_argument("--n-options", type=int, help="options of the --ids prompt")
    js.add_argument("--cases", type=path, help="scripts/jeff_reference.py output (ids + FP32 reference probabilities)")
    js.add_argument("--only", help="comma-separated case names from --cases")
    js.add_argument("--row", type=path, help="JSON decision row {state, question}")
    js.add_argument("--host", action="store_true", help="also run the host DecodeLayer reference")
    js.add_argument("--bench", type=int, default=0, help="extra timed Core AI prefills per case")
    js.add_argument("--out", type=path, help="results JSON")
    js.add_argument("--dry-run", action="store_true")
    jv = sub.add_parser("system1-ane-serve", aliases=["jeff-serve"],
                        help="System 1 ANE decision server (POST /v1/systemone) plus the browser demo; jeff-serve is a deprecated alias")
    jv.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    jv.add_argument("--build", type=path, required=True, help="coreai/ directory from jeff-convert")
    jv.add_argument("--host", default="127.0.0.1")
    jv.add_argument("--port", type=int, default=8787)
    jv.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                    help="extra merged Core AI build, name=coreai-dir. Repeatable. base is --build")
    jv.add_argument("--dry-run", action="store_true")
    jt = sub.add_parser("jeff-train-lora", help="sample LoRA on a Jeff checkpoint (snake oracle, or a JSONL dataset)")
    jt.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    jt.add_argument("--output", type=path, required=True, help="new directory for adapter/, merged/, and report.json")
    jt.add_argument("--dataset", type=path, help="JSONL of {state, options, label, instructions}; default is the synthetic task")
    jt.add_argument("--task", choices=("snake", "tetris"), default="snake")
    jt.add_argument("--rank", type=int, default=16)
    jt.add_argument("--alpha", type=float, default=32)
    jt.add_argument("--train", type=int, default=256)
    jt.add_argument("--heldout", type=int, default=64)
    jt.add_argument("--epochs", type=int, default=2)
    jt.add_argument("--batch", type=int, default=4)
    jt.add_argument("--eval-batch", type=int, help="evaluation batch size; defaults to --batch")
    jt.add_argument("--eval-train-rows", type=int, default=0, help="train-accuracy sample size; 0 uses all rows")
    jt.add_argument("--checkpoint-every", type=int, default=0, help="recovery checkpoint interval in training steps")
    jt.add_argument("--train-until", help="timezone-aware ISO deadline for optimization")
    jt.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    jt.add_argument("--readout-lr", type=float, default=5e-6)
    jt.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    jt.add_argument("--seed", type=int, default=0)
    jt.add_argument("--play", type=int, default=0, help="PyTorch self-play games per model; 0 skips them")
    jt.add_argument("--max-steps", type=int, default=48)
    jt.add_argument("--skip-merge", action="store_true")
    jt.add_argument("--init-adapter", type=path, help="continue from this adapter/ directory")
    jt.add_argument("--heldout-dataset", type=path, help="fixed held-out JSONL")
    jt.add_argument("--eval-only", action="store_true", help="print held-out accuracy and exit without writing")
    jt.add_argument("--dry-run", action="store_true")
    return p


def _text_config(model: Path) -> dict:
    raw = json.loads((model / "config.json").read_text())
    cfg = raw.get("text_config")
    if isinstance(cfg, dict) and cfg.get("num_hidden_layers") and cfg.get("hidden_size"):
        return cfg
    if raw.get("num_hidden_layers") and raw.get("hidden_size"):
        return raw
    return {}


def _is_hybrid(cfg: dict) -> bool:
    types = cfg.get("layer_types") or []
    return "linear_attention" in types and "full_attention" in types


def prepare_jeff(a):
    """Jeff decision path: hybrid Qwen3.5 + readout."""
    if not (a.model / "config.json").is_file():
        raise ValueError(f"Missing checkpoint config: {a.model / 'config.json'}")
    cfg = _text_config(a.model)
    if not _is_hybrid(cfg):
        raise ValueError("jeff-convert / jeff-smoke / system1-ane-serve / jeff-train-lora expect a Qwen3.5 hybrid text config "
                         "(layer_types with linear_attention and full_attention).")
    env = {"MODEL": str(a.model)}
    published_embedding = a.model / "embed_tokens_fp16.npy"
    env["EMBED_NPY"] = str(published_embedding if published_embedding.is_file()
                           else a.model / ".anemll-forge" / "embed_tokens_fp16.npy")
    if a.command == "jeff-convert":
        if a.prefill <= 8 or a.prefill % 8:
            raise ValueError("--prefill must be a multiple of 8 and greater than 8")
        if a.ctx < a.prefill:
            raise ValueError("--ctx must be >= --prefill")
        dest = a.output
        if not a.dry_run and dest.exists() and any(dest.iterdir()):
            raise ValueError(f"Use a new or empty --output directory: {dest}")
        args = ["--model", str(a.model), "--output", str(a.output), "--ctx", str(a.ctx),
                "--prefill", str(a.prefill), "--quant", a.quant, "--chunk-layers", str(a.chunk_layers)]
        if a.prefill_extra:
            args += ["--prefill-extra", a.prefill_extra]
        if a.dry_run:
            args.append("--dry-run")
        python = sys.executable if a.dry_run else coreai_python()
        return [python, str(ROOT / "scripts" / "jeff_coreai_convert.py"), *args], env
    if a.command == "jeff-train-lora":
        args = ["--model", str(a.model), "--output", str(a.output), "--task", a.task,
                "--rank", str(a.rank), "--alpha", str(a.alpha), "--train", str(a.train),
                "--heldout", str(a.heldout), "--epochs", str(a.epochs), "--batch", str(a.batch),
                "--lr", str(a.lr), "--readout-lr", str(a.readout_lr),
                "--device", a.device, "--seed", str(a.seed), "--play", str(a.play),
                "--max-steps", str(a.max_steps)]
        if a.dataset is not None:
            args += ["--dataset", str(a.dataset)]
        if a.init_adapter is not None:
            args += ["--init-adapter", str(a.init_adapter)]
        if a.heldout_dataset is not None:
            args += ["--heldout-dataset", str(a.heldout_dataset)]
        if a.skip_merge:
            args.append("--skip-merge")
        if a.eval_only:
            args.append("--eval-only")
        if a.eval_batch is not None:
            args += ["--eval-batch", str(a.eval_batch)]
        if a.eval_train_rows:
            args += ["--eval-train-rows", str(a.eval_train_rows)]
        if a.checkpoint_every:
            args += ["--checkpoint-every", str(a.checkpoint_every)]
        if a.train_until:
            args += ["--train-until", a.train_until]
        return [sys.executable, str(ROOT / "scripts" / "jeff_lora_train.py"), *args], env
    if a.command in ("system1-ane-serve", "jeff-serve"):
        if a.command == "jeff-serve":
            print("note: jeff-serve is deprecated; use system1-ane-serve", file=sys.stderr, flush=True)
        args = ["--model", str(a.model), "--build", str(a.build), "--host", a.host, "--port", str(a.port)]
        for item in a.adapter or []:
            if item.count("=") != 1 or not item.split("=", 1)[0]:
                raise ValueError("--adapter must be name=path to a coreai directory")
            args += ["--adapter", item]
        return [coreai_python(), str(ROOT / "scripts" / "system1_ane_serve.py"), *args], {
            **env, "TOKENIZER_PYTHON": sys.executable}
    args = ["--model", str(a.model), "--state", a.state, "--options", a.options,
            "--instructions", a.instructions]
    for flag, value in (("--build", a.build), ("--ids", a.ids), ("--n-options", a.n_options), ("--cases", a.cases),
                        ("--only", a.only), ("--row", a.row), ("--out", a.out)):
        if value is not None:
            args += [flag, str(value)]
    args += ["--host"] if a.host else []
    args += ["--bench", str(a.bench)] if a.bench else []
    python = coreai_python() if a.build else sys.executable
    return [python, str(ROOT / "scripts" / "jeff_coreai_smoke.py"), *args], env


def coreai_python() -> str:
    """Interpreter with the Core AI conversion SDK and runtime (coreai_torch, coreai.runtime): COREAI_PYTHON, else
    this one. The host venv often has neither; the SDK ships its own environment."""
    return os.environ.get("COREAI_PYTHON") or sys.executable


def is_jeff_build(build: Path) -> bool:
    try:
        return json.loads((build / "manifest.json").read_text()).get("kind") in ("decision", "jeff-decision")
    except (OSError, ValueError):
        return False


def prepare(a):
    """Return argv and env overrides without importing ML packages or writing files."""
    if a.command in ("jeff-convert", "jeff-smoke", "system1-ane-serve", "jeff-serve", "jeff-train-lora"):
        return prepare_jeff(a)
    raise ValueError(f"unknown command {a.command}")


def run_recovering(command, env):
    """Run a package-loading child. If it dies by a crash signal while loading a package, purge that package's
    cache for this Python and retry once."""
    fd, state = tempfile.mkstemp(prefix="jeff-ane-loading-")
    os.close(fd)
    env = {**env, G.STATE_ENV: state}
    try:
        for attempt in range(2):
            rc = subprocess.call(command, env=env, cwd=ROOT)
            pkg = Path(state).read_text().strip()
            if attempt or -rc not in G.CRASH_SIGNALS or not pkg:
                return rc
            n = G.purge(Path(pkg))
            print(f"{G.TAG} crashed (signal {-rc}) while loading {Path(pkg).name}; purged {n} cached specializations "
                  f"for '{G.process_key()}' and retrying once (it recompiles)", file=sys.stderr, flush=True)
    finally:
        os.unlink(state)


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.command == "compile":
        if not (a.build / "manifest.json").is_file():
            p.error(f"Missing build manifest: {a.build / 'manifest.json'}")
        python = coreai_python() if is_jeff_build(a.build) else sys.executable
        command = [python, str(ROOT / "scripts" / "coreai_compile.py"), "--build", str(a.build)]
        command += ["--force"] if a.force else []
        if a.dry_run:
            print(json.dumps(dict(argv=command), indent=2))
            return 0
        if sys.platform != "darwin":
            p.error("Core AI compilation requires macOS")
        env = os.environ.copy()
        try:
            SOC.apply(strict=True, log=lambda m: None)
        except SOC.UnsupportedSocError as e:
            p.error(str(e))
        return run_recovering(command, env)
    if a.command == "doctor":
        versions = {}
        for name in ("numpy", "torch", "coremltools", "transformers", "safetensors",
                     "scipy", "scikit-learn", "tokenizers", "ml_dtypes", "coreai-core"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        soc = SOC.detect_soc()
        try:
            mode = SOC.apply(strict=False, soc=soc)
        except SOC.UnsupportedSocError:
            mode = None
        print(json.dumps(dict(python=sys.version, platform=platform.platform(), packages=versions,
                             soc={"class": soc.klass, "generation": soc.generation, "source": soc.source},
                             ane_bonded_compile_mode=mode,
                             note="Presence is not compatibility or ANE-placement verification."), indent=2))
        return 0
    try:
        command, overrides = prepare(a)
    except (ValueError, OSError) as e:
        p.error(str(e))
    if getattr(a, "dry_run", False):
        print(json.dumps(dict(argv=command, environment=overrides), indent=2))
        return 0
    linux_ok = a.command in ("jeff-smoke", "jeff-train-lora")
    if not linux_ok and sys.platform != "darwin":
        p.error("Core AI conversion, compile, and serve require macOS "
                "(jeff-smoke host readout and --dry-run run anywhere)")
    env = os.environ.copy()
    env.update(overrides)
    return subprocess.call(command, env=env, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
