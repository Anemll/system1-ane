#!/usr/bin/env python3
"""Stage the Hugging Face upload folders for the Jeff Core AI demo builds (no upload).

    python hf/stage.py --out ~/jeff_release_prep/hf_staging \
        --tetris "$HOME/Models/jeff-tetris-training/overnight-20261008/round-02/build" \
        --snake  "$HOME/Models/jeff-coreai/adapters/snake" \
        --base   "$HOME/Models/jeff-coreai" \
        --base-weights "$HOME/Models/jeff/jeff-base-v1.3/model.safetensors"

Each source is a directory with coreai/ (exported .aimodel packages + manifest.json) and model/
(tokenizer, decision_config.json, readout, embeddings). The staged folder adds a root-level config.json
(Hugging Face counts downloads by it), the model card from hf/<name>/README.md, LICENSE and NOTICE, and
removes absolute local paths from manifest.json. system1-ane-serve --model loads the full backbone checkpoint
(embeddings, final norm, shape checks), so the base build also carries model/model.safetensors (--base-weights).
Snake/Tetris builds are served as --adapter beside the base and do not need it. Files are cloned (APFS clonefile) when possible.

Upload afterwards, only with approval, e.g.:
    hf upload anemll/jeff-ane-tetris-coreai ~/jeff_release_prep/hf_staging/tetris . --repo-type model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ("tetris", "snake", "base")
SKIP = {".DS_Store", ".gitattributes"}


def clone_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if sys.platform == "darwin":
        subprocess.run(["cp", "-cR", str(src), str(dst)], check=True)  # APFS clone, no extra disk
    else:
        shutil.copytree(src, dst)
    for junk in dst.rglob("*"):
        if junk.name in SKIP:
            junk.unlink()


def scrub_manifest(path: Path, label: str) -> None:
    data = json.loads(path.read_text())
    home = str(Path.home())

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str) and (x.startswith("/") or x.startswith(home)):
            return label
        return x

    path.write_text(json.dumps(walk(data), indent=1) + "\n")


DESCRIPTIONS = {
    "tetris": ("Tetris policy: Jeff v1.3 + Tetris LoRA and readout (overnight 2026-10-08, round 02), merged.",
               "jeff-base v1.3 + LoRA (rank 16, alpha 32) + 255-way readout, trained locally on Apple M5 Max (PyTorch MPS)"),
    "snake": ("Snake policy: Jeff v1.3 + Snake LoRA and readout (2026-10-07), merged.",
              "jeff-base v1.3 + LoRA (rank 16) + 255-way readout, trained locally on Apple M5 Max (PyTorch MPS)"),
    "base": ("Jeff v1.3 base decision model, unchanged (routing demo; also the system1-ane-serve --model checkpoint).",
             "mstrasser/jeff-base v1.3, no fine-tuning by Anemll"),
}


def write_descriptor(name: str, dest: Path) -> None:
    """Package descriptor, not a transformers config (the transformers config is model/config.json)."""
    man = json.loads((dest / "coreai" / "manifest.json").read_text())
    conv = man.get("convert", {})
    desc = {
        "schema_version": 1,
        "config_type": "jeff-ane/coreai-package",
        "description": ("Package descriptor for the Core AI (.aimodel) builds in this repo. This is NOT a transformers "
                        "config: transformers' AutoConfig / AutoModel cannot load this repo. The transformers config "
                        "of the Qwen3.5-0.8B backbone is model/config.json. Run the packages with "
                        "https://github.com/Anemll/system1-ane."),
        "library_name": "coreai",
        "format": "coreai .aimodel, FP16, prefill-only, compiled per Mac with `python forge.py compile`",
        "compute": "Apple Neural Engine via Core AI",
        "runtime": "https://github.com/Anemll/system1-ane",
        "build": name,
        "summary": DESCRIPTIONS[name][0],
        "training": DESCRIPTIONS[name][1],
        "base_model": {"id": "mstrasser/jeff-base", "revision": "v1.3", "backbone": "Qwen/Qwen3.5-0.8B"},
        "license": "apache-2.0",
        "model_type": "jeff-decision",
        "layers": conv.get("layers"),
        "hidden_size": conv.get("hidden_size"),
        "context": conv.get("ctx"),
        "prefill_rows": conv.get("prefill_rows"),
        "quant": man.get("quant"),
        "readout": {"codes": 255, "temperature": conv.get("temperature"), "prompt_layout": conv.get("prompt_layout")},
        "packages": [c["file"] for c in man["chunks"]] + [man["head"]["file"]],
        "manifest": "coreai/manifest.json",
        "host": {"path": "model/", "transformers_config": "model/config.json", "tokenizer": "model/tokenizer.json",
                 "decision_config": "model/decision_config.json", "readout": "model/readout.safetensors"},
        "download": "python scripts/download_model.py " + name,
    }
    if name == "base":
        desc["host"]["checkpoint"] = "model/model.safetensors"
    (dest / "config.json").write_text(json.dumps(desc, indent=1) + "\n")


def stage(name: str, src: Path, out: Path, weights: Path | None = None) -> Path:
    src = src.expanduser().resolve()
    for need in ("coreai/manifest.json", "model/config.json", "model/decision_config.json", "model/tokenizer.json"):
        if not (src / need).is_file():
            raise SystemExit(f"{name}: {src / need} missing")
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    for part in ("coreai", "model"):
        clone_tree(src / part, dest / part)
    scrub_manifest(dest / "coreai" / "manifest.json", f"jeff-ane-{name}")
    if weights is not None:
        target = dest / "model" / "model.safetensors"
        if sys.platform == "darwin":
            subprocess.run(["cp", "-c", str(weights.expanduser()), str(target)], check=True)
        else:
            shutil.copyfile(weights.expanduser(), target)
    write_descriptor(name, dest)  # root-level config.json (Hugging Face counts downloads by it)
    card = HERE / name / "README.md"
    if card.is_file():
        shutil.copyfile(card, dest / "README.md")
    for extra in ("LICENSE", "NOTICE"):
        if (HERE / extra).is_file():
            shutil.copyfile(HERE / extra, dest / extra)
    leaks = subprocess.run(["grep", "-rIl", "--exclude=tokenizer.json", home_marker(), str(dest)],
                           capture_output=True, text=True).stdout.split()
    if leaks:
        raise SystemExit(f"{name}: home path still present in {leaks}")
    return dest


def home_marker() -> str:
    return str(Path.home())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    for n in NAMES:
        p.add_argument(f"--{n}", type=Path, help=f"source build dir for {n}")
    p.add_argument("--base-weights", type=Path, help="jeff-base v1.3 model.safetensors, copied into base/model/")
    a = p.parse_args(argv)
    if a.base and not a.base_weights:
        p.error("--base needs --base-weights (system1-ane-serve --model loads the full checkpoint)")
    out = a.out.expanduser()
    done = 0
    for n in NAMES:
        src = getattr(a, n)
        if src:
            dest = stage(n, src, out, a.base_weights if n == "base" else None)
            size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
            print(f"{n}: {dest}  {size / 1e9:.2f} GB")
            done += 1
    if not done:
        p.error("give at least one of --tetris/--snake/--base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
