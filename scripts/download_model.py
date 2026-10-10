#!/usr/bin/env python3
"""Download the Jeff Core AI demo builds from Hugging Face.

    python scripts/download_model.py                 # tetris, snake and base
    python scripts/download_model.py tetris          # one build
    python scripts/download_model.py snake-stock-qwen3.5-unsloth   # optional extra: Unsloth-trained stock Snake (not in the default set)
    python scripts/download_model.py --list

Each build lands in $JEFF_MODELS_DIR/<name>/ (default ~/Models/jeff-ane) with this layout:

    config.json      backbone config (root-level, also what Hugging Face counts downloads by)
    coreai/          exported .aimodel packages + manifest.json   -> forge.py compile / system1-ane-serve --build
    model/           tokenizer, decision_config.json, readout, embeddings -> system1-ane-serve --model
                     (base also has model.safetensors, the full checkpoint the server loads)

`snake-stock-qwen3.5-unsloth` is an optional extra adapter (stock Qwen3.5-0.8B + Unsloth Snake head, 304-token prefill). It is a folder
of the repo `anemll/system1-base-unsloth`: only the root `config.json` and `snake-stock-qwen3.5-unsloth/*` are fetched, and the build lands in
$JEFF_MODELS_DIR/snake-stock-qwen3.5-unsloth/snake-stock-qwen3.5-unsloth/. If the repo cannot be reached, the download stops with a clear message
(NOT_REACHABLE_HINT below). It is never part of the default set; Jeff stays the default.

The packages are exported, not compiled: run `python forge.py compile --build <dir>/coreai` once per Mac.
Needs `huggingface_hub` (pip install huggingface_hub). Set HF_TOKEN only if a repo is private.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODELS = {
    # name: (Hugging Face repo, what it is)
    "tetris": ("anemll/jeff-ane-tetris-coreai", "Tetris continuation (overnight 2026-10-08, round-02), merged LoRA + readout"),
    "snake": ("anemll/jeff-ane-snake-coreai", "Snake LoRA, merged"),
    "base": ("anemll/jeff-ane-base-coreai", "jeff-base v1.3 + full checkpoint (server --model, routing panel)"),
}
DEFAULT_ORDER = ("tetris", "snake", "base")
# Optional extra, served as `--adapter snake-stock=<build>/coreai`; a folder inside its own repo.
EXTRA = "snake-stock-qwen3.5-unsloth"
MODELS[EXTRA] = ("anemll/system1-base-unsloth", "OPTIONAL extra adapter: Unsloth-trained stock Snake (folder %s/; Clef head, 304-token prefill)" % EXTRA)
NOT_REACHABLE_HINT = " The Unsloth model repo may not be reachable yet; check %s." % MODELS[EXTRA][0]  # remove once the upload is confirmed live


def models_dir() -> Path:
    return Path(os.environ.get("JEFF_MODELS_DIR", Path.home() / "Models" / "jeff-ane")).expanduser()


def download(name: str, root: Path, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    repo, _ = MODELS[name]
    env_name = "JEFF_%s_REPO" % name.upper().replace("-", "_").replace(".", "_")
    repo = os.environ.get(env_name, repo)
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    folder = name if name == EXTRA else None
    # Fetch the root config.json on its own first: Hugging Face counts downloads by this file.
    try:
        hf_hub_download(repo, "config.json", revision=revision, local_dir=dest)
    except Exception as error:  # RepositoryNotFound / RevisionNotFound / offline / gated
        if type(error).__name__ in ("RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError",
                                    "HfHubHTTPError", "LocalEntryNotFoundError", "OfflineModeIsEnabled"):
            hint = NOT_REACHABLE_HINT if folder else ""
            raise SystemExit(f"{repo}: cannot download ({type(error).__name__}).{hint} Check the repo name, your network, "
                             f"and HF_TOKEN if the repo is private. Or point {env_name} at another repo.")
        raise
    if folder:
        snapshot_download(repo, revision=revision, local_dir=dest, allow_patterns=["config.json", f"{folder}/*", f"{folder}/**"])
        build = dest / folder
        need = ("coreai/manifest.json", "coreai/clef_head/joint_head.safetensors", "model/decision_config.json", "model/tokenizer.json")
        missing = [p for p in need if not (build / p).is_file()]
        if missing:
            raise SystemExit(f"{repo}: folder {folder}/ is missing {missing} (is the upload complete?)")
        return build
    snapshot_download(repo, revision=revision, local_dir=dest,
                      allow_patterns=["config.json", "README.md", "LICENSE*", "NOTICE*",
                                      "coreai/**", "model/**"])
    missing = [p for p in ("config.json", "coreai/manifest.json", "model/decision_config.json", "model/tokenizer.json",
                *(["model/model.safetensors"] if name == "base" else []))
               if not (dest / p).is_file()]
    if missing:
        raise SystemExit(f"{repo}: download incomplete, missing {missing}")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", metavar="NAME",
                   help=f"one or more of {', '.join(MODELS)} (default: all)")
    p.add_argument("--dir", type=Path, default=None, help="destination root (default $JEFF_MODELS_DIR or ~/Models/jeff-ane)")
    p.add_argument("--revision", default=None, help="tag or commit, e.g. v0.1.0")
    p.add_argument("--list", action="store_true", help="print the builds and their repos, then exit")
    a = p.parse_args(argv)
    if a.list:
        for name, (repo, what) in MODELS.items():
            print(f"{name:7s} {repo:34s} {what}")
        return 0
    bad = [n for n in a.names if n not in MODELS]
    if bad:
        p.error(f"unknown build(s) {bad}; choose from {list(MODELS)}")
    root = (a.dir or models_dir()).expanduser()
    dests = {}
    for name in a.names or DEFAULT_ORDER:
        dests[name] = download(name, root, a.revision)
        print(f"{name}: {dests[name]}")
    print("\nNext, once per Mac (in $COREAI_PYTHON via forge.py):")
    for name in a.names or DEFAULT_ORDER:
        print(f"  python forge.py compile --build {dests[name] / 'coreai'}")
    if EXTRA in dests:
        print(f"Then serve it as an extra adapter: system1-ane-serve --adapter snake-stock={dests[EXTRA] / 'coreai'} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
