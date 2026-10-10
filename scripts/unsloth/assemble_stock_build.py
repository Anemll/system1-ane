"""Finish a stock Snake Core AI build after `forge.py jeff-convert` + `forge.py compile`.

  python scripts/unsloth/assemble_stock_build.py CKPT_DIR BUILD_DIR ADAPTER_HEAD_DIR

CKPT_DIR          the merged checkpoint written by prep_ckpt.py
BUILD_DIR         the --output directory of jeff-convert (contains coreai/ and model/)
ADAPTER_HEAD_DIR  the trained adapter_head/ folder (joint_head.safetensors + configs)

1. BUILD_DIR/model/model.safetensors = the language-model tensors of CKPT_DIR in fp16 (the runtime reads the embedding table and
   final norm from it), plus the config / tokenizer / decision_config / readout files.
2. BUILD_DIR/coreai/clef_head/ = the trained Clef head files; a build that ships this folder is served by ClefEngine.
No weights are modified; both copies are generated next to the build and are not meant for Git."""
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

MODEL_FILES = ["config.json", "decision_config.json", "readout.safetensors", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "vocab.json", "merges.txt"]
HEAD_FILES = ["joint_head.safetensors", "joint_head_config.json", "unsloth_decision_config.json"]


def main(ckpt: Path, build: Path, head: Path):
    model = build / "model"
    model.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt / "model.safetensors"))
    half = {k: v.half().contiguous() for k, v in tensors.items() if k.startswith("model.language_model.")}
    save_file(half, str(model / "model.safetensors"), metadata={"format": "pt"})
    for name in MODEL_FILES:
        if (ckpt / name).exists():
            shutil.copy(ckpt / name, model / name)
    dest = build / "coreai" / "clef_head"
    dest.mkdir(parents=True, exist_ok=True)
    for name in HEAD_FILES:
        shutil.copy(head / name, dest / name)
    print(f"{len(half)} language-model tensors -> {model / 'model.safetensors'}; Clef head -> {dest}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*(Path(a) for a in sys.argv[1:]))
