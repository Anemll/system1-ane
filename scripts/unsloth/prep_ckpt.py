"""Merge the Unsloth Snake LoRA into the stock Qwen3.5-0.8B weights -> checkpoint directory for `forge.py jeff-convert`.

  python scripts/unsloth/prep_ckpt.py ADAPTER_HEAD_DIR DEST_DIR

ADAPTER_HEAD_DIR  the trained `adapter_head/` folder (adapter_config.json + adapter_model.safetensors) from run_decision_a100.py
STOCK_BASE_DIR    (env) local snapshot of `unsloth/Qwen3.5-0.8B` (or Qwen/Qwen3.5-0.8B): *.safetensors, config.json, tokenizer files
DECISION_CONFIG   (env) a `decision_config.json` from a downloaded base build (`python scripts/download_model.py base`,
                  `<models dir>/base/model/decision_config.json`); only its answer-code table is used, the temperature is reset to 1.0

The merge is done in fp32 (a bf16 round trip of the merged weights drops part of the LoRA delta). DEST gets `model.safetensors`
(fp32), a ZERO placeholder `readout.safetensors` (255 x hidden, never used: the trained Clef head runs separately, see
clef_head.py / clef_ane_head.py), `decision_config.json` and the config / tokenizer files. DEST must not exist."""
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CODES = 255  # width of the placeholder readout (the converter expects a readout tensor)
COPY = ["config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "preprocessor_config.json"]


def load_adapter(adapter_dir: Path):
    """-> (lora scale alpha/r, {module name: {"A": tensor, "B": tensor}}) for a plain LoRA (no DoRA / rsLoRA / bias)."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["bias"] == "none" and not cfg.get("use_dora") and not cfg.get("use_rslora")
    layers = {}
    for k, v in load_file(str(adapter_dir / "adapter_model.safetensors")).items():
        assert k.startswith("base_model.model.")
        name, part, _ = k[len("base_model.model."):].rsplit(".", 2)
        layers.setdefault(name, {})["A" if part == "lora_A" else "B"] = v
    assert all(set(p) == {"A", "B"} for p in layers.values())
    return cfg["lora_alpha"] / cfg["r"], layers


def merge_lora(tensors: dict, layers: dict, scale: float):
    """In place: W += scale * B @ A (fp32) for every adapted module. Returns (modules merged, max |delta|)."""
    n, mx = 0, 0.0
    for name, p in layers.items():
        key = name + ".weight"
        assert key in tensors, key
        delta = scale * (p["B"].float() @ p["A"].float())
        mx = max(mx, delta.abs().max().item())
        tensors[key] = (tensors[key] + delta).contiguous()
        n += 1
    return n, mx


def main(argv):
    adapter_dir, dest = Path(argv[1]), Path(argv[2])
    base, decision = Path(os.environ["STOCK_BASE_DIR"]), Path(os.environ["DECISION_CONFIG"])
    scale, layers = load_adapter(adapter_dir)
    tensors = {}
    for f in sorted(base.glob("*.safetensors")):
        tensors.update({k: v.float().contiguous() for k, v in load_file(str(f)).items()})
    n, mx = merge_lora(tensors, layers, scale)
    dest.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(dest / "model.safetensors"), metadata={"format": "pt"})
    hidden = json.loads((base / "config.json").read_text())["text_config"]["hidden_size"]
    save_file({"weight": torch.zeros(CODES, hidden, dtype=torch.float16)}, str(dest / "readout.safetensors"))
    dc = json.loads(decision.read_text())
    dc["temperature"] = 1.0
    dc["note"] = "placeholder zero readout; the Unsloth Clef head runs separately"
    (dest / "decision_config.json").write_text(json.dumps(dc, indent=2))
    for name in COPY:
        if (base / name).exists():
            shutil.copy(base / name, dest / name)
    print(json.dumps({"merged_modules": n, "scale": scale, "max_abs_delta": mx, "dest": str(dest)}))


if __name__ == "__main__":
    if len(sys.argv) != 3 or "STOCK_BASE_DIR" not in os.environ or "DECISION_CONFIG" not in os.environ:
        sys.exit(__doc__)
    main(sys.argv)
