"""Hand-rolled LoRA for a Jeff checkpoint.

PEFT is not required. A linear layer keeps its frozen weight and adds a low-rank
update ``y = Wx + (alpha / rank) B A x``, with B started at zero so the adapter
begins as the base model. The same projection names Jeff adapts are the targets:
attention ``q/k/v/o``, Gated DeltaNet ``in_proj_qkv``, ``in_proj_z``, ``out_proj``,
and the MLP ``gate/up/down``. The tiny DeltaNet ``in_proj_a`` / ``in_proj_b`` gates
stay frozen, and so does the vision tower.

Merging is ``W <- W + (alpha / rank) B A`` in float32, then cast back to the
checkpoint dtype. That merged checkpoint is what ``jeff-convert`` compiles. The
ANE build does not apply LoRA at runtime.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

TARGET_NAMES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
})
TOWERS = frozenset({
    "visual", "vision_tower", "vision_model", "audio_tower", "audio_model",
})
FORMAT = "anemll-jeff-lora-1"


class LoRALinear(nn.Module):
    """Frozen ``base`` plus a rank-r update. ``key`` is the safetensors name of ``base.weight``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, key: str):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if base.weight.ndim != 2:
            raise ValueError(f"{key} is not a 2-D weight")
        self.base = base
        self.key = key
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        out_f, in_f = base.weight.shape
        device = base.weight.device
        # Factors stay in fp32 so Adam and the merge are stable even when the backbone is bf16.
        seed = torch.empty(rank, in_f, dtype=torch.float32, device=device)
        nn.init.kaiming_uniform_(seed, a=math.sqrt(5))
        self.A = nn.Parameter(seed)
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (x.float() @ self.A.T) @ self.B.T
        return self.base(x) + (update * self.scale).to(dtype=self.base.weight.dtype)

    def merged_weight(self) -> torch.Tensor:
        """``W + scale * B @ A`` in the base weight's dtype, on CPU."""
        weight = self.base.weight.detach()
        delta = self.scale * (self.B.detach().float() @ self.A.detach().float())
        return (weight.float().cpu() + delta.cpu()).to(dtype=weight.dtype).contiguous()


def attach_lora(root: nn.Module, rank: int, alpha: float, targets: frozenset[str] | set[str] = TARGET_NAMES) -> list[LoRALinear]:
    """Wrap each matching linear layer. Call this before moving newly created parameters is unnecessary:
    A and B are created on the base weight's device and dtype."""
    chosen: list[tuple[str, nn.Linear]] = []
    for name, module in list(root.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets or not isinstance(module, nn.Linear):
            continue
        if TOWERS.intersection(name.split(".")):
            continue
        chosen.append((name, module))
    if not chosen:
        raise ValueError(f"no linear layers named {sorted(targets)}")
    wrapped: list[LoRALinear] = []
    for name, module in chosen:
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        layer = LoRALinear(module, rank, alpha, key=f"{name}.weight")
        setattr(parent, child, layer)
        wrapped.append(layer)
    return wrapped


def trainable_parameters(layers: list[LoRALinear], readout: nn.Linear) -> list[nn.Parameter]:
    """Keep each wrapped base weight frozen and train the LoRA factors plus the readout."""
    for layer in layers:
        layer.base.requires_grad_(False)
        layer.A.requires_grad_(True)
        layer.B.requires_grad_(True)
    readout.weight.requires_grad_(True)
    if readout.bias is not None:
        readout.bias.requires_grad_(False)
    return [layer.A for layer in layers] + [layer.B for layer in layers] + [readout.weight]


def save_lora(directory: Path, layers: list[LoRALinear], readout: nn.Linear, meta: dict) -> None:
    """Write the factors (not a PEFT folder) plus the trained readout and a JSON description."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer in layers:
        stem = layer.key.removesuffix(".weight")
        tensors[f"{stem}.lora_A"] = layer.A.detach().float().cpu().contiguous()
        tensors[f"{stem}.lora_B"] = layer.B.detach().float().cpu().contiguous()
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    save_file({"weight": readout.weight.detach().cpu().contiguous()}, str(directory / "readout.safetensors"))
    payload = {"format": FORMAT, "rank": layers[0].rank, "scale": layers[0].scale, "layers": len(layers), **meta}
    (directory / "adapter_config.json").write_text(json.dumps(payload, indent=2) + "\n")


def merge_peft_adapter(base: Path, adapter: Path, dest: Path) -> dict:
    """Fold one published Jeff PEFT adapter into a copy of the base checkpoint.

    This is the same update as ``jeff.lora.merge_adapter``: ``W += (alpha / rank) B A`` in
    float32, stored in the base weight's dtype. The adapter's readout replaces the base
    readout, and its fitted temperature is written into ``decision_config.json``. ``dest``
    is what ``jeff-convert`` compiles. The ANE package does not apply LoRA at runtime.
    """
    base, adapter, dest = Path(base), Path(adapter), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Refusing to overwrite checkpoint contents: {dest}")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA" or config.get("bias") != "none":
        raise ValueError(f"{adapter} is not a plain LoRA adapter")
    if config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{adapter}: only a single rank and alpha are merged")
    scale = float(config["lora_alpha"]) / float(config["r"])
    factors = load_file(str(adapter / "adapter_model.safetensors"))
    layers: dict[str, dict[str, torch.Tensor]] = {}
    prefix = "base_model.model."
    for key, value in factors.items():
        if not key.startswith(prefix) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unexpected adapter weight {key}")
        name, part, _rest = key[len(prefix):].rsplit(".", 2)
        layers.setdefault(name, {})["A" if part == "lora_A" else "B"] = value
    if not layers or any(set(parts) != {"A", "B"} for parts in layers.values()):
        raise ValueError(f"{adapter} is missing a LoRA factor")
    dest.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(base / "model.safetensors"))
    missing = []
    for name, parts in layers.items():
        key = f"{name}.weight"
        if key not in tensors and name.startswith("model."):
            key = f"{name[len('model.'):]}.weight"
        if key not in tensors:
            missing.append(name)
            continue
        current = tensors[key]
        delta = scale * (parts["B"].float() @ parts["A"].float())
        tensors[key] = (current.float() + delta).to(dtype=current.dtype).contiguous()
    if missing:
        raise KeyError(f"adapter layers not in {base / 'model.safetensors'}: {missing[:3]}")
    save_file(tensors, str(dest / "model.safetensors"))
    readout = load_file(str(adapter / "readout.safetensors"))["weight"].contiguous()
    save_file({"weight": readout}, str(dest / "readout.safetensors"))
    decision = json.loads((base / "decision_config.json").read_text())
    adapter_decision = json.loads((adapter / "decision_config.json").read_text())
    temperature = float(adapter_decision["temperature"])
    if temperature <= 0:
        raise ValueError(f"{adapter} temperature must be positive")
    decision["temperature"] = temperature
    (dest / "decision_config.json").write_text(json.dumps(decision, indent=2) + "\n")
    for item in base.iterdir():
        if not item.is_file() or item.name in ("model.safetensors", "readout.safetensors", "decision_config.json"):
            continue
        target = dest / item.name
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)
    return {"layers": len(layers), "rank": int(config["r"]), "alpha": float(config["lora_alpha"]),
            "scale": scale, "temperature": temperature}


def save_merged_checkpoint(source: Path, dest: Path, layers: list[LoRALinear], readout: nn.Linear) -> None:
    """Copy a Jeff checkpoint and replace the adapted matrices and the readout.

    Tokenizer, config, and ``decision_config.json`` are linked when the filesystem allows it. ``model.safetensors``
    is rewritten. Vision and frozen text tensors are unchanged.
    """
    source, dest = Path(source), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Refusing to overwrite checkpoint contents: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(source / "model.safetensors"))
    missing = [layer.key for layer in layers if layer.key not in tensors]
    if missing:
        raise KeyError(f"merged weights not in {source / 'model.safetensors'}: {missing[:3]}")
    for layer in layers:
        current = tensors[layer.key]
        tensors[layer.key] = layer.merged_weight().to(dtype=current.dtype).contiguous()
    save_file(tensors, str(dest / "model.safetensors"))
    weight = readout.weight.detach().cpu().contiguous()
    save_file({"weight": weight}, str(dest / "readout.safetensors"))
    for item in source.iterdir():
        if not item.is_file() or item.name in ("model.safetensors", "readout.safetensors"):
            continue
        target = dest / item.name
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)


def merge_saved_lora(source: Path, adapter: Path, dest: Path) -> None:
    """Rebuild a merged checkpoint from a fully published LoRA directory after interruption."""
    source, adapter, dest = Path(source), Path(adapter), Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    temporary = dest.with_name(dest.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        temporary.mkdir(parents=True)
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config["format"] != FORMAT:
            raise ValueError(f"{adapter}: unsupported LoRA format")
        factors = load_file(str(adapter / "adapter_model.safetensors"))
        tensors = load_file(str(source / "model.safetensors"))
        stems = {key.removesuffix(".lora_A") for key in factors if key.endswith(".lora_A")}
        if len(stems) != config["layers"] or set(factors) != {
                f"{stem}.lora_{side}" for stem in stems for side in ("A", "B")}:
            raise ValueError(f"{adapter}: incomplete LoRA factors")
        for stem in stems:
            key = stem + ".weight"
            if key not in tensors:
                raise KeyError(f"{key} missing from {source}")
            a, b = factors[f"{stem}.lora_A"], factors[f"{stem}.lora_B"]
            current = tensors[key]
            tensors[key] = (current.float() + float(config["scale"]) * (b.float() @ a.float())).to(
                dtype=current.dtype).contiguous()
        save_file(tensors, str(temporary / "model.safetensors"))
        shutil.copy2(adapter / "readout.safetensors", temporary / "readout.safetensors")
        for item in source.iterdir():
            if not item.is_file() or item.name in ("model.safetensors", "readout.safetensors"):
                continue
            target = temporary / item.name
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        temporary.replace(dest)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
