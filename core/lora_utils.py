"""
LoRA utilities for AceStep 1.5 LoRA Trainer.
PEFT-based LoRA injection into DiT decoder and single-file safetensors export.
"""

import os
import logging
from typing import Dict, Any, Tuple
from collections import OrderedDict

import torch

logger = logging.getLogger("AceStepTrainer")


def check_peft_available() -> bool:
    """Check if PEFT library is available."""
    try:
        import importlib.util
        return importlib.util.find_spec("peft") is not None
    except Exception:
        return False


def inject_lora(
    model,
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.1,
    target_modules: list = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Inject LoRA adapters into the DiT decoder.

    Args:
        model: The AceStep DiT model (has .decoder attribute)
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha/rank)
        dropout: LoRA dropout
        target_modules: Module names to apply LoRA to

    Returns:
        Tuple of (model_with_lora, info_dict)
    """
    from peft import LoraConfig

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    decoder = model.decoder

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    # Use add_adapter (in-place) instead of get_peft_model (wrapper).
    # get_peft_model wraps in PeftModel which breaks gradient chain for custom models.
    # add_adapter injects LoRA layers directly — this is how the original ACE-Step trains.
    decoder.add_adapter(adapter_config=peft_config, adapter_name="lora_adapter")

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "rank": rank,
        "alpha": alpha,
    }

    print("[AceStep LoRA] Injected into DiT decoder:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable (LoRA): {trainable_params:,} ({info['trainable_ratio']:.2%})")
    print(f"  Rank: {rank}, Alpha: {alpha}")

    return model, info


def extract_lora_state_dict(model, adapter_name: str = "lora_adapter", alpha: int = 128) -> OrderedDict:
    """Extract only LoRA parameters from the model as a flat state dict.

    Strips the adapter name and converts keys to the official ACE-Step LoRA
    format used by ComfyUI's ACEStep15 loader:
      decoder.layers.0.self_attn.q_proj.lora_A.lora_adapter.weight
    becomes:
      base_model.model.layers.0.self_attn.q_proj.lora_A.weight

    ComfyUI's ACEStep15 key mapping (comfy/lora.py line 335-339) maps
    'base_model.model.{key}' to model state dict keys. This matches the
    format produced by the original Gradio training app.

    Also includes per-module .alpha tensors so ComfyUI applies the correct
    scaling (alpha/rank). Without alpha tensors, ComfyUI defaults to 1.0.

    Weights are saved as bfloat16 to match the official format (~86MB).

    Returns an OrderedDict suitable for saving as a single safetensors file.
    """
    lora_state = OrderedDict()
    strip_segment = f".{adapter_name}"
    # Collect module base names for alpha tensors
    alpha_bases = set()
    for name, param in model.named_parameters():
        if "lora_" in name:
            clean_name = name.replace(strip_segment, "")
            # Model params: decoder.layers.0.self_attn.q_proj.lora_A.weight
            # Strip "decoder." and add "base_model.model." prefix
            if clean_name.startswith("decoder."):
                clean_name = clean_name[len("decoder."):]
            clean_name = f"base_model.model.{clean_name}"
            lora_state[clean_name] = param.data.clone().cpu().to(torch.bfloat16)
            # Track base module name for alpha (e.g. base_model.model.layers.0.self_attn.q_proj)
            if ".lora_A." in clean_name:
                base = clean_name.split(".lora_A.")[0]
                alpha_bases.add(base)
    # Add per-module .alpha tensors so ComfyUI applies correct scaling
    for base in sorted(alpha_bases):
        lora_state[f"{base}.alpha"] = torch.tensor(float(alpha))
    return lora_state


def save_lora_safetensors(model, output_path: str, alpha: int = 128, rank: int = 64, dropout: float = 0.1, **kwargs) -> str:
    """Save LoRA weights as a single .safetensors file.

    Uses the official ACE-Step LoRA format (base_model.model.* keys, bfloat16).
    Includes per-module .alpha tensors for correct scaling in ComfyUI.
    Embeds LoRA config as safetensors metadata for portability.

    Args:
        model: Model with LoRA adapters
        output_path: Full path for the output .safetensors file
        alpha: LoRA alpha value (saved as .alpha tensors for ComfyUI)
        rank: LoRA rank (saved in metadata)
        dropout: LoRA dropout (saved in metadata)

    Returns:
        Path to saved file
    """
    import json
    from safetensors.torch import save_file

    lora_state = extract_lora_state_dict(model, alpha=alpha)

    if not lora_state:
        print("[AceStep LoRA] WARNING: No LoRA parameters found to save!")
        return ""

    # Diagnostic: print weight norms to verify training produced meaningful weights
    weight_norms = []
    for k, v in lora_state.items():
        if ".lora_" in k:
            weight_norms.append(v.float().norm().item())
    if weight_norms:
        avg_norm = sum(weight_norms) / len(weight_norms)
        max_norm = max(weight_norms)
        print(f"[AceStep LoRA] Weight stats: avg_norm={avg_norm:.6f}, max_norm={max_norm:.6f}, n_weights={len(weight_norms)}")

    # Embed LoRA config as safetensors metadata for portability
    metadata = {
        "format": "ace_step_lora",
        "lora_rank": str(rank),
        "lora_alpha": str(alpha),
        "lora_dropout": str(dropout),
        "target_modules": json.dumps(["q_proj", "k_proj", "v_proj", "o_proj"]),
        "model": "ACE-Step-1.5-turbo-shift3",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(lora_state, output_path, metadata=metadata)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[AceStep LoRA] Saved {len(lora_state)} tensors to {output_path} ({size_mb:.1f} MB)")
    return output_path


def load_lora_safetensors(model, lora_path: str, rank: int = 64, alpha: int = 128):
    """Load LoRA weights from a single .safetensors file into the model.

    The model must already have LoRA injected (or will be injected here).

    Args:
        model: The base DiT model
        lora_path: Path to .safetensors file
        rank: LoRA rank (needed if LoRA not yet injected)
        alpha: LoRA alpha

    Returns:
        Model with LoRA weights loaded
    """
    from safetensors.torch import load_file

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    # Check if LoRA is already injected
    has_lora = any("lora_" in name for name, _ in model.named_parameters())
    if not has_lora:
        model, _ = inject_lora(model, rank=rank, alpha=alpha)

    # Load weights
    lora_state = load_file(lora_path)

    # Build a mapping from clean keys -> model keys.
    # Saved files use clean keys (no adapter name), but the model
    # uses keys with the adapter name segment (e.g. .lora_A.lora_adapter.weight).
    model_state = model.state_dict()
    adapter_name = "lora_adapter"

    loaded = 0
    for name, param in lora_state.items():
        # Convert saved key to model key by stripping prefix and adding decoder.
        # Handle multiple formats:
        #   base_model.model.layers.X... → decoder.layers.X...  (official format)
        #   diffusion_model.decoder.layers.X... → decoder.layers.X...  (old format)
        #   diffusion_model.layers.X... → decoder.layers.X...  (legacy)
        base_name = name
        if base_name.startswith("base_model.model."):
            base_name = "decoder." + base_name[len("base_model.model."):]
        elif base_name.startswith("diffusion_model."):
            base_name = base_name[len("diffusion_model."):]

        if base_name in model_state:
            model_state[base_name].copy_(param)
            loaded += 1
        else:
            # Try inserting adapter name: lora_A.weight -> lora_A.lora_adapter.weight
            model_key = base_name.replace(".lora_A.", f".lora_A.{adapter_name}.").replace(
                ".lora_B.", f".lora_B.{adapter_name}."
            )
            if model_key in model_state:
                model_state[model_key].copy_(param)
                loaded += 1
            else:
                print(f"[AceStep LoRA] Key not found in model: {name}")

    print(f"[AceStep LoRA] Loaded {loaded}/{len(lora_state)} LoRA tensors from {lora_path}")
    return model
