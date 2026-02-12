"""
Model loader for AceStep 1.5 LoRA Trainer.
Loads DiT, VAE, TextEncoder, and LLM using HuggingFace APIs
with ComfyUI-friendly memory management.
"""

import gc
import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from diffusers.models import AutoencoderOobleck

from .model_downloader import get_models_dir, download_component, MAIN_COMPONENTS

logger = logging.getLogger("AceStepTrainer")

# Global cache to avoid reloading models repeatedly
_model_cache = {}


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_dtype(device: torch.device) -> torch.dtype:
    """Get appropriate dtype for device."""
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def clear_cache():
    """Clear all cached models and free memory."""
    global _model_cache
    _model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[AceStep] Model cache cleared")


def load_vae(device: Optional[torch.device] = None) -> AutoencoderOobleck:
    """Load the AceStep VAE (AutoencoderOobleck from diffusers).

    Returns VAE model on the specified device.
    """
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    if "vae" in _model_cache:
        _model_cache["vae"] = _model_cache["vae"].to(device).to(dtype)
        return _model_cache["vae"]

    vae_path = download_component("vae")
    print(f"[AceStep] Loading VAE from {vae_path}...")
    vae = AutoencoderOobleck.from_pretrained(str(vae_path))
    vae = vae.to(device).to(dtype)
    vae.eval()

    _model_cache["vae"] = vae
    logger.info(f"[AceStep] VAE loaded on {device} ({dtype})")
    return vae


def load_text_encoder(device: Optional[torch.device] = None) -> Tuple[AutoModel, AutoTokenizer]:
    """Load the AceStep text encoder (Qwen3-Embedding-0.6B).

    Returns (text_encoder, text_tokenizer) tuple.
    """
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    if "text_encoder" in _model_cache:
        _model_cache["text_encoder"] = _model_cache["text_encoder"].to(device).to(dtype)
        return _model_cache["text_encoder"], _model_cache["text_tokenizer"]

    te_path = download_component("text_encoder")
    print(f"[AceStep] Loading text encoder from {te_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(te_path))
    encoder = AutoModel.from_pretrained(str(te_path))
    encoder = encoder.to(device).to(dtype)
    encoder.eval()

    _model_cache["text_encoder"] = encoder
    _model_cache["text_tokenizer"] = tokenizer
    logger.info(f"[AceStep] Text encoder loaded on {device} ({dtype})")
    return encoder, tokenizer


def load_dit(device: Optional[torch.device] = None, config_name: str = "dit_turbo",
             quantization_config=None):
    """Load the AceStep DiT model (uses trust_remote_code for custom architecture).

    Args:
        device: Target device
        config_name: Which DiT to load ('dit_turbo' for training target)
        quantization_config: Optional BitsAndBytesConfig for 4-bit/8-bit loading (QLoRA)

    Returns:
        DiT model
    """
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    # Don't use cache for quantized models (they are training-only, one-shot)
    if quantization_config is None:
        cache_key = f"dit_{config_name}"
        if cache_key in _model_cache:
            _model_cache[cache_key] = _model_cache[cache_key].to(device).to(dtype)
            return _model_cache[cache_key]

    dit_path = download_component(config_name)

    if quantization_config is not None:
        # Quantized loading: device_map places model on GPU, skip manual .to()
        device_idx = device.index if device.index is not None else 0
        print(f"[AceStep] Loading DiT model (quantized) from {dit_path}...")
        model = AutoModel.from_pretrained(
            str(dit_path),
            trust_remote_code=True,
            attn_implementation="sdpa",
            dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
            quantization_config=quantization_config,
            device_map={"": device_idx},
        )
        # Do NOT call model.to(device).to(dtype) — breaks quantized tensors
        logger.info(f"[AceStep] DiT model loaded (quantized) on cuda:{device_idx}")
    else:
        print(f"[AceStep] Loading DiT model from {dit_path}...")
        model = AutoModel.from_pretrained(
            str(dit_path),
            trust_remote_code=True,
            attn_implementation="sdpa",
            dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
        )
        model = model.to(device).to(dtype)
        model.eval()

    # Load silence latent
    silence_path = dit_path / "silence_latent.pt"
    silence_latent = None
    if silence_path.exists():
        silence_latent = torch.load(str(silence_path), weights_only=True).transpose(1, 2)
        silence_latent = silence_latent.to(device).to(dtype)
        logger.info("[AceStep] Silence latent loaded")

    # Only cache non-quantized models
    if quantization_config is None:
        cache_key = f"dit_{config_name}"
        _model_cache[cache_key] = model
    _model_cache[f"dit_{config_name}_silence"] = silence_latent
    logger.info(f"[AceStep] DiT model loaded on {device} ({dtype})")
    return model


def get_silence_latent(config_name: str = "dit_turbo") -> Optional[torch.Tensor]:
    """Get the cached silence latent for a DiT model."""
    return _model_cache.get(f"dit_{config_name}_silence")


def load_dit_default(device: Optional[torch.device] = None):
    """Load the default DiT model (from main repo, used for audio encoding/tokenization).

    This is different from dit_turbo (the training target).
    The default DiT is in the main model repo under 'acestep-v15-turbo'.
    """
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    cache_key = "dit_default"
    if cache_key in _model_cache:
        _model_cache[cache_key] = _model_cache[cache_key].to(device).to(dtype)
        return _model_cache[cache_key]

    from .model_downloader import MAIN_MODEL_REPO

    models_dir = get_models_dir()
    dit_name = MAIN_COMPONENTS["dit_turbo"]
    dit_path = models_dir / dit_name

    # Download if needed — check for actual model checkpoint, not just silence_latent.pt
    has_checkpoint = (dit_path / "model.safetensors").exists() or (dit_path / "pytorch_model.bin").exists()
    if not has_checkpoint:
        from huggingface_hub import snapshot_download
        print(f"[AceStep] Downloading default DiT weights from {MAIN_MODEL_REPO}...")
        snapshot_download(
            repo_id=MAIN_MODEL_REPO,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[f"{dit_name}/**"],
        )
        print(f"[AceStep] Default DiT downloaded to {dit_path}")

    print(f"[AceStep] Loading default DiT from {dit_path}...")
    model = AutoModel.from_pretrained(
        str(dit_path),
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
    )
    model = model.to(device).to(dtype)
    model.eval()

    # Load silence latent
    silence_path = dit_path / "silence_latent.pt"
    if silence_path.exists():
        silence_latent = torch.load(str(silence_path), weights_only=True).transpose(1, 2)
        silence_latent = silence_latent.to(device).to(dtype)
        _model_cache[f"{cache_key}_silence"] = silence_latent

    _model_cache[cache_key] = model
    logger.info(f"[AceStep] Default DiT loaded on {device} ({dtype})")
    return model


def load_llm(device: Optional[torch.device] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the AceStep LLM (acestep-5Hz-lm-1.7B) for audio understanding/labeling.

    Returns (llm_model, llm_tokenizer) tuple.
    """
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    if "llm" in _model_cache:
        _model_cache["llm"] = _model_cache["llm"].to(device).to(dtype)
        return _model_cache["llm"], _model_cache["llm_tokenizer"]

    lm_path = download_component("lm")
    print(f"[AceStep] Loading LLM from {lm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(lm_path))
    llm = AutoModelForCausalLM.from_pretrained(
        str(lm_path),
        torch_dtype=dtype,
    )
    llm = llm.to(device)
    llm.eval()

    _model_cache["llm"] = llm
    _model_cache["llm_tokenizer"] = tokenizer
    logger.info(f"[AceStep] LLM loaded on {device} ({dtype})")
    return llm, tokenizer


def offload_to_cpu(*model_keys: str):
    """Move specified cached models to CPU to free GPU memory."""
    for key in model_keys:
        if key in _model_cache and hasattr(_model_cache[key], "to"):
            _model_cache[key] = _model_cache[key].to("cpu")
            logger.info(f"[AceStep] Offloaded {key} to CPU")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_to_device(*model_keys: str, device: Optional[torch.device] = None):
    """Move specified cached models back to GPU."""
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)
    for key in model_keys:
        if key in _model_cache and hasattr(_model_cache[key], "to"):
            _model_cache[key] = _model_cache[key].to(device).to(dtype)
            logger.info(f"[AceStep] Moved {key} to {device}")
