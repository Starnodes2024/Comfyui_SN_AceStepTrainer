"""
Model downloader for AceStep 1.5 LoRA Trainer.
Auto-downloads required models from HuggingFace to ComfyUI models/AceStep/ directory.

Downloads are stored directly as real files (no symlinks, no HF cache dependency).
Existence checks look for actual model files (.safetensors, config.json, etc.)
to avoid triggering unnecessary re-downloads.
"""

import logging
from pathlib import Path

logger = logging.getLogger("AceStepTrainer")

# HuggingFace repos
MAIN_MODEL_REPO = "ACE-Step/Ace-Step1.5"
TURBO_DIT_REPO = "ACE-Step/acestep-v15-turbo-shift3"

# Components inside the main repo
MAIN_COMPONENTS = {
    "vae": "vae",
    "text_encoder": "Qwen3-Embedding-0.6B",
    "lm": "acestep-5Hz-lm-1.7B",
    "dit_turbo": "acestep-v15-turbo",
}

# File extensions that indicate a model is actually downloaded (not just metadata)
MODEL_FILE_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}


def get_models_dir() -> Path:
    """Get ComfyUI models/AceStep/ directory."""
    # Walk up from this file to find ComfyUI/models/
    current = Path(__file__).resolve()
    # Go up: core/ -> ComfyUI-AceStep_15_Trainer/ -> custom_nodes/ -> ComfyUI/
    comfyui_root = current.parent.parent.parent.parent
    models_dir = comfyui_root / "models" / "AceStep"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_output_base_dir() -> Path:
    """Get ComfyUI output/AceLora/ directory."""
    current = Path(__file__).resolve()
    comfyui_root = current.parent.parent.parent.parent
    output_dir = comfyui_root / "output" / "AceLora"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_dataset_dir(name: str = "") -> Path:
    """Get dataset directory. If name provided, returns specific dataset dir."""
    base = get_output_base_dir() / "Dataset"
    base.mkdir(parents=True, exist_ok=True)
    if name:
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        return d
    return base


def get_trained_dir(name: str = "") -> Path:
    """Get trained LoRA output directory."""
    base = get_output_base_dir() / "Trained"
    base.mkdir(parents=True, exist_ok=True)
    if name:
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        return d
    return base


def list_datasets_with_audio() -> list:
    """List dataset names that have audio + txt files."""
    base = get_dataset_dir()
    datasets = []
    if not base.exists():
        return datasets
    for d in sorted(base.iterdir()):
        if d.is_dir():
            has_audio = any(
                f.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".opus"}
                for f in d.iterdir() if f.is_file()
            )
            if has_audio:
                datasets.append(d.name)
    return datasets


def list_datasets_with_tensors() -> list:
    """List dataset names that have preprocessed .pt tensor files."""
    base = get_dataset_dir()
    datasets = []
    if not base.exists():
        return datasets
    for d in sorted(base.iterdir()):
        if d.is_dir():
            tensors_dir = d / "tensors"
            if tensors_dir.exists():
                has_pt = any(f.suffix == ".pt" for f in tensors_dir.iterdir() if f.is_file())
                if has_pt:
                    datasets.append(d.name)
    return datasets


def list_training_checkpoints() -> list:
    """List available training checkpoint .pt files for resume.

    Scans all subdirectories under Trained/ for *_checkpoint.pt files.
    Returns list of strings like 'my_lora/my_lora_500steps_checkpoint.pt'
    (relative to Trained/).
    """
    base = get_trained_dir()
    checkpoints = []
    if not base.exists():
        return checkpoints
    for d in sorted(base.iterdir()):
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.name.endswith("_checkpoint.pt"):
                    checkpoints.append(f"{d.name}/{f.name}")
    return checkpoints


def _has_model_weights(folder: Path) -> bool:
    """Check if a folder contains actual model weight files.

    Looks for .safetensors, .bin, .pt files (not just config/metadata).
    Also checks subdirectories one level deep for sharded models.
    """
    if not folder.exists() or not folder.is_dir():
        return False

    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in MODEL_FILE_EXTENSIONS:
            return True
        # Check one level of subdirs for sharded models
        if f.is_dir():
            for sf in f.iterdir():
                if sf.is_file() and sf.suffix.lower() in MODEL_FILE_EXTENSIONS:
                    return True
    return False


def _has_config_file(folder: Path) -> bool:
    """Check if a folder has a config.json or similar config file."""
    if not folder.exists():
        return False
    for name in ["config.json", "model_index.json", "tokenizer_config.json"]:
        if (folder / name).exists():
            return True
    return False


def _component_is_complete(comp_path: Path) -> bool:
    """Check if a model component is fully downloaded.

    A component is considered complete if it has both:
    1. A config file (config.json, tokenizer_config.json, etc.)
    2. Actual model weight files (.safetensors, .bin, etc.)

    For components like the LM that only have weight files, just weight check suffices.
    """
    if not comp_path.exists():
        return False

    has_weights = _has_model_weights(comp_path)
    has_config = _has_config_file(comp_path)

    # Must have at least weight files. Config is a bonus check.
    # Some components (silence_latent.pt in DiT) are just single .pt files
    # so we check for any non-empty content as fallback.
    if has_weights:
        return True
    if has_config:
        # Config exists but no weights yet — might be partial download
        return False

    # Fallback: check if directory has any substantial files (>1KB)
    for f in comp_path.rglob("*"):
        if f.is_file() and f.stat().st_size > 1024:
            return True
    return False


def download_component(component_key: str, force: bool = False) -> Path:
    """Download a specific model component if not already present.

    Files are stored directly in models/AceStep/ as real files.
    Uses local_dir_use_symlinks=False to avoid HF cache symlink issues.
    Checks for actual model weight files before deciding to download.

    Args:
        component_key: One of 'vae', 'text_encoder', 'lm', 'dit_turbo'
        force: Force re-download even if files exist

    Returns:
        Path to the downloaded component directory
    """
    from huggingface_hub import snapshot_download

    models_dir = get_models_dir()

    if component_key == "dit_turbo":
        # DiT turbo is in its own separate repo
        local_name = "acestep-v15-turbo-shift3"
        comp_path = models_dir / local_name

        if not force and _component_is_complete(comp_path):
            print(f"[AceStep] DiT turbo already present at {comp_path}")
            return comp_path

        print("[AceStep] Downloading DiT turbo model (~2GB)... This may take a while.")
        snapshot_download(
            repo_id=TURBO_DIT_REPO,
            local_dir=str(comp_path),
            local_dir_use_symlinks=False,
        )
        print(f"[AceStep] DiT turbo downloaded to {comp_path}")
        return comp_path

    elif component_key in MAIN_COMPONENTS:
        subfolder = MAIN_COMPONENTS[component_key]
        comp_path = models_dir / subfolder

        if not force and _component_is_complete(comp_path):
            print(f"[AceStep] {component_key} already present at {comp_path}")
            return comp_path

        size_hints = {
            "vae": "~300MB", "text_encoder": "~1.2GB",
            "lm": "~3.4GB", "dit_turbo": "~2GB",
        }
        hint = size_hints.get(component_key, "")
        print(f"[AceStep] Downloading {component_key} ({hint})... This may take a while.")

        # Download only the specific subfolder from the main repo
        # Files go directly into models/AceStep/{subfolder}/ as real copies
        snapshot_download(
            repo_id=MAIN_MODEL_REPO,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[f"{subfolder}/**"],
        )
        print(f"[AceStep] {component_key} downloaded to {comp_path}")
        return comp_path

    else:
        valid = list(MAIN_COMPONENTS.keys()) + ["dit_turbo"]
        raise ValueError(f"Unknown component: {component_key}. Valid: {valid}")


def ensure_models_for_labeling() -> dict:
    """Ensure LM + VAE + DiT models are available for auto-labeling.

    Returns dict with paths to each component.
    """
    paths = {}
    paths["lm"] = download_component("lm")
    paths["vae"] = download_component("vae")
    # DiT default (for audio code extraction via tokenizer)
    dit_subfolder = MAIN_COMPONENTS["dit_turbo"]
    dit_path = get_models_dir() / dit_subfolder
    if not _component_is_complete(dit_path):
        from huggingface_hub import snapshot_download
        print("[AceStep] Downloading default DiT model... This may take a while.")
        snapshot_download(
            repo_id=MAIN_MODEL_REPO,
            local_dir=str(get_models_dir()),
            local_dir_use_symlinks=False,
            allow_patterns=[f"{dit_subfolder}/**"],
        )
    paths["dit"] = dit_path
    return paths


def ensure_models_for_preprocessing() -> dict:
    """Ensure VAE + TextEncoder + DiT are available for preprocessing.

    Returns dict with paths to each component.
    """
    paths = {}
    paths["vae"] = download_component("vae")
    paths["text_encoder"] = download_component("text_encoder")
    paths["dit"] = download_component("dit_turbo")
    # Also need the default DiT from main repo for the condition encoder
    dit_subfolder = MAIN_COMPONENTS["dit_turbo"]
    dit_default_path = get_models_dir() / dit_subfolder
    if not _component_is_complete(dit_default_path):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MAIN_MODEL_REPO,
            local_dir=str(get_models_dir()),
            local_dir_use_symlinks=False,
            allow_patterns=[f"{dit_subfolder}/**"],
        )
    paths["dit_default"] = dit_default_path
    return paths


def ensure_models_for_training() -> dict:
    """Ensure DiT turbo (shift3) model is available for LoRA training.

    Returns dict with path to DiT model.
    """
    paths = {}
    paths["dit"] = download_component("dit_turbo")
    return paths
