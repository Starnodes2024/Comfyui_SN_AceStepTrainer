"""
Node 2: AceStep 1.5 Dataset Preprocessor
Encodes audio + text metadata into .pt tensor files for training.
Can receive dataset_path from Node 1 or user selects from dropdown.
"""

import os
import logging

from ..core.model_downloader import get_dataset_dir, list_datasets_with_audio
from ..core.preprocessing import preprocess_dataset
from ..core.model_loader import offload_to_cpu

logger = logging.getLogger("AceStepTrainer")


class AceStep15DatasetPreprocessor:
    """Preprocesses audio+txt dataset into .pt tensor files for LoRA training."""

    @classmethod
    def INPUT_TYPES(cls):
        datasets = ["(none)"] + list_datasets_with_audio()

        return {
            "required": {
                "existing_dataset": (datasets, {
                    "tooltip": "Pick a dataset you created with the Dataset Builder. Only datasets that contain audio files are shown. If you just created one but don't see it, restart ComfyUI to refresh the list, or connect the dataset_path from Dataset Builder directly.",
                }),
                "max_duration_seconds": ("FLOAT", {
                    "default": 240.0,
                    "min": 10.0,
                    "max": 600.0,
                    "step": 10.0,
                    "tooltip": "Maximum audio length in seconds. Files longer than this are trimmed to fit. Shorter files are used as-is. Default 240s (4 minutes) works well for most music. Longer audio uses more VRAM during encoding.",
                }),
                "custom_activation_tag": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional trigger word added to the front of every caption during encoding. Useful if you forgot to set it in the Dataset Builder, or want to override it. Leave empty to keep the captions exactly as they are in your .txt files.",
                }),
            },
            "optional": {
                "dataset_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect this from the Dataset Builder node's 'dataset_path' output to automatically use that dataset. When connected, this overrides the dropdown selection above.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("status", "tensor_path",)
    FUNCTION = "preprocess"
    CATEGORY = "AceStep/Training"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run so deleted .pt files are detected for reprocessing
        return float("NaN")

    def preprocess(self, existing_dataset, max_duration_seconds, custom_activation_tag,
                   dataset_path=None):
        # Determine which dataset to use
        if dataset_path and os.path.isdir(dataset_path):
            target_dir = dataset_path
        elif existing_dataset and existing_dataset not in ("(none)", "(no datasets found)"):
            target_dir = str(get_dataset_dir(existing_dataset))
        else:
            return ("ERROR: No dataset selected. Either connect Dataset Builder or select from dropdown.", "")

        if not os.path.isdir(target_dir):
            return (f"ERROR: Dataset directory not found: {target_dir}", "")

        print(f"[AceStep Preprocess] Starting preprocessing for: {target_dir}")

        # Run preprocessing
        status = preprocess_dataset(
            dataset_dir=target_dir,
            custom_tag=custom_activation_tag,
            max_duration=max_duration_seconds,
        )

        # Determine tensor output path
        tensor_path = os.path.join(target_dir, "tensors")
        if not os.path.isdir(tensor_path):
            tensor_path = ""

        # Offload models after preprocessing
        offload_to_cpu("vae", "text_encoder", "dit_dit_turbo")

        return (status, tensor_path)
