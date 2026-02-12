"""
Node 3: AceStep 1.5 LoRA Trainer
Trains LoRA adapters on the DiT decoder using preprocessed tensor files.
Blocks ComfyUI while training, logs progress to CMD console.
Saves LoRA as single-file .safetensors at intervals and at completion.
"""

import os
import math
import logging

from comfy.utils import ProgressBar

from ..core.model_downloader import get_dataset_dir, get_trained_dir, list_datasets_with_tensors, list_training_checkpoints
from ..core.training import train_lora

logger = logging.getLogger("AceStepTrainer")


class AceStep15LoRATrainer:
    """Trains LoRA on AceStep DiT decoder. Blocks node, logs to CMD."""

    @classmethod
    def INPUT_TYPES(cls):
        datasets = ["(none)"] + list_datasets_with_tensors()
        checkpoints = ["(none — train from scratch)"] + list_training_checkpoints()

        return {
            "required": {
                "existing_dataset": (datasets, {
                    "tooltip": "Pick a dataset that has been preprocessed (must have a tensors/ subfolder with .pt files). If you just preprocessed but don't see it here, restart ComfyUI to refresh the list, or connect tensor_path from the Preprocessor node.",
                }),
                "lora_name": ("STRING", {
                    "default": "my_lora",
                    "multiline": False,
                    "tooltip": "Base name for your LoRA output files. Checkpoints are saved as '{name}_{step}steps.safetensors' and the final result as '{name}_final.safetensors'. The trained LoRA safetensors can be loaded with the LoRA loader in any ACE-Step ComfyUI node pack.",
                }),
                "lora_rank": ("INT", {
                    "default": 64,
                    "min": 4,
                    "max": 256,
                    "step": 4,
                    "tooltip": "Controls how much the LoRA can learn. Higher rank = more capacity to capture your style, but larger file and higher risk of overfitting on small datasets. Recommended: 16-32 for subtle styles, 64 for general use, 128+ for complex styles with large datasets.",
                }),
                "lora_alpha": ("INT", {
                    "default": 128,
                    "min": 4,
                    "max": 512,
                    "step": 4,
                    "tooltip": "Scaling factor that controls how strongly the LoRA influences the model. A good rule of thumb is to set alpha = 2x rank. Higher alpha makes the LoRA effect stronger during training. Example: rank 64 -> alpha 128.",
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0001,
                    "min": 0.0000001,
                    "max": 0.01,
                    "step": 0.00001,
                    "tooltip": "How fast the model learns. Too high = unstable training, too low = very slow progress. Default 1e-4 (0.0001) is a safe starting point. Try 5e-5 for larger datasets or 2e-4 for very small ones (3-5 files). Watch the loss in the CMD console — if it spikes, lower this.",
                }),
                "epochs": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Number of full passes through your dataset. One epoch = every sample seen once. The total optimizer steps are calculated automatically: steps = epochs × ceil(samples / batch_size) / gradient_accumulation. For small datasets (3-10 files) try 50-200 epochs. For larger ones (10-50 files) try 20-100. The loss graph shows individual steps for fine-grained monitoring.",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of samples processed together in one step. Keep at 1 unless you have lots of VRAM (24GB+) and a large dataset. Higher batch sizes use more VRAM but can speed up training. Use gradient_accumulation instead if you want a larger effective batch without extra VRAM.",
                }),
                "gradient_accumulation": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Simulates a larger batch size without extra VRAM. Gradients are accumulated over this many steps before updating weights. Effective batch = batch_size × this value. Default 4 means the model updates after seeing 4 samples. Increase to 8-16 for smoother, more stable training.",
                }),
                "save_every_n_epochs": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Saves a LoRA checkpoint every N epochs (e.g. my_lora_250steps.safetensors). This way you can stop training anytime and still have usable checkpoints. Set to 0 to only save the final result. Tip: start with 10-25 so you can compare quality at different stages.",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 4294967295,
                    "tooltip": "Random seed for reproducible results. Using the same seed with the same settings and data produces identical training. Change this if you want to try a different random initialization.",
                }),
                "quantization_mode": (["none", "load_in_8bit", "load_in_4bit"], {
                    "tooltip": "QLoRA: Load the base model in reduced precision to save VRAM. 'none' = full bfloat16 (~4GB model). 'load_in_8bit' = 8-bit quantization (~2GB model). 'load_in_4bit' = 4-bit NF4 quantization (~1.2GB model). Requires: pip install bitsandbytes accelerate. CUDA only. LoRA weights stay in full precision — only the frozen base model is quantized.",
                }),
            },
            "optional": {
                "tensor_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect this from the Preprocessor node's 'tensor_path' output to automatically use that dataset's tensors. When connected, this overrides the dropdown selection above.",
                }),
                "resume_from_checkpoint": (checkpoints, {
                    "tooltip": "Resume training from a saved checkpoint. Checkpoints are saved alongside LoRA files at every save interval. Select one to continue training from that step instead of starting over. The LoRA config (rank, alpha) should match the original training run.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "train"
    CATEGORY = "AceStep/Training"
    OUTPUT_NODE = True

    def train(self, existing_dataset, lora_name, lora_rank, lora_alpha,
              learning_rate, epochs, batch_size, gradient_accumulation,
              save_every_n_epochs, seed, quantization_mode="none",
              tensor_path=None, resume_from_checkpoint=None):

        # Determine tensor directory
        if tensor_path and os.path.isdir(tensor_path):
            tensor_dir = tensor_path
        elif existing_dataset and existing_dataset not in ("(none)", "(no preprocessed datasets found)"):
            tensor_dir = str(get_dataset_dir(existing_dataset) / "tensors")
        else:
            return ("ERROR: No preprocessed dataset selected. Run Preprocessor first or select from dropdown.",)

        if not os.path.isdir(tensor_dir):
            return (f"ERROR: Tensor directory not found: {tensor_dir}",)

        # Check for .pt files
        pt_files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        if not pt_files:
            return (f"ERROR: No .pt tensor files found in {tensor_dir}. Run Preprocessor first.",)

        # Calculate total batches for progress bar display
        num_samples = len(pt_files)
        batches_per_epoch = math.ceil(num_samples / batch_size)
        total_batches = epochs * batches_per_epoch

        # Output directory
        output_dir = str(get_trained_dir(lora_name))

        # Resolve checkpoint path
        resume_path = None
        if resume_from_checkpoint and resume_from_checkpoint != "(none — train from scratch)":
            trained_base = get_trained_dir()
            resume_path = str(trained_base / resume_from_checkpoint)
            if not os.path.isfile(resume_path):
                return (f"ERROR: Checkpoint file not found: {resume_path}",)

        print("=" * 70)
        print(f"[AceStep Train] LoRA Training Configuration:")
        print(f"  Dataset: {tensor_dir} ({num_samples} samples)")
        print(f"  Output: {output_dir}")
        print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
        print(f"  Epochs: {epochs}, Batches/epoch: {batches_per_epoch}, Total steps: {total_batches}")
        print(f"  Batch: {batch_size}, Grad accum: {gradient_accumulation}")
        print(f"  LR: {learning_rate}, Save every: {save_every_n_epochs} epochs")
        if quantization_mode != "none":
            print(f"  Quantization: {quantization_mode} (QLoRA)")
        if resume_path:
            print(f"  Resume from: {resume_path}")
        print("=" * 70)

        # Create ComfyUI progress bar (shows on the node during training)
        pbar = ProgressBar(total_batches)
        pbar_pos = 0

        def progress_callback(step, total):
            nonlocal pbar_pos
            delta = step - pbar_pos
            if delta > 0:
                pbar.update(delta)
                pbar_pos = step

        # Run training (this blocks until complete)
        status = train_lora(
            tensor_dir=tensor_dir,
            output_dir=output_dir,
            lora_name=lora_name,
            rank=lora_rank,
            alpha=lora_alpha,
            learning_rate=learning_rate,
            num_epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            save_every_n_epochs=save_every_n_epochs,
            seed=seed,
            num_workers=0,  # 0 avoids Windows multiprocessing issues
            progress_callback=progress_callback,
            resume_checkpoint=resume_path,
            quantization_mode=quantization_mode,
        )

        return (status,)
