"""
Flow matching LoRA training loop for AceStep 1.5.
Pure PyTorch implementation (no Lightning Fabric dependency).
Trains LoRA adapters on the DiT decoder using preprocessed .pt tensor files.
"""

import os
import gc
import math
import time
import random
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from .lora_utils import inject_lora, save_lora_safetensors
from .model_loader import load_dit, _get_device, _get_dtype

logger = logging.getLogger("AceStepTrainer")


def _send_training_event(data: dict):
    """Send a training update event to the frontend via ComfyUI WebSocket."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("ace_step_training_update", data)
    except Exception:
        pass  # Server may not be available in all contexts

# Discrete timesteps for turbo model (shift=3.0, 8 steps)
# The turbo model was distilled to work at exactly these timesteps.
# Training MUST use these same timesteps so LoRA adaptations transfer to inference.
TURBO_SHIFT3_TIMESTEPS = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]


class PreprocessedTensorDataset(Dataset):
    """Dataset that loads preprocessed .pt tensor files."""

    def __init__(self, tensor_dir: str):
        self.tensor_dir = Path(tensor_dir)
        self.files = sorted([
            f for f in self.tensor_dir.iterdir()
            if f.suffix == ".pt" and f.is_file()
        ])
        if not self.files:
            raise ValueError(f"No .pt files found in {tensor_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(str(self.files[idx]), map_location="cpu", weights_only=False)
        return {
            "target_latents": data["target_latents"].float(),
            "attention_mask": data["attention_mask"].float(),
            "encoder_hidden_states": data["encoder_hidden_states"].float(),
            "encoder_attention_mask": data["encoder_attention_mask"].float(),
            "context_latents": data["context_latents"].float(),
        }


def _collate_fn(batch):
    """Collate preprocessed tensors with padding to max length in batch."""
    keys = ["target_latents", "attention_mask", "encoder_hidden_states",
            "encoder_attention_mask", "context_latents"]

    result = {}
    for key in keys:
        tensors = [b[key] for b in batch]
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad_size = max_len - t.shape[0]
                if t.dim() == 1:
                    t = F.pad(t, (0, pad_size), value=0)
                else:
                    t = F.pad(t, (0, 0, 0, pad_size), value=0)
            padded.append(t)
        result[key] = torch.stack(padded)
    return result


def _save_training_checkpoint(path, dit_model, optimizer, scheduler, global_step, epoch, config):
    """Save a full training checkpoint for resume."""
    checkpoint = {
        "global_step": global_step,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "lora_state_dict": {
            k: v.cpu() for k, v in dit_model.state_dict().items()
            if "lora_" in k
        },
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": config,
    }
    torch.save(checkpoint, path)


def train_lora(
    tensor_dir: str,
    output_dir: str,
    lora_name: str = "my_lora",
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    max_steps: int = 500,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    save_every_n_steps: int = 250,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    progress_callback=None,
    resume_checkpoint: Optional[str] = None,
) -> str:
    """Train LoRA adapters on the DiT decoder using preprocessed tensors.

    This function blocks until training is complete.
    Progress is printed to the CMD console.

    Args:
        tensor_dir: Path to directory with preprocessed .pt files
        output_dir: Directory to save LoRA checkpoints
        lora_name: Base name for saved LoRA files
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        learning_rate: Initial learning rate
        max_steps: Total number of optimizer steps to train
        batch_size: Training batch size
        gradient_accumulation: Gradient accumulation steps
        save_every_n_steps: Save checkpoint every N global steps
        warmup_steps: Warmup steps for LR scheduler
        weight_decay: AdamW weight decay
        max_grad_norm: Max gradient norm for clipping
        seed: Random seed
        num_workers: DataLoader workers (0 = main thread, avoids Windows issues)
        device: Target device
        progress_callback: Optional callback(step, total) for UI progress bar
        resume_checkpoint: Optional path to a _checkpoint.pt file to resume from

    Returns:
        Status message string
    """
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Validate tensor directory
    if not os.path.exists(tensor_dir):
        return f"ERROR: Tensor directory not found: {tensor_dir}"

    # Create dataset and dataloader
    try:
        dataset = PreprocessedTensorDataset(tensor_dir)
    except ValueError as e:
        return str(e)

    print(f"[AceStep Train] Loaded {len(dataset)} preprocessed samples from {tensor_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # CRITICAL: ComfyUI wraps node execution in torch.inference_mode() which
    # disables all gradient tracking and marks tensors as "inference tensors".
    # We must exit inference mode BEFORE loading model weights, otherwise the
    # weight tensors can't participate in autograd.
    _inference_ctx = torch.inference_mode(mode=False)
    _inference_ctx.__enter__()

    # Evict cached DiT model — if it was loaded during preprocessing (inside
    # inference_mode), ALL its tensors (params, buffers, internal state) are
    # permanently marked as inference tensors.  Cloning params alone is not
    # enough.  Force a fresh load from disk inside our non-inference context.
    from .model_loader import _model_cache
    for key in list(_model_cache.keys()):
        if key.startswith("dit_dit_turbo"):
            del _model_cache[key]
    torch.cuda.empty_cache()

    # Load DiT model (fresh, outside inference_mode)
    print("[AceStep Train] Loading DiT model...")
    dit_model = load_dit(device, "dit_turbo")

    # Inject LoRA
    print(f"[AceStep Train] Injecting LoRA (rank={rank}, alpha={alpha})...")
    dit_model, lora_info = inject_lora(
        dit_model, rank=rank, alpha=alpha, dropout=dropout,
    )

    # Enable gradient checkpointing to save VRAM (like original ACE-Step training).
    # Can't use gradient_checkpointing_enable() — it calls get_input_embeddings()
    # which AceStepDiTModel doesn't implement. Set the flag directly instead.
    dit_model.decoder.gradient_checkpointing = True

    # Setup optimizer (only LoRA parameters) — betas match original ACE-Step trainer
    trainable_params = [p for p in dit_model.parameters() if p.requires_grad]
    if not trainable_params:
        return "ERROR: No trainable parameters found after LoRA injection!"

    optimizer_kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": (0.8, 0.9),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(trainable_params, **optimizer_kwargs)

    # LR scheduler: linear warmup then linear decay (matches original ACE-Step trainer)
    batches_per_epoch = len(dataloader)
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / gradient_accumulation))
    warmup_steps = min(warmup_steps, max(1, max_steps // 10))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Ensure entire model is on correct device and dtype, then set decoder to train mode
    dit_model = dit_model.to(device=device, dtype=dtype)
    dit_model.decoder.train()

    # Resume from checkpoint if provided
    resume_step = 0
    resume_epoch = 0
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"[AceStep Train] Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, weights_only=False, map_location=device)

        # Validate config compatibility
        saved_cfg = ckpt.get("config", {})
        if saved_cfg.get("rank") != rank or saved_cfg.get("alpha") != alpha:
            print(f"  WARNING: LoRA config mismatch! Checkpoint: rank={saved_cfg.get('rank')}, alpha={saved_cfg.get('alpha')} "
                  f"vs current: rank={rank}, alpha={alpha}. Proceeding anyway but results may be unexpected.")

        # Restore LoRA weights
        lora_sd = ckpt.get("lora_state_dict", {})
        if lora_sd:
            model_sd = dit_model.state_dict()
            model_sd.update({k: v.to(device=device, dtype=dtype) for k, v in lora_sd.items()})
            dit_model.load_state_dict(model_sd, strict=False)
            print(f"  Restored {len(lora_sd)} LoRA weight tensors")

        # Restore optimizer & scheduler
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        resume_step = ckpt.get("global_step", 0)
        resume_epoch = ckpt.get("epoch", 0)

        # Restore RNG states for reproducibility
        rng = ckpt.get("rng_state", {})
        if rng.get("python"):
            random.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.random.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

        print(f"  Resumed at step {resume_step}, epoch {resume_epoch}")
        del ckpt
        gc.collect()

    print("[AceStep Train] Starting training:")
    print(f"  Device: {device}, Precision: {dtype}")
    print(f"  Samples: {len(dataset)}, Batch: {batch_size}, Grad accum: {gradient_accumulation}")
    print(f"  Max steps: {max_steps}, Steps/epoch: {steps_per_epoch}")
    print(f"  LR: {learning_rate}, Warmup: {warmup_steps} steps")
    print(f"  Save every: {save_every_n_steps} steps")
    if save_every_n_steps > 0 and save_every_n_steps > max_steps:
        print(f"  WARNING: save_every_n_steps ({save_every_n_steps}) > max_steps ({max_steps})")
        print("    Only the final LoRA will be saved. Lower save_every_n_steps for intermediate saves.")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # Build config dict for checkpoint saving
    _train_config = {
        "rank": rank, "alpha": alpha, "dropout": dropout,
        "learning_rate": learning_rate, "max_steps": max_steps,
        "batch_size": batch_size, "gradient_accumulation": gradient_accumulation,
        "warmup_steps": warmup_steps, "weight_decay": weight_decay,
        "seed": seed, "lora_name": lora_name,
    }

    # Training loop
    global_step = resume_step
    accumulation_step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    training_start = time.time()
    timesteps_tensor = torch.tensor(TURBO_SHIFT3_TIMESTEPS, device=device, dtype=dtype)

    # Fast-forward the progress bar if resuming
    if resume_step > 0 and progress_callback is not None:
        progress_callback(resume_step, max_steps)

    # Notify frontend that training has started
    _send_training_event({"type": "start", "max_steps": max_steps, "resume_step": resume_step})

    done = False
    if global_step >= max_steps:
        done = True
    epoch = resume_epoch
    while not done:
        epoch += 1

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            target_latents = batch["target_latents"].to(device, dtype=dtype, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, dtype=dtype, non_blocking=True)
            enc_hidden = batch["encoder_hidden_states"].to(device, dtype=dtype, non_blocking=True)
            enc_mask = batch["encoder_attention_mask"].to(device, dtype=dtype, non_blocking=True)

            context_latents = batch["context_latents"].to(device, dtype=dtype, non_blocking=True)

            bsz = target_latents.shape[0]

            # Flow matching: sample noise and interpolate
            x1 = torch.randn_like(target_latents)  # noise
            x0 = target_latents  # data

            # Sample timestep from discrete turbo schedule (matches original ACE-Step trainer)
            t_indices = torch.randint(0, len(TURBO_SHIFT3_TIMESTEPS), (bsz,), device=device)
            t = timesteps_tensor[t_indices]
            t_expanded = t.unsqueeze(-1).unsqueeze(-1)

            # Interpolate: x_t = t * noise + (1-t) * data
            xt = t_expanded * x1 + (1.0 - t_expanded) * x0

            # Enable grad on input so PEFT LoRA layers can compute gradients
            xt = xt.requires_grad_(True)

            # Forward through decoder (model already in bfloat16, no autocast needed)
            decoder_outputs = dit_model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                context_latents=context_latents,
            )

            # Flow matching loss: predict flow field v = x1 - x0
            flow = x1 - x0
            loss = F.mse_loss(decoder_outputs[0], flow)

            # Scale loss for gradient accumulation
            loss = loss.float() / gradient_accumulation

            # Backward
            loss.backward()
            accumulated_loss += loss.item()
            accumulation_step += 1

            # Optimizer step when accumulation is complete
            if accumulation_step >= gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                avg_loss = accumulated_loss / accumulation_step
                accumulated_loss = 0.0
                accumulation_step = 0

                # Update ComfyUI progress bar
                if progress_callback is not None:
                    progress_callback(global_step, max_steps)

                # Compute ETA and LR for logging + WebSocket
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - training_start
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                remaining_steps = max_steps - global_step
                eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_min, eta_s = divmod(int(eta_sec), 60)
                eta_h, eta_min = divmod(eta_min, 60)
                elapsed_min = int(elapsed) // 60
                elapsed_s = int(elapsed) % 60
                eta_str = f"{eta_h}h{eta_min:02d}m" if eta_h > 0 else f"{eta_min}m{eta_s:02d}s"

                # Send every step to the loss graph widget
                _send_training_event({
                    "type": "step",
                    "step": global_step,
                    "loss": avg_loss,
                    "lr": lr,
                    "eta": eta_str,
                })

                # Log progress to console (every 10 steps to avoid spam)
                if global_step % 10 == 0 or global_step == 1:
                    print(
                        f"[AceStep Train] Step {global_step}/{max_steps} | "
                        f"Loss: {avg_loss:.6f} | LR: {lr:.2e} | "
                        f"{steps_per_sec:.2f} steps/s | "
                        f"Elapsed: {elapsed_min}m{elapsed_s:02d}s | ETA: {eta_str}"
                    )

                # Save checkpoint at interval (LoRA weights + training state)
                if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                    ckpt_name = f"{lora_name}_{global_step}steps.safetensors"
                    ckpt_path = os.path.join(output_dir, ckpt_name)
                    save_lora_safetensors(dit_model, ckpt_path, alpha=alpha)
                    # Save training state for resume
                    state_name = f"{lora_name}_{global_step}steps_checkpoint.pt"
                    state_path = os.path.join(output_dir, state_name)
                    _save_training_checkpoint(state_path, dit_model, optimizer, scheduler, global_step, epoch, _train_config)
                    print(f"[AceStep Train] Checkpoint saved: {ckpt_name} + {state_name}")

                # Check if we've reached max_steps
                if global_step >= max_steps:
                    done = True
                    break

        # Flush remaining accumulated gradients at end of data pass
        if not done and accumulation_step > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            avg_loss = accumulated_loss / accumulation_step
            accumulated_loss = 0.0
            accumulation_step = 0

            # Update ComfyUI progress bar
            if progress_callback is not None:
                progress_callback(global_step, max_steps)

            # Compute ETA and LR for logging + WebSocket
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - training_start
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0
            remaining_steps = max_steps - global_step
            eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_min, eta_s = divmod(int(eta_sec), 60)
            eta_h, eta_min = divmod(eta_min, 60)
            elapsed_min = int(elapsed) // 60
            elapsed_s = int(elapsed) % 60
            eta_str = f"{eta_h}h{eta_min:02d}m" if eta_h > 0 else f"{eta_min}m{eta_s:02d}s"

            # Send every step to the loss graph widget
            _send_training_event({
                "type": "step",
                "step": global_step,
                "loss": avg_loss,
                "lr": lr,
                "eta": eta_str,
            })

            # Log progress to console (every 10 steps to avoid spam)
            if global_step % 10 == 0 or global_step == 1:
                print(
                    f"[AceStep Train] Step {global_step}/{max_steps} | "
                    f"Loss: {avg_loss:.6f} | LR: {lr:.2e} | "
                    f"{steps_per_sec:.2f} steps/s | "
                    f"Elapsed: {elapsed_min}m{elapsed_s:02d}s | ETA: {eta_str}"
                )

            # Save checkpoint at interval (LoRA weights + training state)
            if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                ckpt_name = f"{lora_name}_{global_step}steps.safetensors"
                ckpt_path = os.path.join(output_dir, ckpt_name)
                save_lora_safetensors(dit_model, ckpt_path, alpha=alpha)
                # Save training state for resume
                state_name = f"{lora_name}_{global_step}steps_checkpoint.pt"
                state_path = os.path.join(output_dir, state_name)
                _save_training_checkpoint(state_path, dit_model, optimizer, scheduler, global_step, epoch, _train_config)
                print(f"[AceStep Train] Checkpoint saved: {ckpt_name} + {state_name}")

            if global_step >= max_steps:
                done = True

    # Notify frontend that training is done
    _send_training_event({"type": "done"})

    # Save final LoRA
    total_time = time.time() - training_start
    final_name = f"{lora_name}_final.safetensors"
    final_path = os.path.join(output_dir, final_name)
    save_lora_safetensors(dit_model, final_path, alpha=alpha)

    print("=" * 70)
    print("[AceStep Train] Training complete!")
    print(f"  Total steps: {global_step}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Final LoRA: {final_path}")
    print("=" * 70)

    # Restore inference mode (re-enter ComfyUI's default context)
    _inference_ctx.__exit__(None, None, None)

    # Cleanup
    del optimizer, scheduler, dataloader, dataset, _train_config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f"Training complete! {global_step} steps in {total_time/60:.1f}min. LoRA saved to: {final_path}"
