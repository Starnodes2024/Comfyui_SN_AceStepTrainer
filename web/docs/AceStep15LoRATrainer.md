# ⭐SN AceStep 1.5 LoRA Trainer

## Description
Trains LoRA (Low-Rank Adaptation) adapters on the ACE-Step DiT decoder using preprocessed tensor files. The node **blocks ComfyUI** during training and logs detailed progress (epoch, step, loss, learning rate) to the CMD console window. LoRA checkpoints are saved as single `.safetensors` files at configurable intervals, so you can stop training early and still use intermediate results.

## Inputs

### Required
- **existing_dataset**: Dropdown of datasets that have been preprocessed (contain a `tensors/` subfolder with `.pt` files).
- **lora_name**: Base name for the output LoRA files. Checkpoints are named `{lora_name}_{step}steps.safetensors` and the final file is `{lora_name}_final.safetensors`.
- **lora_rank**: LoRA rank (default: 64). Higher rank = more expressive capacity but larger file. Common values: 8, 16, 32, 64, 128.
- **lora_alpha**: LoRA alpha scaling factor (default: 128). A good rule of thumb is 2x the rank.
- **learning_rate**: Initial learning rate (default: 1e-4). Lower values (5e-5) for larger datasets, higher (2e-4) for very small datasets.
- **epochs**: Number of full passes through your dataset (default: 100). One epoch = every sample seen once. The total optimizer steps are calculated automatically. See "Epochs vs Steps" below.
- **batch_size**: Number of samples per training step (default: 1). Keep at 1 for small datasets or limited VRAM.
- **gradient_accumulation**: Number of steps to accumulate gradients before updating weights (default: 4). Effective batch size = batch_size x gradient_accumulation.
- **save_every_n_epochs**: Save a LoRA checkpoint every N epochs (default: 10). Set to 0 to only save the final result. Files are named like `my_lora_250steps.safetensors`.
- **seed**: Random seed for reproducibility (default: 42).

### Optional
- **tensor_path**: String input that overrides the dropdown. Connect this from the Preprocessor node's `tensor_path` output for a seamless pipeline.
- **resume_from_checkpoint**: Resume training from a previously saved checkpoint. Checkpoints (`.pt` files) are saved alongside LoRA `.safetensors` at every save interval. Select one to continue training from that step instead of starting over. The LoRA config (rank, alpha) should match the original training run.

## Outputs
- **status**: Summary of the training run (total steps, time, final LoRA path).

## Usage
1. Make sure you have preprocessed your dataset (Preprocessor node)
2. Select the dataset from the dropdown, or connect from Preprocessor
3. Configure training parameters (see Training Tips below)
4. Queue the node — **ComfyUI will be blocked** until training finishes
5. Watch the CMD console for live progress updates
6. Find your LoRA files in `output/AceLora/Trained/{lora_name}/`

## Epochs vs Steps

The trainer uses **epochs** as the primary training duration setting. Here's how they relate:

- **Epoch**: One full pass through your entire dataset. If you have 10 audio files, one epoch processes all 10.
- **Step**: One optimizer weight update. With `batch_size=1` and `gradient_accumulation=4`, one step processes 4 samples.
- **Steps per epoch**: `ceil(num_samples / batch_size) / gradient_accumulation`. For example, 10 samples with batch=1 and grad_accum=4 gives ~3 steps per epoch.
- **Total steps**: `epochs x steps_per_epoch`. The loss graph shows individual steps for fine-grained monitoring.

The CMD console and loss graph always display the current **step** count, so you can monitor training at a granular level even though you set **epochs** in the node.

## Training Tips

### Small Datasets (3-10 files)
- Rank: 32-64
- Epochs: 100-200
- Learning rate: 1e-4
- Gradient accumulation: 4-8

### Medium Datasets (10-50 files)
- Rank: 64-128
- Epochs: 50-100
- Learning rate: 5e-5
- Gradient accumulation: 4

### General Advice
- **Start with `save_every_n_epochs: 10`** so you can test early checkpoints
- **Watch the loss** in the CMD console — it should steadily decrease
- If loss stops decreasing, you can stop training and use the last saved checkpoint
- Higher rank captures more detail but risks overfitting on small datasets
- Use an **activation tag** in Dataset Builder so the LoRA only triggers with your keyword

## Console Output Example
```
[AceStep Train] Starting training:
  Device: cuda, Precision: torch.bfloat16
  Samples: 5, Batch: 1, Grad accum: 4
  Max steps: 1000, Steps/epoch: 2
  LR: 0.0001, Warmup: 100 steps
  Save every: 250 steps
======================================================================
[AceStep Train] Step 10/1000 | Loss: 0.043521 | LR: 1.00e-05 | 2.15 steps/s | Elapsed: 0m04s | ETA: 7m42s
[AceStep Train] Checkpoint saved: my_lora_250steps.safetensors + my_lora_250steps_checkpoint.pt
...
[AceStep Train] Training complete!
  Total steps: 1000
  Final LoRA: output/AceLora/Trained/my_lora/my_lora_final.safetensors
```

## Technical Details
- LoRA is injected into DiT decoder attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Training uses flow matching loss with 8 discrete turbo timesteps
- Optimizer: AdamW (default betas 0.9, 0.999) with linear warmup + cosine annealing schedule
- Mixed precision: bfloat16 autocast on CUDA
- Requires ~8 GB VRAM minimum (DiT model + LoRA gradients)

## Using Your Trained LoRA
The output `.safetensors` files can be loaded directly in ComfyUI using the standard **LoRA Loader** node from any ACE-Step inference node pack (e.g. ComfyUI_RH_ACE-Step). Simply:
1. Copy or move your `my_lora_final.safetensors` (or any checkpoint) into the LoRA path expected by your ACE-Step node pack
2. In your ACE-Step generation workflow, use the LoRA loading option and select your trained file
3. If you used an **activation tag** during dataset building, include that tag in your caption/prompt to activate the LoRA style
4. Adjust the LoRA strength (0.0–1.0) to control how strongly your style is applied

You can test different checkpoints (e.g. `my_lora_250steps.safetensors` vs `my_lora_500steps.safetensors`) to find the sweet spot between style fidelity and flexibility.

## Resume Training
Training can be paused and resumed later:
1. Train with `epochs=100`, `save_every_n_epochs=10` — checkpoints saved every 10 epochs
2. Stop training (or let it finish early)
3. Later: select a checkpoint from the `resume_from_checkpoint` dropdown (e.g. `my_lora/my_lora_500steps_checkpoint.pt`)
4. Set `epochs=100` again — training continues from the saved step with the same optimizer state

Checkpoint `.pt` files contain: LoRA weights, optimizer state, LR scheduler state, RNG states, and training config.

## Notes
- The node **intentionally blocks** ComfyUI to prevent GPU conflicts during training
- Progress is shown via the **⭐SN AceStep Loss Graph** node in real-time and printed to the CMD console
- If you close ComfyUI during training, the last saved checkpoint is still usable
- Intermediate checkpoints let you compare quality at different training stages — earlier checkpoints are more flexible, later ones capture more detail
