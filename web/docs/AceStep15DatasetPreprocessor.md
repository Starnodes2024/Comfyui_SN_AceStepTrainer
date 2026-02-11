# ⭐SN AceStep 1.5 Dataset Preprocessor

## Description
Encodes audio files and their `.txt` metadata into `.pt` tensor files that the LoRA Trainer can consume directly. This is the bridge between your human-readable dataset (audio + text) and the GPU-optimized training data.

The preprocessing pipeline for each sample:
1. Load audio and resample to 48 kHz stereo
2. VAE encode audio to latent representation
3. Text-encode the caption prompt with the Qwen3 text encoder
4. Text-encode the lyrics with the Qwen3 text encoder
5. Run the DiT condition encoder to produce final conditioning tensors
6. Build context latents from the silence template
7. Save everything as a single `.pt` file

## Inputs

### Required
- **existing_dataset**: Dropdown of datasets found in `output/AceLora/Dataset/` that contain audio files. Select the dataset you want to preprocess.
- **max_duration_seconds**: Maximum audio duration in seconds (default: 240). Audio longer than this is trimmed. Shorter audio is kept as-is.
- **custom_activation_tag**: Optional tag prepended to all captions during encoding (e.g. `mystyle`). Leave empty to use the tags already in your `.txt` files.

### Optional
- **dataset_path**: String input that overrides the dropdown. Connect this from the Dataset Builder node's `dataset_path` output for a seamless pipeline.

## Outputs
- **status**: Summary of preprocessing results (successes, failures, output path).
- **tensor_path**: Path to the `tensors/` subfolder containing the `.pt` files. Connect this to the Trainer node.

## Usage
1. Make sure you have run the Dataset Builder first (or manually created audio + `.txt` pairs)
2. Select the dataset from the dropdown, or connect from Dataset Builder
3. Queue the node — each audio+txt pair is encoded into a `.pt` tensor file
4. Progress is logged to the CMD console

## Notes
- Only audio files that have a matching `.txt` metadata file are processed
- Already-existing `.pt` files are **overwritten** (re-run is safe but re-encodes everything)
- Requires ~6 GB VRAM (loads VAE + Text Encoder + DiT). Models are auto-downloaded on first use.
- Tensor files are saved to `output/AceLora/Dataset/{name}/tensors/`
- After preprocessing, models are offloaded from GPU memory automatically
- If a file fails to preprocess, it is skipped and reported in the status output
