# ⭐SN AceStep 1.5 Dataset Builder

## Description
Scans a folder of audio files (MP3, WAV, FLAC, OGG, Opus), copies them into a structured dataset directory, and optionally auto-labels each file using the ACE-Step 5Hz LLM. The result is a set of audio files paired with editable `.txt` metadata files, ready for the Preprocessor node.

## Inputs

### Required
- **mp3_source_path**: Full path to the folder containing your audio training files. All supported audio formats are scanned (MP3, WAV, FLAC, OGG, Opus).
- **dataset_name**: A name for this dataset. Creates a folder at `output/AceLora/Dataset/{name}/`.
- **custom_activation_tag**: An optional unique keyword that gets prepended to every caption (e.g. `mystyle`). This lets you trigger the LoRA during inference by including the tag in your prompt. Leave empty to skip.
- **all_instrumental**: Set to `True` if none of your tracks have vocals. This sets lyrics to `[Instrumental]` for all files.
- **skip_llm_labeling**: Set to `True` to skip the LLM auto-labeling step. Empty `.txt` templates will be created instead, which you can fill in manually.
- **llm_model**: Which LLM to use for auto-labeling. Options: `acestep-5Hz-lm-0.6B` (~1.2 GB, lighter), `acestep-5Hz-lm-1.7B` (default, ~3.4 GB, balanced), `acestep-5Hz-lm-4B` (~8 GB, best quality). Models are downloaded automatically on first use.

## Outputs
- **status**: A summary of what was done (files copied, labels generated, errors).
- **dataset_path**: The full path to the created dataset folder. Connect this to the Preprocessor node.

## Usage
1. Place your training audio files in a folder on your system
2. Enter the folder path in `mp3_source_path`
3. Give the dataset a descriptive name
4. Optionally set an activation tag and instrumental flag
5. Queue the node — files are copied and labeled
6. **After running**, open the `.txt` files in the dataset folder and review/edit the auto-generated labels before preprocessing

## Metadata Format
Each `.txt` file contains editable fields:
```
caption: A mellow lo-fi hip hop beat with warm Rhodes chords
genre: lo-fi, hip hop, chillhop
bpm: 85
key: D minor
timesignature: 4
language: unknown
lyrics: [Instrumental]
```

## Notes
- Already-copied files are skipped (safe to re-run)
- Already-labeled files (with existing `.txt`) are not overwritten
- LLM labeling requires ~5 GB VRAM (loads VAE + DiT + LLM). Models are auto-downloaded on first use.
- The LLM labels are a starting point — **always review and edit** them for best training results
- If labeling fails for a file, an empty template is created so you can fill it in manually
