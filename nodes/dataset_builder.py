"""
Node 1: AceStep 1.5 Dataset Builder
Scans audio files, copies them to dataset folder, runs LLM auto-labeling,
and saves .txt metadata files that users can edit before preprocessing.
"""

import os
import shutil
import logging

from ..core.model_downloader import get_dataset_dir
from ..core.audio_utils import (
    SUPPORTED_AUDIO_FORMATS, scan_audio_files,
    get_txt_path_for_audio, write_metadata_txt,
)
from ..core.labeling import extract_audio_codes, label_audio_with_llm
from ..core.model_loader import offload_to_cpu

logger = logging.getLogger("AceStepTrainer")


class AceStep15DatasetBuilder:
    """Scans MP3/audio files, copies to dataset folder, auto-labels with LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mp3_source_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path to the folder with your training audio files (MP3, WAV, FLAC, OGG, Opus). Example: C:/Music/MySamples or /home/user/music. All supported audio files in this folder will be scanned and copied into the dataset.",
                }),
                "dataset_name": ("STRING", {
                    "default": "my_dataset",
                    "multiline": False,
                    "tooltip": "A unique name for your dataset. This creates a folder at output/AceLora/Dataset/{name}/ where all audio files and their .txt label files are stored. Use something descriptive like 'lofi_beats' or 'rock_guitar'.",
                }),
                "custom_activation_tag": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional unique keyword prepended to every caption. When you later generate music, including this tag in your prompt activates the LoRA style. Example: 'mystyle'. Leave empty if you don't need a trigger word.",
                }),
                "all_instrumental": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable this if none of your training tracks contain vocals. Sets lyrics to [Instrumental] for all files. If your tracks have vocals, disable this and the LLM will attempt to detect/transcribe lyrics.",
                }),
                "skip_llm_labeling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip the AI auto-labeling step and create blank .txt template files instead. Use this if you prefer to write all labels manually, or if you don't have enough VRAM for the LLM (~5GB needed). You can always edit the auto-generated labels afterwards too.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("status", "dataset_path",)
    FUNCTION = "build_dataset"
    CATEGORY = "AceStep/Training"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run so deleted/edited .txt files are detected
        return float("NaN")

    def build_dataset(self, mp3_source_path, dataset_name, custom_activation_tag,
                      all_instrumental, skip_llm_labeling):
        # Validate source path
        if not mp3_source_path or not os.path.isdir(mp3_source_path):
            return (f"ERROR: Source path not found: {mp3_source_path}", "")

        # Scan for audio files
        audio_files = scan_audio_files(mp3_source_path)
        if not audio_files:
            exts = ", ".join(SUPPORTED_AUDIO_FORMATS)
            return (f"ERROR: No audio files found in {mp3_source_path}. Supported: {exts}", "")

        # Create dataset directory
        dataset_dir = get_dataset_dir(dataset_name)
        dataset_path = str(dataset_dir)

        print(f"[AceStep Dataset] Found {len(audio_files)} audio files in {mp3_source_path}")
        print(f"[AceStep Dataset] Creating dataset: {dataset_path}")

        # Copy audio files to dataset folder
        copied = 0
        for src in audio_files:
            fname = os.path.basename(src)
            dst = dataset_dir / fname
            if not dst.exists():
                shutil.copy2(src, str(dst))
                copied += 1
            else:
                print(f"[AceStep Dataset] Already exists, skipping: {fname}")

        print(f"[AceStep Dataset] Copied {copied} new files to dataset folder")

        # Get list of audio files now in dataset folder
        dataset_audio = scan_audio_files(dataset_path)

        if skip_llm_labeling:
            # Create empty .txt templates
            created = 0
            for af in dataset_audio:
                txt_path = get_txt_path_for_audio(af)
                if not os.path.exists(txt_path):
                    metadata = {
                        "caption": f"[EDIT THIS] Description of {os.path.basename(af)}",
                        "genre": "",
                        "bpm": "",
                        "key": "",
                        "timesignature": "",
                        "language": "unknown",
                        "lyrics": "[Instrumental]" if all_instrumental else "[EDIT THIS] Add lyrics here",
                    }
                    write_metadata_txt(txt_path, metadata)
                    created += 1

            status = (
                f"Dataset created: {dataset_path}\n"
                f"Audio files: {len(dataset_audio)}\n"
                f"Text templates created: {created}\n"
                f"LLM labeling: SKIPPED\n"
                f"Please edit the .txt files manually before running the Preprocessor node."
            )
            print(f"[AceStep Dataset] {status}")
            return (status, dataset_path)

        # Auto-label with LLM
        # Counts missing .txt files so user sees what will be processed
        needs_label = sum(1 for af in dataset_audio if not os.path.exists(get_txt_path_for_audio(af)))
        print(f"[AceStep Dataset] Starting LLM auto-labeling... ({needs_label} missing, {len(dataset_audio) - needs_label} already done)")
        labeled = 0
        skipped = 0
        failed = 0

        for i, af in enumerate(dataset_audio):
            fname = os.path.basename(af)
            txt_path = get_txt_path_for_audio(af)

            # Skip if already properly labeled (not a template)
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "[EDIT THIS]" not in content and "caption: \n" not in content:
                    print(f"[AceStep Dataset] [{i+1}/{len(dataset_audio)}] Already labeled: {fname}")
                    skipped += 1
                    continue
                else:
                    print(f"[AceStep Dataset] [{i+1}/{len(dataset_audio)}] Re-labeling template: {fname}")

            try:
                print(f"[AceStep Dataset] [{i+1}/{len(dataset_audio)}] Labeling: {fname}")

                # Extract audio codes (VAE + DiT stay on GPU throughout)
                codes = extract_audio_codes(af)

                if codes is None:
                    print(f"[AceStep Dataset] FAILED to extract codes for {fname}")
                    # Create template instead
                    metadata = {
                        "caption": f"[EDIT THIS] Description of {fname}",
                        "genre": "", "bpm": "", "key": "", "timesignature": "",
                        "language": "unknown",
                        "lyrics": "[Instrumental]" if all_instrumental else "",
                    }
                    write_metadata_txt(txt_path, metadata)
                    failed += 1
                    continue

                # LLM labeling
                metadata = label_audio_with_llm(codes)

                # Apply custom tag
                if custom_activation_tag:
                    existing_caption = metadata.get("caption", "")
                    if existing_caption:
                        metadata["caption"] = custom_activation_tag + ", " + existing_caption
                    else:
                        metadata["caption"] = custom_activation_tag

                # Force instrumental if requested
                if all_instrumental:
                    metadata["lyrics"] = "[Instrumental]"

                write_metadata_txt(txt_path, metadata)
                labeled += 1
                print(f"[AceStep Dataset] Labeled: {fname} -> {metadata.get('caption', '')[:80]}")

            except Exception as e:
                print(f"[AceStep Dataset] ERROR labeling {fname}: {e}")
                logger.exception(f"Error labeling {fname}")
                failed += 1

        # Offload LLM models to free GPU memory
        offload_to_cpu("llm", "dit_default", "vae")

        status = (
            f"Dataset created: {dataset_path}\n"
            f"Audio files: {len(dataset_audio)}\n"
            f"Already labeled (kept): {skipped}\n"
            f"Newly labeled: {labeled}\n"
            f"Failed: {failed}\n"
            f"Delete any .txt file and re-run to re-label it."
        )
        print(f"[AceStep Dataset] Done! {status}")
        return (status, dataset_path)
