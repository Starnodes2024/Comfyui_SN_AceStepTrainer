"""
Audio utilities for AceStep 1.5 LoRA Trainer.
Handles audio loading, resampling, and duration detection.
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
import soundfile as sf

logger = logging.getLogger("AceStepTrainer")

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".opus"}
TARGET_SAMPLE_RATE = 48000


def _load_audio_sf(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio using soundfile, bypassing torchaudio's torchcodec backend.

    Returns:
        Tuple of (waveform [channels, samples], sample_rate)
    """
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # soundfile returns [samples, channels], convert to [channels, samples]
    waveform = torch.from_numpy(data.T)
    return waveform, sr


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds."""
    info = sf.info(audio_path)
    return info.duration


def load_audio_stereo(
    audio_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    max_duration: float = 240.0,
) -> Tuple[torch.Tensor, int]:
    """Load audio file and convert to stereo at target sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default 48000)
        max_duration: Maximum duration in seconds to load

    Returns:
        Tuple of (audio_tensor [2, samples], sample_rate)
    """
    waveform, sr = _load_audio_sf(audio_path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Convert to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    # Trim to max duration
    max_samples = int(max_duration * sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    return waveform, sr


def scan_audio_files(directory: str) -> list:
    """Scan directory for supported audio files.

    Returns list of absolute paths sorted alphabetically.
    """
    audio_files = []
    directory = Path(directory)

    if not directory.exists():
        return audio_files

    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
            audio_files.append(str(f.resolve()))

    return audio_files


def get_txt_path_for_audio(audio_path: str) -> str:
    """Get the corresponding .txt metadata file path for an audio file."""
    return str(Path(audio_path).with_suffix(".txt"))


def parse_metadata_txt(txt_path: str) -> dict:
    """Parse a metadata .txt file into a dictionary.

    Expected format (one key: value per line):
        caption: A melodic piano piece
        genre: classical, piano
        bpm: 120
        key: C major
        timesignature: 4/4
        language: unknown
        lyrics: [Instrumental]
    """
    metadata = {
        "caption": "",
        "genre": "",
        "bpm": "",
        "key": "",
        "timesignature": "",
        "language": "unknown",
        "lyrics": "[Instrumental]",
    }

    if not os.path.exists(txt_path):
        return metadata

    with open(txt_path, "r", encoding="utf-8") as f:
        current_key = None
        multiline_value = []

        for line in f:
            line = line.rstrip("\n")

            # Check if this line starts a new key
            if ": " in line and not line.startswith(" ") and not line.startswith("\t"):
                # Save previous multiline value
                if current_key and current_key == "lyrics":
                    metadata[current_key] = "\n".join(multiline_value)

                key, _, value = line.partition(": ")
                key = key.strip().lower()
                if key in metadata:
                    current_key = key
                    if key == "lyrics":
                        multiline_value = [value] if value else []
                    else:
                        metadata[key] = value.strip()
                else:
                    # Unknown key, could be continuation of lyrics
                    if current_key == "lyrics":
                        multiline_value.append(line)
            else:
                # Continuation line (for lyrics)
                if current_key == "lyrics":
                    multiline_value.append(line)

        # Final flush for lyrics
        if current_key == "lyrics":
            metadata["lyrics"] = "\n".join(multiline_value)

    return metadata


def write_metadata_txt(txt_path: str, metadata: dict) -> None:
    """Write metadata dictionary to a .txt file."""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"caption: {metadata.get('caption', '')}\n")
        f.write(f"genre: {metadata.get('genre', '')}\n")
        f.write(f"bpm: {metadata.get('bpm', '')}\n")
        f.write(f"key: {metadata.get('key', '')}\n")
        f.write(f"timesignature: {metadata.get('timesignature', '')}\n")
        f.write(f"language: {metadata.get('language', 'unknown')}\n")
        lyrics = metadata.get("lyrics", "[Instrumental]")
        f.write(f"lyrics: {lyrics}\n")
