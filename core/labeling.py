"""
Audio labeling for AceStep 1.5 LoRA Trainer.
Extracts audio codes from audio files and uses the AceStep LLM
to generate captions, BPM, key, time signature, and genre tags.
"""

import logging
import os
import yaml
import re
from typing import Optional, Dict, Any

import torch
import torchaudio
import soundfile as sf

from .model_loader import (
    load_vae, load_dit_default, load_llm,
    get_silence_latent, _get_device, _get_dtype,
)

logger = logging.getLogger("AceStepTrainer")

# LLM understand instruction (from ACE-Step constants)
LM_UNDERSTAND_INSTRUCTION = (
    "Understand the given musical conditions and describe the audio semantics accordingly:"
)


def _normalize_audio_to_stereo_48k(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """Normalize audio to stereo 48kHz."""
    target_sr = 48000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio


def extract_audio_codes(audio_path: str, device: Optional[torch.device] = None) -> Optional[str]:
    """Extract audio codes from an audio file using VAE + DiT tokenizer.

    Pipeline: audio -> VAE encode -> latents -> DiT tokenize -> code indices -> formatted string

    Args:
        audio_path: Path to audio file
        device: Target device

    Returns:
        Formatted audio codes string like '<|audio_code_123|><|audio_code_456|>...'
        or None on failure
    """
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    try:
        # Load audio
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(data.T)  # [channels, samples]
        audio = _normalize_audio_to_stereo_48k(audio, sr)

        # Load models
        vae = load_vae(device)
        dit = load_dit_default(device)
        silence_latent = get_silence_latent("default")

        if silence_latent is None:
            print("[AceStep Labeling] ERROR: Silence latent not available for dit_default")
            return None

        with torch.no_grad():
            # Chunked VAE encode to avoid OOM on long audio files.
            # A 5-min song at 48kHz stereo = ~28M samples; encoding all at
            # once creates huge intermediate activations.  We split into
            # ~30-second chunks, encode each, and concatenate latents.
            CHUNK_SECONDS = 30
            chunk_samples = CHUNK_SECONDS * 48000  # 48kHz after resampling
            total_samples = audio.shape[1]

            all_latents = []
            for start in range(0, total_samples, chunk_samples):
                end = min(start + chunk_samples, total_samples)
                chunk = audio[:, start:end].unsqueeze(0).to(device).to(vae.dtype)
                chunk_dist = vae.encode(chunk)
                chunk_latent = chunk_dist.latent_dist.sample().to(dtype)
                all_latents.append(chunk_latent)
                del chunk, chunk_dist
                torch.cuda.empty_cache()

            latents = torch.cat(all_latents, dim=2)  # cat along time dim
            del all_latents
            # Transpose: [1, d, T] -> [1, T, d]
            latents = latents.transpose(1, 2)

            # Create attention mask
            attention_mask = torch.ones(1, latents.shape[1], dtype=torch.bool, device=device)

            # Tokenize using DiT model's tokenizer
            _, indices, _ = dit.tokenize(latents, silence_latent, attention_mask)

            # Format as code string
            indices_flat = indices.flatten().cpu().tolist()
            codes_string = "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])

            print(f"[AceStep Labeling] Generated {len(indices_flat)} audio codes for {os.path.basename(audio_path)}")
            return codes_string

    except Exception as e:
        import traceback
        print(f"[AceStep Labeling] ERROR extracting audio codes from {audio_path}:")
        traceback.print_exc()
        return None


def label_audio_with_llm(
    audio_codes: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Use the AceStep LLM to generate metadata from audio codes.

    Args:
        audio_codes: Formatted audio codes string
        device: Target device

    Returns:
        Dictionary with keys: caption, genre, bpm, keyscale, timesignature, language, lyrics
    """
    if device is None:
        device = _get_device()

    default_result = {
        "caption": "",
        "genre": "",
        "bpm": "",
        "keyscale": "",
        "timesignature": "",
        "language": "unknown",
        "lyrics": "[Instrumental]",
    }

    try:
        llm, tokenizer = load_llm(device)

        # Build the understand prompt using chat template
        system_content = f"# Instruction\n{LM_UNDERSTAND_INSTRUCTION}\n\n"
        user_content = audio_codes

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Parse the output
        metadata = _parse_llm_output(output_text)
        return metadata

    except Exception as e:
        import traceback
        print(f"[AceStep Labeling] ERROR during LLM labeling: {e}")
        traceback.print_exc()
        return default_result


def _parse_llm_output(output_text: str) -> Dict[str, Any]:
    """Parse LLM output to extract metadata.

    The LLM outputs in a <think>...</think> format with YAML-like content,
    followed by optional lyrics.
    """
    result = {
        "caption": "",
        "genre": "",
        "bpm": "",
        "keyscale": "",
        "timesignature": "",
        "language": "unknown",
        "lyrics": "[Instrumental]",
    }

    try:
        # Extract content between <think> and </think>
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", output_text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            try:
                parsed = yaml.safe_load(think_content)
                if isinstance(parsed, dict):
                    result["caption"] = str(parsed.get("caption", ""))
                    result["genre"] = str(parsed.get("genres", parsed.get("genre", "")))
                    bpm_val = parsed.get("bpm")
                    if bpm_val is not None and str(bpm_val) != "N/A":
                        result["bpm"] = str(bpm_val)
                    result["keyscale"] = str(parsed.get("keyscale", ""))
                    ts = parsed.get("timesignature", "")
                    if ts and str(ts) != "N/A":
                        result["timesignature"] = str(ts)
                    result["language"] = str(parsed.get("language", "unknown"))
            except yaml.YAMLError:
                # Try line-by-line parsing as fallback
                for line in think_content.split("\n"):
                    line = line.strip()
                    if ":" in line:
                        key, _, value = line.partition(":")
                        key = key.strip().lower()
                        value = value.strip()
                        if key == "caption":
                            result["caption"] = value
                        elif key in ("genre", "genres"):
                            result["genre"] = value
                        elif key == "bpm" and value != "N/A":
                            result["bpm"] = value
                        elif key == "keyscale":
                            result["keyscale"] = value
                        elif key == "timesignature" and value != "N/A":
                            result["timesignature"] = value
                        elif key == "language":
                            result["language"] = value

        # Extract lyrics after </think>
        after_think = output_text.split("</think>")[-1].strip() if "</think>" in output_text else ""
        if after_think:
            # Clean up: remove EOS tokens and trailing whitespace
            after_think = re.sub(r"<\|endoftext\|>.*", "", after_think).strip()
            after_think = re.sub(r"<\|im_end\|>.*", "", after_think).strip()
            if after_think and after_think != "[Instrumental]":
                result["lyrics"] = after_think

    except Exception as e:
        print(f"[AceStep Labeling] WARNING: Error parsing LLM output: {e}")

    return result
