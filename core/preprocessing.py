"""
Preprocessing for AceStep 1.5 LoRA Trainer.
Encodes audio + text metadata into .pt tensor files for efficient training.

Matches the original ACE-Step source preprocessing exactly:
- preprocess_vae.py: VAE encode -> transpose
- preprocess_text.py: tokenize(padding=max_length, max_length=256), encoder(input_ids) no attn mask
- preprocess_lyrics.py: tokenize(padding=max_length, max_length=512), embed_tokens only (NO forward)
- preprocess_encoder.py: full model.encoder() call with dummy timbre zeros
- preprocess_context.py: silence + chunk_masks -> 128ch context
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

import torch
import yaml

from .audio_utils import load_audio_stereo, scan_audio_files, get_txt_path_for_audio, parse_metadata_txt
from .model_loader import (
    load_vae, load_text_encoder, load_dit,
    get_silence_latent, _get_device, _get_dtype,
)

logger = logging.getLogger("AceStepTrainer")

DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"


def _build_text_prompt(metadata: dict, custom_tag: str = "") -> str:
    caption = metadata.get("caption", "")
    if custom_tag:
        caption = f"{custom_tag}, {caption}" if caption else custom_tag

    metas = {}
    if metadata.get("bpm"):
        try:
            metas["bpm"] = int(metadata["bpm"])
        except (ValueError, TypeError):
            pass
    if metadata.get("key"):
        metas["keyscale"] = metadata["key"]
    if metadata.get("timesignature"):
        ts = metadata["timesignature"]
        if isinstance(ts, str) and ts.endswith("/4"):
            ts = ts.split("/")[0]
        metas["timesignature"] = ts

    metas_str = yaml.dump(metas, allow_unicode=True, sort_keys=True).strip() if metas else ""

    # Match SFT_GEN_PROMPT from original source (ends with <|endoftext|>)
    return (
        f"# Instruction\n{DIT_INSTRUCTION}\n\n"
        f"# Caption\n{caption}\n\n"
        f"# Metas\n{metas_str}<|endoftext|>\n"
    )


def _encode_caption(text_encoder, text_tokenizer, text: str, device, dtype):
    inputs = text_tokenizer(
        text,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device).to(dtype)

    text_dev = next(text_encoder.parameters()).device
    if input_ids.device != text_dev:
        input_ids = input_ids.to(text_dev)
        attention_mask = attention_mask.to(text_dev)

    with torch.no_grad():
        outputs = text_encoder(input_ids)
        hidden_states = outputs.last_hidden_state.to(dtype)

    return hidden_states, attention_mask


def _encode_lyrics(text_encoder, text_tokenizer, lyrics: str, device, dtype):
    inputs = text_tokenizer(
        lyrics,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device).to(dtype)

    text_dev = next(text_encoder.parameters()).device
    if input_ids.device != text_dev:
        input_ids = input_ids.to(text_dev)
        attention_mask = attention_mask.to(text_dev)

    with torch.no_grad():
        hidden_states = text_encoder.embed_tokens(input_ids).to(dtype)

    return hidden_states, attention_mask


def preprocess_dataset(
    dataset_dir: str,
    custom_tag: str = "",
    max_duration: float = 240.0,
    device: Optional[torch.device] = None,
) -> str:
    """Preprocess all audio+txt pairs in a dataset directory into .pt tensor files."""
    if device is None:
        device = _get_device()
    dtype = _get_dtype(device)

    dataset_path = Path(dataset_dir)
    tensors_dir = dataset_path / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    audio_files = scan_audio_files(str(dataset_path))
    if not audio_files:
        return "No audio files found in dataset directory."

    pairs = []
    for af in audio_files:
        txt = get_txt_path_for_audio(af)
        if os.path.exists(txt):
            pairs.append((af, txt))

    if not pairs:
        return "No audio files with matching .txt metadata found. Run Dataset Builder first."

    needs_processing = []
    already_done = 0
    for af, _txt in pairs:
        pt_path = tensors_dir / (Path(af).stem + ".pt")
        if pt_path.exists():
            already_done += 1
        else:
            needs_processing.append((af, _txt))

    if not needs_processing:
        return f"All {len(pairs)} files already preprocessed. Delete .pt files from tensors/ to reprocess."

    print(f"[AceStep Preprocess] {len(needs_processing)} to process, {already_done} already done. Loading models...")

    vae = load_vae(device)
    text_encoder, text_tokenizer = load_text_encoder(device)
    dit = load_dit(device, "dit_turbo")
    silence_latent = get_silence_latent("dit_turbo")

    if silence_latent is None:
        return "ERROR: silence_latent not found in DiT model checkpoint."

    success = 0
    failed = 0

    for i, (audio_path, txt_path) in enumerate(needs_processing):
        fname = os.path.basename(audio_path)
        try:
            print(f"[AceStep Preprocess] [{i+1}/{len(needs_processing)}] Processing: {fname}")

            metadata = parse_metadata_txt(txt_path)

            audio, _sr = load_audio_stereo(audio_path, max_duration=max_duration)

            # VAE encode in chunks to avoid OOM on long audio
            CHUNK_SECONDS = 30
            chunk_samples = CHUNK_SECONDS * 48000
            total_samples = audio.shape[1]

            all_latents = []
            with torch.no_grad():
                for start in range(0, total_samples, chunk_samples):
                    end = min(start + chunk_samples, total_samples)
                    chunk = audio[:, start:end].unsqueeze(0).to(device).to(vae.dtype)
                    chunk_dist = vae.encode(chunk)
                    chunk_latent = chunk_dist.latent_dist.sample().to(dtype)
                    all_latents.append(chunk_latent)
                    del chunk, chunk_dist

                target_latents = torch.cat(all_latents, dim=2)
                del all_latents
                target_latents = target_latents.transpose(1, 2)  # [1, 64, T] -> [1, T, 64]

            latent_length = target_latents.shape[1]
            attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

            # Caption encoding
            text_prompt = _build_text_prompt(metadata, custom_tag)
            text_hidden_states, text_attention_mask = _encode_caption(
                text_encoder, text_tokenizer, text_prompt, device, dtype
            )

            # Lyrics encoding: raw lyrics string (no wrapping)
            lyrics = metadata.get("lyrics") or "[Instrumental]"
            lyric_hidden_states, lyric_attention_mask = _encode_lyrics(
                text_encoder, text_tokenizer, lyrics, device, dtype
            )

            # Run full condition encoder (includes timbre_encoder with dummy zeros)
            with torch.no_grad():
                model_device = next(dit.parameters()).device
                model_dtype = next(dit.parameters()).dtype

                th = text_hidden_states.to(model_device, model_dtype)
                tm = text_attention_mask.to(model_device)
                lh = lyric_hidden_states.to(model_device, model_dtype)
                lm = lyric_attention_mask.to(model_device)

                refer_audio_hidden = torch.zeros(1, 1, 64, device=model_device, dtype=model_dtype)
                refer_audio_order_mask = torch.zeros(1, device=model_device, dtype=torch.long)

                encoder_hidden_states, encoder_attention_mask = dit.encoder(
                    text_hidden_states=th,
                    text_attention_mask=tm,
                    lyric_hidden_states=lh,
                    lyric_attention_mask=lm,
                    refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                    refer_audio_order_mask=refer_audio_order_mask,
                )

            # Build context_latents = cat([src_latents, chunk_masks]) = 128ch
            sl = silence_latent  # [1, T_silence, 64]
            if sl.shape[1] >= latent_length:
                src_latents = sl[:, :latent_length, :]
            else:
                n_repeats = (latent_length // sl.shape[1]) + 1
                src_latents = sl.repeat(1, n_repeats, 1)[:, :latent_length, :]
            src_latents = src_latents.to(device, dtype)
            chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)
            context_latents = torch.cat([src_latents, chunk_masks], dim=-1)  # [1, T, 128]

            sample_id = Path(audio_path).stem
            output_data = {
                "target_latents": target_latents.squeeze(0).cpu(),
                "attention_mask": attention_mask.squeeze(0).cpu(),
                "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),
                "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),
                "context_latents": context_latents.squeeze(0).cpu(),
                "metadata": {
                    "audio_path": audio_path,
                    "filename": fname,
                    "caption": metadata.get("caption", ""),
                    "lyrics": lyrics,
                    "language": metadata.get("language", "unknown"),
                },
            }

            output_path = tensors_dir / (sample_id + ".pt")
            torch.save(output_data, str(output_path))
            success += 1

            del output_data
            del target_latents, attention_mask
            del text_hidden_states, text_attention_mask
            del lyric_hidden_states, lyric_attention_mask
            del encoder_hidden_states, encoder_attention_mask
            del context_latents, src_latents, chunk_masks
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[AceStep Preprocess] FAILED {fname}: {e}")
            logger.exception(f"Error preprocessing {fname}")
            failed += 1
            torch.cuda.empty_cache()

    status = f"Preprocessing complete: {success}/{len(needs_processing)} succeeded"
    if already_done > 0:
        status += f" ({already_done} already done, kept)"
    if failed > 0:
        status += f" ({failed} failed)"
    status += f"\nTensors saved to: {tensors_dir}"
    print(f"[AceStep Preprocess] {status}")
    return status
