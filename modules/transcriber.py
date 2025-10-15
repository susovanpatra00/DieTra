import os
import json
import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from typing import Tuple, Dict, Any

from config import (
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
    CHUNK_OVERLAP,
    TARGET_SR,
    DEFAULT_MODEL_SIZE,
    DEFAULT_LANGUAGE
)


# -----------------------------------------------------------------------------
# Audio Utilities
# -----------------------------------------------------------------------------

def get_audio_duration(audio_path: str) -> float:
    """Return duration of the given audio file."""
    try:
        return sf.info(audio_path).duration
    except Exception as e:
        print(f"[WARN] Failed to read duration for {audio_path}: {e}")
        return 0.0


def load_and_validate_audio(audio_path: str) -> Tuple[np.ndarray, int, float, bool, str]:
    """Load, validate, and resample audio to target sample rate."""
    try:
        data, sr = sf.read(audio_path)
        duration = len(data) / sr

        if duration < MIN_AUDIO_DURATION:
            return None, sr, duration, False, f"Too short ({duration:.2f}s)"

        if np.max(np.abs(data)) < 1e-6:
            return None, sr, duration, False, "Silent or corrupted audio"

        if sr != TARGET_SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        return data, sr, duration, True, ""
    except Exception as e:
        return None, 0, 0.0, False, f"Audio load error: {e}"


def split_audio(data: np.ndarray, sr: int, max_len: float, overlap: float):
    """Split long audio into overlapping segments."""
    total_duration = len(data) / sr
    if total_duration <= max_len:
        return [(data, 0.0, total_duration)]

    chunks = []
    step = int((max_len - overlap) * sr)
    size = int(max_len * sr)
    start = 0

    while start < len(data):
        end = min(start + size, len(data))
        chunk = data[start:end]
        chunks.append((chunk, start / sr, end / sr))
        if end >= len(data):
            break
        start += step

    return chunks


def estimate_max_tokens(duration: float) -> int:
    """Estimate max token length for transcription."""
    est_words = duration * 3.5
    est_tokens = int(est_words * 1.3)
    return min(max(100, int(est_tokens * 1.5)), 2048)


# -----------------------------------------------------------------------------
# Voxtral Setup and Inference
# -----------------------------------------------------------------------------

def init_voxtral_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Initialize Voxtral model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_map = {
        "mini": "mistralai/Voxtral-Mini-3B-2507",
        "small": "mistralai/Voxtral-Small-24B-2507"
    }

    repo_id = model_map.get(model_size.lower())
    if not repo_id:
        raise ValueError(f"Unknown model size: {model_size}")

    print(f"[INFO] Loading Voxtral model ({model_size}) on {device}...")
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )

    return model, processor, device, repo_id


def run_voxtral_transcription(
    audio_path: str,
    model,
    processor,
    device: str,
    repo_id: str,
    language: str = DEFAULT_LANGUAGE
) -> Dict[str, Any]:
    """Run transcription for one file, auto-splitting if too long."""
    try:
        data, sr, duration, valid, err = load_and_validate_audio(audio_path)
        if not valid:
            return {"success": False, "error": err}

        if duration > MAX_AUDIO_DURATION:
            print(f"[INFO] Long audio ({duration:.2f}s) → splitting...")
            segments = split_audio(data, sr, MAX_AUDIO_DURATION, CHUNK_OVERLAP)
            transcripts = []

            for i, (chunk, start_t, end_t) in enumerate(segments):
                tmp_file = f"/tmp/vox_chunk_{os.getpid()}_{i}.wav"
                sf.write(tmp_file, chunk, sr)
                try:
                    chunk_dur = end_t - start_t
                    max_tokens = estimate_max_tokens(chunk_dur)

                    inputs = processor.apply_transcription_request(
                        language=language,
                        audio=tmp_file,
                        model_id=repo_id
                    ).to(device, dtype=torch.bfloat16)

                    output = model.generate(**inputs, max_new_tokens=max_tokens)
                    text = processor.batch_decode(
                        output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )[0]

                    transcripts.append(text)
                finally:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)

            return {"success": True, "text": " ".join(transcripts).strip(), "duration": duration}

        # Normal case
        inputs = processor.apply_transcription_request(
            language=language,
            audio=audio_path,
            model_id=repo_id
        ).to(device, dtype=torch.bfloat16)

        max_tokens = estimate_max_tokens(duration)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return {"success": True, "text": decoded[0], "duration": duration}

    except Exception as e:
        return {"success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# Main Transcription Entry
# -----------------------------------------------------------------------------

def transcribe_single_file(
    audio_path: str,
    output_file: str = None,
    language: str = DEFAULT_LANGUAGE,
    model_size: str = DEFAULT_MODEL_SIZE
) -> Dict[str, Any]:
    """Transcribe a single audio file with Voxtral."""
    model, processor, device, repo_id = init_voxtral_model(model_size)
    print(f"[INFO] Transcribing {os.path.basename(audio_path)}")

    result = run_voxtral_transcription(audio_path, model, processor, device, repo_id, language)

    if not result.get("success"):
        print(f"[ERROR] {result.get('error')}")
        return result

    text = result["text"]
    duration = result["duration"]

    print(f"[DONE] Duration: {duration:.2f}s\n{text[:400]}{'...' if len(text) > 400 else ''}")

    # Save result
    out_path = output_file or f"{os.path.splitext(audio_path)[0]}_transcription.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio_file": audio_path,
                "transcription": text,
                "duration": duration,
                "language": language,
                "model": repo_id
            },
            f,
            indent=4
        )

    print(f"[SAVED] {out_path}")
    return result


# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Voxtral transcription on a single file.")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", help="Optional path to save transcription JSON")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Language code (default: en)")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE, choices=["mini", "small"], help="Model size")

    args = parser.parse_args()
    transcribe_single_file(
        audio_path=args.input,
        output_file=args.output,
        language=args.language,
        model_size=args.model
    )
