import os
import json
import uuid
import glob
import shutil
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv()

 
from config import DEFAULT_LANGUAGE, DEFAULT_MODEL_SIZE
from s3_upload import S3AudioUploader
from denoiser import DeepFilterNetNoiseReducer
from diarization import DiarizationPipeline
from transcription import (
    init_voxtral_model,
    run_voxtral_transcription,
    estimate_max_tokens
)

# Environment variables for S3
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


class AudioPipeline:
    """End-to-end pipeline: Denoise → Diarize → Transcribe → JSON + optional S3"""

    def __init__(
        self,
        device: str = "cuda",
        model_size: str = DEFAULT_MODEL_SIZE,
        language: str = DEFAULT_LANGUAGE,
        s3_bucket: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None
    ):
        self.device = device
        self.language = language

        # Initialize diarization and Voxtral
        self.diarizer = DiarizationPipeline(device=device)
        self.model, self.processor, self.device, self.repo_id = init_voxtral_model(model_size)

        # Initialize optional S3 uploader
        self.s3_uploader = None
        if s3_bucket:
            self.s3_uploader = S3AudioUploader(
                bucket_name=s3_bucket,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name="ap-south-1"
            )

        print(f"[INIT] Pipeline ready with Voxtral-{model_size} on {self.device}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_conversation_id(self, base_name: Optional[str] = None) -> str:
        uid = uuid.uuid4().hex[:8]
        return f"conv_{base_name}_{uid}" if base_name else f"conv_{uid}"

    def _upload_to_s3(self, output_dir: str, base_name: str):
        """Upload audio chunks and JSON for one processed file."""
        if not self.s3_uploader:
            return {"uploaded": False, "reason": "S3 not configured"}

        stats = {"uploaded": True, "audio_chunks": [], "json_files": []}

        try:
            # Upload chunks
            chunk_dir = os.path.join(output_dir, "Audio_Chunks", base_name)
            if os.path.exists(chunk_dir):
                print("Uploading chunks to S3...")
                uploaded = self.s3_uploader.upload_directory(
                    local_dir=chunk_dir,
                    s3_prefix=f"Audio_Chunks/{base_name}",
                    file_extensions=['.wav']
                )
                stats["audio_chunks"] = uploaded

            # Upload JSON
            json_file = os.path.join(output_dir, "JSON", f"{base_name}.json")
            if os.path.exists(json_file):
                s3_key = f"JSON/{base_name}.json"
                s3_uri = self.s3_uploader.upload_file(json_file, s3_key)
                stats["json_files"].append({"s3_uri": s3_uri, "file": s3_key})

        except Exception as e:
            stats.update({"uploaded": False, "error": str(e)})

        return stats

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    def _transcribe_segments(self, segments: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Transcribe diarized audio segments and create message + alignment."""
        messages = []
        alignment = []

        for idx, seg in enumerate(segments):
            speaker = seg["speaker"]
            start, end = seg["start_time"], seg["end_time"]
            duration = end - start

            print(f"→ Transcribing {speaker} ({start:.2f}s–{end:.2f}s)")

            max_tokens = estimate_max_tokens(duration)
            result = run_voxtral_transcription(
                audio_path=seg["full_path"],
                model=self.model,
                processor=self.processor,
                device=self.device,
                repo_id=self.repo_id,
                language=self.language
            )

            if not result["success"]:
                print(f"  ⚠ Failed: {result['error']}")
                continue

            text = result["text"].strip()
            if not text:
                continue

            segment_id = f"seg_{idx+1:03d}"

            messages.append({
                "speaker": speaker,
                "segment_id": segment_id,
                "text": text
            })

            alignment.append({
                "segment_id": segment_id,
                "start_ms": int(start * 1000),
                "end_ms": int(end * 1000),
                "chunk_file": os.path.basename(seg["full_path"])
            })

        return messages, alignment

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    def process_single_audio(
        self,
        input_audio: str,
        output_dir: str,
        conversation_id: Optional[str] = None,
        upload_to_s3: bool = False
    ) -> str:
        """Run full pipeline for one audio file and export to JSON."""

        base_name = Path(input_audio).stem
        conversation_id = conversation_id or self._generate_conversation_id(base_name)

        # Directory setup
        chunk_dir = os.path.join(output_dir, "Audio_Chunks", base_name)
        json_dir = os.path.join(output_dir, "JSON")
        tmp_dir = os.path.join(output_dir, "Temp")

        for d in [chunk_dir, json_dir, tmp_dir]:
            os.makedirs(d, exist_ok=True)

        print(f"\n{'='*70}\nProcessing: {base_name}\nConversation ID: {conversation_id}\n{'='*70}")

        # Step 1 — Denoising
        denoised_audio = input_audio
        print("[Step 1] Denoising...")
        denoiser = DeepFilterNetNoiseReducer(device=self.device)
        denoised_path = os.path.join(tmp_dir, f"{base_name}_denoised.wav")

        try:
            denoiser.process_audio(input_audio, denoised_path)
            denoised_audio = denoised_path
            print("✓ Denoised successfully\n")
        except Exception as e:
            print(f"⚠ Denoising failed: {e}\nContinuing with original...\n")
        del denoiser

        # Step 2 — Diarization
        print("[Step 2] Running diarization...")
        segments = self.diarizer.process_audio(denoised_audio, output_dir=chunk_dir, batch_size=1)
        print(f"✓ Found {len(segments)} segments\n")

        # Convert to speaker_0..N format
        for s in segments:
            s["speaker"] = f"speaker_{s['speaker']}"

        # Step 3 — Transcription
        print("[Step 3] Transcribing segments...")
        messages, alignment = self._transcribe_segments(segments)

        # Step 4 — JSON assembly
        total_dur = int(segments[-1]["end_time"] * 1000) if segments else 0
        output_json = {
            "conversation_id": conversation_id,
            "messages": messages,
            "alignment": alignment,
            "metadata": {
                "audio_file": os.path.basename(input_audio),
                "language": self.language,
                "processed_on": datetime.now().isoformat(),
                "duration_ms": total_dur,
                "model": self.repo_id
            }
        }

        out_file = os.path.join(json_dir, f"{base_name}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

        print(f"✓ JSON saved: {out_file}")

        # Step 5 — Upload to S3
        if upload_to_s3:
            print("[Step 5] Uploading to S3...")
            stats = self._upload_to_s3(output_dir, base_name)
            if stats.get("uploaded"):
                print("✓ Upload completed\n")
            else:
                print(f"⚠ Upload skipped or failed: {stats.get('error', stats.get('reason'))}\n")

        # Cleanup
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        print(f"✓ Done with {base_name}\n")
        return out_file

    # -------------------------------------------------------------------------
    # Batch Mode
    # -------------------------------------------------------------------------

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        upload_to_s3: bool = False
    ):
        """Process all audio files in a folder."""
        audio_exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
        files = [f for f in Path(input_dir).glob("*") if f.suffix.lower() in audio_exts]

        if not files:
            print(f"No audio found in {input_dir}")
            return []

        results = []
        for i, f in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing {f.name}")
            try:
                out_path = self.process_single_audio(str(f), output_dir, upload_to_s3=upload_to_s3)
                results.append(out_path)
            except Exception as e:
                print(f"✗ Failed: {e}\n")

        return results


# -----------------------------------------------------------------------------
# CLI Entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT = "/home/user/voice/speechriv_transcribe/input_audio/sample.mp3"
    OUTPUT = "reforged_output"

    pipeline = AudioPipeline(
        s3_bucket=S3_BUCKET,
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_key=AWS_SECRET_KEY
    )

    pipeline.process_single_audio(INPUT, OUTPUT)
