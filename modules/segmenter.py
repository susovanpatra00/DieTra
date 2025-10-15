import os
import torch
import librosa
import soundfile as sf
from typing import List, Optional


class SpeakerSegmenter:
    """
    Speaker segmentation using NVIDIA NeMo Streaming Sortformer model.
    Splits long audio into speaker-specific chunks for downstream processing.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the Sortformer model for streaming speaker diarization.

        Args:
            device: "cpu" or "cuda" (default: "cuda")
        """
        self.device = device
        self.model = None
        self._load_model()

    # -------------------------------------------------------------------------
    def _load_model(self):
        """Load NVIDIA NeMo Sortformer diarization model."""
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
            print("Loading NVIDIA Sortformer diarization model...")
            self.model = SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_streaming_sortformer_4spk-v2"
            )
            self.model.eval()
            if self.device == "cuda":
                self.model = self.model.cuda()
        except ImportError:
            raise ImportError(
                "NVIDIA NeMo is not installed. Please install it with: pip install nemo_toolkit[asr]"
            )

    # -------------------------------------------------------------------------
    def segment_speakers(
        self,
        audio_path: str,
        batch_size: int = 1,
        speakers: Optional[int] = 2,
    ) -> List[str]:
        """
        Run speaker diarization and return time-stamped segments.

        Args:
            audio_path: Path to the audio file.
            batch_size: Inference batch size.
            speakers: Expected number of speakers (default: 2)

        Returns:
            List of strings: "start end speaker"
        """
        segments = self.model.diarize(audio=audio_path, batch_size=batch_size)
        if not segments:
            print("⚠ No segments returned by model.")
            return []

        return segments[0]

    # -------------------------------------------------------------------------
    def export_chunks(
        self,
        audio_path: str,
        segments: List[str],
        output_dir: str = "speaker_segments",
        save_manifest: bool = True,
    ) -> None:
        """
        Save audio chunks based on diarization output.

        Args:
            audio_path: Path to the input audio file.
            segments: List of "start end speaker" strings.
            output_dir: Destination directory.
            save_manifest: Whether to save a text file of all segments.
        """
        os.makedirs(output_dir, exist_ok=True)
        audio, sr = librosa.load(audio_path, sr=None)

        manifest_lines = []

        for idx, segment in enumerate(segments):
            try:
                start, end, speaker = segment.split()
                start_t = float(start)
                end_t = float(end)
            except Exception as e:
                print(f"Skipping malformed segment '{segment}': {e}")
                continue

            clip = audio[int(start_t * sr): int(end_t * sr)]
            filename = f"{speaker}_{start_t:.2f}_{end_t:.2f}_{idx:03d}.wav"
            out_path = os.path.join(output_dir, filename)

            sf.write(out_path, clip, sr)
            manifest_lines.append(f"{out_path}\t{speaker}\t{start_t:.2f}-{end_t:.2f}")

        if save_manifest:
            manifest_path = os.path.join(output_dir, "segments_manifest.txt")
            with open(manifest_path, "w") as f:
                f.write("\n".join(manifest_lines))
            print(f"✓ Segment manifest saved to {manifest_path}")

    # -------------------------------------------------------------------------
    def process(
        self,
        audio_path: str,
        output_dir: str = "speaker_segments",
        batch_size: int = 1,
        speakers: int = 2,
    ) -> List[str]:
        """
        Full speaker segmentation pipeline:
        - Diarize
        - Extract speaker-specific clips
        - Return segment list
        """
        print(f"\n🔹 Running diarization on: {os.path.basename(audio_path)}")
        segments = self.segment_speakers(audio_path, batch_size, speakers)

        if not segments:
            print("No segments detected.")
            return []

        self.export_chunks(audio_path, segments, output_dir)
        print(f"✓ Diarization complete. Chunks saved in: {output_dir}")
        return segments


# -----------------------------------------------------------------------------
# Example usage (CLI-style execution)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example file for testing (replace with your own)
    test_file = "/home/user/voiceflow_ai/input_audio/example.mp3"

    segmenter = SpeakerSegmenter(device="cuda")
    segments = segmenter.process(test_file)
    print(f"\nDetected {len(segments)} segments total.")
