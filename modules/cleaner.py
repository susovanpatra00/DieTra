import os
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf

try:
    from df.enhance import enhance, init_df
    from df.io import load_audio, save_audio
    DEEPFILTERNET_OK = True
except ImportError:
    DEEPFILTERNET_OK = False
    raise ImportError("DeepFilterNet not found. Install it with: pip install deepfilternet")


class AudioCleaner:
    """
    High-quality noise reduction using DeepFilterNet.
    Recommended for full-band 48kHz audio.
    """

    SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    def __init__(self, device: str = "cuda"):
        """
        Initialize the DeepFilterNet model.

        Args:
            device: Either 'cpu' or 'cuda'.
        """
        self.device = device
        self.model, self.state, _ = init_df()

    # -------------------------------------------------------------------------
    def _split_audio(self, data, sr, chunk_secs=600):
        """
        Split audio into smaller chunks (default 10 minutes each).
        """
        total_samples = len(data)
        chunk_size = int(chunk_secs * sr)
        return [data[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    # -------------------------------------------------------------------------
    def _process_file(self, input_file: str, output_file: str, chunk_secs=600):
        """
        Apply noise reduction to a single audio file.
        """
        audio, _ = load_audio(input_file, sr=self.state.sr())
        sr = self.state.sr()
        duration = len(audio) / sr

        if duration > chunk_secs:
            print(f"→ Long file ({duration/60:.1f} min). Splitting into chunks...")
            segments = self._split_audio(audio, sr, chunk_secs)

            processed = []
            for i, seg in enumerate(segments, 1):
                print(f"   Chunk {i}/{len(segments)} ({len(seg)/sr/60:.1f} min)")
                clean_seg = enhance(self.model, self.state, seg)
                processed.append(clean_seg)

            cleaned_audio = np.concatenate(processed)
        else:
            cleaned_audio = enhance(self.model, self.state, audio)

        save_audio(output_file, cleaned_audio, sr)
        return output_file

    # -------------------------------------------------------------------------
    def clean_directory(self, input_dir: str, output_dir: str):
        """
        Process all supported audio files inside a directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        files = [
            f for f in Path(input_dir).iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTS
        ]

        if not files:
            print(f"No valid audio files in {input_dir}")
            return

        print(f"Found {len(files)} files to clean.\n")
        for idx, file in enumerate(files, 1):
            out_file = Path(output_dir) / f"cleaned_{file.stem}.wav"
            try:
                print(f"[{idx}/{len(files)}] Cleaning {file.name}...")
                self._process_file(str(file), str(out_file))
                print(f"✓ {file.name} → {out_file.name}")
            except Exception as e:
                print(f"✗ Failed {file.name}: {e}")

    # -------------------------------------------------------------------------
    def clean_single(self, input_file: str, output_dir: str):
        """
        Clean one file and save result to output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"cleaned_{Path(input_file).stem}.wav"
        self._process_file(input_file, str(output_path))
        print(f"✓ Cleaned file saved at: {output_path}")
        return output_path


# -----------------------------------------------------------------------------
# CLI interface
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Audio cleaner using DeepFilterNet (denoising for high-quality speech)"
    )
    parser.add_argument("--input", "-i", help="Input audio file path")
    parser.add_argument("--output", "-o", default="cleaned_audio", help="Output directory")
    parser.add_argument("--batch", "-b", help="Process all audio files in this directory")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda"], default="cuda")

    args = parser.parse_args()

    cleaner = AudioCleaner(device=args.device)

    if args.batch:
        print(f"Batch cleaning: {args.batch}\n")
        cleaner.clean_directory(args.batch, args.output)
        print(f"\nAll files saved to {args.output}")
    elif args.input:
        cleaner.clean_single(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
