# DieTra
*Denoising • Diarization • Transcription*

**DieTra** is an end-to-end audio processing pipeline that combines the three core pillars of audio analysis: **De**noising, D**i**arization, and **Tra**nscription. Built with state-of-the-art AI models, DieTra converts multi-speaker audio files into structured JSON outputs for high-quality audio analysis workflows.

## 🚀 Features

- **Audio Denoising**: High-quality noise reduction using DeepFilterNet
- **Speaker Diarization**: Efficient streaming speaker segmentation using NVIDIA NeMo Sortformer (optimized for long audio files)
- **Speech Transcription**: Advanced transcription with Mistral's Voxtral models
- **Cloud Storage**: Optional AWS S3 integration for storing processed audio and results
- **Batch Processing**: Process single files or entire directories
- **Structured Output**: Export results as JSON with speaker alignment and metadata

## 🏗️ Architecture

DieTra follows a modular architecture with two main components:

### Core Pipeline (`core/`)
- [`pipeline.py`](core/pipeline.py) - Main orchestration pipeline
- [`config.py`](core/config.py) - Configuration parameters and defaults
- [`utils.py`](core/utils.py) - Utility functions

### Processing Modules (`modules/`)
- [`cleaner.py`](modules/cleaner.py) - Audio denoising with DeepFilterNet
- [`segmenter.py`](modules/segmenter.py) - Speaker diarization with NVIDIA NeMo Streaming Sortformer
- [`transcriber.py`](modules/transcriber.py) - Speech transcription with Voxtral
- [`cloud_uploader.py`](modules/cloud_uploader.py) - AWS S3 integration

> **Performance Note**: DieTra uses streaming diarization which is highly memory-efficient. For a 1-hour audio file, traditional diarization would require 80GB+ VRAM, but our streaming approach processes it with just 10-12GB VRAM, making it practical for consumer-grade GPUs.

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU with **12GB+ VRAM** (minimum 10GB for processing, up to 20GB for very long audio files >3 hours)
- 16GB+ RAM
- 20GB+ free disk space

### Key Dependencies
- PyTorch with CUDA support
- Transformers (>=4.57.0) for Voxtral models
- DeepFilterNet for audio denoising
- NVIDIA NeMo for streaming speaker diarization
- Mistral Common with audio support
- AWS SDK for cloud storage

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/DieTra.git
cd DieTra
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install --upgrade pip
pip install torch torchaudio numpy soundfile librosa deepfilternet "transformers>=4.57.0" nemo_toolkit[asr] python-dotenv boto3
pip install --upgrade mistral_common[audio]
```

   **If you encounter NeMo toolkit version issues**, run:
   ```bash
   pip install --upgrade torch torchaudio transformers
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   S3_BUCKET=<your-s3-bucket-name>
   AWS_ACCESS_KEY_ID=<your-access-key-id>
   AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
   ```

## 🎯 Usage

### Basic Usage

```python
from core.pipeline import AudioPipeline

# Initialize pipeline
pipeline = AudioPipeline(
    device="cuda",  # or "cpu"
    model_size="mini",  # or "small"
    language="en"
)

# Process single audio file
output_file = pipeline.process_single_audio(
    input_audio="path/to/audio.mp3",
    output_dir="output",
    upload_to_s3=False
)

print(f"Results saved to: {output_file}")
```

### Batch Processing

```python
# Process entire directory
results = pipeline.process_folder(
    input_dir="input_audio/",
    output_dir="output/",
    upload_to_s3=False
)
```

### Command Line Usage

#### Individual Modules

**Audio Cleaning**:
```bash
python modules/cleaner.py --input audio.mp3 --output cleaned_audio/
python modules/cleaner.py --batch input_dir/ --output cleaned_audio/
```

**Speaker Segmentation**:
```bash
python modules/segmenter.py  # Edit the test file path in __main__
```

**Transcription**:
```bash
python modules/transcriber.py --input audio.mp3 --language en --model mini
```

**Cloud Upload**:
```bash
python modules/cloud_uploader.py  # Configure S3 credentials first
```

#### Full Pipeline

```bash
python core/pipeline.py  # Edit INPUT/OUTPUT paths in __main__
```

### Configuration

Key parameters in [`config.py`](core/config.py):

```python
# Audio processing
MIN_AUDIO_DURATION = 0.5       # Minimum valid duration (seconds)
MAX_AUDIO_DURATION = 30.0      # Maximum per chunk for Voxtral
CHUNK_OVERLAP = 1.0            # Overlap between chunks (seconds)
TARGET_SR = 16000              # Required sampling rate for Voxtral

# Model defaults
DEFAULT_MODEL_SIZE = "mini"    # "mini" or "small"
DEFAULT_LANGUAGE = "en"        # Language code

# AWS S3
DEFAULT_S3_REGION = "us-east-1"
```

## 📊 Output Format

DieTra generates structured JSON output with the following format:

```json
{
  "conversation_id": "conv_sample_a1b2c3d4",
  "messages": [
    {
      "speaker": "speaker_0",
      "segment_id": "seg_001",
      "text": "Hello, how can I help you today?"
    },
    {
      "speaker": "speaker_1", 
      "segment_id": "seg_002",
      "text": "I need assistance with my account."
    }
  ],
  "alignment": [
    {
      "segment_id": "seg_001",
      "start_ms": 0,
      "end_ms": 2500,
      "chunk_file": "speaker_0_0.00_2.50_000.wav"
    }
  ],
  "metadata": {
    "audio_file": "sample.mp3",
    "language": "en",
    "processed_on": "2024-01-15T10:30:00",
    "duration_ms": 120000,
    "model": "mistralai/Voxtral-Mini-3B-2507"
  }
}
```

## 🔧 Advanced Configuration

### Custom Pipeline Setup

```python
from core.pipeline import AudioPipeline

# Advanced configuration
pipeline = AudioPipeline(
    device="cuda",
    model_size="small",  # Better quality, slower processing
    language="es",       # Spanish transcription
    s3_bucket="my-bucket",
    aws_access_key="...",
    aws_secret_key="..."
)

# Process with custom conversation ID
output = pipeline.process_single_audio(
    input_audio="meeting.wav",
    output_dir="results/",
    conversation_id="meeting_2024_01_15",
    upload_to_s3=True
)
```

### Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`) 
- FLAC (`.flac`)
- M4A (`.m4a`)
- OGG (`.ogg`)

### Model Options

**Voxtral Models**:
- `mini`: Faster processing, good quality (3B parameters)
- `small`: Higher quality, slower processing (24B parameters)

## 🤝 Contributions

Developed by [@susovanpatra00](https://github.com/susovanpatra00) and [@anand-therattil](https://github.com/anand-therattil).




## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepFilterNet** for high-quality audio denoising
- **NVIDIA NeMo** for speaker diarization capabilities
- **Mistral AI** for the Voxtral transcription models
- **Hugging Face** for model hosting and transformers library

---

**DieTra** - *Transforming audio conversations into structured insights*