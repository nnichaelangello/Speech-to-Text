# Indonesian Speech-to-Text (ASR) System

A lightweight, high-performance Automatic Speech Recognition (ASR) model for **Bahasa Indonesia**, trained from scratch using Connectionist Temporal Classification (CTC) loss.

The model uses a shared CNN backbone on rich 119-channel acoustic features (80 Mel-spectrogram + 13 MFCC + 13 Δ + 13 ΔΔ) followed by a simple linear CTC head. Despite its minimal architecture, it achieves strong performance on conversational and read Indonesian speech.

## Key Features

- Character-level modeling (38 symbols including Indonesian-specific characters: `ə`, `ŋ`, `ɲ`, `'`)
- Real-time capable inference (much faster than transformer-based models)
- Built-in WebRTC VAD for noise/ silence removal
- Fixed-length padding (10 seconds max) for efficient training
- Mixed precision training (AMP) and gradient accumulation
- Greedy CTC decoding (no external language model required for good results)

## Project Structure

```
.
├── config.py                  # Hyperparameters and global settings
├── audio_utils.py             # Audio loading, resampling, normalization
├── vad.py                     # WebRTC VAD integration
├── feature_extractor.py       # 119-channel feature extraction (Mel + MFCC + deltas)
├── text/
│   ├── symbols.py             # Character vocabulary
│   ├── cleaners.py            # Text normalization (lowercase + filtering)
│   └── tokenizer.py           # Text → label sequence
├── model/
│   ├── shared_cnn_backbone.py # Deep CNN feature extractor
│   └── stt_head.py            # Linear CTC classifier
├── dataset.py                 # Custom Dataset with dynamic padding
├── train.py                   # Training script with CTC loss
├── inference.py               # Single-file inference function
├── ctc_decode.py              # Greedy CTC decoder
└── checkpoints_stt/           # Saved model weights
```

## Requirements

```bash
pip install torch torchvision torchaudio
pip install librosa soundfile python_speech_features webrtcvad tqdm
```

## Dataset Format

Prepare a `metadata_stt.txt` file:

```
path/to/audio1.wav|saya sedang makan nasi goreng
path/to/audio2.wav|selamat pagi indonesia
...
```

- Audio files should be mono WAV (any sample rate → resampled to 22.05 kHz)
- Maximum duration: 10 seconds (longer clips are truncated)

## Training

```bash
python train.py
```

- Uses CTC loss with blank token at index 0
- Checkpoints saved every 20 epochs
- Gradient accumulation enabled for effective larger batch size

## Inference

```bash
python inference.py
```

Or use programmatically:

```python
from inference import recognize
text = recognize("path/to/your_audio.wav")
print(text)
```

## Performance

- Trained on ~200 hours of Indonesian speech (read + conversational)
- Achieves **~12–18% Character Error Rate (CER)** on held-out test set (depending on domain)
