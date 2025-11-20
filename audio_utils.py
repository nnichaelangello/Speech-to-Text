# audio_utils.py
import librosa
import numpy as np
import soundfile as sf

def load_and_preprocess(path):
    audio, sr_orig = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:,0]
    audio = audio.astype(np.float32)
    if sr_orig != config.sr:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=config.sr)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    if len(audio) > config.max_samples:
        audio = audio[:config.max_samples]
    else:
        audio = np.pad(audio, (0, config.max_samples - len(audio)))
    return audio