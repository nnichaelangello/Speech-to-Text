# vad.py
import webrtcvad
import numpy as np
import config

vad = webrtcvad.Vad(3)

def apply_vad(audio):
    audio_int16 = np.int16(audio * 32767)
    frame_duration = 30
    frames = [audio_int16[i:i+int(config.sr*0.03)] for i in range(0, len(audio_int16), int(config.sr*0.03))]
    voiced = [vad.is_speech(frame.tobytes(), config.sr) for frame in frames]
    mask = np.repeat(voiced, int(config.sr*0.03))
    mask = np.pad(mask, (0, len(audio) - len(mask)), constant_values=False)
    return audio[mask]