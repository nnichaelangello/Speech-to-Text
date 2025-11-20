# config.py
class Config:
    sr = 22050
    max_seconds = 10.0
    max_samples = int(sr * max_seconds)
    n_mels = 80
    n_mfcc = 13
    total_channels = 119
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    fmin = 0
    fmax = 8000
    batch_size = 32
    lr = 3e-4
    epochs = 200
    grad_accum = 2
    num_classes = 38
    blank_id = 0
    checkpoint_dir = "checkpoints_stt"

config = Config()