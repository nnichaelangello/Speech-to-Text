# inference.py
import torch
from model.shared_cnn_backbone import SharedCNNBackbone
from model.stt_head import STTHead
from audio_utils import load_and_preprocess
from vad import apply_vad
from feature_extractor import extract_119_features
from ctc_decode import greedy_ctc_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = SharedCNNBackbone().to(device)
head = STTHead().to(device)
model = nn.Sequential(backbone, head).to(device)
model.load_state_dict(torch.load("checkpoints_stt/stt_model_200.pt", map_location=device))
model.eval()

def recognize(wav_path):
    audio = load_and_preprocess(wav_path)
    audio = apply_vad(audio)
    tensor = torch.from_numpy(audio).unsqueeze(0)
    feat = extract_119_features(tensor).to(device)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(feat), dim=-1)
    text = greedy_ctc_decode(log_probs)
    print("Hasil STT:", text)
    return text

recognize("test.wav")