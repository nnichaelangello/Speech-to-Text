# dataset.py
import torch
from torch.utils.data import Dataset
from audio_utils import load_and_preprocess
from vad import apply_vad
from feature_extractor import extract_119_features
from text.tokenizer import text_to_labels

class STTDataset(Dataset):
    def __init__(self, metadata_path):
        self.items = []
        with open(metadata_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                wav = parts[0]
                text = parts[1]
                self.items.append((wav, text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text = self.items[idx]
        audio = load_and_preprocess(wav_path)
        audio = apply_vad(audio)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        features = extract_119_features(audio_tensor)
        labels = torch.LongTensor(text_to_labels(text))
        return features, labels

    def collate_fn(self, batch):
        feats = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        input_lengths = torch.LongTensor([f.shape[2] for f in feats])
        label_lengths = torch.LongTensor([len(l) for l in labels])
        max_feat = max(input_lengths)
        max_label = max(label_lengths)
        feat_padded = torch.zeros(len(batch), config.total_channels, max_feat, 1)
        label_padded = torch.zeros(len(batch), max_label, dtype=torch.long)
        for i in range(len(batch)):
            feat_padded[i,:,:feats[i].shape[2],:] = feats[i]
            label_padded[i,:len(labels[i])] = labels[i]
        return feat_padded, label_padded, input_lengths, label_lengths