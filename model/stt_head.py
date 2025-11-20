# model/stt_head.py
import torch.nn as nn

class STTHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(512, config.num_classes)

    def forward(self, x):
        return self.classifier(x)