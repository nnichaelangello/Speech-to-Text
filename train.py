# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.shared_cnn_backbone import SharedCNNBackbone
from model.stt_head import STTHead
from dataset import STTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = SharedCNNBackbone().to(device)
head = STTHead().to(device)
model = nn.Sequential(backbone, head).to(device)

criterion = nn.CTCLoss(blank=config.blank_id, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=config.lr)
scaler = GradScaler()

dataset = STTDataset("metadata_stt.txt")
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8)

for epoch in range(1, config.epochs+1):
    model.train()
    total_loss = 0
    for feats, labels, input_len, label_len in tqdm(loader, desc=f"Epoch {epoch}"):
        feats = feats.to(device)
        labels = labels.to(device)
        input_len = input_len.to(device)
        label_len = label_len.to(device)
        with autocast():
            log_probs = torch.log_softmax(model(feats), dim=-1)
            log_probs = log_probs.transpose(0,1)
            loss = criterion(log_probs, labels, input_len//4, label_len)
        scaler.scale(loss / config.grad_accum).backward()
        if (tqdm._instances[0].n + 1) % config.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch} - CTC Loss: {total_loss/len(loader):.4f}")
    if epoch % 20 == 0:
        torch.save(model.state_dict(), f"{config.checkpoint_dir}/stt_model_{epoch}.pt")