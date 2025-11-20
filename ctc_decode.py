# ctc_decode.py
import torch

def greedy_ctc_decode(log_probs):
    indices = torch.argmax(log_probs, dim=-1)
    decoded = []
    prev = -1
    for i in indices[0]:
        if i != prev and i != config.blank_id:
            decoded.append(id_to_symbol[i.item()])
        prev = i
    return ''.join(decoded)