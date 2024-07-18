import torch
from torch import nn
import math
from config import *

class TimePositionEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.half_emb_size = embedding_size // 2
        num = torch.arange(self.half_emb_size, dtype=torch.float32)
        self.div_term = torch.exp(-math.log(10000.0) * num / (self.half_emb_size - 1)).to(DEVICE)

    def forward(self, t):
        t = t.float().unsqueeze(1)  #  [N, 1]
        encodings = t * self.div_term.unsqueeze(0) # [N, half_emb_size]
        encodings = torch.cat([torch.sin(encodings), torch.cos(encodings)], dim=-1).to(DEVICE)
        return encodings


