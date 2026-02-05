import os
import sys
import torch.nn as nn

# Add models directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from swin_encoder import SwinEncoder
from cross_attention import CrossAttention
from prediction_head import PredictionHead

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SwinEncoder()
        D = self.encoder.embed_dim

        self.meta_encoder = nn.Sequential(
            nn.Linear(3, D),
            nn.LayerNorm(D),
            nn.ReLU()
        )

        self.cross_attn = CrossAttention(D)
        self.head = PredictionHead(D)

    def forward(self, image, metadata):
        img_feat = self.encoder(image).unsqueeze(1)
        meta_feat = self.meta_encoder(metadata).unsqueeze(1)

        fused = self.cross_attn(img_feat, meta_feat).squeeze(1)
        return self.head(fused)
