import torch.nn as nn
from .swin_encoder import SwinEncoder
from .prediction_head import PredictionHead

class ImageOnlyModel(nn.Module):
    """
    Baseline model:
    Image → Swin Transformer → Severity Score
    """

    def __init__(self):
        super().__init__()

        self.encoder = SwinEncoder()
        feature_dim = self.encoder.embed_dim

        self.head = PredictionHead(feature_dim)

    def forward(self, image):
        features = self.encoder(image)   # (B, D)
        return self.head(features)        # (B, 1)
