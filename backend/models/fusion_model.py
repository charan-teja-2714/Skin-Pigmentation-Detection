import torch
import torch.nn as nn
from .swin_encoder import SwinEncoder
from .cross_attention import CrossAttention
from .prediction_head import PredictionHead

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders (do NOT pass feature_dim manually)
        self.clinical_encoder = SwinEncoder()
        self.dermoscopy_encoder = SwinEncoder()
        self.multispectral_encoder = SwinEncoder()

        feature_dim = self.clinical_encoder.embed_dim

        self.cross_attention = CrossAttention(
            feature_dim=feature_dim,
            num_heads=8
        )

        self.prediction_head = PredictionHead(feature_dim)

    def forward(self, clinical_img, dermoscopy_img=None, multispectral_img=None):
        # Encode clinical image (Query)
        clinical_features = self.clinical_encoder(clinical_img)  # (B, D)

        key_value_features = []

        # Encode dermoscopy image
        if dermoscopy_img is not None:
            dermo_features = self.dermoscopy_encoder(dermoscopy_img)
            key_value_features.append(dermo_features)

        # Encode multispectral image
        if multispectral_img is not None:
            multi_features = self.multispectral_encoder(multispectral_img)
            key_value_features.append(multi_features)

        # If no auxiliary modality is provided (fallback)
        if len(key_value_features) == 0:
            key_value_features.append(torch.zeros_like(clinical_features))

        # Stack dermoscopy & multispectral as separate tokens
        key_value_features = torch.stack(key_value_features, dim=1)  # (B, N, D)

        # Prepare query
        query = clinical_features.unsqueeze(1)  # (B, 1, D)

        # Cross-attention fusion
        fused_features = self.cross_attention(query, key_value_features)
        fused_features = fused_features.squeeze(1)  # (B, D)

        # Prediction
        score = self.prediction_head(fused_features)
        return score
