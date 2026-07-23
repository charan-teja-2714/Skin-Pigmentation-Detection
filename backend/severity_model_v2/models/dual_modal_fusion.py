import os
import sys
import torch
import torch.nn as nn

# Add models directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from swin_encoder import SwinEncoder
from cross_attention import CrossAttention
from prediction_head import PredictionHead


class DualModalFusionModel(nn.Module):
    """
    Dual-modal fusion model with separate Swin encoders for clinical and dermoscopy images.

    Architecture:
    - Clinical Swin Encoder: processes unmasked clinical images
    - Dermoscopy Swin Encoder: processes masked dermoscopy images
    - Cross-Attention Fusion: fuses features from both modalities
    - Prediction Head: outputs severity score
    """

    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()

        # Separate encoders for each modality
        self.clinical_encoder = SwinEncoder(model_name=model_name, pretrained=pretrained)
        self.dermoscopy_encoder = SwinEncoder(model_name=model_name, pretrained=pretrained)

        # Get embedding dimension
        self.embed_dim = self.clinical_encoder.embed_dim

        # Cross-attention for fusion
        self.cross_attn = CrossAttention(self.embed_dim)

        # Prediction head
        self.head = PredictionHead(self.embed_dim)

    def forward(self, clinical_img, dermoscopy_img):
        """
        Args:
            clinical_img: (B, 3, 224, 224) - unmasked clinical images
            dermoscopy_img: (B, 3, 224, 224) - masked dermoscopy images

        Returns:
            severity_score: (B, 1) - predicted severity scores
        """
        # Encode both modalities separately
        clinical_feat = self.clinical_encoder(clinical_img)  # (B, embed_dim)
        dermo_feat = self.dermoscopy_encoder(dermoscopy_img)  # (B, embed_dim)

        # Add sequence dimension for cross-attention
        clinical_feat = clinical_feat.unsqueeze(1)  # (B, 1, embed_dim)
        dermo_feat = dermo_feat.unsqueeze(1)  # (B, 1, embed_dim)

        # Fuse features using cross-attention
        # Query: clinical, Key/Value: dermoscopy
        fused_feat = self.cross_attn(clinical_feat, dermo_feat)  # (B, 1, embed_dim)

        # Remove sequence dimension
        fused_feat = fused_feat.squeeze(1)  # (B, embed_dim)

        # Predict severity score
        severity_score = self.head(fused_feat)  # (B, 1)

        return severity_score
