import torch.nn as nn
from timm import create_model

class SwinEncoder(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True):
        super().__init__()
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        self.embed_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)
