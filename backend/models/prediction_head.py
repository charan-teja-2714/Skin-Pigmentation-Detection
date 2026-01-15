import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()   # output in range [0, 1]
        )

    def forward(self, x):
        """
        x: (B, feature_dim)
        return: (B, 1) pigmentation severity score
        """
        return self.regressor(x)
