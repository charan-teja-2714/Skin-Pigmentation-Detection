import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, feature_dim=768, num_heads=8):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, query, key_value):
        """
        query:      (B, Nq, D)
        key_value:  (B, Nk, D)
        """
        B = query.size(0)

        Q = self.query_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.feature_dim)

        output = self.output_proj(attn_output)

        # Residual + Norm (VERY IMPORTANT)
        return self.norm(output + query)
