import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.q = nn.Linear(feature_dim, feature_dim)
        self.k = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Linear(feature_dim, feature_dim)

        self.out = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, query, key_value):
        B = query.size(0)

        Q = self.q(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)

        return self.norm(self.out(out) + query)
