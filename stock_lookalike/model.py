"""Transformer 编码器：将时序窗口映射为 L2 归一化的 embedding"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class AttentionPooling(nn.Module):
    """用注意力权重对序列做加权平均，比直接取 [CLS] 或 mean pool 更灵活"""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        w = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        return (w * x).sum(dim=1)               # [B, D]


class StockTransformerEncoder(nn.Module):
    """
    Architecture:
        [B, T, 5]
        → Linear(5→d_model) + LayerNorm
        → LearnablePositionalEncoding
        → TransformerEncoderLayer × num_layers (Pre-LN)
        → AttentionPooling → [B, d_model]
        → ProjectionHead: Linear→ReLU→Linear → [B, embed_dim]
        → L2 normalize
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 128,
        window_size: int = 30,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = LearnablePositionalEncoding(window_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.pool = AttentionPooling(d_model)

        self.proj_head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 5]
        x = self.input_proj(x)        # [B, T, d_model]
        x = self.pos_enc(x)
        x = self.transformer(x)       # [B, T, d_model]
        x = self.pool(x)              # [B, d_model]
        x = self.proj_head(x)         # [B, embed_dim]
        x = F.normalize(x, dim=-1)    # L2 归一化
        return x


def build_model(config: dict) -> StockTransformerEncoder:
    m = config.get("model", config)
    return StockTransformerEncoder(
        input_dim=m.get("input_dim", 5),
        d_model=m.get("d_model", 64),
        nhead=m.get("nhead", 4),
        num_layers=m.get("num_layers", 3),
        ffn_dim=m.get("ffn_dim", 256),
        dropout=m.get("dropout", 0.1),
        embed_dim=m.get("embed_dim", 128),
        window_size=config.get("data", {}).get("window_size", 30),
    )
