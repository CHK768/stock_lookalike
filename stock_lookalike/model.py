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


class DualChannelStockEncoder(nn.Module):
    """
    双通道架构：价格（OHLC）和成交量（Volume）各走独立 Transformer 分支。

    Architecture:
        Input [B, T, 5]
        ├── OHLC   [B,T,4] → Linear(4→64)+LN → PosEnc → Transformer×3 → AttnPool → Linear(64→96)  ─┐
        └── Volume [B,T,1] → Linear(1→32)+LN → PosEnc → Transformer×2 → AttnPool → Linear(32→32)  ─┘
                                                                               concat [B, 128]
                                                               ProjectionHead(128→128→128)
                                                                     L2 normalize → [B, 128]
    """

    def __init__(
        self,
        price_d_model: int = 64,  price_nhead: int = 4, price_num_layers: int = 3,
        price_ffn_dim: int = 256, price_embed: int = 96,
        vol_d_model: int = 32,   vol_nhead: int = 4,   vol_num_layers: int = 2,
        vol_ffn_dim: int = 128,   vol_embed: int = 32,
        dropout: float = 0.1, embed_dim: int = 128, window_size: int = 30,
    ):
        super().__init__()

        # ── Price branch (OHLC, 4 features) ──────────────────────────────────
        self.price_proj = nn.Sequential(
            nn.Linear(4, price_d_model),
            nn.LayerNorm(price_d_model),
        )
        self.price_pos_enc = LearnablePositionalEncoding(window_size, price_d_model)
        price_layer = nn.TransformerEncoderLayer(
            d_model=price_d_model, nhead=price_nhead,
            dim_feedforward=price_ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.price_transformer = nn.TransformerEncoder(
            price_layer, num_layers=price_num_layers, enable_nested_tensor=False,
        )
        self.price_pool = AttentionPooling(price_d_model)
        self.price_embed = nn.Linear(price_d_model, price_embed)

        # ── Volume branch (1 feature) ─────────────────────────────────────────
        self.vol_proj = nn.Sequential(
            nn.Linear(1, vol_d_model),
            nn.LayerNorm(vol_d_model),
        )
        self.vol_pos_enc = LearnablePositionalEncoding(window_size, vol_d_model)
        vol_layer = nn.TransformerEncoderLayer(
            d_model=vol_d_model, nhead=vol_nhead,
            dim_feedforward=vol_ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.vol_transformer = nn.TransformerEncoder(
            vol_layer, num_layers=vol_num_layers, enable_nested_tensor=False,
        )
        self.vol_pool = AttentionPooling(vol_d_model)
        self.vol_embed = nn.Linear(vol_d_model, vol_embed)

        # ── Cross-branch attention（双向量价交互） ────────────────────────────
        # price attend to vol：价格 token 感知成交量方向
        self.cross_attn_pv = nn.MultiheadAttention(
            embed_dim=price_d_model,
            num_heads=price_nhead,
            kdim=vol_d_model,
            vdim=vol_d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_p = nn.LayerNorm(price_d_model)
        # vol attend to price：成交量 token 感知关键价格时间点
        self.cross_attn_vp = nn.MultiheadAttention(
            embed_dim=vol_d_model,
            num_heads=vol_nhead,
            kdim=price_d_model,
            vdim=price_d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_v = nn.LayerNorm(vol_d_model)

        # ── Fusion head ───────────────────────────────────────────────────────
        fused_dim = price_embed + vol_embed  # 96 + 32 = 128
        self.proj_head = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 5]
        price = x[..., :4]   # [B, T, 4]
        vol   = x[..., 4:]   # [B, T, 1]

        # Price path
        p = self.price_proj(price)
        p = self.price_pos_enc(p)
        p = self.price_transformer(p)   # [B, T, price_d_model]

        # Volume path
        v = self.vol_proj(vol)
        v = self.vol_pos_enc(v)
        v = self.vol_transformer(v)     # [B, T, vol_d_model]

        # 双向 cross-branch attention
        cross_pv, _ = self.cross_attn_pv(query=p, key=v, value=v)
        p = self.cross_norm_p(p + cross_pv)   # price 感知量价方向
        cross_vp, _ = self.cross_attn_vp(query=v, key=p, value=p)
        v = self.cross_norm_v(v + cross_vp)   # vol 感知关键价格时间点

        p = self.price_pool(p)
        p = self.price_embed(p)   # [B, price_embed]
        v = self.vol_pool(v)
        v = self.vol_embed(v)     # [B, vol_embed]

        # Fuse & project
        fused = torch.cat([p, v], dim=-1)   # [B, 128]
        out = self.proj_head(fused)          # [B, embed_dim]
        return F.normalize(out, dim=-1)


def build_model(config: dict) -> nn.Module:
    m = config.get("model", config)
    model_type = m.get("model_type", "single")
    window_size = config.get("data", {}).get("window_size", 30)

    if model_type == "dual_channel":
        return DualChannelStockEncoder(
            price_d_model=m.get("price_d_model", 64),
            price_nhead=m.get("price_nhead", 4),
            price_num_layers=m.get("price_num_layers", 3),
            price_ffn_dim=m.get("price_ffn_dim", 256),
            price_embed=m.get("price_embed", 96),
            vol_d_model=m.get("vol_d_model", 32),
            vol_nhead=m.get("vol_nhead", 4),
            vol_num_layers=m.get("vol_num_layers", 2),
            vol_ffn_dim=m.get("vol_ffn_dim", 128),
            vol_embed=m.get("vol_embed", 32),
            dropout=m.get("dropout", 0.1),
            embed_dim=m.get("embed_dim", 128),
            window_size=window_size,
        )
    return StockTransformerEncoder(
        input_dim=m.get("input_dim", 5),
        d_model=m.get("d_model", 64),
        nhead=m.get("nhead", 4),
        num_layers=m.get("num_layers", 3),
        ffn_dim=m.get("ffn_dim", 256),
        dropout=m.get("dropout", 0.1),
        embed_dim=m.get("embed_dim", 128),
        window_size=window_size,
    )
