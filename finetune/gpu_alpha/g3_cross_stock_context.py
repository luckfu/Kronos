"""G3: Cross-Stock Peer & Market-Context Attention Network over daily hidden states.

Unlike per-stock independent Kronos scoring, G3 processes an entire trading day's
cross-section [N_stocks, D] simultaneously by pooling:
1) Daily market global summary tokens (mean, std, q25, q75 across all N stocks)
2) Same-sector peer centroid + cross-attention over sector prototypes (86 sectors)
so every stock is scored relative to its same-day sector peers and market regime.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossStockPeerRanker(nn.Module):
    """Linear skip + zero-init Sector/Market Peer Context Attention for daily cross-sections."""

    def __init__(self, in_dim: int = 512, proj_dim: int = 128, num_sectors: int = 86, num_heads: int = 4, dropout: float = 0.35):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_sectors = int(num_sectors)
        self.skip = nn.Linear(in_dim, 1)

        self.stock_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            nn.SiLU(),
        )
        self.sector_emb = nn.Embedding(num_sectors + 1, proj_dim)
        self.mkt_proj = nn.Linear(in_dim * 2, proj_dim)
        self.peer_attn = nn.MultiheadAttention(proj_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ctx_norm = nn.LayerNorm(proj_dim * 3)
        self.ctx_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 3, 1),
        )
        # Start cleanly from the strong linear skip baseline
        nn.init.zeros_(self.ctx_head[-1].weight)
        nn.init.zeros_(self.ctx_head[-1].bias)

    def forward(self, x: torch.Tensor, sector_id: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim] daily cross-sectionally z-scored hidden states; sector_id: [N] int64."""
        n = x.shape[0]
        base = self.skip(x).squeeze(-1)
        sec_clamped = sector_id.clamp(0, self.num_sectors - 1)
        z = self.stock_proj(x) + self.sector_emb(sec_clamped)  # [N, P]

        # Build same-day sector centroids via scatter_add -> O(N)
        sec_sum = torch.zeros((self.num_sectors, z.shape[1]), device=x.device, dtype=z.dtype)
        sec_cnt = torch.zeros((self.num_sectors, 1), device=x.device, dtype=z.dtype)
        sec_sum.index_add_(0, sec_clamped, z)
        sec_cnt.index_add_(0, sec_clamped, torch.ones((n, 1), device=x.device, dtype=z.dtype))
        present = (sec_cnt.squeeze(-1) > 0)
        sec_mean = sec_sum / sec_cnt.clamp_min(1.0)
        own_sec = sec_mean[sec_clamped]  # [N, P]

        # Global market summary token (mean + std of x projected into P)
        mkt_stat = torch.cat([x.mean(dim=0, keepdim=True), x.std(dim=0, unbiased=False, keepdim=True)], dim=-1)
        mkt_tok = self.mkt_proj(mkt_stat)  # [1, P]

        # Keys/values = active sector prototypes + global market token -> <= 87 tokens
        kv_tokens = torch.cat([sec_mean[present], mkt_tok], dim=0).unsqueeze(0)  # [1, K, P]
        attn_out, _ = self.peer_attn(z.unsqueeze(0), kv_tokens, kv_tokens, need_weights=False)
        attn_out = attn_out.squeeze(0)  # [N, P]

        ctx = self.ctx_norm(torch.cat([z - own_sec, attn_out, z * mkt_tok], dim=-1))
        return base + self.ctx_head(ctx).squeeze(-1)
