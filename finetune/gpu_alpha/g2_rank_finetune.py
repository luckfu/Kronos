"""G2: End-to-end top-block Cosine-IC + Multi-Horizon + C2-KL-Anchor fine-tuning.

Freezes tokenizer, embeddings, condition layers (sector/size), and lower Transformer
blocks (0..freeze_until-1). Unfreezes only top Transformer blocks (freeze_until..n_layers-1),
final norm, and a linear multi-view rank head over [h_last, h_m5].

Supports two arms:
- fixed L=120 ('l120')
- variable lookback L in {60, 90, 120} ('varlen'), evaluated via L=60 + L=120 ensemble.
"""
from __future__ import annotations

import copy
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from finetune.gpu_alpha.g1_multiscale_extract_and_fit import (
        FEATURES,
        TRAIN_FIT_END,
        VAL_SEL_END,
        VAL_SEL_START,
        lookup_conditions_for_symbol,
        orthogonalize_target,
        phase,
        rank_pct_centered,
        standardize_1d,
    )
except ImportError:
    from g1_multiscale_extract_and_fit import (
        FEATURES,
        TRAIN_FIT_END,
        VAL_SEL_END,
        VAL_SEL_START,
        lookup_conditions_for_symbol,
        orthogonalize_target,
        phase,
        rank_pct_centered,
        standardize_1d,
    )

TRAIN_START_G2 = "2020-01-01"


class TopBlockRankAdapter(nn.Module):
    """Wraps Kronos with frozen lower layers and trainable top layers + linear rank head."""

    def __init__(self, kronos_model, freeze_until: int = 5):
        super().__init__()
        self.freeze_until = int(freeze_until)
        # Keep full model reference so weights save/merge seamlessly into Kronos state_dict
        self.model = kronos_model
        # Freeze everything first
        for p in self.model.parameters():
            p.requires_grad = False
        # Unfreeze only top Transformer blocks + final norm
        for idx in range(self.freeze_until, len(self.model.transformer)):
            for p in self.model.transformer[idx].parameters():
                p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True

        # Frozen teacher copies of top blocks + norm + s1 head for KL anchor (eval only)
        self.teacher_blocks = nn.ModuleList([
            copy.deepcopy(self.model.transformer[idx]).eval()
            for idx in range(self.freeze_until, len(self.model.transformer))
        ])
        self.teacher_norm = copy.deepcopy(self.model.norm).eval()
        self.teacher_head = copy.deepcopy(self.model.head).eval()
        for m in (self.teacher_blocks, self.teacher_norm, self.teacher_head):
            for p in m.parameters():
                p.requires_grad = False

        # Keep snapshot of original C2 top params for L2 anchor regularization
        self.c2_anchor = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        d_model = int(self.model.d_model)
        self.rank_norm = nn.LayerNorm(2 * d_model)
        self.rank_drop = nn.Dropout(0.3)
        self.rank_head = nn.Linear(2 * d_model, 1)

    @torch.no_grad()
    def encode_frozen_lower(
        self,
        tokenizer,
        raw_hist: torch.Tensor,
        stamp_hist: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize trailing raw_hist [B, L, 6] and run frozen tokenizer + layers 0..freeze_until-1."""
        device = raw_hist.device
        mean = raw_hist.mean(dim=1, keepdim=True)
        std = raw_hist.std(dim=1, unbiased=False, keepdim=True)
        x_norm = torch.clamp((raw_hist - mean) / (std + 1e-5), -5.0, 5.0)
        # Keep frozen modules deterministic in eval mode
        was_training = self.model.training
        self.model.eval()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            tokens = tokenizer.encode(x_norm, half=True)
            h = self.model.embedding([tokens[0], tokens[1]])
            if stamp_hist is not None:
                h = h + self.model.time_emb(stamp_hist)
            for idx in range(self.freeze_until):
                h = self.model.transformer[idx](h, key_padding_mask=None)
        if was_training:
            self.model.train()
            # Re-freeze batch-like eval behavior on frozen condition/embed layers
            self.model.embedding.eval()
            self.model.time_emb.eval()
            for idx in range(self.freeze_until):
                self.model.transformer[idx].eval()
            if self.model.sector_emb is not None:
                self.model.sector_emb.eval()
            if self.model.size_mlp is not None:
                self.model.size_mlp.eval()
        return h.detach()

    def forward_top(
        self,
        h_lower: torch.Tensor,
        sector_id: torch.Tensor,
        size_pct: torch.Tensor,
        with_teacher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run trainable layers freeze_until..end and optionally frozen teacher s1 logits."""
        device = h_lower.device
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = h_lower
            for idx in range(self.freeze_until, len(self.model.transformer)):
                if self.model.context_layer > 0 and idx == self.model.context_layer:
                    h = self.model._add_context(h, sector_id=sector_id, size_percentile=size_pct)
                h = self.model.transformer[idx](h, key_padding_mask=None)
            h_out = self.model.norm(h).float()
            s1_ft = self.model.head(h_out[:, -1, :].to(h.dtype)).float()

        feat = torch.cat([h_out[:, -1, :], h_out[:, -5:, :].mean(dim=1)], dim=-1)
        score = self.rank_head(self.rank_drop(self.rank_norm(feat))).squeeze(-1)

        s1_c2 = None
        if with_teacher:
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ht = h_lower
                for rel_idx, layer in enumerate(self.teacher_blocks):
                    abs_idx = self.freeze_until + rel_idx
                    if self.model.context_layer > 0 and abs_idx == self.model.context_layer:
                        ht = self.model._add_context(ht, sector_id=sector_id, size_percentile=size_pct)
                    ht = layer(ht, key_padding_mask=None)
                ht = self.teacher_norm(ht)
                s1_c2 = self.teacher_head(ht[:, -1, :]).float()
        return score, s1_ft, s1_c2

    def weight_l2_anchor(self) -> torch.Tensor:
        """Mean squared deviation from original C2 top-block parameters."""
        total = None
        count = 0
        for name, p in self.model.named_parameters():
            if name in self.c2_anchor:
                diff = (p - self.c2_anchor[name].to(p.device)).pow(2).sum()
                total = diff if total is None else (total + diff)
                count += p.numel()
        if total is None or count == 0:
            return torch.tensor(0.0)
        return total / float(count)

    def extract_top_state_dict(self) -> dict[str, torch.Tensor]:
        """Return trainable top-block + norm + rank_head weights cleanly on CPU."""
        out = {}
        for name, p in self.model.named_parameters():
            if name in self.c2_anchor:
                out[f"model.{name}"] = p.detach().half().cpu()
        for k, v in self.rank_norm.state_dict().items():
            out[f"rank_norm.{k}"] = v.detach().float().cpu()
        for k, v in self.rank_head.state_dict().items():
            out[f"rank_head.{k}"] = v.detach().float().cpu()
        return out


def cosine_ic_loss_vec(pred: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """1 - cross-sectional Pearson correlation + mild variance stabilization."""
    p = pred - pred.mean()
    t = target_z - target_z.mean()
    cos = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
    p_std = p / p.std(unbiased=False).clamp_min(1e-6)
    return (1.0 - cos) + 0.05 * F.mse_loss(p_std, t)


def kl_logits_anchor(s1_c2: torch.Tensor, s1_ft: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    """Forward KL(p_C2 || p_FT) preventing generative manifold collapse."""
    log_p_c2 = F.log_softmax(s1_c2 / temperature, dim=-1)
    log_p_ft = F.log_softmax(s1_ft / temperature, dim=-1)
    p_c2 = log_p_c2.exp()
    return (p_c2 * (log_p_c2 - log_p_ft)).sum(dim=-1).mean() * (temperature ** 2)


def build_g2_date_batches(
    pkl_path: Path,
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    date_from: str,
    date_to: str,
    min_cross_section: int = 100,
    default_sector: int = 85,
) -> tuple[list[tuple], list[tuple]]:
    """Build per-symbol compact arrays and per-date multi-horizon cross-sectional targets."""
    with pkl_path.open("rb") as f:
        raw_panel: dict[str, pd.DataFrame] = pickle.load(f)

    sym_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    date_records: dict[str, list[tuple[int, int, int, float, float, float, float, float]]] = {}

    for symbol, df in raw_panel.items():
        df = df.sort_index()
        if len(df) < 131:
            continue
        idx = pd.DatetimeIndex(df.index)
        raw_vals = df[FEATURES].to_numpy(dtype=np.float32)
        stamps = np.column_stack([idx.minute, idx.hour, idx.weekday, idx.day, idx.month]).astype(np.float32)
        dates_str = idx.strftime("%Y-%m-%d").to_numpy()
        dates_ns = idx.to_numpy(dtype="datetime64[ns]").view(np.int64)
        sec_arr, pct_arr = lookup_conditions_for_symbol(lookup, str(symbol), dates_ns, default_sector)
        if "size_percentile" in df.columns:
            inline_pct = pd.to_numeric(df["size_percentile"], errors="coerce").to_numpy(dtype=np.float32)
            pct_arr = np.where(np.isfinite(inline_pct), inline_pct, pct_arr)
        sym_arrays.append((raw_vals, stamps))
        s_i = len(sym_arrays) - 1
        closes = raw_vals[:, 3].astype(np.float64)

        valid_t = np.where((dates_str >= date_from) & (dates_str <= date_to))[0]
        for t in valid_t:
            if t < 119 or t + 10 >= len(df):
                continue
            c0 = closes[t]
            cp = closes[t - 10]
            if c0 <= 1e-8 or cp <= 1e-8:
                continue
            r3 = float(closes[t + 3] / c0 - 1.0)
            r5 = float(closes[t + 5] / c0 - 1.0)
            r10 = float(closes[t + 10] / c0 - 1.0)
            if not (math.isfinite(r3) and math.isfinite(r5) and math.isfinite(r10)):
                continue
            rev10_past = float(-(c0 / cp - 1.0))
            date_records.setdefault(str(dates_str[t]), []).append(
                (s_i, int(t), int(sec_arr[t]), float(pct_arr[t]), r3, r5, r10, rev10_past)
            )

    batches = []
    for d_str in sorted(date_records.keys()):
        recs = date_records[d_str]
        if len(recs) < min_cross_section:
            continue
        sec_np = np.asarray([r[2] for r in recs], dtype=np.int64)
        pct_np = np.asarray([r[3] for r in recs], dtype=np.float64)
        r3 = np.asarray([r[4] for r in recs], dtype=np.float64)
        r5 = np.asarray([r[5] for r in recs], dtype=np.float64)
        r10 = np.asarray([r[6] for r in recs], dtype=np.float64)
        rev10 = np.asarray([r[7] for r in recs], dtype=np.float64)
        y_multi = standardize_1d(
            0.15 * standardize_1d(rank_pct_centered(r3))
            + 0.25 * standardize_1d(rank_pct_centered(r5))
            + 0.60 * standardize_1d(rank_pct_centered(r10))
        ).astype(np.float32)
        y_raw10, y_orth10 = orthogonalize_target(r10, pct_np, rev10, sec_np)
        s_idx = np.asarray([r[0] for r in recs], dtype=np.int32)
        t_idx = np.asarray([r[1] for r in recs], dtype=np.int32)
        batches.append((
            d_str,
            s_idx,
            t_idx,
            sec_np,
            pct_np.astype(np.float32),
            y_multi,
            y_orth10.astype(np.float32),
            r10.astype(np.float32),
        ))
    return sym_arrays, batches
