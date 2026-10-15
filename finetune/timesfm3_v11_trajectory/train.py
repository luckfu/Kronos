"""TimesFM-3 LoRA + Sector Embedding V11: 走势大模型 (Rank-16 扩容 + 余弦退火调度版)。

核心升级:
1. LoRA 参数表达容量翻倍 (rank=16, alpha=32)，突破低秩瓶颈，充分容纳全市场 5000+ 股票复杂走势;
2. 学习率余弦退火调度 (CosineAnnealingLR: 1e-4 -> 1e-5)，在后期沉淀极小值，消除走势预测震荡;
3. 8 通道端到端特征注入 (OHLCV + 市值百分位 + 可学习证监会行业风格嵌入);
4. 价格去量纲锚定 (P0=1.0) + 收益率 Huber 损失 + 趋势方向感知损失;
5. 专为 12 小时长周期深度微调设计，支持断点状态接力。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

STARTED = time.monotonic()
INPUT_ROOT = Path(os.environ.get("RELAY_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("RELAY_OUTPUT", "/kaggle/working/relay"))

SWANLAB_API_KEY = "fmEPDGk4IItxgqSZKGLi8"
SWANLAB_PROJECT = "finance"
SWANLAB_WORKSPACE = "roc_fu"
SWANLAB_RUN_ID = "tfm3v11rank16powerlaw"  # V11 幂律重要性采样 + 振幅方差正则
EXPERIMENT_NAME = "timesfm3-v11-rank16-powerlaw"

QUANTILE_INDEX = 4
DATA_FEATURES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "size_percentile",
)
PRICE_INDICES = (0, 1, 2, 3)  # open, high, low, close
COVARIATE_INDICES = (0, 1, 2, 4, 5, 6)
MODEL_CHANNEL_INDICES = (3, *COVARIATE_INDICES)  # 与 build_forward_inputs 的通道 0~6 顺序一致
NAN_BATCH_DUMP_LIMIT = 32


def log(event: str, **fields) -> None:
    line = json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                      ensure_ascii=False, default=float)
    print(line, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def gpu_metrics() -> dict[str, float]:
    import torch
    if not torch.cuda.is_available():
        return {"gpu/count": 0.0}
    allocated = [torch.cuda.memory_allocated(i) / (1024 ** 3) for i in range(torch.cuda.device_count())]
    reserved = [torch.cuda.memory_reserved(i) / (1024 ** 3) for i in range(torch.cuda.device_count())]
    metrics = {"gpu/count": float(len(allocated)),
               "gpu/max_allocated_gb": float(max(allocated)),
               "gpu/max_reserved_gb": float(max(reserved))}
    for i, (a, r) in enumerate(zip(allocated, reserved)):
        metrics[f"gpu/{i}_allocated_gb"] = float(round(a, 3))
    return metrics


# ----------------------------------------------------------------- 数据集加载与去量纲采样

def discover_dataset(root: Path) -> tuple[Path, Path, Path]:
    train_pkls = sorted(root.glob("**/train_data.pkl"))
    val_pkls = sorted(root.glob("**/val_data.pkl"))
    splits = sorted(root.glob("**/symbol_split.csv"))
    if not train_pkls or not val_pkls or not splits:
        raise FileNotFoundError(f"在 {root} 下未找到完整的原生数据集 (需要 train_data.pkl, val_data.pkl, symbol_split.csv)")
    return train_pkls[0], val_pkls[0], splits[0]


class FastStockPool:
    """全市场时序动态池 (C04 有上限的幂律加权混合采样版):
    1. 窗口级别拉平池 (Window-level Pool)，消除次新股偏见;
    2. 路径最大波幅 A_i = max(h=1..10, abs(close[119+h]/close[119] - 1.0));
    3. 40% 均匀基底 + 60% 幂律加权 (4.0倍硬封顶);
    4. 训练前预冻结确定性抽样计划 (带 SHA-256 哈希校验);
    5. 独立 UNKNOWN_SECTOR_ID，彻底消除 0 号行业冲突。
    """

    def __init__(self, train_pkl: Path, symbol_split_csv: Path):
        import pandas as pd
        df_split = pd.read_csv(symbol_split_csv)
        self.sectors = sorted(df_split["sector"].dropna().unique().tolist())
        self.sector2id = {s: i for i, s in enumerate(self.sectors)}
        self.unknown_sector_id = len(self.sectors)
        self.symbol2sector = {
            row["symbol"]: self.sector2id.get(row["sector"], self.unknown_sector_id)
            for _, row in df_split.iterrows()
        }

        with train_pkl.open("rb") as f:
            panel = pickle.load(f)

        self.stocks = []
        self.windows = []  # 存放 (stock_idx, start_offset)
        amplitudes = []
        lookback, horizon = 120, 10
        window_len = lookback + horizon

        t0 = time.time()
        for symbol, df in panel.items():
            if not all(col in df.columns for col in DATA_FEATURES):
                continue
            df = df.sort_index()
            vals = df[list(DATA_FEATURES)].to_numpy(dtype=np.float32)
            if len(vals) < window_len:
                continue

            sec_id = self.symbol2sector.get(symbol, self.unknown_sector_id)
            s_idx = len(self.stocks)
            self.stocks.append((vals, sec_id))

            closes = vals[:, 3]
            max_start = len(vals) - window_len
            for start in range(max_start + 1):
                p0 = closes[start + lookback - 1]
                if p0 <= 1e-4:
                    continue
                w_sub = vals[start : start + window_len]
                if not (np.isfinite(w_sub).all() and (w_sub[:, :4] > 0).all() and (w_sub[:, 4] > 0).all()):
                    continue

                future_closes = closes[start + lookback : start + window_len]
                amp = np.max(np.abs(future_closes / p0 - 1.0))
                self.windows.append((s_idx, start))
                amplitudes.append(amp)

        if not self.stocks or not self.windows:
            raise RuntimeError("没有加载到合法的股票训练数据")

        self.amplitudes = np.asarray(amplitudes, dtype=np.float64)
        print(f"[Data Index] 全市场合法滑动窗口构建完成: {len(self.windows)} 个窗口 (来自 {len(self.stocks)} 只股票, 耗时 {time.time() - t0:.2f}s)", flush=True)

    def build_plan(self, total_samples: int, seed: int, ratio: float = 0.6, reference: float = 0.05, exponent: float = 1.0, cap: float = 4.0):
        """生成 C04 确定性有上限幂律加权混合采样计划并生成 SHA-256 校验码。"""
        weights = np.minimum(np.maximum(self.amplitudes / reference, 1.0) ** exponent, cap)
        probabilities = (1.0 - ratio) / len(weights) + ratio * weights / weights.sum()

        self.order = np.random.default_rng(seed).choice(
            len(weights), size=total_samples, replace=True, p=probabilities
        )
        self.sha256 = hashlib.sha256(self.order.astype("<i8").tobytes()).hexdigest()
        self.plan_ptr = 0
        self.total_samples = total_samples

        selected_amps = self.amplitudes[self.order]
        print(f"[Sampling Plan] 预冻结 C04 采样计划生成完成:", flush=True)
        print(f"  - 计划抽样数: {total_samples}, 候选窗口总数: {len(weights)}", flush=True)
        print(f"  - SHA-256 校验指纹: {self.sha256}", flush=True)
        print(f"  - 自然分布波幅 >= 5% 占比: {np.mean(self.amplitudes >= 0.05)*100:.2f}%, 均值: {np.mean(self.amplitudes)*100:.2f}%", flush=True)
        print(f"  - 抽样后波幅 >= 5% 占比: {np.mean(selected_amps >= 0.05)*100:.2f}%, 均值: {np.mean(selected_amps)*100:.2f}%", flush=True)

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """零拒绝、零重试、确定性高能批次输出。"""
        if self.plan_ptr + batch_size > len(self.order):
            self.plan_ptr = 0  # 循环复用
        batch_indices = self.order[self.plan_ptr : self.plan_ptr + batch_size]
        self.plan_ptr += batch_size

        contexts, targets, sectors = [], [], []
        for b_idx in batch_indices:
            s_idx, start = self.windows[b_idx]
            vals, sec = self.stocks[s_idx]
            window = vals[start : start + 130].copy()
            p0 = window[119, 3]
            window[:, list(PRICE_INDICES)] /= p0
            vol_med = np.median(window[:120, 4]) + 1e-4
            amt_med = np.median(window[:120, 5]) + 1e-4
            window[:, 4] /= vol_med
            window[:, 5] /= amt_med

            # 输入通道物理有界防护: 允许高达 50 倍中位数天量与极大价格动态范围，彻底阻断 Layer 0 注意力点积上溢
            window[:, 4] = np.clip(window[:, 4], 0.0, 50.0)
            window[:, 5] = np.clip(window[:, 5], 0.0, 50.0)
            window[:, list(PRICE_INDICES)] = np.clip(window[:, list(PRICE_INDICES)], 0.05, 10.0)

            contexts.append(window[:120])
            targets.append(window[120:130, 3])
            sectors.append(sec)

        return (
            np.asarray(contexts, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(sectors, dtype=np.int64),
        )


def build_validation_pool(val_pkl: Path, symbol_split_csv: Path, max_samples: int = 4096, seed: int = 20261009):
    import pandas as pd

    df_split = pd.read_csv(symbol_split_csv)
    sectors = sorted(df_split["sector"].dropna().unique().tolist())
    sector2id = {s: i for i, s in enumerate(sectors)}
    unknown_sector_id = len(sectors)
    symbol2sector = dict(zip(df_split["symbol"], df_split["sector"].map(sector2id).fillna(unknown_sector_id).astype(int)))

    with val_pkl.open("rb") as f:
        panel = pickle.load(f)

    start_date, end_date = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-07-02")
    contexts, targets, val_sectors = [], [], []
    sector_samples = defaultdict(list)

    sample_counter = 0
    for symbol, df in panel.items():
        if not all(col in df.columns for col in DATA_FEATURES):
            continue
        df = df.sort_index()
        vals = df[list(DATA_FEATURES)].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(df.index)
        sec = symbol2sector.get(symbol, unknown_sector_id)

        valid_mask = (dates >= start_date) & (dates <= end_date)
        valid_indices = np.where(valid_mask)[0]
        for idx in valid_indices:
            if idx >= 119 and idx + 10 < len(vals):
                w = vals[idx - 119 : idx + 11].copy()
                if np.isfinite(w).all() and (w[:, :4] > 0).all() and (w[:, 4] > 0).all():
                    p0 = w[119, 3]
                    if p0 > 1e-4:
                        w[:, list(PRICE_INDICES)] /= p0
                        vol_med = np.median(w[:120, 4]) + 1e-4
                        amt_med = np.median(w[:120, 5]) + 1e-4
                        w[:, 4] /= vol_med
                        w[:, 5] /= amt_med
                        w[:, 4] = np.clip(w[:, 4], 0.0, 50.0)
                        w[:, 5] = np.clip(w[:, 5], 0.0, 50.0)
                        w[:, list(PRICE_INDICES)] = np.clip(w[:, list(PRICE_INDICES)], 0.05, 10.0)

                        contexts.append(w[:120])
                        targets.append(w[120:130, 3])
                        val_sectors.append(sec)
                        sector_samples[sec].append(sample_counter)
                        sample_counter += 1

    contexts = np.asarray(contexts, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    val_sectors = np.asarray(val_sectors, dtype=np.int64)

    total_available = len(contexts)
    if max_samples and max_samples < total_available:
        base_quota = 10
        allocated = {}
        remaining_budget = max_samples
        for sec, idx_list in sector_samples.items():
            alloc = min(len(idx_list), base_quota)
            allocated[sec] = alloc
            remaining_budget -= alloc

        if remaining_budget > 0:
            for sec, idx_list in sector_samples.items():
                ratio = len(idx_list) / total_available
                extra = int(remaining_budget * ratio)
                allocated[sec] = min(len(idx_list), allocated[sec] + extra)

        current_total = sum(allocated.values())
        if current_total < max_samples:
            sorted_secs = sorted(sector_samples.keys(), key=lambda s: len(sector_samples[s]), reverse=True)
            for s in sorted_secs:
                if current_total >= max_samples:
                    break
                if allocated[s] < len(sector_samples[s]):
                    allocated[s] += 1
                    current_total += 1

        selected_indices = []
        for sec, quota in allocated.items():
            idx_list = np.array(sector_samples[sec])
            if quota >= len(idx_list):
                chosen = idx_list
            else:
                step = len(idx_list) / quota
                picks = [int(i * step) for i in range(quota)]
                chosen = idx_list[picks]
            selected_indices.extend(chosen)

        choice = np.sort(np.array(selected_indices, dtype=np.int64))
        log("val_stratified_sampled", total_val=total_available, sampled=len(choice), num_sectors=len(sector_samples))
        return contexts[choice], targets[choice], val_sectors[choice]

    return contexts, targets, val_sectors


def find_relay_state(root: Path) -> Path | None:
    best, best_step = None, -1
    candidates = list(root.glob("**/relay/relay.json")) + list(root.glob("**/relay.json"))
    for meta in candidates:
        try:
            info = json.loads(meta.read_text())
            if info.get("run_id") == SWANLAB_RUN_ID and info.get("global_step", -1) > best_step:
                target_state = meta.parent / "state.pt"
                if target_state.exists():
                    best, best_step = target_state, info["global_step"]
        except Exception:
            continue
    return best


# ----------------------------------------------------------------- 模型与损失函数

def build_forward_inputs(contexts, sectors, model, horizon: int, sector_embedding):
    import torch
    import torch.nn.functional as F

    batch, context, _ = contexts.shape
    patch_len = int(model.input_patch_len)
    ctx_padding = (patch_len - context % patch_len) % patch_len
    num_context_patches = (context + ctx_padding) // patch_len

    if getattr(model, "use_stitching", False):
        overlap = int(model._stitching_extract_len) - patch_len
        num_forecast_patches = max((horizon - overlap + patch_len - 1) // patch_len, 1)
        num_horizon_patches = num_forecast_patches + int(model.rolls) - 1
        padded_horizon = num_horizon_patches * patch_len
    else:
        out_len = int(model.output_patch_len)
        padded_horizon = (horizon + out_len - 1) // out_len * out_len
        num_horizon_patches = padded_horizon // patch_len

    total_patches = num_context_patches + num_horizon_patches

    # 1. 行业特征通道
    sec_vec = sector_embedding(sectors)  # (batch, patch_len)
    sec_channel = sec_vec[:, None, None, :].expand(batch, 1, total_patches, patch_len)

    # 2. 连续特征 (通道 0 是相对 close=1.0 锚定, 通道 1~6 为协变量)
    close = F.pad(contexts[..., 3:4].transpose(1, 2), (ctx_padding, padded_horizon), value=0.0)
    covariates = F.pad(contexts[..., list(COVARIATE_INDICES)].transpose(1, 2), (ctx_padding, padded_horizon), value=0.0)
    continuous = torch.cat([close, covariates], dim=1).reshape(batch, 7, total_patches, patch_len)

    # 3. 拼接得到完整的 8 维通道
    values = torch.cat([continuous, sec_channel], dim=1)

    masks = torch.zeros(batch, context + ctx_padding + padded_horizon, dtype=torch.bool, device=contexts.device)
    masks[:, :ctx_padding] = True
    masks[:, context + ctx_padding:] = True
    masks = masks[:, None, :].expand(batch, 8, -1).reshape(batch, 8, -1, patch_len)

    patch_is_target = torch.zeros(batch, 8, total_patches, dtype=torch.bool, device=contexts.device)
    patch_is_target[:, 0, :] = True

    cpm = torch.zeros(batch, total_patches, dtype=torch.bool, device=contexts.device)
    cpm[:, num_context_patches:] = True

    return {"values": values, "masks": masks, "patch_is_target": patch_is_target}, cpm, num_context_patches


def forecast(model, contexts, sectors, horizon: int, sector_embedding):
    inputs, cpm, ctx_patches = build_forward_inputs(contexts, sectors, model, horizon, sector_embedding)
    freeze = ctx_patches - 1 if getattr(model, "use_frozen_running_stats", False) else None
    logits = model(inputs, freeze_after=freeze, patch_cpm_mask=cpm)["logits"]
    return logits[:, 0, ctx_patches - 1, :, QUANTILE_INDEX][:, :horizon]


def alpha_loss(pred_rel_price, target_rel_price, dir_loss_weight: float = 0.5, amp_loss_weight: float = 0.1):
    """收益率路径 Huber 损失 + 趋势方向感知损失 + 振幅方差校准正则。"""
    import torch
    import torch.nn.functional as F

    pred_ret = pred_rel_price - 1.0
    true_ret = target_rel_price - 1.0
    # 温和 Winsorize: 过滤极端单点异常波动，防止极端数据冲击求导
    true_ret = torch.clamp(true_ret, min=-0.85, max=3.0)

    ret_loss = F.smooth_l1_loss(pred_ret, true_ret, beta=0.02)

    soft_sign_pred = torch.tanh(pred_ret * 20.0)
    soft_sign_true = torch.tanh(true_ret * 20.0)
    dir_alignment = (soft_sign_pred * soft_sign_true).mean()
    dir_loss = 1.0 - dir_alignment

    # 振幅宽带校准正则: 允许在 0.7x ~ 1.3x 真实波动带内自由贝叶斯收缩
    # 下限防过度萎缩 (解决 std~0.02 导致的 NRMSE 倒挂)，上限防暴走，且不除以 true_std 避免放大极小波动冲击方向分
    pred_std = torch.sqrt(torch.var(pred_ret, dim=-1, unbiased=False) + 1e-7).mean()
    true_std = torch.sqrt(torch.var(true_ret, dim=-1, unbiased=False) + 1e-7).mean()
    amp_loss = F.relu(pred_std - 1.3 * true_std) + F.relu(0.7 * true_std - pred_std)

    total_loss = ret_loss + dir_loss_weight * dir_loss + amp_loss_weight * amp_loss
    return total_loss, {
        "ret_loss": float(ret_loss.detach().cpu()) if torch.isfinite(ret_loss) else 0.0,
        "dir_loss": float(dir_loss.detach().cpu()) if torch.isfinite(dir_loss) else 0.0,
        "dir_alignment": float(dir_alignment.detach().cpu()) if torch.isfinite(dir_alignment) else 0.0,
        "amp_loss": float(amp_loss.detach().cpu()) if torch.isfinite(amp_loss) else 0.0,
    }


def attach_lora(model, rank: int, alpha: int, dropout: float):
    """将 LoRA 注入到核心投影层 (V11 支持 rank=16 扩容)。"""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            base.weight.requires_grad_(False)
            if base.bias is not None:
                base.bias.requires_grad_(False)
            kw = {"device": base.weight.device, "dtype": base.weight.dtype}
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **kw))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **kw))
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            self.scale, self.dropout = alpha / rank, nn.Dropout(dropout)

        def forward(self, x):
            return self.base(x) + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scale

    for p in model.parameters():
        p.requires_grad_(False)
    targets = {"query_proj", "key_proj", "value_proj", "out_proj", "ff0", "ff1"}
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child))
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("LoRA 没有匹配到任何 Linear 层")
    return model


def get_trainable_state(model, sector_embedding) -> dict:
    from torch import nn
    base = model.module if isinstance(model, nn.DataParallel) else model
    st = {f"lora.{k}": v.detach().cpu() for k, v in base.named_parameters() if v.requires_grad}
    st.update({f"sector.{k}": v.detach().cpu() for k, v in sector_embedding.named_parameters()})
    return st


# ----------------------------------------------------------------- 评估与训练

def evaluate(model, val, device, horizon, batch_size, dir_loss_weight, sector_embedding, amp_loss_weight: float = 0.1) -> dict:
    import torch

    model.eval()
    contexts_all, targets_all, sectors_all = val
    sums = {"loss": 0.0, "nrmse_sq": 0.0, "naive_nrmse_sq": 0.0,
            "direction_acc": 0.0, "trajectory_corr_sum": 0.0}
    count = 0
    total_batches = (len(targets_all) + batch_size - 1) // batch_size
    print(f"[Eval Start] 评估开始: 样本量={len(targets_all)}, BatchSize={batch_size}, 批次数={total_batches}", flush=True)

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(targets_all), batch_size), 1):
            c = torch.from_numpy(contexts_all[i:i + batch_size]).to(device)
            t = torch.from_numpy(targets_all[i:i + batch_size]).to(device)
            s = torch.from_numpy(sectors_all[i:i + batch_size]).to(device)
            p = forecast(model, c, s, horizon, sector_embedding)

            loss, _ = alpha_loss(p, t, dir_loss_weight, amp_loss_weight)
            pred_ret = p - 1.0
            true_ret = t - 1.0

            err = pred_ret - true_ret
            naive_err = true_ret
            n = len(t)

            sums["loss"] += float(loss) * n
            sums["nrmse_sq"] += float(err.square().mean()) * n
            sums["naive_nrmse_sq"] += float(naive_err.square().mean()) * n

            same_dir = (torch.sign(pred_ret) == torch.sign(true_ret))
            sums["direction_acc"] += float(same_dir.float().mean()) * n

            # 单票走势轨迹形态相关性 (Trajectory Shape Correlation)
            vx = pred_ret - pred_ret.mean(dim=-1, keepdim=True)
            vy = true_ret - true_ret.mean(dim=-1, keepdim=True)
            std_x = torch.sqrt((vx ** 2).sum(dim=-1) + 1e-8)
            std_y = torch.sqrt((vy ** 2).sum(dim=-1) + 1e-8)
            traj_corr = (vx * vy).sum(dim=-1) / (std_x * std_y)
            sums["trajectory_corr_sum"] += float(traj_corr.mean()) * n

            count += n
            if batch_idx % 8 == 0 or batch_idx == total_batches:
                print(f"[Eval Progress] Batch {batch_idx}/{total_batches} ({min(i + batch_size, len(targets_all))}/{len(targets_all)})", flush=True)

    model.train()
    metrics = {
        "loss": sums["loss"] / count,
        "nrmse": (sums["nrmse_sq"] / count) ** 0.5,
        "naive_nrmse": (sums["naive_nrmse_sq"] / count) ** 0.5,
        "direction_acc": sums["direction_acc"] / count,
        "trajectory_corr": sums["trajectory_corr_sum"] / count,
    }
    metrics["skill_nrmse"] = 1.0 - metrics["nrmse"] / max(metrics["naive_nrmse"], 1e-12)
    # 单票未来10天走势轨迹综合得分: 兼顾方向胜率 (大趋势) 与 TrajCorr (曲线形态同频度)
    metrics["trajectory_score"] = float(metrics["direction_acc"] + 0.5 * max(metrics.get("trajectory_corr", 0.0), 0.0))
    print(f"[Eval Done] DirAcc={metrics['direction_acc']:.4f}, TrajCorr={metrics['trajectory_corr']:+.4f}, "
          f"Score={metrics['trajectory_score']:.4f}, NRMSE={metrics['nrmse']:.4f} (naive={metrics['naive_nrmse']:.4f})", flush=True)
    return metrics


def compute_trajectory_score(m: dict) -> float:
    return float(m.get("direction_acc", 0.0) + 0.5 * max(m.get("trajectory_corr", 0.0), 0.0))


def save_state(model, sector_embedding, optimizer, scheduler, global_step, best, history, args, chunk_index):
    import torch

    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": SWANLAB_RUN_ID, "global_step": global_step, "best": best, "best_score": best, "history": history,
        "weights": get_trainable_state(model, sector_embedding),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": vars(args), "chunk_index": chunk_index,
    }
    tmp = OUTPUT / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(OUTPUT / "state.pt")
    (OUTPUT / "relay.json").write_text(json.dumps({
        "run_id": SWANLAB_RUN_ID, "global_step": global_step, "chunk_index": chunk_index,
        "best_score": best, "best_val_nrmse": history[-1].get("nrmse", 0.0) if history else 0.0,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    (OUTPUT / "history.json").write_text(json.dumps(history, indent=2))


def train(args) -> None:
    import swanlab
    import torch
    from torch import nn
    from timesfm3 import TimesFM3Forecaster

    train_pkl, val_pkl, split_csv = discover_dataset(INPUT_ROOT)
    resume_path = find_relay_state(INPUT_ROOT)
    state = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None
    chunk_index = state["chunk_index"] + 1 if state else 1

    log("start", chunk_index=chunk_index, resume_from=str(resume_path) if state else None,
        train_source=str(train_pkl), gpu_detected=torch.cuda.device_count() if torch.cuda.is_available() else 0)

    swanlab.login(api_key=SWANLAB_API_KEY)
    run = swanlab.init(project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE,
                       experiment_name=EXPERIMENT_NAME, id=SWANLAB_RUN_ID, resume="allow",
                       config={**vars(args), "chunk": chunk_index})
    log("swanlab_ready", run_id=SWANLAB_RUN_ID, chunk_index=chunk_index)

    log("loading_data_pools")
    pool = FastStockPool(train_pkl, split_csv)
    total_plan_samples = min(args.max_steps * args.batch_size * 2, 4000000)
    pool.build_plan(total_samples=total_plan_samples, seed=args.seed + chunk_index, ratio=args.power_law_ratio)
    val = build_validation_pool(val_pkl, split_csv, max_samples=args.val_samples, seed=args.seed)
    log("data_ready", num_sectors=len(pool.sectors), val_size=len(val[0]), plan_sha256=pool.sha256)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = TimesFM3Forecaster.from_pretrained(args.model, per_core_batch_size=args.batch_size)
    model = forecaster.model.to(device)

    # V11 核心升级: LoRA Rank 扩容至 16
    model = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    patch_len = int(model.input_patch_len)
    sector_embedding = nn.Embedding(len(pool.sectors) + 5, patch_len).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(sector_embedding.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # 余弦退火学习率调度器 (按 3 小时有效长训步数约 12,000 步精确退火)
    anneal_steps = 12000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=anneal_steps, eta_min=args.min_lr)

    global_step, best_score, history = 0, -1.0, []
    if state:
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        saved_weights = state["weights"]
        lora_weights = {k.replace("lora.", ""): v for k, v in saved_weights.items() if k.startswith("lora.")}
        sec_weights = {k.replace("sector.", ""): v for k, v in saved_weights.items() if k.startswith("sector.")}

        # 导师硬审核防御断言: 严格校验 400 个 LoRA 参数数量
        lora_params_in_model = {k: v for k, v in base_model.named_parameters() if "lora_" in k}
        assert len(lora_params_in_model) == len(lora_weights) == 400, f"LoRA 数量不匹配! 模型: {len(lora_params_in_model)}, 权重: {len(lora_weights)}"
        missing, unexpected = base_model.load_state_dict(lora_weights, strict=False)
        assert len(unexpected) == 0, f"发现多余未匹配权重: {unexpected}"

        sector_embedding.load_state_dict(sec_weights, strict=True)

        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") and scheduler:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"[Scheduler Warning] 重置 scheduler: {e}")
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        global_step, history = state["global_step"], state.get("history", [])
        if "best_score" in state:
            best_score = float(state["best_score"])
        elif history:
            best_score = float(max(compute_trajectory_score(h) for h in history))
        else:
            best_score = -1.0
        log("resumed_successfully", global_step=global_step, best_score=best_score, lora_defense_passed=True)

    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        base_attrs = {
            name: getattr(model, name)
            for name in (
                "input_patch_len",
                "output_patch_len",
                "use_stitching",
                "_stitching_extract_len",
                "rolls",
                "use_frozen_running_stats",
            )
            if hasattr(model, name)
        }
        model = nn.DataParallel(model)
        for name, value in base_attrs.items():
            setattr(model, name, value)
        log("multi_gpu_enabled", count=torch.cuda.device_count())

    # Step 0 基线体检
    if not state:
        metrics = evaluate(model, val, device, args.horizon, args.eval_batch_size, args.dir_loss_weight, sector_embedding, args.amp_loss_weight)
        best_score = metrics["trajectory_score"]
        history.append({"step": 0, **metrics})
        base_log = {f"val/{k}": v for k, v in metrics.items()}
        base_log.update(gpu_metrics())
        run.log(base_log, step=0)
        log("baseline", **metrics, **gpu_metrics())

    deadline = args.budget_seconds - args.reserve_seconds
    run.log({"relay/chunk_index": chunk_index}, step=max(global_step, 1))
    stop = "max_steps"
    model.train()
    rng = np.random.default_rng(args.seed + chunk_index)
    tic, chunk_start_step = time.monotonic(), global_step
    skipped = 0
    nan_batches_dumped = 0
    hook_diagnosed = False  # 首次遇到非有限梯度时懒挂 Hook 深度诊断，记完即卸载，平时零开销

    while global_step < args.max_steps:
        if time.monotonic() - STARTED >= deadline:
            stop = "time_budget"
            break

        c, t, s = pool.sample_batch(args.batch_size)
        c, t, s = torch.from_numpy(c).to(device), torch.from_numpy(t).to(device), torch.from_numpy(s).to(device)
        pred = forecast(model, c, s, args.horizon, sector_embedding)
        loss, parts = alpha_loss(pred, t, args.dir_loss_weight, args.amp_loss_weight)

        if not torch.isfinite(loss):
            log("warning_skip_non_finite_loss", step=global_step + 1)
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

        if not torch.isfinite(grad):
            pred_ret = (pred - 1.0).detach()
            true_ret = (t - 1.0).detach()
            nan_params = [
                name for name, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            sec_nan = sector_embedding.weight.grad is not None and not torch.isfinite(sector_embedding.weight.grad).all()

            # CPM RevIN 修正在上下文 std == 0 的通道上会对 sqrt(0) 求导得到 NaN, 按模型通道顺序统计零方差样本数
            ctx_std = c[..., list(MODEL_CHANNEL_INDICES)].std(dim=1, unbiased=False)
            zero_std = ctx_std == 0

            diag_event = {
                "step": global_step + 1,
                "amp_loss": parts.get("amp_loss", 0.0),
                "pred_std": float(torch.sqrt(torch.var(pred_ret, dim=-1, unbiased=False) + 1e-7).mean()),
                "true_std": float(torch.sqrt(torch.var(true_ret, dim=-1, unbiased=False) + 1e-7).mean()),
                "max_abs_pred_ret": float(pred_ret.abs().max()),
                "max_abs_true_ret": float(true_ret.abs().max()),
                "nan_count": len(nan_params),
                "total_trainable": len(trainable_params),
                "nan_layers_sample": nan_params[:10],
                "sec_nan": bool(sec_nan),
                "zero_std_samples": int(zero_std.any(dim=1).sum()),
                "zero_std_per_channel": zero_std.sum(dim=0).tolist(),
            }

            if nan_batches_dumped < NAN_BATCH_DUMP_LIMIT:
                dump_dir = OUTPUT / "nan_batches"
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / f"step{global_step + 1}_skip{skipped + 1}.pt"
                torch.save({"step": global_step + 1, "contexts": c.cpu(), "targets": t.cpu(), "sectors": s.cpu()},
                           dump_path)
                nan_batches_dumped += 1
                diag_event["dump_path"] = str(dump_path)

            # 首次出现非有限梯度: 懒挂钩子单次深入定位 (Two-shot Lazy Hook Diagnosis)
            if not hook_diagnosed:
                hook_diagnosed = True
                first_forward_anomaly = None
                f_handles = []

                def make_f_hook(mod_name):
                    def f_hook(module, inputs, output):
                        nonlocal first_forward_anomaly
                        if first_forward_anomaly is not None:
                            return
                        out_tensor = output[0] if isinstance(output, (tuple, list)) else output
                        if isinstance(out_tensor, torch.Tensor) and not torch.isfinite(out_tensor).all():
                            first_forward_anomaly = {
                                "module": mod_name,
                                "stage": "forward",
                                "out_shape": list(out_tensor.shape),
                            }
                    return f_hook

                for name, mod in model.named_modules():
                    if len(list(mod.children())) == 0:
                        f_handles.append(mod.register_forward_hook(make_f_hook(name)))

                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    _ = forecast(model, c, s, args.horizon, sector_embedding)

                for h in f_handles:
                    h.remove()

                if first_forward_anomaly:
                    diag_event["forward_clean"] = False
                    diag_event["first_anomaly"] = first_forward_anomaly
                else:
                    diag_event["forward_clean"] = True
                    # 第二枪: 反传 Hook 抓取 (若前向 clean，则深入反传链路)
                    first_backward_anomaly = None
                    b_handles = []

                    def make_b_hook(mod_name):
                        def b_hook(module, grad_input, grad_output):
                            nonlocal first_backward_anomaly
                            if first_backward_anomaly is not None:
                                return
                            for idx, g in enumerate(grad_output):
                                if g is not None and not torch.isfinite(g).all():
                                    first_backward_anomaly = {
                                        "module": mod_name,
                                        "stage": "backward_grad_output",
                                        "idx": idx,
                                        "shape": list(g.shape),
                                    }
                                    return
                        return b_hook

                    for name, mod in model.named_modules():
                        if len(list(mod.children())) == 0:
                            b_handles.append(mod.register_full_backward_hook(make_b_hook(name)))

                    optimizer.zero_grad(set_to_none=True)
                    p_retry = forecast(model, c, s, args.horizon, sector_embedding)
                    l_retry, _ = alpha_loss(p_retry, t, args.dir_loss_weight, args.amp_loss_weight)
                    l_retry.backward()

                    for h in b_handles:
                        h.remove()

                    if first_backward_anomaly:
                        diag_event["backward_clean"] = False
                        diag_event["first_anomaly"] = first_backward_anomaly
                    else:
                        diag_event["backward_clean"] = True

            log("warning_skip_non_finite_grad", **diag_event)
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % args.log_every == 0:
            rate = (global_step - chunk_start_step) / max(time.monotonic() - tic, 1e-6)
            cur_lr = optimizer.param_groups[0]["lr"]
            row = {"train/loss": float(loss), "train/grad_norm": float(grad),
                   "train/lr": float(cur_lr),
                   **{f"train/{k}": v for k, v in parts.items()},
                   "train/steps_per_sec": rate, "train/skipped_batches": skipped, "relay/chunk_index": chunk_index,
                   **gpu_metrics()}
            run.log(row, step=global_step)
            log("train", step=global_step, **row)

        if global_step % args.eval_every == 0:
            metrics = evaluate(model, val, device, args.horizon, args.eval_batch_size, args.dir_loss_weight, sector_embedding, args.amp_loss_weight)
            history.append({"step": global_step, **metrics})
            eval_row = {f"val/{k}": v for k, v in metrics.items()}
            eval_row.update(gpu_metrics())
            run.log(eval_row, step=global_step)
            log("val", step=global_step, **metrics, **gpu_metrics())
            score = metrics["trajectory_score"]
            if score > best_score:
                best_score = score
                torch.save(get_trainable_state(model, sector_embedding), OUTPUT / "best_weights.pt")
                (OUTPUT / "best_metrics.json").write_text(json.dumps(history[-1], indent=2))
                log("new_best_trajectory_model", step=global_step, trajectory_score=best_score,
                    direction_acc=metrics["direction_acc"], trajectory_corr=metrics.get("trajectory_corr", 0.0), nrmse=metrics["nrmse"])

        if global_step % args.save_every == 0:
            save_state(model, sector_embedding, optimizer, scheduler, global_step, best_score, history, args, chunk_index)
            log("saved", step=global_step)

    save_state(model, sector_embedding, optimizer, scheduler, global_step, best_score, history, args, chunk_index)
    torch.save(get_trainable_state(model, sector_embedding), OUTPUT / "last_weights.pt")
    best_dir = max((h.get("direction_acc", 0.0) for h in history), default=0.0)
    best_tc = max((h.get("trajectory_corr", -1.0) for h in history), default=0.0)
    summary = {
        "chunk_index": chunk_index, "global_step": global_step, "stop": stop,
        "best_trajectory_score": best_score,
        "best_direction_acc": best_dir,
        "best_trajectory_corr": best_tc,
        "best_val_nrmse": min((h.get("nrmse", 999.0) for h in history), default=0.0),
        "next": "push 下一个 chunk 并以本 kernel 为 kernel_source" if stop == "time_budget" else "done"
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    run.log({"relay/last_step_of_chunk": global_step, **gpu_metrics()}, step=global_step)
    log("chunk_done", **summary)
    swanlab.finish()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/timesfm-3.0-pytorch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--multi-gpu", action="store_true", default=True)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dir-loss-weight", type=float, default=0.5)
    p.add_argument("--amp-loss-weight", type=float, default=0.1)
    p.add_argument("--power-law-ratio", type=float, default=0.6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=16)   # 扩容至 16
    p.add_argument("--lora-alpha", type=int, default=32)  # 对应翻倍至 32
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--val-samples", type=int, default=0)  # 0 为 100% 全量 12.3 万条
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--budget-seconds", type=int, default=43200)  # 默认 12 小时大 Chunk
    p.add_argument("--reserve-seconds", type=int, default=300)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--save-every", type=int, default=2500)
    p.set_defaults(**CHUNK_CONFIG)
    return p.parse_args(argv)


CHUNK_CONFIG: dict = {}


if __name__ == "__main__":
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/google-research/timesfm.git", "swanlab==0.10.1"])
    train(parse_args())
