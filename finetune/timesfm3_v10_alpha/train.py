"""TimesFM-3 LoRA + Sector Embedding V10: 收益率与 Alpha 驱动版。

核心突破:
1. 价格输入相对化归一 (以 t=120 为锚点 P0，消除高低价股量纲悬殊);
2. 收益率路径损失 (Return Huber Loss) + 方向感知可微损失 (Direction Correlation Loss);
3. 彻底粉碎保守模型"预测未来价格走平"的局部极小陷阱;
4. 深度量化指标: NRMSE, Naive 走平基线, Skill 分数, 方向胜率 (DirAcc), 信息系数 (IC).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
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
SWANLAB_RUN_ID = "tfm3v10alpha20261010a"  # V10 全新独立 Run ID
EXPERIMENT_NAME = "timesfm3-v10-alpha-2xt4"

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
    """全市场时序动态池 (输入相对归一化版): 以每个窗口历史末端 P0 锚定去量纲。"""

    def __init__(self, train_pkl: Path, symbol_split_csv: Path):
        import pandas as pd
        df_split = pd.read_csv(symbol_split_csv)
        self.sectors = sorted(df_split["sector"].dropna().unique().tolist())
        self.sector2id = {s: i for i, s in enumerate(self.sectors)}
        self.symbol2sector = dict(zip(df_split["symbol"], df_split["sector"].map(self.sector2id).fillna(0).astype(int)))

        with train_pkl.open("rb") as f:
            panel = pickle.load(f)

        self.stocks = []
        for symbol, df in panel.items():
            if not all(col in df.columns for col in DATA_FEATURES):
                continue
            df = df.sort_index()
            vals = df[list(DATA_FEATURES)].to_numpy(dtype=np.float32)
            if len(vals) >= 131:
                sec_id = self.symbol2sector.get(symbol, 0)
                self.stocks.append((vals, sec_id))
        if not self.stocks:
            raise RuntimeError("没有加载到合法的股票训练数据")

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        contexts, targets, sectors = [], [], []
        while len(contexts) < batch_size:
            idx = rng.integers(0, len(self.stocks))
            vals, sec = self.stocks[idx]
            max_start = len(vals) - 130
            start = rng.integers(0, max_start)
            window = vals[start:start + 130].copy()

            # 剔除停牌/脏数据 (全特征有限, 价格正值, 成交量正值)
            if not (np.isfinite(window).all() and (window[:, :4] > 0).all() and (window[:, 4] > 0).all()):
                continue

            p0 = window[119, 3]  # context 最后一个交易日的收盘价
            if p0 <= 1e-4:
                continue

            # 核心创新 1: 价格去量纲归一 (OHLC 全部除以 P0，锚定在 1.0)
            window[:, list(PRICE_INDICES)] /= p0

            # 成交量与金额以历史窗口中位数归一，消除绝对股数差异
            vol_med = np.median(window[:120, 4]) + 1e-4
            amt_med = np.median(window[:120, 5]) + 1e-4
            window[:, 4] /= vol_med
            window[:, 5] /= amt_med

            contexts.append(window[:120])
            # target 为未来 10 天相对价格 (1.0 + 未来收益率)
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
    symbol2sector = dict(zip(df_split["symbol"], df_split["sector"].map(sector2id).fillna(0).astype(int)))

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
        sec = symbol2sector.get(symbol, 0)

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
    for meta in root.glob("**/relay/relay.json"):
        try:
            info = json.loads(meta.read_text())
            if info.get("run_id") == SWANLAB_RUN_ID and info.get("global_step", -1) > best_step:
                best, best_step = meta.parent / "state.pt", info["global_step"]
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


def alpha_loss(pred_rel_price, target_rel_price, dir_loss_weight: float = 0.5):
    """
    核心突破 2: 收益率路径与方向感知联合损失。
    pred_rel_price: 预测的相对价格 (1.0 + pred_return)
    target_rel_price: 真实的相对价格 (1.0 + true_return)
    """
    import torch
    import torch.nn.functional as F

    pred_ret = pred_rel_price - 1.0
    true_ret = target_rel_price - 1.0

    # 1. 收益率路径 SmoothL1 (Huber) 损失 (以 0.02 即 2% 为过渡带)
    ret_loss = F.smooth_l1_loss(pred_ret, true_ret, beta=0.02)

    # 2. 趋势方向感知惩罚 (Direction Correlation Penalty)
    # 当预测收益率与真实收益率同号时奖励，异号时严惩，迫使模型走出"只预测 0"的怠工区
    soft_sign_pred = torch.tanh(pred_ret * 20.0)
    soft_sign_true = torch.tanh(true_ret * 20.0)
    dir_alignment = (soft_sign_pred * soft_sign_true).mean()
    dir_loss = 1.0 - dir_alignment  # 越对齐越接近 0，完全相反接近 2.0

    total_loss = ret_loss + dir_loss_weight * dir_loss
    return total_loss, {
        "ret_loss": float(ret_loss.detach().cpu()) if torch.isfinite(ret_loss) else 0.0,
        "dir_loss": float(dir_loss.detach().cpu()) if torch.isfinite(dir_loss) else 0.0,
        "dir_alignment": float(dir_alignment.detach().cpu()) if torch.isfinite(dir_alignment) else 0.0,
    }


def attach_lora(model, rank: int, alpha: int, dropout: float):
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

def evaluate(model, val, device, horizon, batch_size, dir_loss_weight, sector_embedding) -> dict:
    import torch

    model.eval()
    contexts_all, targets_all, sectors_all = val
    sums = {"loss": 0.0, "nrmse_sq": 0.0, "naive_nrmse_sq": 0.0,
            "direction_acc": 0.0, "ic_sum": 0.0}
    count = 0
    total_batches = (len(targets_all) + batch_size - 1) // batch_size
    print(f"[Eval Start] 评估开始: 样本量={len(targets_all)}, BatchSize={batch_size}, 批次数={total_batches}", flush=True)

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(targets_all), batch_size), 1):
            c = torch.from_numpy(contexts_all[i:i + batch_size]).to(device)
            t = torch.from_numpy(targets_all[i:i + batch_size]).to(device)
            s = torch.from_numpy(sectors_all[i:i + batch_size]).to(device)
            p = forecast(model, c, s, horizon, sector_embedding)

            loss, _ = alpha_loss(p, t, dir_loss_weight)
            pred_ret = p - 1.0
            true_ret = t - 1.0

            err = pred_ret - true_ret
            naive_err = true_ret  # 朴素走平: pred_ret = 0.0
            n = len(t)

            sums["loss"] += float(loss) * n
            sums["nrmse_sq"] += float(err.square().mean()) * n
            sums["naive_nrmse_sq"] += float(naive_err.square().mean()) * n

            same_dir = (torch.sign(pred_ret) == torch.sign(true_ret))
            sums["direction_acc"] += float(same_dir.float().mean()) * n

            # 计算横截面最终收益率相关系数 (IC)
            p_final, t_final = pred_ret[:, -1], true_ret[:, -1]
            if len(p_final) > 1 and p_final.std() > 1e-6 and t_final.std() > 1e-6:
                vx = p_final - p_final.mean()
                vy = t_final - t_final.mean()
                ic = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum() * (vy ** 2).sum()) + 1e-8)
                sums["ic_sum"] += float(ic) * n

            count += n
            if batch_idx % 8 == 0 or batch_idx == total_batches:
                print(f"[Eval Progress] Batch {batch_idx}/{total_batches} ({min(i + batch_size, len(targets_all))}/{len(targets_all)})", flush=True)

    model.train()
    metrics = {
        "loss": sums["loss"] / count,
        "nrmse": (sums["nrmse_sq"] / count) ** 0.5,
        "naive_nrmse": (sums["naive_nrmse_sq"] / count) ** 0.5,
        "direction_acc": sums["direction_acc"] / count,
        "ic": sums["ic_sum"] / count,
    }
    metrics["skill_nrmse"] = 1.0 - metrics["nrmse"] / max(metrics["naive_nrmse"], 1e-12)
    print(f"[Eval Done] NRMSE={metrics['nrmse']:.4f} (naive={metrics['naive_nrmse']:.4f}, skill={metrics['skill_nrmse']:+.4f}), "
          f"DirAcc={metrics['direction_acc']:.4f}, IC={metrics['ic']:+.4f}", flush=True)
    return metrics


def save_state(model, sector_embedding, optimizer, global_step, best, history, args, chunk_index):
    import torch

    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": SWANLAB_RUN_ID, "global_step": global_step, "best": best, "history": history,
        "weights": get_trainable_state(model, sector_embedding), "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": vars(args), "chunk_index": chunk_index,
    }
    tmp = OUTPUT / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(OUTPUT / "state.pt")
    (OUTPUT / "relay.json").write_text(json.dumps({
        "run_id": SWANLAB_RUN_ID, "global_step": global_step, "chunk_index": chunk_index,
        "best_val_nrmse": best, "saved_at": datetime.now(timezone.utc).isoformat(),
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
    val = build_validation_pool(val_pkl, split_csv, max_samples=args.val_samples, seed=args.seed)
    log("data_ready", num_sectors=len(pool.sectors), val_size=len(val[0]))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = TimesFM3Forecaster.from_pretrained(args.model, per_core_batch_size=args.batch_size)
    model = forecaster.model.to(device)
    model = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    patch_len = int(model.input_patch_len)
    sector_embedding = nn.Embedding(len(pool.sectors) + 5, patch_len).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(sector_embedding.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step, best, history = 0, float("inf"), []
    if state:
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        saved_weights = state["weights"]
        lora_weights = {k.replace("lora.", ""): v for k, v in saved_weights.items() if k.startswith("lora.")}
        sec_weights = {k.replace("sector.", ""): v for k, v in saved_weights.items() if k.startswith("sector.")}
        base_model.load_state_dict(lora_weights, strict=False)
        sector_embedding.load_state_dict(sec_weights)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        global_step, best, history = state["global_step"], state["best"], state["history"]
        log("resumed_successfully", global_step=global_step, best_nrmse=best)

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
        metrics = evaluate(model, val, device, args.horizon, args.eval_batch_size, args.dir_loss_weight, sector_embedding)
        best = metrics["nrmse"]
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

    while global_step < args.max_steps:
        if time.monotonic() - STARTED >= deadline:
            stop = "time_budget"
            break

        c, t, s = pool.sample_batch(args.batch_size, rng)
        c, t, s = torch.from_numpy(c).to(device), torch.from_numpy(t).to(device), torch.from_numpy(s).to(device)
        pred = forecast(model, c, s, args.horizon, sector_embedding)
        loss, parts = alpha_loss(pred, t, args.dir_loss_weight)

        if not torch.isfinite(loss):
            log("warning_skip_non_finite_loss", step=global_step + 1)
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

        if not torch.isfinite(grad):
            log("warning_skip_non_finite_grad", step=global_step + 1)
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        global_step += 1

        if global_step % args.log_every == 0:
            rate = (global_step - chunk_start_step) / max(time.monotonic() - tic, 1e-6)
            row = {"train/loss": float(loss), "train/grad_norm": float(grad),
                   **{f"train/{k}": v for k, v in parts.items()},
                   "train/steps_per_sec": rate, "train/skipped_batches": skipped, "relay/chunk_index": chunk_index,
                   **gpu_metrics()}
            run.log(row, step=global_step)
            log("train", step=global_step, **row)

        if global_step % args.eval_every == 0:
            metrics = evaluate(model, val, device, args.horizon, args.eval_batch_size, args.dir_loss_weight, sector_embedding)
            history.append({"step": global_step, **metrics})
            eval_row = {f"val/{k}": v for k, v in metrics.items()}
            eval_row.update(gpu_metrics())
            run.log(eval_row, step=global_step)
            log("val", step=global_step, **metrics, **gpu_metrics())
            if metrics["nrmse"] < best:
                best = metrics["nrmse"]
                torch.save(get_trainable_state(model, sector_embedding), OUTPUT / "best_weights.pt")
                (OUTPUT / "best_metrics.json").write_text(json.dumps(history[-1], indent=2))

        if global_step % args.save_every == 0:
            save_state(model, sector_embedding, optimizer, global_step, best, history, args, chunk_index)
            log("saved", step=global_step)

    save_state(model, sector_embedding, optimizer, global_step, best, history, args, chunk_index)
    torch.save(get_trainable_state(model, sector_embedding), OUTPUT / "last_weights.pt")
    summary = {"chunk_index": chunk_index, "global_step": global_step, "stop": stop,
               "best_val_nrmse": best,
               "next": "push 下一个 chunk 并以本 kernel 为 kernel_source" if stop == "time_budget" else "done"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    run.log({"relay/last_step_of_chunk": global_step, **gpu_metrics()}, step=global_step)
    log("chunk_done", **summary)
    swanlab.finish()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/timesfm-3.0-pytorch")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--multi-gpu", action="store_true", default=True)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dir-loss-weight", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--val-samples", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--budget-seconds", type=int, default=1800)
    p.add_argument("--reserve-seconds", type=int, default=300)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=200)
    p.set_defaults(**CHUNK_CONFIG)
    return p.parse_args(argv)


CHUNK_CONFIG: dict = {}


if __name__ == "__main__":
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/google-research/timesfm.git", "swanlab==0.10.1"])
    train(parse_args())
