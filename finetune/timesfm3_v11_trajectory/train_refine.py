"""TimesFM 3.0 V11 Clean C07-Refine: 甜点学习率 (7.92e-5) 恒定平台精修实验 (5小时独立分支)。

设计架构:
1. 起点: 自动加载 C07 best_weights.pt 与 last_weights.pt 并做 1:1 SWA 权重空间融合作为初始参数;
2. 优化器: 重置 AdamW 优化器动量历史，消除前序接力惯性干扰;
3. 学习率策略 A: 甜点恒定平台微调 (基准 7.92e-5, 极窄幅底噪带 7.0e-5 ~ 7.92e-5);
4. 验证策略: 日常每 2500 步快考 (8192 样本, 耗时 30s), 5 小时终点前执行 100% 全量 12.3 万条大考结算;
5. 预算: 5 小时 (18,000 秒), 预留 900 秒安全收尾。
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

REFINE_CONFIG: dict = {}

SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", "fmEPDGk4IItxgqSZKGLi8")
SWANLAB_PROJECT = os.environ.get("SWANLAB_PROJECT", "finance")
SWANLAB_WORKSPACE = os.environ.get("SWANLAB_WORKSPACE", "roc_fu")
SWANLAB_RUN_ID = "tfm3v11-clean-c07-refine-sweetspot"
EXPERIMENT_NAME = "timesfm3-v11-clean-c07-refine"

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
PRICE_INDICES = (0, 1, 2, 3)
COVARIATE_INDICES = (0, 1, 2, 4, 5, 6)
MODEL_CHANNEL_INDICES = (3, *COVARIATE_INDICES)
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


def discover_dataset(root: Path) -> tuple[Path, Path, Path]:
    train_pkls = sorted(root.glob("**/train_data.pkl"))
    val_pkls = sorted(root.glob("**/val_data.pkl"))
    splits = sorted(root.glob("**/symbol_split.csv"))
    if not train_pkls or not val_pkls or not splits:
        raise FileNotFoundError(f"在 {root} 下未找到完整数据集 (train_data.pkl, val_data.pkl, symbol_split.csv)")
    return train_pkls[0], val_pkls[0], splits[0]


def discover_c07_initial_weights(root: Path):
    best_weights = sorted(root.glob("**/best_weights.pt"))
    last_weights = sorted(root.glob("**/last_weights.pt"))
    if not best_weights:
        raise FileNotFoundError(f"未在 {root} 找到上游 C07 初始权重 best_weights.pt")
    p_best = best_weights[-1]
    p_last = last_weights[-1] if last_weights else p_best
    return p_best, p_last


class FastStockPool:
    def __init__(self, train_pkl: Path, symbol_split_csv: Path):
        import pandas as pd
        df_split = pd.read_csv(symbol_split_csv)
        self.sectors = sorted(df_split["sector"].dropna().unique().tolist())
        self.sector2id = {s: i for i, s in enumerate(self.sectors)}
        self.unknown_sector_id = len(self.sectors)
        self.symbol2sector = dict(zip(df_split["symbol"],
                                      df_split["sector"].map(self.sector2id).fillna(self.unknown_sector_id).astype(int)))
        with train_pkl.open("rb") as f:
            panel = pickle.load(f)

        windows, amplitudes, sym_ids, sec_ids = [], [], [], []
        print("[Data Loading] 开始构建样本采样池...", flush=True)

        for s_idx, (symbol, df) in enumerate(panel.items()):
            if not all(col in df.columns for col in DATA_FEATURES):
                continue
            df = df.sort_index()
            vals = df[list(DATA_FEATURES)].to_numpy(dtype=np.float32)
            n_rows = len(vals)
            sec = self.symbol2sector.get(symbol, self.unknown_sector_id)
            for idx in range(119, n_rows - 10):
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

                        c_future = w[120:130, 3]
                        amp = float(np.max(np.abs(c_future - 1.0)))

                        windows.append(w)
                        amplitudes.append(amp)
                        sym_ids.append(s_idx)
                        sec_ids.append(sec)

        self.windows = np.asarray(windows, dtype=np.float32)
        self.amplitudes = np.asarray(amplitudes, dtype=np.float32)
        self.sym_ids = np.asarray(sym_ids, dtype=np.int32)
        self.sec_ids = np.asarray(sec_ids, dtype=np.int64)
        print(f"[Data Ready] 采样池就绪: 共 {len(self.windows):,} 个有效窗口", flush=True)

    def generate_plan(self, total_samples: int, power_law_ratio: float, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        N = len(self.windows)
        p_uniform = np.full(N, 1.0 / N, dtype=np.float64)

        amps = np.maximum(self.amplitudes, 1e-6)
        p_power = amps / np.sum(amps)
        cap = 4.0 / N
        p_power = np.minimum(p_power, cap)
        p_power /= np.sum(p_power)

        p_mix = (1.0 - power_law_ratio) * p_uniform + power_law_ratio * p_power
        p_mix /= np.sum(p_mix)

        return rng.choice(N, size=total_samples, replace=True, p=p_mix).astype(np.int64)


def load_val_dataset(val_pkl: Path, symbol_split_csv: Path, max_samples: int = 0):
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

    c_arr = np.asarray(contexts, dtype=np.float32)
    t_arr = np.asarray(targets, dtype=np.float32)
    s_arr = np.asarray(val_sectors, dtype=np.int64)

    if 0 < max_samples < len(c_arr):
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(c_arr), size=max_samples, replace=False)
        return c_arr[idxs], t_arr[idxs], s_arr[idxs]

    return c_arr, t_arr, s_arr


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
    return model


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

    sec_vec = sector_embedding(sectors)
    sec_channel = sec_vec[:, None, None, :].expand(batch, 1, total_patches, patch_len)

    close = F.pad(contexts[..., 3:4].transpose(1, 2), (ctx_padding, padded_horizon), value=0.0)
    covariates = F.pad(contexts[..., list(COVARIATE_INDICES)].transpose(1, 2), (ctx_padding, padded_horizon), value=0.0)
    continuous = torch.cat([close, covariates], dim=1).reshape(batch, 7, total_patches, patch_len)
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


def compute_loss(pred_p, target_p, dir_weight=0.5, amp_weight=0.1):
    import torch
    import torch.nn.functional as F

    pred_ret = pred_p - 1.0
    true_ret = target_p - 1.0

    ret_loss = F.huber_loss(pred_ret, true_ret, delta=0.05)
    dir_alignment = torch.tanh(pred_ret * 20.0) * torch.tanh(true_ret * 20.0)
    dir_loss = torch.mean(1.0 - dir_alignment)

    pred_amp = torch.std(pred_ret, dim=-1)
    true_amp = torch.std(true_ret, dim=-1)
    amp_excess = F.relu(pred_amp - 1.2 * true_amp)
    amp_loss = torch.mean(amp_excess ** 2)

    total_loss = ret_loss + dir_weight * dir_loss + amp_weight * amp_loss
    return total_loss, ret_loss, dir_loss, torch.mean(dir_alignment), amp_loss


def evaluate(model, c_val, t_val, s_val, sector_embedding, device, batch_size=512):
    import torch

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(0, len(t_val), batch_size):
            c = torch.from_numpy(c_val[i : i + batch_size]).to(device)
            t = torch.from_numpy(t_val[i : i + batch_size]).to(device)
            s = torch.from_numpy(s_val[i : i + batch_size]).to(device)
            p = forecast(model, c, s, 10, sector_embedding)
            all_pred.append((p - 1.0).cpu().numpy())
            all_true.append((t - 1.0).cpu().numpy())

    pred_ret = np.concatenate(all_pred, axis=0)
    true_ret = np.concatenate(all_true, axis=0)

    p_10d, t_10d = pred_ret[:, -1], true_ret[:, -1]
    dir_acc = float(np.mean(np.sign(p_10d) == np.sign(t_10d)))

    corrs = []
    step = max(1, len(pred_ret) // 5000)
    for idx in range(0, len(pred_ret), step):
        px, tx = pred_ret[idx], true_ret[idx]
        if np.std(px) > 1e-6 and np.std(tx) > 1e-6:
            r = np.corrcoef(px, tx)[0, 1]
            if np.isfinite(r):
                corrs.append(r)
    traj_corr = float(np.mean(corrs)) if corrs else 0.0

    nrmse = float(np.sqrt(np.mean((pred_ret - true_ret) ** 2)))
    naive_nrmse = float(np.sqrt(np.mean(true_ret ** 2)))
    skill = float(1.0 - nrmse / max(naive_nrmse, 1e-12))
    score = float(dir_acc + 0.5 * max(traj_corr, 0.0))

    pred_std = float(np.mean(np.std(pred_ret, axis=-1)))
    true_std = float(np.mean(np.std(true_ret, axis=-1)))
    amp_ratio = float(pred_std / max(true_std, 1e-12))

    return {
        "direction_acc": dir_acc,
        "trajectory_corr": traj_corr,
        "nrmse": nrmse,
        "skill_nrmse": skill,
        "trajectory_score": score,
        "amp_ratio": amp_ratio,
    }


def get_trainable_state(model, sector_embedding):
    from torch import nn
    raw = model.module if isinstance(model, nn.DataParallel) else model
    st = {}
    for k, v in raw.named_parameters():
        if "lora_" in k:
            st[f"lora.{k}"] = v.detach().cpu()
    for k, v in sector_embedding.named_parameters():
        st[f"sector.{k}"] = v.detach().cpu()
    return st


def train(args):
    import swanlab
    import torch
    from torch import nn
    from timesfm3 import TimesFM3Forecaster

    train_pkl, val_pkl, split_csv = discover_dataset(INPUT_ROOT)
    p_best, p_last = discover_c07_initial_weights(INPUT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Model Init] 加载 TimesFM 3.0 基础模型...", flush=True)
    forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=args.eval_batch_size)
    model = forecaster.model.to(device)
    model = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    pool = FastStockPool(train_pkl, split_csv)
    patch_len = int(model.input_patch_len)
    sector_embedding = nn.Embedding(len(pool.sectors) + 5, patch_len).to(device)

    # 1. 执行 C07 SWA 权重空间融合并载入
    print(f"[Initial Weights] 正在将 C07 best ({p_best.name}) 与 last ({p_last.name}) 进行 1:1 SWA 融合并初始化...", flush=True)
    w_best = torch.load(p_best, map_location="cpu", weights_only=False)
    w_last = torch.load(p_last, map_location="cpu", weights_only=False)
    w_swa = {k: 0.5 * w_best[k].to(torch.float32) + 0.5 * w_last[k].to(torch.float32) for k in w_best.keys()}

    raw_model = model
    lora_w = {k.replace("lora.", ""): v for k, v in w_swa.items() if k.startswith("lora.")}
    sec_w = {k.replace("sector.", ""): v for k, v in w_swa.items() if k.startswith("sector.")}
    raw_model.load_state_dict(lora_w, strict=False)
    sector_embedding.load_state_dict(sec_w, strict=True)
    print(f"[Init OK] C07 SWA 权重成功载入！以全场最优状态起跑！", flush=True)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
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
        print(f"[Multi-GPU] 启用双卡 DataParallel 并行加速 (GPU Count: {torch.cuda.device_count()})", flush=True)

    # 2. 优化器与策略 A【甜点恒定平台学习率】
    # 基准学习率锁定在 7.92e-5 (用户的甜点假设)
    sweet_spot_lr = args.learning_rate  # 默认 7.92e-5
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(sector_embedding.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=sweet_spot_lr, weight_decay=args.weight_decay)

    # 策略 A: 极温和窄幅退火 (7.92e-5 -> 7.00e-5, 维持在甜点平台，几乎恒定，避免剧烈震荡)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15000, eta_min=args.min_lr)

    print(f"\n[Strategy A Ready] 学习率策略 A: 甜点恒定平台微调 | 起始 LR = {sweet_spot_lr:.2e}, 极小下限 = {args.min_lr:.2e}")

    # 3. 准备验证集数据
    print(f"[Validation Ready] 加载快速抽检验证集 (日常采样数={args.val_samples})...", flush=True)
    c_val_fast, t_val_fast, s_val_fast = load_val_dataset(val_pkl, split_csv, max_samples=args.val_samples)
    print(f"[Val Fast] 快速抽检集就绪: {len(c_val_fast):,} 条样本", flush=True)

    run = None
    if SWANLAB_API_KEY:
        try:
            swanlab.login(api_key=SWANLAB_API_KEY)
            run = swanlab.init(project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE,
                               experiment_name=EXPERIMENT_NAME, id=SWANLAB_RUN_ID, resume="allow",
                               config={**vars(args), "branch": "c07_refine_sweetspot_strategy_a"})
        except Exception as e:
            print(f"[SwanLab Warning] 初始化跳过: {e}", flush=True)
            run = None
    else:
        print("[SwanLab Info] 未提供 SWANLAB_API_KEY，采用本地 run.log 追踪", flush=True)

    total_plan_samples = min(args.max_steps * args.batch_size * 2, 4000000)
    plan = pool.generate_plan(total_plan_samples, args.power_law_ratio, args.seed)
    plan_ptr = 0

    best_score = -1.0
    history = []
    global_step = 0
    stop = "max_steps"

    print(f"\n[Training Start] 5小时甜点精修训练正式启动 (预算秒数: {args.budget_seconds}s)...", flush=True)

    while global_step < args.max_steps:
        elapsed = time.monotonic() - STARTED
        if elapsed >= args.budget_seconds - args.reserve_seconds:
            print(f"\n[Time Budget] 已达到时间预算保护线 (已运行 {elapsed:.1f}s), 启动最终大考与安全收尾！", flush=True)
            stop = "time_budget"
            break

        model.train()
        if plan_ptr + args.batch_size > len(plan):
            plan = pool.generate_plan(total_plan_samples, args.power_law_ratio, args.seed + global_step)
            plan_ptr = 0

        batch_indices = plan[plan_ptr : plan_ptr + args.batch_size]
        plan_ptr += args.batch_size

        batch_windows = pool.windows[batch_indices]
        batch_secs = pool.sec_ids[batch_indices]

        c = torch.from_numpy(batch_windows[:, :120]).to(device)
        target = torch.from_numpy(batch_windows[:, 120:130, 3]).to(device)
        s = torch.from_numpy(batch_secs).to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = forecast(model, c, s, args.horizon, sector_embedding)

        loss, ret_loss, dir_loss, dir_align, amp_loss = compute_loss(
            pred, target, dir_weight=args.dir_loss_weight, amp_weight=args.amp_loss_weight
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        global_step += 1
        cur_lr = optimizer.param_groups[0]["lr"]

        if global_step % args.log_every == 0 and run is not None:
            run.log({
                "train/loss": float(loss.item()),
                "train/lr": float(cur_lr),
                "train/ret_loss": float(ret_loss.item()),
                "train/dir_loss": float(dir_loss.item()),
                "train/amp_loss": float(amp_loss.item()),
            }, step=global_step)

        if global_step % args.eval_every == 0:
            ev = evaluate(model, c_val_fast, t_val_fast, s_val_fast, sector_embedding, device, args.eval_batch_size)
            ev["step"] = global_step
            ev["lr"] = cur_lr
            history.append(ev)
            if run is not None:
                run.log({
                    "val/direction_acc": ev["direction_acc"],
                    "val/trajectory_corr": ev["trajectory_corr"],
                    "val/trajectory_score": ev["trajectory_score"],
                    "val/amp_ratio": ev["amp_ratio"],
                    "val/skill_nrmse": ev["skill_nrmse"],
                }, step=global_step)

            print(f"[Eval Step {global_step:,}] DirAcc: {ev['direction_acc']*100:.2f}%, Corr: {ev['trajectory_corr']:+.4f}, Score: {ev['trajectory_score']:.4f}, AmpRatio: {ev['amp_ratio']:.4f}, LR: {cur_lr:.2e}", flush=True)

            if ev["trajectory_score"] > best_score:
                best_score = ev["trajectory_score"]
                torch.save(get_trainable_state(model, sector_embedding), OUTPUT / "best_weights.pt")
                (OUTPUT / "best_metrics.json").write_text(json.dumps(ev, indent=2))
                print(f"★ 突破最高分: {best_score:.4f}，已保存 best_weights.pt！", flush=True)

    # 4. 终极大考结算 (全量 100% 验证集 12.3 万条样本大考)
    print("\n" + "=" * 70)
    print(">>> 5小时精修收官: 执行全量 123,839 条 100% 验证集终极大考结算 <<<")
    print("=" * 70, flush=True)
    c_val_full, t_val_full, s_val_full = load_val_dataset(val_pkl, split_csv, max_samples=0)
    final_ev = evaluate(model, c_val_full, t_val_full, s_val_full, sector_embedding, device, args.eval_batch_size)
    final_ev["step"] = global_step
    final_ev["lr"] = optimizer.param_groups[0]["lr"]

    torch.save(get_trainable_state(model, sector_embedding), OUTPUT / "last_weights.pt")
    if not (OUTPUT / "best_weights.pt").exists() or final_ev["trajectory_score"] > best_score:
        torch.save(get_trainable_state(model, sector_embedding), OUTPUT / "best_weights.pt")
        (OUTPUT / "best_metrics.json").write_text(json.dumps(final_ev, indent=2))

    summary = {
        "global_step": global_step,
        "stop_reason": stop,
        "final_full_val_metrics": final_ev,
        "sweet_spot_lr_used": sweet_spot_lr,
        "strategy": "Constant Sweet-Spot Platform Tune (Strategy A)",
        "best_score": best_score,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Done] 5小时甜点精修训练圆满收官! 总步数: {global_step:,}, 最终全量验证集胜率: {final_ev['direction_acc']*100:.2f}%, 相关度: {final_ev['trajectory_corr']:+.4f}")
    log("refine_complete", **summary)
    if run is not None:
        swanlab.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=7.92e-5, help="甜点基准学习率 (默认 7.92e-5)")
    p.add_argument("--min-lr", type=float, default=7.00e-5, help="恒定平台底噪下限 (默认 7.00e-5, 维持在甜点窄带)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dir-loss-weight", type=float, default=0.5)
    p.add_argument("--amp-loss-weight", type=float, default=0.1)
    p.add_argument("--power-law-ratio", type=float, default=0.6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--val-samples", type=int, default=8192, help="日常抽验样本数 (默认 8192, 耗时仅 30s)")
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--budget-seconds", type=int, default=18000, help="严格 5 小时预算 (18,000 秒)")
    p.add_argument("--reserve-seconds", type=int, default=900, help="保留 15 分钟终极大考与安全收尾")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=2500)
    args = p.parse_args()
    if REFINE_CONFIG:
        for k, v in REFINE_CONFIG.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


if __name__ == "__main__":
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/google-research/timesfm.git", "swanlab==0.10.1"])
    train(parse_args())
