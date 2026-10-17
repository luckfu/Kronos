"""TimesFM-3 V11 Clean: 原生合并大模型 (零 LoRA) 全量 True OOS (17.5万条) 盲测验证。

验证目标:
1. 完全不注入任何 LoRA 补丁模块，以 100% 官方原生 TimesFM 3.0 稠密大模型架构直接推断;
2. 加载云端导出的 timesfm3_v11_<MODEL_TAG>_native_merged.pt;
3. 对全量 175,428 条真实封箱交易日进行盲测推断与预测压缩落盘;
4. 严格比对该原生合并模型的指标是否精准复现甚至超越预期 (Pooled IC ~0.161, 胜率 ~58.3%)。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STARTED = time.monotonic()
INPUT_ROOT = Path(os.environ.get("EVAL_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("EVAL_OUTPUT", "/kaggle/working/eval_results"))
MAX_SAFE_SECONDS = 2400

# 由 push_eval_native_merged.py 注入
MODEL_TAG = "c07"
MODEL_LABEL = "C07"

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


def log(event: str, **fields) -> None:
    line = json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                      ensure_ascii=False, default=float)
    print(line, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "eval.log").open("a") as handle:
        handle.write(line + "\n")


def discover_inputs(root: Path):
    splits = sorted(root.glob("**/symbol_split.csv"))
    oos_panels = sorted(root.glob("**/evaluation_panel.pkl"))
    oos_samples = sorted(root.glob("**/evaluation_samples.jsonl"))

    # 寻找导出的原生合并模型文件
    merged_models = sorted(root.glob(f"**/timesfm3_v11_{MODEL_TAG}_native_merged.pt"))
    if not merged_models:
        merged_models = sorted(root.glob("**/*native_merged*.pt"))
    if not merged_models:
        merged_models = sorted(root.glob("**/*merged*.pt"))

    if not splits:
        raise FileNotFoundError(f"未找到 symbol_split.csv 于 {root}")
    if not merged_models:
        raise FileNotFoundError(f"未在 {root} 找到导出的原生合并模型 timesfm3_v11_{MODEL_TAG}_native_merged.pt")

    return splits[0], merged_models[-1], oos_panels[0] if oos_panels else None, oos_samples[0] if oos_samples else None


def extract_full_oos_set(oos_panel_pkl: Path, oos_samples_jsonl: Path, sectors: list[str]):
    import pickle
    import pandas as pd

    sector2id = {s: i for i, s in enumerate(sectors)}
    with oos_panel_pkl.open("rb") as f:
        panel = pickle.load(f)

    contexts, targets, oos_sectors, asof_dates, symbols = [], [], [], [], []
    print("[Data Loading] 开始提取全量 175,428 条 OOS 封箱交易日测试集...", flush=True)

    with oos_samples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sym = item["symbol"]
            start_idx = item["start_index"]
            asof_date = item.get("asof_date", "")

            if sym not in panel:
                continue
            df = panel[sym]
            if start_idx + 130 > len(df):
                continue

            vals = df[list(DATA_FEATURES)].iloc[start_idx : start_idx + 130].to_numpy(dtype=np.float32)
            if not (np.isfinite(vals).all() and (vals[:, :4] > 0).all() and (vals[:, 4] > 0).all()):
                continue

            p0 = vals[119, 3]
            if p0 <= 1e-4:
                continue

            w = vals.copy()
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
            sec_name = item.get("sector", "unknown")
            unknown_sector_id = len(sectors)
            oos_sectors.append(sector2id.get(sec_name, unknown_sector_id))
            asof_dates.append(asof_date)
            symbols.append(sym)

    print(f"[Data Ready] 全量 OOS 测试集提取完毕: 共 {len(contexts):,} 条实际交易日样本", flush=True)
    return (
        np.asarray(contexts, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(oos_sectors, dtype=np.int64),
        asof_dates,
        symbols,
    )


def build_forward_inputs(contexts, sectors, model, horizon: int, sector_embedding):
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


def run_full_inference(model, contexts_all, targets_all, sectors_all, device, sector_embedding, batch_size=512):
    model.eval()
    all_pred_ret, all_true_ret = [], []
    total_batches = (len(targets_all) + batch_size - 1) // batch_size
    print(f"[Inference Start] 样本总数={len(targets_all):,}, BatchSize={batch_size}, 批次数={total_batches}", flush=True)

    with torch.no_grad():
        for b_idx, i in enumerate(range(0, len(targets_all), batch_size), 1):
            elapsed_total = time.monotonic() - STARTED
            if elapsed_total > MAX_SAFE_SECONDS:
                print(f"\n[Safety Sentry] 超出安全阈值，立即安全截断！", flush=True)
                break

            c = torch.from_numpy(contexts_all[i : i + batch_size]).to(device)
            t = torch.from_numpy(targets_all[i : i + batch_size]).to(device)
            s = torch.from_numpy(sectors_all[i : i + batch_size]).to(device)

            p = forecast(model, c, s, 10, sector_embedding)
            all_pred_ret.append((p - 1.0).cpu().numpy())
            all_true_ret.append((t - 1.0).cpu().numpy())

            if b_idx % 25 == 0 or b_idx == total_batches:
                processed = min(i + batch_size, len(targets_all))
                pct = processed / len(targets_all) * 100
                print(f"[Progress] {processed:,}/{len(targets_all):,} ({pct:.1f}%) 完成 (耗时: {elapsed_total:.1f}s)", flush=True)

    pred_ret = np.concatenate(all_pred_ret, axis=0)
    true_ret = np.concatenate(all_true_ret, axis=0)
    return pred_ret, true_ret


def calculate_metrics(pred_ret: np.ndarray, true_ret: np.ndarray, asof_dates=None):
    p_10d = pred_ret[:, -1]
    t_10d = true_ret[:, -1]

    # 方向与胜率
    same_dir_10d = (np.sign(p_10d) == np.sign(t_10d))
    direction_acc_10d = float(np.mean(same_dir_10d))
    daily_accs = [float(np.mean(np.sign(pred_ret[:, d]) == np.sign(true_ret[:, d]))) for d in range(10)]

    # 平衡准确率 (Balanced Accuracy)
    p_up, t_up = p_10d > 0, t_10d > 0
    tp = float(np.sum(p_up & t_up))
    fp = float(np.sum(p_up & ~t_up))
    fn = float(np.sum(~p_up & t_up))
    tn = float(np.sum(~p_up & ~t_up))
    tpr = tp / max(tp + fn, 1e-12)
    tnr = tn / max(tn + fp, 1e-12)
    balanced_acc = float(0.5 * (tpr + tnr))
    pred_pos_ratio = float(np.mean(p_up))
    market_up_rate = float(np.mean(t_10d > 0))

    # Pooled IC
    vx = p_10d - np.mean(p_10d)
    vy = t_10d - np.mean(t_10d)
    denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2)) + 1e-12
    pooled_ic_10d = float(np.sum(vx * vy) / denom)

    # 轨迹波形相关度
    traj_corrs = []
    step = max(1, len(pred_ret) // 10000)
    for i in range(0, len(pred_ret), step):
        px, tx = pred_ret[i], true_ret[i]
        if np.std(px) > 1e-6 and np.std(tx) > 1e-6:
            r = np.corrcoef(px, tx)[0, 1]
            if np.isfinite(r):
                traj_corrs.append(r)
    mean_traj_corr = float(np.mean(traj_corrs)) if traj_corrs else 0.0

    # NRMSE & 振幅比
    err = pred_ret - true_ret
    nrmse = float(np.sqrt(np.mean(err ** 2)))
    naive_nrmse = float(np.sqrt(np.mean(true_ret ** 2)))
    skill_nrmse = float(1.0 - nrmse / max(naive_nrmse, 1e-12))
    pred_std_mean = float(np.mean(np.std(pred_ret, axis=-1)))
    true_std_mean = float(np.mean(np.std(true_ret, axis=-1)))
    amp_ratio = float(pred_std_mean / max(true_std_mean, 1e-12))
    trajectory_score = float(direction_acc_10d + 0.5 * max(pooled_ic_10d, 0.0))

    # 日度 IC & ICIR
    daily_ic_mean, daily_icir, positive_ic_ratio = 0.0, 0.0, 0.0
    if asof_dates is not None and len(asof_dates) == len(p_10d):
        date_groups = defaultdict(list)
        for idx, dt in enumerate(asof_dates):
            date_groups[dt].append((p_10d[idx], t_10d[idx]))

        day_ics = []
        for dt, pairs in date_groups.items():
            if len(pairs) >= 50:
                p_sub = np.array([x[0] for x in pairs])
                t_sub = np.array([x[1] for x in pairs])
                if np.std(p_sub) > 1e-6 and np.std(t_sub) > 1e-6:
                    ic_val = np.corrcoef(p_sub, t_sub)[0, 1]
                    if np.isfinite(ic_val):
                        day_ics.append(ic_val)

        if day_ics:
            daily_ic_mean = float(np.mean(day_ics))
            daily_ic_std = float(np.std(day_ics)) + 1e-12
            daily_icir = float(daily_ic_mean / daily_ic_std)
            positive_ic_ratio = float(np.mean([ic > 0 for ic in day_ics]))

    return {
        "direction_acc_10d": direction_acc_10d,
        "balanced_acc": balanced_acc,
        "market_natural_up_rate": market_up_rate,
        "pred_pos_ratio": pred_pos_ratio,
        "alpha_win_rate_edge": direction_acc_10d - market_up_rate,
        "pooled_ic_10d": pooled_ic_10d,
        "mean_trajectory_corr": mean_traj_corr,
        "daily_ic_mean": daily_ic_mean,
        "daily_icir": daily_icir,
        "annualized_icir": daily_icir * np.sqrt(252),
        "positive_ic_ratio": positive_ic_ratio,
        "amp_ratio": amp_ratio,
        "nrmse": nrmse,
        "skill_nrmse": skill_nrmse,
        "trajectory_score": trajectory_score,
        "daily_horizon_accs": daily_accs,
    }


def main():
    split_csv, merged_model_path, oos_panel, oos_sample = discover_inputs(INPUT_ROOT)
    log("native_eval_start", merged_model=str(merged_model_path))

    import pandas as pd
    sectors = sorted(pd.read_csv(split_csv)["sector"].dropna().unique().tolist())

    # 1. 提取全量 True OOS 盲测样本
    c_oos, t_oos, s_oos, oos_dates, oos_symbols = extract_full_oos_set(oos_panel, oos_sample, sectors)

    # 2. 从官方原生骨架加载基础模型 (注意: 100% 纯原生, 完全不调用 attach_lora)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Model Loading] 正在载入官方原生 TimesFM 3.0 PyTorch 骨架...", flush=True)
    from timesfm3 import TimesFM3Forecaster
    forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=512)
    model = forecaster.model.to(device)

    # 3. 加载导出的原生烘焙模型权重
    print(f"[Loading Merged Weights] 加载纯原生合并大模型: {merged_model_path}...", flush=True)
    ckpt = torch.load(merged_model_path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[Check OK] 原生基础模型 100% 严格匹配载入成功 (完全零 LoRA 依赖)！", flush=True)

    patch_len = int(model.input_patch_len)
    num_sectors = ckpt["metadata"]["num_sectors"]
    sector_embedding = nn.Embedding(num_sectors, patch_len).to(device)
    sector_embedding.load_state_dict(ckpt["sector_embedding_state_dict"], strict=True)
    print(f"[Sector OK] 行业嵌入权重匹配载入成功!", flush=True)

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
        print(f"[Multi-GPU] 启用双卡 DataParallel 并行加速推断 (GPU Count: {torch.cuda.device_count()})", flush=True)

    # 4. 执行全量 175,428 条盲测推断
    print("\n" + "=" * 70)
    print(">>> 聚焦终极大考: 原生烘焙合并模型 100% 全量 True OOS 盲测 <<<")
    print("=" * 70, flush=True)

    oos_pred, oos_true = run_full_inference(model, c_oos, t_oos, s_oos, device, sector_embedding, batch_size=512)
    valid_dates = oos_dates[:len(oos_pred)] if oos_dates is not None else None
    valid_symbols = oos_symbols[:len(oos_pred)] if oos_symbols is not None else None

    # 5. 落盘预测文件与报告
    pred_npz_path = OUTPUT / f"eval_predictions_{MODEL_TAG}_native.npz"
    np.savez_compressed(
        pred_npz_path,
        pred_ret=oos_pred.astype(np.float32),
        true_ret=oos_true.astype(np.float32),
        dates=np.array(valid_dates),
        symbols=np.array(valid_symbols),
    )
    print(f"[Saved] 原生模型逐样本预测落盘成功: {pred_npz_path} ({pred_npz_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    m = calculate_metrics(oos_pred, oos_true, asof_dates=valid_dates)

    report = {
        "model_architecture": "TimesFM-3.0 Native Dense Merged (Zero LoRA)",
        "model_tag": MODEL_TAG,
        "model_label": MODEL_LABEL,
        "source_checkpoint": str(merged_model_path.name),
        "evaluation_time": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(oos_pred),
        "oos_metrics": m,
    }
    (OUTPUT / f"audit_report_{MODEL_TAG}_native.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # 6. 打印验收摘要
    print("\n" + "=" * 70)
    print(f"            {MODEL_LABEL} 原生烘焙合并模型 OOS 盲测验收")
    print("=" * 70)
    print(f"{'评估指标':<26} | {'原生模型实测':<16}")
    print("-" * 70)
    for label, value in [
        ("未来10天走势胜率", f"{m['direction_acc_10d']*100:.2f}%"),
        ("平衡准确率", f"{m['balanced_acc']*100:.2f}%"),
        ("预测上涨率", f"{m['pred_pos_ratio']*100:.2f}%"),
        ("10天 Pooled IC", f"{m['pooled_ic_10d']:+.4f}"),
        ("10天形态相关度", f"{m['mean_trajectory_corr']:+.4f}"),
        ("日均 IC", f"{m['daily_ic_mean']:+.4f}"),
        ("年化 ICIR (未做重叠修正)", f"{m['annualized_icir']:.2f}"),
        ("正 IC 交易日占比", f"{m['positive_ic_ratio']*100:.1f}%"),
        ("振幅比", f"{m['amp_ratio']:.4f}"),
        ("skill_nrmse", f"{m['skill_nrmse']:.4f}"),
        ("走势综合分", f"{m['trajectory_score']:.4f}"),
    ]:
        print(f"{label:<26} | {value:<16}")
    print("=" * 70 + "\n")

    log("native_eval_complete", report=report)


if __name__ == "__main__":
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/google-research/timesfm.git"])
    main()
