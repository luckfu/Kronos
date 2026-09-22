"""Compare autoregressive and teacher-forced close-path vol on the same windows.

No training. Uses the last four causal-validation signal dates and C2 best.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CLOSE = 3
LOOKBACK = 120
HORIZON = 10
WINDOW = LOOKBACK + HORIZON + 1
N_DATES = 4
DECODE_SEED = 20260906


def last_signal_days(day_ids, n=N_DATES):
    unique = sorted({int(day) for day in day_ids})
    if len(unique) < n:
        raise ValueError(f"Need at least {n} signal days, found {len(unique)}")
    return unique[-n:]


def day_to_date(day):
    return str(np.datetime64("1970-01-01") + np.timedelta64(int(day), "D"))


def _ratio(realized, predicted):
    realized = np.asarray(realized, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = np.isfinite(realized) & np.isfinite(predicted) & (realized > 0) & (predicted > 0)
    if not mask.any():
        raise ValueError("No positive vol pairs")
    realized = realized[mask]
    predicted = predicted[mask]
    return {
        "samples": int(mask.sum()),
        "mean_realized_path_vol": float(realized.mean()),
        "mean_predicted_path_vol": float(predicted.mean()),
        "calibration_ratio": float(np.mean(realized / predicted)),
    }


def summarize(rows):
    keys = (
        "ar_t065_n5",
        "tf_t065_n5_weighted",
        "tf_t065_n5_uniform",
        "tf_t1_n16_weighted",
    )
    out = {key: _ratio([row["realized"] for row in rows], [row[key] for row in rows]) for key in keys}
    ar = out["ar_t065_n5"]["calibration_ratio"]
    tf = out["tf_t065_n5_weighted"]["calibration_ratio"]
    out["decision"] = {
        "ar_under_vol": ar > 1.2,
        "teacher_forced_not_under_vol": tf < 1.05,
        "gap_ar_minus_tf5": ar - tf,
        "confirmed_exposure_gap": bool(ar > 1.2 and tf < 1.05 and (ar - tf) > 0.3),
    }
    return out


def _extract_windows(dataset, day_ids):
    chosen = set(day_ids)
    inverse = {int(source): index for index, source in enumerate(dataset.active_positions)}
    indices = [
        inverse[position]
        for position, day in enumerate(dataset.signal_date_ids)
        if int(day) in chosen and position in inverse
    ]
    if not indices:
        raise RuntimeError("No validation windows on the selected dates")
    xs, stamps, sectors, percentiles, means, stds, days = [], [], [], [], [], [], []
    for index in indices:
        item = dataset[index]
        xs.append(item[0].numpy())
        stamps.append(item[1].numpy())
        sectors.append(int(item[2]))
        percentiles.append(float(item[4]))
        means.append(item[-2].numpy())
        stds.append(item[-1].numpy())
        source = int(dataset.active_positions[index])
        days.append(int(dataset.signal_date_ids[source]))
    return {
        "x": np.stack(xs).astype(np.float32),
        "stamp": np.stack(stamps).astype(np.float32),
        "sector": np.asarray(sectors, dtype=np.int64),
        "percentile": np.asarray(percentiles, dtype=np.float32),
        "mean": np.stack(means).astype(np.float32),
        "std": np.stack(stds).astype(np.float32),
        "day": np.asarray(days, dtype=np.int64),
    }


def _teacher_forced_vols(predictor, tokenizer, batch, device):
    import torch
    from finetune.stage3_vol_alignment import (
        VolAlignmentConfig, denorm_close, daily_returns_from_close,
        expected_path_vol, production_style_mixture_decode,
    )

    x = torch.as_tensor(batch["x"], device=device)
    stamp = torch.as_tensor(batch["stamp"], device=device)
    sector = torch.as_tensor(batch["sector"], device=device)
    percentile = torch.as_tensor(batch["percentile"], device=device)
    mean = torch.as_tensor(batch["mean"], device=device)
    std = torch.as_tensor(batch["std"], device=device)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x.float(), half=True)
        context = predictor.encode_context(
            s1[:, :-1], s2[:, :-1], stamp[:, :-1],
            sector_id=sector, size_percentile=percentile,
        )
        start = LOOKBACK - 1
        logits1 = predictor.predict_s1(context[:, start:start + HORIZON])
        close_mean = mean[:, CLOSE]
        close_std = std[:, CLOSE]
        last_px = denorm_close(x[:, LOOKBACK - 1, CLOSE], close_mean, close_std)
        target_px = denorm_close(x[:, LOOKBACK:LOOKBACK + HORIZON, CLOSE], close_mean[:, None], close_std[:, None])
        realized = daily_returns_from_close(target_px, last_px).std(dim=-1, unbiased=False)
        result = {"realized": realized.detach().cpu().numpy()}
        configs = {
            "tf_t065_n5": VolAlignmentConfig(temperature=0.65, top_p=0.8, candidates=5, weight=0.0),
            "tf_t1_n16": VolAlignmentConfig(temperature=1.0, top_p=1.0, candidates=16, support_k=16, weight=0.0),
        }
        for name, config in configs.items():
            _, weights, extra = production_style_mixture_decode(
                predictor, tokenizer, context, logits1, config,
                position_start=start, is_causal=True,
            )
            candidate_z = extra["candidate_paths"][:, :, CLOSE, :]
            candidate_px = denorm_close(candidate_z, close_mean[:, None, None], close_std[:, None, None])
            weighted, candidate_vol, _ = expected_path_vol(candidate_px, weights, last_px)
            result[name + "_weighted"] = weighted.detach().cpu().numpy()
            result[name + "_uniform"] = candidate_vol.mean(-1).detach().cpu().numpy()
    return result


def _autoregressive_vol(predictor, tokenizer, batch, device):
    import torch
    from finetune.stage3_vol_alignment import denorm_close, daily_returns_from_close
    from model.kronos import auto_regressive_inference

    x = torch.as_tensor(batch["x"], device=device)
    stamp = torch.as_tensor(batch["stamp"], device=device)
    sector = torch.as_tensor(batch["sector"], device=device)
    percentile = torch.as_tensor(batch["percentile"], device=device)
    mean = torch.as_tensor(batch["mean"], device=device)
    std = torch.as_tensor(batch["std"], device=device)
    forecast = auto_regressive_inference(
        tokenizer, predictor,
        x[:, :LOOKBACK], stamp[:, :LOOKBACK], stamp[:, LOOKBACK:LOOKBACK + HORIZON],
        max_context=512, pred_len=HORIZON, clip=5,
        T=0.65, top_k=0, top_p=0.8, sample_count=5, verbose=False,
        sector_id=sector, size_percentile=percentile, return_samples=True,
    )
    future_z = torch.as_tensor(forecast[:, :, -HORIZON:, CLOSE], device=device)
    close_mean = mean[:, CLOSE]
    close_std = std[:, CLOSE]
    last_px = denorm_close(x[:, LOOKBACK - 1, CLOSE], close_mean, close_std)
    future_px = denorm_close(future_z, close_mean[:, None, None], close_std[:, None, None])
    daily = daily_returns_from_close(future_px, last_px)
    return daily.std(dim=-1, unbiased=False).mean(-1).detach().cpu().numpy()


def score_shard(predictor, tokenizer, windows, device, batch_size=8):
    import torch

    predictor.eval()
    tokenizer.eval()
    torch.manual_seed(DECODE_SEED)
    rows = []
    count = len(windows["x"])
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = {key: windows[key][start:stop] for key in ("x", "stamp", "sector", "percentile", "mean", "std")}
        forced = _teacher_forced_vols(predictor, tokenizer, batch, device)
        sampled = _autoregressive_vol(predictor, tokenizer, batch, device)
        for index in range(stop - start):
            rows.append({
                "day": int(windows["day"][start + index]),
                "date": day_to_date(windows["day"][start + index]),
                "realized": float(forced["realized"][index]),
                "ar_t065_n5": float(sampled[index]),
                "tf_t065_n5_weighted": float(forced["tf_t065_n5_weighted"][index]),
                "tf_t065_n5_uniform": float(forced["tf_t065_n5_uniform"][index]),
                "tf_t1_n16_weighted": float(forced["tf_t1_n16_weighted"][index]),
            })
        print(json.dumps({"phase": "batch_done", "scored": stop, "total": count}), flush=True)
        del forced
        torch.cuda.empty_cache()
    return rows
