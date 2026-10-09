"""Small TimesFM-3 zero-shot price-forecast benchmark.

The first pass is intentionally close-only: 120 observed closes -> 10 future
closes. This isolates the value of the pretrained forecaster before adding
OHLCVA covariates or market features.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_panel(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise TypeError(f"Expected dict panel, got {type(panel)!r}")
    return panel


def choose_windows(
    panel: dict[str, pd.DataFrame],
    symbols: int,
    signal_start: str,
    signal_end: str,
    context: int,
    horizon: int,
) -> list[dict]:
    windows: list[dict] = []
    for symbol in sorted(panel)[:symbols]:
        frame = panel[symbol].copy()
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        if "close" not in frame:
            continue
        dates = frame.index[
            (frame.index >= pd.Timestamp(signal_start))
            & (frame.index <= pd.Timestamp(signal_end))
        ]
        for asof in dates:
            end = frame.index.get_loc(asof)
            if end < context - 1 or end + horizon >= len(frame):
                continue
            values = frame["close"].to_numpy(dtype=np.float32)
            history = values[end - context + 1 : end + 1]
            future = values[end + 1 : end + horizon + 1]
            covariate_columns = ["open", "high", "low", "volume", "amount"]
            if not all(column in frame for column in covariate_columns):
                continue
            covariates = frame[covariate_columns].to_numpy(dtype=np.float32)
            price_scale = max(float(history[-1]), 1e-6)
            covariates[:, :3] /= price_scale
            for column_index in (3, 4):
                scale = max(
                    float(np.nanmedian(covariates[: end + 1, column_index])),
                    1e-6,
                )
                covariates[:, column_index] = np.log1p(
                    np.maximum(covariates[:, column_index], 0.0) / scale
                )
            # TimesFM-3 expects covariates as (num_covariates, context).
            past_covariates = covariates[
                end - context + 1 : end + 1
            ].T
            if not np.isfinite(history).all() or not np.isfinite(future).all():
                continue
            windows.append(
                {
                    "symbol": symbol,
                    "asof": asof.strftime("%Y-%m-%d"),
                    "last_close": float(history[-1]),
                    "history": history,
                    "future": future,
                    "past_covariates": past_covariates,
                    "ohlcva": frame[
                        ["open", "high", "low", "volume", "amount"]
                    ].to_numpy(dtype=np.float32)[
                        end - context + 1 : end + 1
                    ],
                }
            )
    return windows


def make_inputs(item: dict, arm: str, context: int) -> tuple[np.ndarray, np.ndarray]:
    hist = item["history"].astype(np.float32)
    ohlcva = item["ohlcva"].copy()
    last = max(float(hist[-1]), 1e-6)
    if arm == "raw_ohlcva":
        cov = ohlcva.T
        return hist, cov
    if arm == "norm_ohlcva":
        cov = ohlcva.copy()
        cov[:, :3] /= last
        for col in (3, 4):
            scale = max(float(np.nanmedian(cov[:, col])), 1e-6)
            cov[:, col] = np.log1p(np.maximum(cov[:, col], 0) / scale)
        return hist, cov.T
    log_price = np.log(np.maximum(hist, 1e-6))
    returns = np.diff(log_price, prepend=log_price[0]).astype(np.float32)
    price_ret = np.diff(np.log(np.maximum(ohlcva[:, :3], 1e-6)), axis=0, prepend=np.log(np.maximum(ohlcva[:1, :3], 1e-6)))
    volume_ret = np.diff(np.log1p(np.maximum(ohlcva[:, 3:], 0)), axis=0, prepend=np.log1p(np.maximum(ohlcva[:1, 3:], 0)))
    features = np.concatenate([price_ret, volume_ret], axis=1).T.astype(np.float32)
    if arm == "return_features":
        return returns, features
    return returns, price_ret.T.astype(np.float32)


def run_arm(forecaster, windows: list[dict], arm: str, args: argparse.Namespace) -> dict:
    rows = []
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset : offset + args.batch_size]
        prepared = [make_inputs(item, arm, args.context) for item in batch]
        output = forecaster.predict_batch(
            [x[0] for x in prepared],
            horizon=args.horizon,
            return_quantiles=True,
            past_only_covariates=[x[1] for x in prepared],
            use_znorm=True,
        )
        for item, prediction, (target, _) in zip(batch, list(output), prepared):
            forecast = np.asarray(prediction.forecast)[: args.horizon]
            q = np.asarray(prediction.quantiles)[: args.horizon]
            if arm in {"return_features", "return_price"}:
                base = np.log(item["last_close"])
                actual = np.log(item["future"] / item["last_close"])
                median = np.exp(base + np.cumsum(forecast))
                q10 = np.exp(base + np.cumsum(q[:, 0]))
                q90 = np.exp(base + np.cumsum(q[:, -1]))
            else:
                actual = item["future"]
                median, q10, q90 = forecast, q[:, 0], q[:, -1]
            pred_ret = float(median[-1] / item["last_close"] - 1)
            actual_ret = float(item["future"][-1] / item["last_close"] - 1)
            rows.append({
                "symbol": item["symbol"], "asof": item["asof"],
                "pred_return_d10": pred_ret, "actual_return_d10": actual_ret,
                "mae_10d": float(np.mean(np.abs(median - item["future"]))),
                "rmse_10d": float(np.sqrt(np.mean((median - item["future"]) ** 2))),
                "nmae_10d": float(
                    np.mean(np.abs(median - item["future"])) / item["last_close"]
                ),
                "nrmse_10d": float(
                    np.sqrt(np.mean((median - item["future"]) ** 2))
                    / item["last_close"]
                ),
                "endpoint_abs_pct_error": float(
                    abs(median[-1] - item["future"][-1]) / item["last_close"]
                ),
                "endpoint_signed_pct_error": float(
                    (median[-1] - item["future"][-1]) / item["last_close"]
                ),
                "path_corr": float(
                    np.corrcoef(median, item["future"])[0, 1]
                    if np.std(median) > 0 and np.std(item["future"]) > 0
                    else 0.0
                ),
                "path_direction_accuracy": float(
                    np.mean(
                        np.sign(np.diff(median))
                        == np.sign(np.diff(item["future"]))
                    )
                ),
                "direction_correct": int(np.sign(pred_ret) == np.sign(actual_ret)),
                "interval_coverage": float(np.mean((item["future"] >= q10) & (item["future"] <= q90))),
            })
    frame = pd.DataFrame(rows)
    daily = frame.groupby("asof").apply(
        lambda x: x["pred_return_d10"].corr(x["actual_return_d10"], method="spearman")
    )
    return {
        "arm": arm, "windows": len(frame),
        "mae": float(frame.mae_10d.mean()), "rmse": float(frame.rmse_10d.mean()),
        "normalized_mae": float(frame.nmae_10d.mean()),
        "normalized_rmse": float(frame.nrmse_10d.mean()),
        "endpoint_abs_pct_error": float(frame.endpoint_abs_pct_error.mean()),
        "endpoint_signed_pct_error": float(frame.endpoint_signed_pct_error.mean()),
        "path_correlation": float(frame.path_corr.mean()),
        "path_direction_accuracy": float(frame.path_direction_accuracy.mean()),
        "direction_accuracy": float(frame.direction_correct.mean()),
        "return_bias": float((frame.pred_return_d10 - frame.actual_return_d10).mean()),
        "return_correlation": float(frame.pred_return_d10.corr(frame.actual_return_d10)),
        "pooled_return_rank_ic": float(frame.pred_return_d10.corr(frame.actual_return_d10, method="spearman")),
        "mean_daily_return_rank_ic": float(daily.mean()),
        "mean_interval_coverage": float(frame.interval_coverage.mean()),
    }


def run(args: argparse.Namespace) -> None:
    from timesfm3 import TimesFM3Forecaster

    panel = load_panel(Path(args.panel))
    windows = choose_windows(
        panel,
        args.symbols,
        args.signal_start,
        args.signal_end,
        args.context,
        args.horizon,
    )
    if not windows:
        raise RuntimeError("No valid windows found")

    forecaster = TimesFM3Forecaster.from_pretrained(
        args.model,
        per_core_batch_size=args.batch_size,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"model": args.model, "windows": len(windows), "arms": [
        run_arm(forecaster, windows, arm, args)
        for arm in ("raw_ohlcva", "norm_ohlcva", "return_features", "return_price")
    ]}
    out.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel")
    parser.add_argument("--output", default="outputs/timesfm3_zero_shot")
    parser.add_argument("--model", default="google/timesfm-3.0-pytorch")
    parser.add_argument("--symbols", type=int, default=256)
    parser.add_argument("--signal-start", default="2026-07-17")
    parser.add_argument("--signal-end", default="2026-08-10")
    parser.add_argument("--context", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.panel is None:
        candidates = list(
            Path("/kaggle/input").glob(
                "**/evaluation_oos_20260914/evaluation_panel.pkl"
            )
        )
        if not candidates:
            candidates = list(Path("/kaggle/input").glob("**/evaluation_panel.pkl"))
        if not candidates:
            raise FileNotFoundError("No evaluation_panel.pkl found under /kaggle/input")
        args.panel = str(candidates[0])
    return args


if __name__ == "__main__":
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "git+https://github.com/google-research/timesfm.git",
        ]
    )
    run(parse_args())
