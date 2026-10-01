"""Continuous regression + cross-sectional feature ablations (CPU, val panel).

Follow-up to Phase A/B/C: test whether continuous mfe10/mae10 / first-touch
proxies and cheap cross-section ranks lift effect size enough to justify a
short neural probe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernbert_finance.ablations._panel_io import load_panel
from modernbert_finance.ablations.label_time_diagnostics import BINARY_HEADS
from modernbert_finance.ablations.simple_baseline import summarize_history_windows
from modernbert_finance.build_dataset import (
    FEATURES,
    FIRST_TOUCH_DOWNSIDE_THRESHOLD,
    FIRST_TOUCH_UPSIDE_THRESHOLD,
    HORIZON,
    LOOKBACK,
    prepare_frame,
)
from modernbert_finance.build_targets import WINDOW

EPS = 1e-4
SEED = 20261001


def _prior_ll(y: np.ndarray) -> float:
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    probs = np.column_stack([np.full(len(y), 1.0 - p), np.full(len(y), p)])
    return float(log_loss(y, probs, labels=[0, 1]))


def _first_touch_day(
    future_high: np.ndarray,
    future_low: np.ndarray,
    close: float,
    *,
    upside: bool,
    threshold: float,
) -> float:
    """1-indexed day of first touch within horizon; horizon+1 if never."""
    for i in range(len(future_high)):
        if upside:
            if future_high[i] / close - 1.0 >= threshold:
                return float(i + 1)
        else:
            if 1.0 - future_low[i] / close >= threshold:
                return float(i + 1)
    return float(HORIZON + 1)


def _first_touch_class(
    future_high: np.ndarray,
    future_low: np.ndarray,
    close: float,
) -> int:
    """0=upside_first, 1=downside_first, 2=neither (downside wins same-day)."""
    up_th = FIRST_TOUCH_UPSIDE_THRESHOLD
    dn_th = FIRST_TOUCH_DOWNSIDE_THRESHOLD
    for high, low in zip(future_high, future_low):
        upside = high / close - 1.0 >= up_th
        downside = 1.0 - low / close >= dn_th
        if downside:
            return 1
        if upside:
            return 0
    return 2


def build_enriched_matrix(
    panel: dict[str, pd.DataFrame] | Path,
    targets: pd.DataFrame,
    *,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
) -> dict[str, Any]:
    """Build base summaries + per-row meta for cross-section joins + FT proxies."""
    if isinstance(panel, (str, Path)):
        panel = load_panel(Path(panel))
    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None

    # Align target rows to the same signal window used when enumerating
    # feature windows. Sidecar parquet is typically full-panel length; feature
    # build skips asof outside [signal_start, signal_end]. Without this filter,
    # train→val with a recent window mismatches (e.g. 2.35M vs 9.01M on train).
    # Order is preserved (sorted symbol, start_index) so row i still matches.
    if signal_start is not None or signal_end is not None:
        if "asof_date" not in targets.columns:
            raise ValueError(
                "targets missing asof_date; cannot filter to signal window"
            )
        asof_ts = pd.to_datetime(targets["asof_date"])
        mask = np.ones(len(targets), dtype=bool)
        if signal_start is not None:
            mask &= asof_ts >= pd.Timestamp(signal_start)
        if signal_end is not None:
            mask &= asof_ts <= pd.Timestamp(signal_end)
        targets = targets.loc[mask].reset_index(drop=True)

    histories: list[np.ndarray] = []
    symbols: list[str] = []
    asofs: list[str] = []
    sectors: list[str] = []
    ret5: list[float] = []
    ret20: list[float] = []
    vol20: list[float] = []
    ft_up_day: list[float] = []
    ft_dn_day: list[float] = []
    ft_class: list[int] = []
    ft_signed: list[float] = []  # +(11-up_day) if up first, -(11-dn_day) if dn first, 0 else

    close_idx = 3  # FEATURES order: open,high,low,close,volume,amount

    for symbol in sorted(panel):
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
        dates = frame.index
        sector_col = (
            frame["sector"].astype(str).to_numpy()
            if "sector" in frame.columns
            else np.full(len(frame), "UNK", dtype=object)
        )
        high = frame["high"].to_numpy(dtype=np.float64)
        low = frame["low"].to_numpy(dtype=np.float64)
        close = frame["close"].to_numpy(dtype=np.float64)
        max_start = len(frame) - WINDOW + 1
        for start in range(max_start):
            asof_pos = start + LOOKBACK - 1
            asof = pd.Timestamp(dates[asof_pos]).date()
            if start_date and asof < start_date:
                continue
            if end_date and asof > end_date:
                continue
            hist = values[start : start + LOOKBACK]
            histories.append(hist)
            symbols.append(str(symbol))
            asofs.append(asof.isoformat())
            sectors.append(str(sector_col[asof_pos]))

            c = hist[:, close_idx]
            last = float(c[-1])
            r5 = last / max(float(c[-6]), 1e-8) - 1.0 if len(c) >= 6 else 0.0
            r20 = last / max(float(c[-21]), 1e-8) - 1.0 if len(c) >= 21 else 0.0
            log_ret = np.diff(np.log(np.maximum(c, 1e-8)))
            v20 = float(np.std(log_ret[-20:])) if len(log_ret) >= 20 else 0.0
            ret5.append(r5)
            ret20.append(r20)
            vol20.append(v20)

            fut_h = high[asof_pos + 1 : asof_pos + 1 + HORIZON]
            fut_l = low[asof_pos + 1 : asof_pos + 1 + HORIZON]
            cur = float(close[asof_pos])
            up_d = _first_touch_day(
                fut_h, fut_l, cur, upside=True, threshold=FIRST_TOUCH_UPSIDE_THRESHOLD
            )
            dn_d = _first_touch_day(
                fut_h, fut_l, cur, upside=False, threshold=FIRST_TOUCH_DOWNSIDE_THRESHOLD
            )
            cls = _first_touch_class(fut_h, fut_l, cur)
            ft_up_day.append(up_d)
            ft_dn_day.append(dn_d)
            ft_class.append(cls)
            if cls == 0:
                ft_signed.append(float(HORIZON + 1 - up_d))
            elif cls == 1:
                ft_signed.append(float(-(HORIZON + 1 - dn_d)))
            else:
                ft_signed.append(0.0)

    if len(histories) != len(targets):
        raise ValueError(
            f"feature windows {len(histories)} != targets {len(targets)}; "
            "check signal date bounds / panel"
        )

    hist = np.stack(histories, axis=0)
    base = summarize_history_windows(hist)

    meta = pd.DataFrame(
        {
            "symbol": symbols,
            "asof_date": asofs,
            "sector": sectors,
            "ret5": ret5,
            "ret20": ret20,
            "vol20": vol20,
        }
    )
    # Cross-section ranks within asof_date (percentile 0..1)
    for col in ("ret5", "ret20", "vol20"):
        meta[f"cs_rank_{col}"] = meta.groupby("asof_date")[col].rank(pct=True)
        # Industry-relative: value - sector median on same day
        sector_med = meta.groupby(["asof_date", "sector"])[col].transform("median")
        meta[f"ind_rel_{col}"] = meta[col] - sector_med

    xsec_cols = [
        "cs_rank_ret5",
        "cs_rank_ret20",
        "cs_rank_vol20",
        "ind_rel_ret5",
        "ind_rel_ret20",
        "ind_rel_vol20",
    ]
    xsec = meta[xsec_cols].to_numpy(dtype=np.float64)
    # Fill any NaN (singleton sector-date) with 0
    xsec = np.nan_to_num(xsec, nan=0.0)

    arrays = {h: targets[h].to_numpy() for h in BINARY_HEADS}
    arrays["mfe10"] = targets["mfe10"].to_numpy(dtype=np.float64)
    arrays["mae10"] = targets["mae10"].to_numpy(dtype=np.float64)
    arrays["mae10_abs"] = (-arrays["mae10"]).astype(np.float64)  # positive downside excursion
    arrays["net_excursion"] = (arrays["mfe10"] + arrays["mae10"]).astype(np.float64)
    arrays["range_excursion"] = (arrays["mfe10"] - arrays["mae10"]).astype(np.float64)
    arrays["ft_up_day"] = np.asarray(ft_up_day, dtype=np.float64)
    arrays["ft_dn_day"] = np.asarray(ft_dn_day, dtype=np.float64)
    arrays["ft_signed"] = np.asarray(ft_signed, dtype=np.float64)
    arrays["ft_upside_first"] = (np.asarray(ft_class) == 0).astype(np.int64)
    arrays["ft_downside_first"] = (np.asarray(ft_class) == 1).astype(np.int64)

    return {
        "base_features": base,
        "xsec_features": xsec,
        "combined_features": np.concatenate([base, xsec], axis=1),
        "targets": arrays,
        "meta": meta,
        "xsec_cols": xsec_cols,
        "n_samples": int(len(base)),
        "base_dim": int(base.shape[1]),
        "xsec_dim": int(xsec.shape[1]),
    }


def _eval_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    alpha: float = 1.0,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if len(y) < 50:
        return {"skipped": True, "reason": "too_few_rows"}
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=seed)
    naive_pred = np.full_like(y_te, float(np.mean(y_tr)))
    naive_mae = float(mean_absolute_error(y_te, naive_pred))
    naive_r2 = float(r2_score(y_te, naive_pred))  # ~0 by construction
    # Linear (Ridge for stability)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=seed))
    ridge.fit(x_tr, y_tr)
    pred = ridge.predict(x_te)
    mae = float(mean_absolute_error(y_te, pred))
    r2 = float(r2_score(y_te, pred))
    return {
        "skipped": False,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "naive_mae": naive_mae,
        "naive_r2": naive_r2,
        "ridge_mae": mae,
        "ridge_r2": r2,
        "mae_lift_vs_naive": naive_mae - mae,  # positive = better
        "r2_lift_vs_naive": r2 - naive_r2,
        "beats_naive_mae": bool(mae < naive_mae - 1e-8),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
    }


def _fit_eval_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_iter: int = 400,
) -> dict[str, Any]:
    y = y.astype(np.int64).reshape(-1)
    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "single_class"}
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed),
    )
    clf.fit(x_tr, y_tr)
    proba = clf.predict_proba(x_te)[:, 1]
    prior = _prior_ll(y_te)
    ll = float(log_loss(y_te, np.column_stack([1 - proba, proba]), labels=[0, 1]))
    return {
        "skipped": False,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "prior_log_loss": prior,
        "model_log_loss": ll,
        "delta_model_minus_prior": ll - prior,
        "beats_prior": bool(ll < prior - EPS),
        "positive_rate_test": float(np.mean(y_te)),
    }


def run_continuous_and_xsection(
    *,
    val_panel: Path,
    val_targets: Path,
    phase_b_json: Path | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    targets = pd.read_parquet(val_targets)
    packed = build_enriched_matrix(val_panel, targets)
    base = packed["base_features"]
    xsec = packed["xsec_features"]
    comb = packed["combined_features"]
    arrays = packed["targets"]

    continuous_heads = [
        "mfe10",
        "mae10",
        "mae10_abs",
        "net_excursion",
        "range_excursion",
        "ft_up_day",
        "ft_dn_day",
        "ft_signed",
    ]
    continuous_base = {
        h: _eval_regression(base, arrays[h], seed=seed) for h in continuous_heads
    }
    continuous_comb = {
        h: _eval_regression(comb, arrays[h], seed=seed) for h in continuous_heads
    }

    # Also logistic on first-touch binary proxies
    ft_logistic = {
        "ft_upside_first_base": _fit_eval_logistic(base, arrays["ft_upside_first"], seed=seed),
        "ft_upside_first_comb": _fit_eval_logistic(comb, arrays["ft_upside_first"], seed=seed),
        "ft_downside_first_base": _fit_eval_logistic(
            base, arrays["ft_downside_first"], seed=seed
        ),
        "ft_downside_first_comb": _fit_eval_logistic(
            comb, arrays["ft_downside_first"], seed=seed
        ),
    }

    # 8-head logistic: base vs base+xsection
    logistic_base = {h: _fit_eval_logistic(base, arrays[h], seed=seed) for h in BINARY_HEADS}
    logistic_xsec_only = {
        h: _fit_eval_logistic(xsec, arrays[h], seed=seed) for h in BINARY_HEADS
    }
    logistic_comb = {h: _fit_eval_logistic(comb, arrays[h], seed=seed) for h in BINARY_HEADS}

    def _macro_delta(per_head: dict[str, Any]) -> float:
        deltas = [
            v["delta_model_minus_prior"]
            for v in per_head.values()
            if not v.get("skipped") and "delta_model_minus_prior" in v
        ]
        return float(np.mean(deltas)) if deltas else float("nan")

    macro_base = _macro_delta(logistic_base)
    macro_xsec = _macro_delta(logistic_xsec_only)
    macro_comb = _macro_delta(logistic_comb)

    phase_b_macro = None
    if phase_b_json and Path(phase_b_json).is_file():
        pb = json.loads(Path(phase_b_json).read_text(encoding="utf-8"))
        phase_b_macro = (
            pb.get("structure", {})
            .get("dilution_summary", {})
            .get("independent_8head_macro_delta")
        )

    # Decision: separate magnitude (vol/range) from direction.
    directional_r2 = {
        h: continuous_comb[h]["ridge_r2"]
        for h in ("mfe10", "mae10", "net_excursion", "ft_signed")
        if not continuous_comb[h].get("skipped")
    }
    magnitude_r2 = {
        h: continuous_comb[h]["ridge_r2"]
        for h in ("range_excursion", "ft_up_day", "ft_dn_day", "mae10_abs")
        if not continuous_comb[h].get("skipped")
    }
    best_cont_r2 = max(
        (v.get("ridge_r2") or -999 for v in continuous_comb.values() if not v.get("skipped")),
        default=float("nan"),
    )
    best_dir_r2 = max(directional_r2.values()) if directional_r2 else float("nan")
    best_mag_r2 = max(magnitude_r2.values()) if magnitude_r2 else float("nan")
    best_cont_mae_lift = max(
        (
            v.get("mae_lift_vs_naive") or -999
            for v in continuous_comb.values()
            if not v.get("skipped")
        ),
        default=float("nan"),
    )
    lift_vs_b = None if phase_b_macro is None else float(macro_comb - phase_b_macro)
    justify_neural = bool(
        (macro_comb is not None and macro_comb < -0.04)
        or (
            isinstance(best_dir_r2, float)
            and best_dir_r2 > 0.10
            and lift_vs_b is not None
            and abs(lift_vs_b) > 0.015
        )
    )
    borderline_probe = bool(
        not justify_neural
        and macro_comb is not None
        and macro_comb < -0.025
        and isinstance(best_dir_r2, float)
        and best_dir_r2 > 0.05
    )
    recommend_relabel = bool(
        (not isinstance(best_dir_r2, float) or best_dir_r2 < 0.12)
        and (macro_comb is None or abs(macro_comb) < 0.04)
    )
    verdict = (
        "xsection_helps_modestly_but_direction_still_weak__prefer_relabel"
        if recommend_relabel and not justify_neural
        else ("justify_short_neural" if justify_neural else "inconclusive")
    )

    return {
        "phase": "D_continuous_xsection",
        "n_samples": packed["n_samples"],
        "base_dim": packed["base_dim"],
        "xsec_dim": packed["xsec_dim"],
        "xsec_cols": packed["xsec_cols"],
        "seed": seed,
        "continuous_regression": {
            "base_features": continuous_base,
            "base_plus_xsection": continuous_comb,
        },
        "first_touch_logistic": ft_logistic,
        "logistic_8head": {
            "base": logistic_base,
            "xsec_only": logistic_xsec_only,
            "base_plus_xsection": logistic_comb,
            "macro_delta_base": macro_base,
            "macro_delta_xsec_only": macro_xsec,
            "macro_delta_base_plus_xsection": macro_comb,
            "phase_b_independent_8head_macro_delta": phase_b_macro,
            "delta_lift_vs_phase_b": lift_vs_b,
        },
        "decision": {
            "best_continuous_ridge_r2": best_cont_r2,
            "best_directional_ridge_r2": best_dir_r2,
            "best_magnitude_ridge_r2": best_mag_r2,
            "directional_r2": directional_r2,
            "magnitude_r2": magnitude_r2,
            "best_continuous_mae_lift_vs_naive": best_cont_mae_lift,
            "macro_delta_base": macro_base,
            "macro_delta_with_xsection": macro_comb,
            "macro_lift_vs_phase_b": lift_vs_b,
            "justify_short_neural_probe": justify_neural,
            "borderline_sanity_neural_probe_ok": borderline_probe,
            "recommend_change_10d_mfe_mae_label": recommend_relabel,
            "verdict": verdict,
            "rationale": (
                "Separate magnitude (range/FT-day, partly vol) from direction "
                "(mfe/net/ft_signed). Justify short neural only if directional R²>0.10 "
                "with clear logistic lift, or macro Δ<-0.04. Otherwise change 10D "
                "MFE/MAE label definition; optional tiny probe only as linear-harvest sanity."
            ),
        },
    }


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
    lg = payload["logistic_8head"]
    dec = payload["decision"]
    cont_b = payload["continuous_regression"]["base_features"]
    cont_c = payload["continuous_regression"]["base_plus_xsection"]

    def _row(name: str, d: dict[str, Any]) -> str:
        if d.get("skipped"):
            return f"| {name} | — | — | — | — | skipped |"
        return (
            f"| {name} | {d['naive_mae']:.6f} | {d['ridge_mae']:.6f} | "
            f"{d['ridge_r2']:.6f} | {d['mae_lift_vs_naive']:+.6f} | "
            f"{'Y' if d['beats_naive_mae'] else 'N'} |"
        )

    lines = [
        "# Kairos 连续目标 / 截面特征消融（Phase D）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
    ]
    if dec["recommend_change_10d_mfe_mae_label"] and not dec["justify_short_neural_probe"]:
        lines.append(
            f"**不值得上短神经探针。** 截面+连续后 8-head macro Δ=`{lg['macro_delta_base_plus_xsection']:.5f}`"
            f"（Phase B 基线 `{lg['phase_b_independent_8head_macro_delta']}`），"
            f"最好连续 Ridge R²=`{dec['best_continuous_ridge_r2']:.4f}` —— "
            "效应量仍弱，应先改 10D MFE/MAE 标签定义。"
        )
    elif dec["justify_short_neural_probe"]:
        lines.append(
            f"**可考虑短神经探针。** 截面后 macro Δ=`{lg['macro_delta_base_plus_xsection']:.5f}`，"
            f"最好连续 R²=`{dec['best_continuous_ridge_r2']:.4f}`。"
        )
    else:
        lines.append(
            f"**信号仍弱。** macro Δ(comb)=`{lg['macro_delta_base_plus_xsection']:.5f}`，"
            f"R²=`{dec['best_continuous_ridge_r2']:.4f}`；优先改标签，神经探针非优先。"
        )

    lines += [
        "",
        "## 设定",
        "",
        f"- n_samples=`{payload['n_samples']}`，base_dim=`{payload['base_dim']}`，"
        f"xsec_dim=`{payload['xsec_dim']}`，seed=`{payload['seed']}`",
        f"- 截面列：`{', '.join(payload['xsec_cols'])}`（同日横截面分位 + 行业相对）",
        "- 连续目标：mfe10 / mae10 / |mae| / net / range / first-touch day & signed",
        "- 模型：StandardScaler + Ridge(α=1) 或 Logistic；75/25 分层或随机切分",
        "- **未**重启 R2；**未**动同事 TPU WIP",
        "",
        "## 1) 连续回归（Ridge vs 训练均值 naive）",
        "",
        "### Base OHLCVA 汇总",
        "",
        "| Target | naive MAE | ridge MAE | ridge R² | MAE lift | beats |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for h in cont_b:
        lines.append(_row(h, cont_b[h]))
    lines += [
        "",
        "### Base + 截面",
        "",
        "| Target | naive MAE | ridge MAE | ridge R² | MAE lift | beats |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for h in cont_c:
        lines.append(_row(h, cont_c[h]))

    ft = payload["first_touch_logistic"]
    lines += [
        "",
        "## 2) First-touch 二分类 logistic（参考）",
        "",
        "| Head | prior LL | model LL | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, d in ft.items():
        if d.get("skipped"):
            lines.append(f"| {name} | — | — | skipped |")
        else:
            lines.append(
                f"| {name} | {d['prior_log_loss']:.6f} | {d['model_log_loss']:.6f} | "
                f"{d['delta_model_minus_prior']:+.6f} |"
            )

    lines += [
        "",
        "## 3) 八头 logistic：Base vs 仅截面 vs Base+截面",
        "",
        f"| 设定 | macro Δ(model−prior) |",
        f"| --- | ---: |",
        f"| Phase B 独立八头（对照） | {lg['phase_b_independent_8head_macro_delta']} |",
        f"| Base（复现） | {lg['macro_delta_base']:.6f} |",
        f"| 仅截面 | {lg['macro_delta_xsec_only']:.6f} |",
        f"| Base+截面 | {lg['macro_delta_base_plus_xsection']:.6f} |",
        f"| Base+截面 − Phase B | {lg['delta_lift_vs_phase_b']} |",
        "",
        "### 各头 Δ（Base+截面）",
        "",
        "| Head | prior | model | Δ | beats |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for h, d in lg["base_plus_xsection"].items():
        if d.get("skipped"):
            continue
        lines.append(
            f"| {h} | {d['prior_log_loss']:.6f} | {d['model_log_loss']:.6f} | "
            f"{d['delta_model_minus_prior']:+.6f} | {'Y' if d['beats_prior'] else 'N'} |"
        )

    lines += [
        "",
        "## 4) 决策",
        "",
        f"- justify_short_neural_probe = `{dec['justify_short_neural_probe']}`",
        f"- recommend_change_10d_mfe_mae_label = `{dec['recommend_change_10d_mfe_mae_label']}`",
        f"- best continuous ridge R² = `{dec['best_continuous_ridge_r2']:.6f}`",
        f"- best MAE lift vs naive = `{dec['best_continuous_mae_lift_vs_naive']:.6f}`",
        f"- 判据：{dec['rationale']}",
        "",
        "## 证据强度",
        "",
        "| 主张 | 强度 | 依据 |",
        "| --- | --- | --- |",
        "| 连续 mfe/mae 可线性预测 | 弱–中 | R² / MAE lift 见表 |",
        "| 截面特征抬升八头 Δ | 见 macro 表 | vs Phase B |",
        "| 值得短神经探针 | "
        + ("中" if dec["justify_short_neural_probe"] else "弱/否")
        + " | 阈值未/已满足 |",
        "| 应改 10D 标签 | "
        + ("强" if dec["recommend_change_10d_mfe_mae_label"] else "弱")
        + " | 效应量门槛 |",
        "",
    ]
    text = "\n".join(lines) + "\n"
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-panel", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument(
        "--phase-b-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_b_structure.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_d_continuous_xsection.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_d_continuous_xsection_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    payload = run_continuous_and_xsection(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        phase_b_json=args.phase_b_json,
        seed=args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_cn_memo(payload, args.out_md)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "decision": payload["decision"],
                "logistic_macros": {
                    "base": payload["logistic_8head"]["macro_delta_base"],
                    "xsec_only": payload["logistic_8head"]["macro_delta_xsec_only"],
                    "comb": payload["logistic_8head"]["macro_delta_base_plus_xsection"],
                    "phase_b": payload["logistic_8head"][
                        "phase_b_independent_8head_macro_delta"
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
