"""Helpers for Kairos short ranking probe (Phase M/N).

Predict continuous path-mfe10 (or within-day CS ranks) with a light
identity / freeze+rank head. Phase M: MSE; Phase N: pairwise/listwise.
Primary metrics: daily Rank IC and TopK lift.
NOT binary mfe≥10%. NOT 22-layer R2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MFE10_DEF = "max(high[T+1:T+10]) / close[T] - 1"
TARGET_NAME = "mfe10_continuous_rank"
# Phase L gate spirit: clear ranking signal.
RANK_IC_BAR = 0.05
TOPK_FRAC = 0.20
TOPK_LIFT_BAR = 0.05
# Ridge mfe10_comb reference from Phase L (honest continuous arm).
PHASE_L_RIDGE_MFE_COMB_RANK_IC = 0.2734396296793431
PHASE_L_RIDGE_MFE_COMB_TOPK_LIFT = 0.17436270462689324
PHASE_M_MSE_RANK_IC = 0.08525129172480793
PHASE_M_MSE_TOPK_LIFT = 0.05686906618296045
LOSS_MODES = ("mse", "pairwise", "listwise")
DEFAULT_LOSS_MODE_PHASE_N = "pairwise"
PAIRWISE_MIN_GAP = 0.005
LISTWISE_TEMPERATURE = 0.05


def cs_rank(dates: np.ndarray | pd.Series, y: np.ndarray) -> np.ndarray:
    """Within-day percentile ranks of y (1.0 = highest)."""
    meta = pd.DataFrame({"asof_date": np.asarray(dates), "_y": np.asarray(y, dtype=np.float64)})
    return meta.groupby("asof_date")["_y"].rank(pct=True).to_numpy(dtype=np.float64)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 5:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    ra = pd.Series(aa).rank().to_numpy(dtype=np.float64)
    rb = pd.Series(bb).rank().to_numpy(dtype=np.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def daily_rank_ic(
    dates: np.ndarray | pd.Series,
    score: np.ndarray,
    label: np.ndarray,
    *,
    min_names: int = 30,
) -> dict[str, Any]:
    dates_arr = np.asarray(dates)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    ics: list[float] = []
    for day in pd.unique(dates_arr):
        idx = np.where(dates_arr == day)[0]
        if len(idx) < min_names:
            continue
        r = safe_spearman(score[idx], label[idx])
        if np.isfinite(r):
            ics.append(float(r))
    arr = np.asarray(ics, dtype=np.float64)
    return {
        "n_days": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "std": float(np.std(arr)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
    }


def topk_hit_rate(
    dates: np.ndarray | pd.Series,
    score: np.ndarray,
    label: np.ndarray,
    *,
    frac: float = TOPK_FRAC,
    min_names: int = 30,
) -> dict[str, Any]:
    dates_arr = np.asarray(dates)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    hits: list[float] = []
    for day in pd.unique(dates_arr):
        idx = np.where(dates_arr == day)[0]
        n = len(idx)
        if n < min_names:
            continue
        k = max(1, int(round(frac * n)))
        score_top = set(idx[np.argpartition(-score[idx], k - 1)[:k]].tolist())
        label_top = set(idx[np.argpartition(-label[idx], k - 1)[:k]].tolist())
        hits.append(len(score_top & label_top) / k)
    arr = np.asarray(hits, dtype=np.float64)
    chance = float(frac)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    return {
        "n_days": int(len(arr)),
        "frac": float(frac),
        "mean_hit": mean,
        "chance": chance,
        "lift_vs_chance": float(mean - chance) if np.isfinite(mean) else float("nan"),
    }


def clears_rank_gate(
    rank_ic_mean: float,
    topk_lift: float,
    *,
    rank_ic_bar: float = RANK_IC_BAR,
    topk_lift_bar: float = TOPK_LIFT_BAR,
) -> bool:
    ic_ok = bool(np.isfinite(rank_ic_mean) and rank_ic_mean >= rank_ic_bar)
    topk_ok = bool(np.isfinite(topk_lift) and topk_lift >= topk_lift_bar)
    return ic_ok or topk_ok


def ranking_eval_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray | pd.Series,
    *,
    topk_frac: float = TOPK_FRAC,
) -> dict[str, Any]:
    """Single ranking eval block with Rank IC / TopK always present."""
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(labels, dtype=np.float64).reshape(-1)
    rank_ic = daily_rank_ic(dates, score, label)
    topk = topk_hit_rate(dates, score, label, frac=topk_frac)
    mse = float(np.mean(np.square(score - label))) if score.size else float("nan")
    pooled = safe_spearman(score, label)
    gate = clears_rank_gate(rank_ic["mean"], topk["lift_vs_chance"])
    return {
        "samples": int(score.size),
        "mse": mse,
        "pooled_spearman": pooled,
        "rank_ic": rank_ic,
        "topk": topk,
        "rank_ic_bar": RANK_IC_BAR,
        "topk_lift_bar": TOPK_LIFT_BAR,
        "gate_passed": gate,
        "target": TARGET_NAME,
        "mfe10_def": MFE10_DEF,
        "phase_l_ridge_mfe_comb_rank_ic": PHASE_L_RIDGE_MFE_COMB_RANK_IC,
        "phase_l_ridge_mfe_comb_topk_lift": PHASE_L_RIDGE_MFE_COMB_TOPK_LIFT,
    }


def target_contract() -> dict[str, Any]:
    return {
        "target": TARGET_NAME,
        "mfe10_def": MFE10_DEF,
        "not_binary_mfe10": True,
        "not_multi_head_r2": True,
        "not_22_layer_binary": True,
        "preferred_backbone": "identity_or_freeze_rank_head",
        "preferred_loss_modes": list(LOSS_MODES),
        "default_loss_mode_phase_n": DEFAULT_LOSS_MODE_PHASE_N,
        "metrics": ["rank_ic_mean", "topk_lift"],
        "rank_ic_bar": RANK_IC_BAR,
        "topk_lift_bar": TOPK_LIFT_BAR,
        "phase_m_mse_rank_ic": PHASE_M_MSE_RANK_IC,
    }
