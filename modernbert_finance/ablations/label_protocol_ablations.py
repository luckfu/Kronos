"""Phase E: alternative label protocols vs constant/mean baselines (CPU).

Protocols (built from val panel + existing mfe/mae targets):
  A) Forward return (close[T+H]/close[T]-1) — continuous + sign logistic
  B) Vol-residualized / vol-scaled MFE/MAE — strip vol-driven magnitude
  C) Cleaner first-touch — same-day both→neither; decisive subsample; 8% FT

Compares effect sizes to Phase B/D; no Kaggle train / no R2 restart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernbert_finance.ablations._panel_io import load_panel
from modernbert_finance.ablations.continuous_xsection_ablations import (
    _eval_regression,
    _fit_eval_logistic,
    _first_touch_day,
    build_enriched_matrix,
)
from modernbert_finance.build_dataset import (
    FIRST_TOUCH_DOWNSIDE_THRESHOLD,
    FIRST_TOUCH_UPSIDE_THRESHOLD,
    HORIZON,
    LOOKBACK,
    prepare_frame,
)
from modernbert_finance.build_targets import WINDOW

EPS = 1e-4
SEED = 20261001
SQRT_H = float(np.sqrt(HORIZON))


def _first_touch_class_clean(
    future_high: np.ndarray,
    future_low: np.ndarray,
    close: float,
    *,
    up_th: float,
    dn_th: float,
) -> int:
    """0=upside_first, 1=downside_first, 2=neither.

    Same-day both hits → neither (cleaner than downside-wins).
    """
    for high, low in zip(future_high, future_low):
        upside = high / close - 1.0 >= up_th
        downside = 1.0 - low / close >= dn_th
        if upside and downside:
            return 2
        if upside:
            return 0
        if downside:
            return 1
    return 2


def build_protocol_labels(
    panel: dict[str, pd.DataFrame] | Path,
    targets: pd.DataFrame,
    *,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
) -> dict[str, Any]:
    """Reuse Phase D feature pack; add forward-return + cleaner FT labels."""
    packed = build_enriched_matrix(
        panel, targets, signal_start=signal_start, signal_end=signal_end
    )
    if isinstance(panel, (str, Path)):
        panel = load_panel(Path(panel))

    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None

    fwd5: list[float] = []
    fwd10: list[float] = []
    ft_clean_5: list[int] = []
    ft_clean_8: list[int] = []
    ft_up_day_8: list[float] = []
    ft_dn_day_8: list[float] = []

    for symbol in sorted(panel):
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        dates = frame.index
        close = frame["close"].to_numpy(dtype=np.float64)
        high = frame["high"].to_numpy(dtype=np.float64)
        low = frame["low"].to_numpy(dtype=np.float64)
        max_start = len(frame) - WINDOW + 1
        for start in range(max_start):
            asof_pos = start + LOOKBACK - 1
            asof = pd.Timestamp(dates[asof_pos]).date()
            if start_date and asof < start_date:
                continue
            if end_date and asof > end_date:
                continue
            cur = float(close[asof_pos])
            # Forward closes: day 5 and day 10 within horizon (1-indexed).
            c5 = float(close[asof_pos + 5]) if asof_pos + 5 < len(close) else cur
            c10 = float(close[asof_pos + HORIZON])
            fwd5.append(c5 / max(cur, 1e-8) - 1.0)
            fwd10.append(c10 / max(cur, 1e-8) - 1.0)

            fut_h = high[asof_pos + 1 : asof_pos + 1 + HORIZON]
            fut_l = low[asof_pos + 1 : asof_pos + 1 + HORIZON]
            ft_clean_5.append(
                _first_touch_class_clean(
                    fut_h,
                    fut_l,
                    cur,
                    up_th=FIRST_TOUCH_UPSIDE_THRESHOLD,
                    dn_th=FIRST_TOUCH_DOWNSIDE_THRESHOLD,
                )
            )
            ft_clean_8.append(
                _first_touch_class_clean(fut_h, fut_l, cur, up_th=0.08, dn_th=0.08)
            )
            ft_up_day_8.append(
                _first_touch_day(fut_h, fut_l, cur, upside=True, threshold=0.08)
            )
            ft_dn_day_8.append(
                _first_touch_day(fut_h, fut_l, cur, upside=False, threshold=0.08)
            )

    n = packed["n_samples"]
    if len(fwd10) != n:
        raise ValueError(f"fwd labels {len(fwd10)} != features {n}")

    meta = packed["meta"].copy()
    vol20 = meta["vol20"].to_numpy(dtype=np.float64)
    mfe = packed["targets"]["mfe10"].astype(np.float64)
    mae = packed["targets"]["mae10"].astype(np.float64)
    mae_abs = (-mae).astype(np.float64)
    net = (mfe + mae).astype(np.float64)

    fwd5_a = np.asarray(fwd5, dtype=np.float64)
    fwd10_a = np.asarray(fwd10, dtype=np.float64)
    # Cross-section rank of forward return (label only; not used as feature).
    meta["_fwd10"] = fwd10_a
    fwd10_cs = meta.groupby("asof_date")["_fwd10"].rank(pct=True).to_numpy(dtype=np.float64)
    meta.drop(columns=["_fwd10"], inplace=True)

    ft5 = np.asarray(ft_clean_5, dtype=np.int64)
    ft8 = np.asarray(ft_clean_8, dtype=np.int64)
    # Signed speed: +(H+1-day) if up, -(H+1-day) if dn, else 0
    up_d8 = np.asarray(ft_up_day_8, dtype=np.float64)
    dn_d8 = np.asarray(ft_dn_day_8, dtype=np.float64)
    ft_signed_8 = np.zeros(n, dtype=np.float64)
    ft_signed_8[ft8 == 0] = (HORIZON + 1 - up_d8[ft8 == 0]).astype(np.float64)
    ft_signed_8[ft8 == 1] = (-(HORIZON + 1 - dn_d8[ft8 == 1])).astype(np.float64)

    # Vol-scaled excursions (closed-form; no fold fit needed)
    vol_scale = np.maximum(vol20 * SQRT_H, 1e-8)
    mfe_vs = mfe / vol_scale
    mae_vs = mae_abs / vol_scale
    net_vs = net / vol_scale

    labels: dict[str, Any] = {
        # Protocol A — forward return
        "fwd_ret_5": fwd5_a,
        "fwd_ret_10": fwd10_a,
        "fwd_sign_up": (fwd10_a > 0.0).astype(np.int64),
        "fwd_cs_top20": (fwd10_cs >= 0.80).astype(np.int64),
        "fwd_cs_bot20": (fwd10_cs <= 0.20).astype(np.int64),
        # Protocol B — vol scaled (residuals built per-fold in eval)
        "mfe10": mfe,
        "mae10_abs": mae_abs,
        "net_excursion": net,
        "mfe_vol_scaled": mfe_vs,
        "mae_vol_scaled": mae_vs,
        "net_vol_scaled": net_vs,
        "vol20": vol20,
        # Protocol C — cleaner FT
        "ft_clean5_up": (ft5 == 0).astype(np.int64),
        "ft_clean5_dn": (ft5 == 1).astype(np.int64),
        "ft_clean5_decisive": (ft5 != 2).astype(np.int64),
        "ft_clean5_class": ft5,
        "ft_clean8_up": (ft8 == 0).astype(np.int64),
        "ft_clean8_dn": (ft8 == 1).astype(np.int64),
        "ft_clean8_decisive": (ft8 != 2).astype(np.int64),
        "ft_clean8_class": ft8,
        "ft_signed_8": ft_signed_8,
        "ft_up_day_8": up_d8,
        "ft_dn_day_8": dn_d8,
    }

    return {
        **packed,
        "protocol_labels": labels,
        "protocol_notes": {
            "A_forward_return": (
                "fwd_ret_H = close[T+H]/close[T]-1; sign and CS top/bot quintile binaries"
            ),
            "B_vol_residualized": (
                "mfe/mae / (vol20*sqrt(10)); plus train-fold OLS residual vs vol20"
            ),
            "C_cleaner_first_touch": (
                "same-day both→neither; 5% and 8% thresholds; decisive-only direction"
            ),
        },
    }


def _eval_regression_residualized(
    x: np.ndarray,
    y: np.ndarray,
    resid_covar: np.ndarray,
    *,
    seed: int,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Ridge on y residualized against resid_covar (fit on train only)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(resid_covar, dtype=np.float64).reshape(-1, 1)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1) & np.isfinite(z[:, 0])
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if len(y) < 50:
        return {"skipped": True, "reason": "too_few_rows"}
    x_tr, x_te, y_tr, y_te, z_tr, z_te = train_test_split(
        x, y, z, test_size=0.25, random_state=seed
    )
    lr = LinearRegression()
    lr.fit(z_tr, y_tr)
    y_tr_r = y_tr - lr.predict(z_tr)
    y_te_r = y_te - lr.predict(z_te)
    naive_pred = np.full_like(y_te_r, float(np.mean(y_tr_r)))
    naive_mae = float(mean_absolute_error(y_te_r, naive_pred))
    naive_r2 = float(r2_score(y_te_r, naive_pred))
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=seed))
    ridge.fit(x_tr, y_tr_r)
    pred = ridge.predict(x_te)
    mae = float(mean_absolute_error(y_te_r, pred))
    r2 = float(r2_score(y_te_r, pred))
    return {
        "skipped": False,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "naive_mae": naive_mae,
        "naive_r2": naive_r2,
        "ridge_mae": mae,
        "ridge_r2": r2,
        "mae_lift_vs_naive": naive_mae - mae,
        "r2_lift_vs_naive": r2 - naive_r2,
        "beats_naive_mae": bool(mae < naive_mae - 1e-8),
        "y_mean": float(np.mean(y_tr_r)),
        "y_std": float(np.std(y_tr_r)),
        "resid_coef_vol": float(lr.coef_[0]),
        "resid_intercept": float(lr.intercept_),
    }


def _fit_eval_logistic_mask(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Logistic on a boolean row mask (e.g. decisive first-touch only)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 50:
        return {"skipped": True, "reason": "too_few_masked_rows", "n_masked": int(mask.sum())}
    return _fit_eval_logistic(x[mask], y[mask], seed=seed)


def run_phase_e(
    *,
    val_panel: Path,
    val_targets: Path,
    phase_b_json: Path | None = None,
    phase_d_json: Path | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    targets = pd.read_parquet(val_targets)
    packed = build_protocol_labels(val_panel, targets)
    base = packed["base_features"]
    comb = packed["combined_features"]
    lab = packed["protocol_labels"]

    # --- Protocol A: forward return ---
    protocol_a_reg = {
        "fwd_ret_5_base": _eval_regression(base, lab["fwd_ret_5"], seed=seed),
        "fwd_ret_5_comb": _eval_regression(comb, lab["fwd_ret_5"], seed=seed),
        "fwd_ret_10_base": _eval_regression(base, lab["fwd_ret_10"], seed=seed),
        "fwd_ret_10_comb": _eval_regression(comb, lab["fwd_ret_10"], seed=seed),
    }
    protocol_a_clf = {
        "fwd_sign_up_base": _fit_eval_logistic(base, lab["fwd_sign_up"], seed=seed),
        "fwd_sign_up_comb": _fit_eval_logistic(comb, lab["fwd_sign_up"], seed=seed),
        "fwd_cs_top20_base": _fit_eval_logistic(base, lab["fwd_cs_top20"], seed=seed),
        "fwd_cs_top20_comb": _fit_eval_logistic(comb, lab["fwd_cs_top20"], seed=seed),
        "fwd_cs_bot20_base": _fit_eval_logistic(base, lab["fwd_cs_bot20"], seed=seed),
        "fwd_cs_bot20_comb": _fit_eval_logistic(comb, lab["fwd_cs_bot20"], seed=seed),
    }

    # --- Protocol B: vol-scaled + vol-residualized ---
    protocol_b_scaled = {
        "mfe_vol_scaled_base": _eval_regression(base, lab["mfe_vol_scaled"], seed=seed),
        "mfe_vol_scaled_comb": _eval_regression(comb, lab["mfe_vol_scaled"], seed=seed),
        "mae_vol_scaled_base": _eval_regression(base, lab["mae_vol_scaled"], seed=seed),
        "mae_vol_scaled_comb": _eval_regression(comb, lab["mae_vol_scaled"], seed=seed),
        "net_vol_scaled_base": _eval_regression(base, lab["net_vol_scaled"], seed=seed),
        "net_vol_scaled_comb": _eval_regression(comb, lab["net_vol_scaled"], seed=seed),
    }
    protocol_b_resid = {
        "mfe_resid_vol_base": _eval_regression_residualized(
            base, lab["mfe10"], lab["vol20"], seed=seed
        ),
        "mfe_resid_vol_comb": _eval_regression_residualized(
            comb, lab["mfe10"], lab["vol20"], seed=seed
        ),
        "mae_resid_vol_base": _eval_regression_residualized(
            base, lab["mae10_abs"], lab["vol20"], seed=seed
        ),
        "mae_resid_vol_comb": _eval_regression_residualized(
            comb, lab["mae10_abs"], lab["vol20"], seed=seed
        ),
        "net_resid_vol_base": _eval_regression_residualized(
            base, lab["net_excursion"], lab["vol20"], seed=seed
        ),
        "net_resid_vol_comb": _eval_regression_residualized(
            comb, lab["net_excursion"], lab["vol20"], seed=seed
        ),
    }
    # Binary: vol-scaled mfe above train median → use logistic on full sample
    # with label = (mfe_vol_scaled > 0) as a cheap direction-ish proxy,
    # plus high-scaled exceedance.
    # Absolute vol-scaled thresholds (no full-sample median → no test-label leak).
    protocol_b_clf = {
        "mfe_vs_gt1_base": _fit_eval_logistic(
            base, (lab["mfe_vol_scaled"] > 1.0).astype(np.int64), seed=seed
        ),
        "mfe_vs_gt1_comb": _fit_eval_logistic(
            comb, (lab["mfe_vol_scaled"] > 1.0).astype(np.int64), seed=seed
        ),
        "net_vs_pos_base": _fit_eval_logistic(
            base, (lab["net_vol_scaled"] > 0).astype(np.int64), seed=seed
        ),
        "net_vs_pos_comb": _fit_eval_logistic(
            comb, (lab["net_vol_scaled"] > 0).astype(np.int64), seed=seed
        ),
    }

    # --- Protocol C: cleaner first-touch ---
    decisive5 = lab["ft_clean5_decisive"].astype(bool)
    decisive8 = lab["ft_clean8_decisive"].astype(bool)
    # On decisive subsample, predict up-first (vs down-first).
    protocol_c_clf = {
        "ft_clean5_up_all_base": _fit_eval_logistic(base, lab["ft_clean5_up"], seed=seed),
        "ft_clean5_up_all_comb": _fit_eval_logistic(comb, lab["ft_clean5_up"], seed=seed),
        "ft_clean5_up_decisive_base": _fit_eval_logistic_mask(
            base, lab["ft_clean5_up"], decisive5, seed=seed
        ),
        "ft_clean5_up_decisive_comb": _fit_eval_logistic_mask(
            comb, lab["ft_clean5_up"], decisive5, seed=seed
        ),
        "ft_clean8_up_all_base": _fit_eval_logistic(base, lab["ft_clean8_up"], seed=seed),
        "ft_clean8_up_all_comb": _fit_eval_logistic(comb, lab["ft_clean8_up"], seed=seed),
        "ft_clean8_up_decisive_base": _fit_eval_logistic_mask(
            base, lab["ft_clean8_up"], decisive8, seed=seed
        ),
        "ft_clean8_up_decisive_comb": _fit_eval_logistic_mask(
            comb, lab["ft_clean8_up"], decisive8, seed=seed
        ),
    }
    protocol_c_reg = {
        "ft_signed_8_base": _eval_regression(base, lab["ft_signed_8"], seed=seed),
        "ft_signed_8_comb": _eval_regression(comb, lab["ft_signed_8"], seed=seed),
        "ft_up_day_8_base": _eval_regression(base, lab["ft_up_day_8"], seed=seed),
        "ft_up_day_8_comb": _eval_regression(comb, lab["ft_up_day_8"], seed=seed),
        "ft_dn_day_8_base": _eval_regression(base, lab["ft_dn_day_8"], seed=seed),
        "ft_dn_day_8_comb": _eval_regression(comb, lab["ft_dn_day_8"], seed=seed),
    }

    # Prevalence stats for FT clean
    ft_stats = {
        "ft_clean5_up_rate": float(np.mean(lab["ft_clean5_up"])),
        "ft_clean5_dn_rate": float(np.mean(lab["ft_clean5_dn"])),
        "ft_clean5_neither_rate": float(1.0 - np.mean(lab["ft_clean5_decisive"])),
        "ft_clean5_decisive_n": int(decisive5.sum()),
        "ft_clean8_up_rate": float(np.mean(lab["ft_clean8_up"])),
        "ft_clean8_dn_rate": float(np.mean(lab["ft_clean8_dn"])),
        "ft_clean8_neither_rate": float(1.0 - np.mean(lab["ft_clean8_decisive"])),
        "ft_clean8_decisive_n": int(decisive8.sum()),
        "fwd_sign_up_rate": float(np.mean(lab["fwd_sign_up"])),
        "fwd_ret_10_mean": float(np.mean(lab["fwd_ret_10"])),
        "fwd_ret_10_std": float(np.std(lab["fwd_ret_10"])),
    }

    # Load Phase B/D baselines for comparison
    phase_b_macro = None
    if phase_b_json and Path(phase_b_json).is_file():
        pb = json.loads(Path(phase_b_json).read_text(encoding="utf-8"))
        phase_b_macro = (
            pb.get("structure", {})
            .get("dilution_summary", {})
            .get("independent_8head_macro_delta")
        )
    phase_d = {}
    if phase_d_json and Path(phase_d_json).is_file():
        phase_d = json.loads(Path(phase_d_json).read_text(encoding="utf-8"))
    phase_d_dir_r2 = (
        phase_d.get("decision", {}).get("best_directional_ridge_r2") if phase_d else None
    )
    phase_d_macro = (
        phase_d.get("decision", {}).get("macro_delta_with_xsection") if phase_d else None
    )
    phase_d_ft_up_delta = None
    if phase_d:
        ft = phase_d.get("first_touch_logistic", {}).get("ft_upside_first_comb", {})
        if ft and not ft.get("skipped"):
            phase_d_ft_up_delta = ft.get("delta_model_minus_prior")

    def _best_r2(block: dict[str, Any], keys: list[str] | None = None) -> float:
        vals = []
        for k, v in block.items():
            if keys is not None and k not in keys:
                continue
            if not v.get("skipped") and "ridge_r2" in v:
                vals.append(v["ridge_r2"])
        return float(max(vals)) if vals else float("nan")

    def _best_delta(block: dict[str, Any], keys: list[str] | None = None) -> float:
        vals = []
        for k, v in block.items():
            if keys is not None and k not in keys:
                continue
            if not v.get("skipped") and "delta_model_minus_prior" in v:
                vals.append(v["delta_model_minus_prior"])
        return float(min(vals)) if vals else float("nan")  # more negative = better

    a_best_r2 = _best_r2(
        protocol_a_reg, ["fwd_ret_5_comb", "fwd_ret_10_comb"]
    )
    a_best_delta = _best_delta(
        protocol_a_clf,
        ["fwd_sign_up_comb", "fwd_cs_top20_comb", "fwd_cs_bot20_comb"],
    )
    b_best_r2 = max(
        _best_r2(
            protocol_b_scaled,
            ["mfe_vol_scaled_comb", "mae_vol_scaled_comb", "net_vol_scaled_comb"],
        ),
        _best_r2(
            protocol_b_resid,
            ["mfe_resid_vol_comb", "mae_resid_vol_comb", "net_resid_vol_comb"],
        ),
    )
    b_best_delta = _best_delta(
        protocol_b_clf, ["mfe_vs_gt1_comb", "net_vs_pos_comb"]
    )
    c_best_r2 = _best_r2(
        protocol_c_reg, ["ft_signed_8_comb", "ft_up_day_8_comb", "ft_dn_day_8_comb"]
    )
    c_best_delta = _best_delta(
        protocol_c_clf,
        [
            "ft_clean5_up_decisive_comb",
            "ft_clean8_up_decisive_comb",
            "ft_clean5_up_all_comb",
            "ft_clean8_up_all_comb",
        ],
    )
    # Directional focus for C: signed FT R² (days are magnitude-ish)
    c_dir_r2 = protocol_c_reg["ft_signed_8_comb"].get("ridge_r2", float("nan"))
    c_decisive_delta = protocol_c_clf["ft_clean5_up_decisive_comb"].get(
        "delta_model_minus_prior", float("nan")
    )
    c_decisive8_delta = protocol_c_clf["ft_clean8_up_decisive_comb"].get(
        "delta_model_minus_prior", float("nan")
    )

    # Lift thresholds vs Phase D (directional R²~0.11, FT logistic Δ~-0.009, macro~-0.03)
    # Justify next step if a protocol clearly beats Phase D direction OR hits absolute bars.
    DIR_R2_BAR = 0.15
    LOGIT_DELTA_BAR = -0.04
    LIFT_R2_VS_D = 0.04  # absolute R² points above Phase D directional
    LIFT_DELTA_VS_D_FT = 0.015  # more negative than Phase D FT Δ by this much

    def _lifts(r2: float, delta: float, *, is_directional_r2: bool = True) -> dict[str, Any]:
        beats_abs_r2 = isinstance(r2, float) and r2 == r2 and r2 >= DIR_R2_BAR
        beats_abs_delta = isinstance(delta, float) and delta == delta and delta <= LOGIT_DELTA_BAR
        beats_d_r2 = (
            is_directional_r2
            and phase_d_dir_r2 is not None
            and isinstance(r2, float)
            and r2 == r2
            and r2 >= float(phase_d_dir_r2) + LIFT_R2_VS_D
        )
        beats_d_ft = (
            phase_d_ft_up_delta is not None
            and isinstance(delta, float)
            and delta == delta
            and delta <= float(phase_d_ft_up_delta) - LIFT_DELTA_VS_D_FT
        )
        enough = bool(beats_abs_r2 or beats_abs_delta or beats_d_r2 or beats_d_ft)
        return {
            "beats_abs_r2_bar": beats_abs_r2,
            "beats_abs_delta_bar": beats_abs_delta,
            "beats_phase_d_r2_by_lift": beats_d_r2,
            "beats_phase_d_ft_delta_by_lift": beats_d_ft,
            "lifts_enough": enough,
        }

    a_lift = _lifts(a_best_r2, a_best_delta)
    # For B, vol-scaled R² may still be magnitude; use net_vol / resid as more directional
    b_dir_r2 = max(
        protocol_b_scaled["net_vol_scaled_comb"].get("ridge_r2", -999),
        protocol_b_resid["net_resid_vol_comb"].get("ridge_r2", -999),
        protocol_b_resid["mfe_resid_vol_comb"].get("ridge_r2", -999),
    )
    b_lift = _lifts(float(b_dir_r2), b_best_delta)
    c_lift = _lifts(float(c_dir_r2), float(c_decisive_delta))

    winners = []
    if a_lift["lifts_enough"]:
        winners.append("A_forward_return")
    if b_lift["lifts_enough"]:
        winners.append("B_vol_residualized")
    if c_lift["lifts_enough"]:
        winners.append("C_cleaner_first_touch")

    if not winners:
        verdict = "none_lift_enough__do_not_justify_relabel_train_yet"
        recommend_next = (
            "No alternative protocol beats Phase D directional bars enough. "
            "Do not start a new Kaggle/label-rebuild train on these defs yet; "
            "either stop Kairos cheap path or invent a qualitatively different label "
            "(e.g. longer horizon, industry-neutral fwd return with cost model)."
        )
        best_protocol = None
    else:
        # Pick the strongest by |logistic delta| then R²
        scored = []
        for name, r2, delta in (
            ("A_forward_return", a_best_r2, a_best_delta),
            ("B_vol_residualized", float(b_dir_r2), b_best_delta),
            ("C_cleaner_first_touch", float(c_dir_r2), float(c_decisive_delta)),
        ):
            if name in winners:
                scored.append((name, delta, r2))
        scored.sort(key=lambda t: (t[1], -t[2]))  # most negative delta first
        best_protocol = scored[0][0]
        verdict = f"{best_protocol}__lifts_enough_for_next_step"
        recommend_next = (
            f"Adopt {best_protocol} as the next label definition for a short "
            "rebuild+linear confirm; still no full R2 8-head restart."
        )

    decision = {
        "best_protocol": best_protocol,
        "winners": winners,
        "verdict": verdict,
        "recommend_next": recommend_next,
        "thresholds": {
            "dir_r2_bar": DIR_R2_BAR,
            "logit_delta_bar": LOGIT_DELTA_BAR,
            "lift_r2_vs_phase_d": LIFT_R2_VS_D,
            "lift_delta_vs_phase_d_ft": LIFT_DELTA_VS_D_FT,
        },
        "phase_b_macro_delta": phase_b_macro,
        "phase_d_best_directional_r2": phase_d_dir_r2,
        "phase_d_macro_delta_xsection": phase_d_macro,
        "phase_d_ft_upside_first_comb_delta": phase_d_ft_up_delta,
        "protocol_A": {
            "best_ridge_r2": a_best_r2,
            "best_logit_delta": a_best_delta,
            **a_lift,
        },
        "protocol_B": {
            "best_directional_ridge_r2": float(b_dir_r2),
            "best_any_ridge_r2": b_best_r2,
            "best_logit_delta": b_best_delta,
            **b_lift,
        },
        "protocol_C": {
            "best_signed_ridge_r2": float(c_dir_r2),
            "best_any_ridge_r2": c_best_r2,
            "best_logit_delta": c_best_delta,
            "decisive5_delta": float(c_decisive_delta)
            if c_decisive_delta == c_decisive_delta
            else None,
            "decisive8_delta": float(c_decisive8_delta)
            if c_decisive8_delta == c_decisive8_delta
            else None,
            **c_lift,
        },
        "rationale": (
            "Phase E asks whether forward-return, vol-residualized MFE/MAE, or cleaner "
            "first-touch labels raise linear effect size enough vs Phase B/D to justify "
            "rebuilding training labels. Absolute bars: directional R²≥0.15 or logistic "
            "Δ≤-0.04; relative: +0.04 R² vs Phase D directional or FT Δ improved by 0.015."
        ),
    }

    return {
        "phase": "E_label_protocols",
        "n_samples": packed["n_samples"],
        "base_dim": packed["base_dim"],
        "xsec_dim": packed["xsec_dim"],
        "xsec_cols": packed["xsec_cols"],
        "seed": seed,
        "protocol_notes": packed["protocol_notes"],
        "label_prevalence": ft_stats,
        "protocol_A_forward_return": {
            "regression": protocol_a_reg,
            "logistic": protocol_a_clf,
        },
        "protocol_B_vol_residualized": {
            "vol_scaled_regression": protocol_b_scaled,
            "vol_resid_regression": protocol_b_resid,
            "logistic": protocol_b_clf,
        },
        "protocol_C_cleaner_first_touch": {
            "regression": protocol_c_reg,
            "logistic": protocol_c_clf,
        },
        "decision": decision,
    }


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
    dec = payload["decision"]
    prev = payload["label_prevalence"]
    a = payload["protocol_A_forward_return"]
    b = payload["protocol_B_vol_residualized"]
    c = payload["protocol_C_cleaner_first_touch"]

    def _rrow(name: str, d: dict[str, Any]) -> str:
        if d.get("skipped"):
            return f"| {name} | — | — | — | skipped |"
        return (
            f"| {name} | {d['naive_mae']:.6f} | {d['ridge_mae']:.6f} | "
            f"{d['ridge_r2']:.6f} | {'Y' if d['beats_naive_mae'] else 'N'} |"
        )

    def _lrow(name: str, d: dict[str, Any]) -> str:
        if d.get("skipped"):
            return f"| {name} | — | — | skipped |"
        return (
            f"| {name} | {d['prior_log_loss']:.6f} | {d['model_log_loss']:.6f} | "
            f"{d['delta_model_minus_prior']:+.6f} |"
        )

    lines = [
        "# Kairos 替代标签协议消融（Phase E）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
    ]
    if dec["best_protocol"]:
        lines.append(
            f"**{dec['best_protocol']} 效应量足够进入下一步。** "
            f"verdict=`{dec['verdict']}`。"
        )
    else:
        lines.append(
            f"**三种替代标签均未相对 Phase D 抬升到可开训门槛。** "
            f"verdict=`{dec['verdict']}`。"
        )
    lines += [
        "",
        f"- 建议：{dec['recommend_next']}",
        "",
        "## 设定",
        "",
        f"- n_samples=`{payload['n_samples']}`，seed=`{payload['seed']}`，"
        f"base_dim=`{payload['base_dim']}`，xsec_dim=`{payload['xsec_dim']}`",
        "- 特征：Phase D 同款 Base OHLCVA 汇总 + 截面（默认看 comb）",
        "- 协议 A：前向收益 fwd_ret_5/10；符号 / 截面顶底 20% 二分类",
        "- 协议 B：MFE/MAE 对 vol20 标准化与 train-fold OLS 残差化",
        "- 协议 C：同日双边触达→neither；5%/8% first-touch；decisive 子样本方向",
        "- **未**重启 R2 / **未** Kaggle 长训；**未**动同事 TPU WIP",
        "",
        "## 对照基线（Phase B/D）",
        "",
        f"| 指标 | 值 |",
        f"| --- | ---: |",
        f"| Phase B 八头 macro Δ | {dec['phase_b_macro_delta']} |",
        f"| Phase D 截面八头 macro Δ | {dec['phase_d_macro_delta_xsection']} |",
        f"| Phase D 最好方向 Ridge R² | {dec['phase_d_best_directional_r2']} |",
        f"| Phase D FT upside_first comb Δ | {dec['phase_d_ft_upside_first_comb_delta']} |",
        "",
        f"门槛：R²≥`{dec['thresholds']['dir_r2_bar']}` 或 logistic Δ≤`{dec['thresholds']['logit_delta_bar']}`；"
        f"或相对 Phase D 方向 R²+`{dec['thresholds']['lift_r2_vs_phase_d']}` / FT Δ 再改善 "
        f"`{dec['thresholds']['lift_delta_vs_phase_d_ft']}`。",
        "",
        "## 标签分布",
        "",
        f"- fwd_sign_up_rate=`{prev['fwd_sign_up_rate']:.4f}`，"
        f"fwd_ret_10 mean/std=`{prev['fwd_ret_10_mean']:.4f}`/`{prev['fwd_ret_10_std']:.4f}`",
        f"- FT clean 5%: up=`{prev['ft_clean5_up_rate']:.4f}` dn=`{prev['ft_clean5_dn_rate']:.4f}` "
        f"neither=`{prev['ft_clean5_neither_rate']:.4f}` decisive_n=`{prev['ft_clean5_decisive_n']}`",
        f"- FT clean 8%: up=`{prev['ft_clean8_up_rate']:.4f}` dn=`{prev['ft_clean8_dn_rate']:.4f}` "
        f"neither=`{prev['ft_clean8_neither_rate']:.4f}` decisive_n=`{prev['ft_clean8_decisive_n']}`",
        "",
        "## 1) 协议 A — 前向收益",
        "",
        "| Target | naive MAE | ridge MAE | ridge R² | beats |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for k, v in a["regression"].items():
        lines.append(_rrow(k, v))
    lines += [
        "",
        "| Head | prior LL | model LL | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in a["logistic"].items():
        lines.append(_lrow(k, v))

    lines += [
        "",
        "## 2) 协议 B — 波动残差化 / 标准化 MFE·MAE",
        "",
        "### Vol-scaled",
        "",
        "| Target | naive MAE | ridge MAE | ridge R² | beats |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for k, v in b["vol_scaled_regression"].items():
        lines.append(_rrow(k, v))
    lines += [
        "",
        "### Train-fold residual vs vol20",
        "",
        "| Target | naive MAE | ridge MAE | ridge R² | beats |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for k, v in b["vol_resid_regression"].items():
        lines.append(_rrow(k, v))
    lines += [
        "",
        "| Head | prior LL | model LL | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in b["logistic"].items():
        lines.append(_lrow(k, v))

    lines += [
        "",
        "## 3) 协议 C — 更干净 first-touch",
        "",
        "| Target | naive MAE | ridge MAE | ridge R² | beats |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for k, v in c["regression"].items():
        lines.append(_rrow(k, v))
    lines += [
        "",
        "| Head | prior LL | model LL | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in c["logistic"].items():
        lines.append(_lrow(k, v))

    pa_, pb_, pc_ = dec["protocol_A"], dec["protocol_B"], dec["protocol_C"]
    lines += [
        "",
        "## 4) 与 Phase D 对比 + 决策",
        "",
        "| 协议 | 方向 R² | 最好 Δ | lifts_enough |",
        "| --- | ---: | ---: | :---: |",
        f"| A 前向收益 | {pa_['best_ridge_r2']:.6f} | {pa_['best_logit_delta']:+.6f} | "
        f"{'Y' if pa_['lifts_enough'] else 'N'} |",
        f"| B 波动残差 | {pb_['best_directional_ridge_r2']:.6f} | {pb_['best_logit_delta']:+.6f} | "
        f"{'Y' if pb_['lifts_enough'] else 'N'} |",
        f"| C 干净 FT | {pc_['best_signed_ridge_r2']:.6f} | {pc_['best_logit_delta']:+.6f} | "
        f"{'Y' if pc_['lifts_enough'] else 'N'} |",
        "",
        f"- best_protocol = `{dec['best_protocol']}`",
        f"- winners = `{dec['winners']}`",
        f"- verdict = `{dec['verdict']}`",
        f"- 判据：{dec['rationale']}",
        "",
        "### 建议下一步（每条一句）",
        "",
    ]
    if dec["best_protocol"]:
        lines += [
            f"1. **主推**：按 `{dec['best_protocol']}` 重建 sidecar 标签并做短线性确认。",
            "2. **不要**重启同配置 R2 八头 BCE 长训。",
            "3. 截面特征可继续默认带上（Phase D 已证 +Δ）。",
        ]
    else:
        lines += [
            "1. **不要**基于这三种定义立刻开标签重建长训。",
            "2. 若继续研究：换更长 horizon / 行业中性 fwd / 成本感知标签，而非微调 10D MFE 阈值。",
            "3. **不要**重启同配置 R2；短神经探针仍非主路径。",
        ]

    lines += [
        "",
        "## 证据强度",
        "",
        "| 主张 | 强度 | 依据 |",
        "| --- | --- | --- |",
        f"| A 前向收益可开下一步 | {'中' if pa_['lifts_enough'] else '弱/否'} | R²/Δ 见表 |",
        f"| B 残差化可开下一步 | {'中' if pb_['lifts_enough'] else '弱/否'} | R²/Δ 见表 |",
        f"| C 干净 FT 可开下一步 | {'中' if pc_['lifts_enough'] else '弱/否'} | R²/Δ 见表 |",
        "| 应维持停 R2 | 强 | 与 Phase C/D 一致 |",
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
        "--phase-d-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_d_continuous_xsection.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_e_label_protocols.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_e_label_protocols_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    payload = run_phase_e(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        phase_b_json=args.phase_b_json,
        phase_d_json=args.phase_d_json,
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
                "label_prevalence": payload["label_prevalence"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
