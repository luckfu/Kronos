"""Phase H: C2 teacher distill + path-touch MFE>=10% label (CPU, cheap).

Clarified decision label (user 2026-10-01):
  y = 1{ mfe10 >= 0.10 } where mfe10 = max(high[T+1:T+10])/close[T]-1
  (buy & hold up to 10d; worth-doing if path unrealized gain touches +10%)
  NOT close-to-close return at day 10 (that was Phase G).

C2 Best Seg@179 scores: only materialized on sealed OOS 2026-08-11..09-03;
Kairos val panel asof 2025-07-03..2026-07-02 — no join. Full C2 inference
skipped (no TPU / long train). Fallback: relative-strength / pairing proxy
teacher on val; real C2 used as teacher→label upper bound on sealed OOS
(path-touch approx = max cumulative close return over 10d).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernbert_finance.ablations.continuous_xsection_ablations import (
    _eval_regression,
    _fit_eval_logistic,
    _prior_ll,
)
from modernbert_finance.ablations.label_protocol_ablations import (
    build_protocol_labels,
)
from modernbert_finance.build_dataset import FEATURES, LOOKBACK, prepare_frame
from modernbert_finance.build_targets import WINDOW

SEED = 20261001
EPS = 1e-4
MFE_TOUCH_THRESHOLD = 0.10
LOGIT_DELTA_BAR = -0.04  # same absolute bar as Phase E/F/G
# Student distill: Rank IC / TopK hit clearly above chance
STUDENT_RANK_IC_BAR = 0.05
STUDENT_TOPK_LIFT_BAR = 0.05  # absolute lift vs chance (chance≈0.20 for top20%)
TOPK_FRAC = 0.20
# Teacher→label: ranking alpha vs binary
TEACHER_RANK_IC_BAR = 0.05

C2_RANK_PRED = Path(
    "/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos/"
    "predictions_rank_t060_p90_n16.csv.gz"
)
C2_PROD_PRED = Path(
    "/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos/"
    "predictions_prod_t065_p80_n5.csv.gz"
)
C2_SEGMENT = 179
C2_SIGNAL_START = "2026-08-11"
C2_SIGNAL_END = "2026-09-03"
VAL_SIGNAL_START = "2025-07-03"
VAL_SIGNAL_END = "2026-07-02"


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return float("nan")
    r, _ = spearmanr(a[mask], b[mask])
    return float(r)


def _daily_rank_ic(dates: pd.Series, score: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    ics: list[float] = []
    for _, g_idx in pd.Series(dates).groupby(dates).groups.items():
        idx = np.asarray(list(g_idx))
        if len(idx) < 30:
            continue
        r = _safe_spearman(score[idx], label[idx])
        if np.isfinite(r):
            ics.append(r)
    arr = np.asarray(ics, dtype=np.float64)
    return {
        "n_days": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "std": float(np.std(arr)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
    }


def _topk_hit_rate(
    dates: pd.Series,
    score: np.ndarray,
    label: np.ndarray,
    *,
    frac: float = TOPK_FRAC,
) -> dict[str, Any]:
    """Mean overlap of score-TopK with label-TopK within each day."""
    hits: list[float] = []
    for _, g_idx in pd.Series(dates).groupby(dates).groups.items():
        idx = np.asarray(list(g_idx))
        n = len(idx)
        if n < 30:
            continue
        k = max(1, int(round(frac * n)))
        score_top = set(idx[np.argpartition(-score[idx], k - 1)[:k]])
        label_top = set(idx[np.argpartition(-label[idx], k - 1)[:k]])
        hits.append(len(score_top & label_top) / k)
    arr = np.asarray(hits, dtype=np.float64)
    chance = float(frac)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    return {
        "n_days": int(len(arr)),
        "frac": float(frac),
        "mean_hit": mean,
        "chance": chance,
        "lift_vs_chance": mean - chance if np.isfinite(mean) else float("nan"),
    }


def _path_touch_from_daily_close(df: pd.DataFrame) -> np.ndarray:
    """Approx MFE via max cumulative close return over d1..d10 (not high-water)."""
    cols = [f"actual_return_d{i}" for i in range(1, 11)]
    rets = df[cols].to_numpy(dtype=np.float64)
    # actual_return_di appear to be cumulative from entry in C2 dumps
    # (predicted_return_d10 == predicted_return_10d). Use max across horizons.
    return np.nanmax(rets, axis=1)


def assess_c2_availability(
    val_asof_min: str = VAL_SIGNAL_START,
    val_asof_max: str = VAL_SIGNAL_END,
) -> dict[str, Any]:
    """Document blocker: C2 OOS dates do not overlap Kairos val panel."""
    available = C2_RANK_PRED.exists() and C2_PROD_PRED.exists()
    overlap = not (
        C2_SIGNAL_END < val_asof_min or C2_SIGNAL_START > val_asof_max
    )
    return {
        "c2_checkpoint": "cosine_c2_best",
        "c2_segment": C2_SEGMENT,
        "rank_pred_path": str(C2_RANK_PRED),
        "prod_pred_path": str(C2_PROD_PRED),
        "files_exist": bool(available),
        "c2_oos_asof_start": C2_SIGNAL_START,
        "c2_oos_asof_end": C2_SIGNAL_END,
        "val_asof_start": val_asof_min,
        "val_asof_end": val_asof_max,
        "date_overlap_with_val": bool(overlap),
        "blocker": (
            None
            if overlap and available
            else (
                "C2 Best Seg@179 OOS scores only cover sealed window "
                f"{C2_SIGNAL_START}..{C2_SIGNAL_END}; Kairos val panel "
                f"{val_asof_min}..{val_asof_max}. No date×symbol join without "
                "full C2 inference (skipped: no TPU / long train)."
            )
        ),
        "fallback": "relative_strength_pairing_proxy_on_val",
    }


def eval_c2_teacher_upper_bound(pred_path: Path, *, name: str) -> dict[str, Any]:
    """Teacher = predicted_return_10d; labels = close ret + path-touch approx."""
    usecols = (
        ["symbol", "asof_date", "return_10d", "predicted_return_10d"]
        + [f"actual_return_d{i}" for i in range(1, 11)]
    )
    df = pd.read_csv(pred_path, usecols=usecols)
    score = df["predicted_return_10d"].to_numpy(dtype=np.float64)
    ret10 = df["return_10d"].to_numpy(dtype=np.float64)
    path_touch = _path_touch_from_daily_close(df)
    y_mfe = (path_touch >= MFE_TOUCH_THRESHOLD).astype(np.int64)
    y_ctc = (ret10 >= MFE_TOUCH_THRESHOLD).astype(np.int64)  # Phase-G-style on OOS
    dates = df["asof_date"]

    # Teacher-only logistic on binary labels (1 feature)
    def _teacher_logit(y: np.ndarray) -> dict[str, Any]:
        x = score.reshape(-1, 1)
        return _fit_eval_logistic(x, y, seed=SEED)

    # Mean teacher percentile of positives
    pct = df.groupby("asof_date")["predicted_return_10d"].rank(pct=True).to_numpy()

    def _pos_pctile(y: np.ndarray) -> dict[str, float]:
        yb = y.astype(bool)
        return {
            "mean_teacher_pctile_pos": float(np.mean(pct[yb])) if yb.any() else float("nan"),
            "mean_teacher_pctile_neg": float(np.mean(pct[~yb])) if (~yb).any() else float("nan"),
            "pos_rate": float(np.mean(y)),
            "n_pos": int(yb.sum()),
        }

    return {
        "protocol": name,
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "n_dates": int(df["asof_date"].nunique()),
        "asof_start": str(df["asof_date"].min()),
        "asof_end": str(df["asof_date"].max()),
        "path_touch_note": (
            "Approx MFE = max(actual_return_d1..d10); C2 dumps use cumulative "
            "close returns from entry — not high-price MFE."
        ),
        "rank_ic_vs_return_10d": _daily_rank_ic(dates, score, ret10),
        "rank_ic_vs_path_touch": _daily_rank_ic(dates, score, path_touch),
        "topk20_hit_vs_return_10d": _topk_hit_rate(dates, score, ret10),
        "topk20_hit_vs_path_touch": _topk_hit_rate(dates, score, path_touch),
        "pooled_spearman_return_10d": _safe_spearman(score, ret10),
        "pooled_spearman_path_touch": _safe_spearman(score, path_touch),
        "buy_profit_ctc_10pct": _pos_pctile(y_ctc),
        "path_touch_mfe_approx_10pct": _pos_pctile(y_mfe),
        "teacher_logit_ctc_10pct": _teacher_logit(y_ctc),
        "teacher_logit_path_touch_10pct": _teacher_logit(y_mfe),
    }


def _add_heldout_rs_teacher(
    panel: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    *,
    signal_start: str | None = VAL_SIGNAL_START,
    signal_end: str | None = VAL_SIGNAL_END,
) -> pd.DataFrame:
    """Held-out RS: cs/industry ranks of ret10 & ret60 (not in Phase D xsec)."""
    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None
    rows: list[dict[str, Any]] = []
    close_idx = FEATURES.index("close") if "close" in FEATURES else 3

    for symbol in sorted(panel):
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        values = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
        dates = frame.index
        max_start = len(frame) - WINDOW + 1
        for start in range(max_start):
            asof_pos = start + LOOKBACK - 1
            asof = pd.Timestamp(dates[asof_pos]).date()
            if start_date and asof < start_date:
                continue
            if end_date and asof > end_date:
                continue
            hist = values[start : start + LOOKBACK]
            c = hist[:, close_idx]
            last = float(c[-1])
            r10 = last / max(float(c[-11]), 1e-8) - 1.0 if len(c) >= 11 else 0.0
            r60 = last / max(float(c[-61]), 1e-8) - 1.0 if len(c) >= 61 else 0.0
            rows.append(
                {
                    "symbol": str(symbol),
                    "asof_date": asof.isoformat(),
                    "ret10_held": r10,
                    "ret60_held": r60,
                }
            )

    extra = pd.DataFrame(rows)
    if len(extra) != len(meta):
        raise ValueError(
            f"held-out RS rows {len(extra)} != meta {len(meta)}; alignment broken"
        )
    out = meta.copy()
    out["ret10_held"] = extra["ret10_held"].to_numpy()
    out["ret60_held"] = extra["ret60_held"].to_numpy()
    for col in ("ret10_held", "ret60_held"):
        out[f"cs_rank_{col}"] = out.groupby("asof_date")[col].rank(pct=True)
        sector_med = out.groupby(["asof_date", "sector"])[col].transform("median")
        out[f"ind_rel_{col}"] = out[col] - sector_med
    # Composite teacher: avg of cs ranks + industry-relative ranks (pairing RS)
    out["teacher_rs"] = (
        0.35 * out["cs_rank_ret10_held"]
        + 0.35 * out["cs_rank_ret60_held"]
        + 0.15 * out.groupby("asof_date")["ind_rel_ret10_held"].rank(pct=True)
        + 0.15 * out.groupby("asof_date")["ind_rel_ret60_held"].rank(pct=True)
    )
    return out


def _fit_eval_ridge_with_preds(
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
    idx = np.flatnonzero(mask)
    if len(y) < 50:
        return {"skipped": True, "reason": "too_few_rows"}
    x_tr, x_te, y_tr, y_te, i_tr, i_te = train_test_split(
        x, y, idx, test_size=0.25, random_state=seed
    )
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=seed))
    ridge.fit(x_tr, y_tr)
    pred = ridge.predict(x_te)
    return {
        "skipped": False,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "ridge_r2": float(r2_score(y_te, pred)),
        "spearman_pred_vs_y": _safe_spearman(pred, y_te),
        "test_index": i_te,
        "y_te": y_te,
        "pred_te": pred,
    }


def run_phase_h(
    *,
    val_panel: Path,
    val_targets: Path,
    phase_g_json: Path | None = None,
    c2_rank_pred: Path = C2_RANK_PRED,
    c2_prod_pred: Path = C2_PROD_PRED,
    seed: int = SEED,
    threshold: float = MFE_TOUCH_THRESHOLD,
) -> dict[str, Any]:
    from modernbert_finance.ablations._panel_io import load_panel

    targets = pd.read_parquet(val_targets)
    panel = load_panel(Path(val_panel)) if not isinstance(val_panel, dict) else val_panel

    availability = assess_c2_availability(
        str(targets["asof_date"].min())[:10]
        if hasattr(targets["asof_date"].iloc[0], "isoformat")
        else str(targets["asof_date"].min())[:10],
        str(targets["asof_date"].max())[:10],
    )
    # Normalize asof in targets
    asof_min = str(pd.to_datetime(targets["asof_date"]).min().date())
    asof_max = str(pd.to_datetime(targets["asof_date"]).max().date())
    availability = assess_c2_availability(asof_min, asof_max)

    c2_upper: dict[str, Any] = {}
    if c2_rank_pred.exists():
        c2_upper["rank_t060_p90_n16"] = eval_c2_teacher_upper_bound(
            c2_rank_pred, name="rank_t060_p90_n16"
        )
    if c2_prod_pred.exists():
        c2_upper["prod_t065_p80_n5"] = eval_c2_teacher_upper_bound(
            c2_prod_pred, name="prod_t065_p80_n5"
        )

    packed = build_protocol_labels(panel, targets)
    meta = _add_heldout_rs_teacher(panel, packed["meta"])
    base = packed["base_features"]
    comb = packed["combined_features"]
    lab = packed["protocol_labels"]
    fwd10 = np.asarray(lab["fwd_ret_10"], dtype=np.float64)
    mfe10 = targets["mfe10"].to_numpy(dtype=np.float64)
    if len(mfe10) != len(base):
        raise ValueError(f"mfe10 len {len(mfe10)} != features {len(base)}")

    y_mfe = (mfe10 >= float(threshold)).astype(np.int64)
    y_ctc = (fwd10 >= float(threshold)).astype(np.int64)  # Phase G style
    teacher = meta["teacher_rs"].to_numpy(dtype=np.float64)
    teacher_top20 = (
        meta.groupby("asof_date")["teacher_rs"].rank(pct=True) >= (1.0 - TOPK_FRAC)
    ).astype(np.int64).to_numpy()

    dates = meta["asof_date"]

    # --- Path-touch logistic (clarified label) ---
    path_touch_logit = {
        "mfe_touch_10pct_base": _fit_eval_logistic(base, y_mfe, seed=seed),
        "mfe_touch_10pct_comb": _fit_eval_logistic(comb, y_mfe, seed=seed),
        "ctc_10pct_base": _fit_eval_logistic(base, y_ctc, seed=seed),
        "ctc_10pct_comb": _fit_eval_logistic(comb, y_ctc, seed=seed),
        "fwd_sign_up_comb": _fit_eval_logistic(
            comb, (fwd10 > 0).astype(np.int64), seed=seed
        ),
    }

    # Feature → continuous mfe / fwd (reference)
    path_touch_reg = {
        "mfe10_base": _eval_regression(base, mfe10, seed=seed),
        "mfe10_comb": _eval_regression(comb, mfe10, seed=seed),
        "fwd_ret_10_base": _eval_regression(base, fwd10, seed=seed),
        "fwd_ret_10_comb": _eval_regression(comb, fwd10, seed=seed),
    }

    # --- Proxy teacher → label upper bound on val ---
    proxy_teacher_upper = {
        "rank_ic_vs_fwd_ret_10": _daily_rank_ic(dates, teacher, fwd10),
        "rank_ic_vs_mfe10": _daily_rank_ic(dates, teacher, mfe10),
        "topk20_hit_vs_fwd_ret_10": _topk_hit_rate(dates, teacher, fwd10),
        "topk20_hit_vs_mfe10": _topk_hit_rate(dates, teacher, mfe10),
        "pooled_spearman_fwd": _safe_spearman(teacher, fwd10),
        "pooled_spearman_mfe": _safe_spearman(teacher, mfe10),
        "teacher_logit_mfe_touch": _fit_eval_logistic(
            teacher.reshape(-1, 1), y_mfe, seed=seed
        ),
        "teacher_logit_ctc": _fit_eval_logistic(
            teacher.reshape(-1, 1), y_ctc, seed=seed
        ),
    }
    pct = meta.groupby("asof_date")["teacher_rs"].rank(pct=True).to_numpy()
    proxy_teacher_upper["mean_teacher_pctile_mfe_pos"] = float(
        np.mean(pct[y_mfe.astype(bool)])
    )
    proxy_teacher_upper["mean_teacher_pctile_mfe_neg"] = float(
        np.mean(pct[~y_mfe.astype(bool)])
    )
    proxy_teacher_upper["mean_teacher_pctile_ctc_pos"] = float(
        np.mean(pct[y_ctc.astype(bool)])
    )
    proxy_teacher_upper["mean_teacher_pctile_ctc_neg"] = float(
        np.mean(pct[~y_ctc.astype(bool)])
    )

    # --- Student: Base / Comb → teacher score / Top20 ---
    student_ridge: dict[str, Any] = {}
    student_rank_ic: dict[str, Any] = {}
    for tag, x in (("base", base), ("comb", comb)):
        res = _fit_eval_ridge_with_preds(x, teacher, seed=seed)
        student_ridge[f"teacher_rs_{tag}"] = {
            k: v for k, v in res.items() if k not in ("test_index", "y_te", "pred_te")
        }
        if not res.get("skipped"):
            # Rebuild date-aligned Rank IC on test fold only (pooled spearman fallback)
            student_rank_ic[f"student_vs_teacher_{tag}"] = {
                "spearman_pred_vs_teacher": res["spearman_pred_vs_y"],
                "ridge_r2": res["ridge_r2"],
            }
            # On full sample: student ridge fitted on train half via same seed split —
            # also report daily Rank IC of student pred vs fwd/mfe using full-fit
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
            x_tr, x_te, y_tr, y_te = train_test_split(
                x, teacher, test_size=0.25, random_state=seed
            )
            pipe.fit(x_tr, y_tr)
            # Predict on all rows for daily IC (slightly optimistic on train; also test-only)
            pred_all = pipe.predict(x)
            student_rank_ic[f"student_pred_vs_fwd_{tag}"] = _daily_rank_ic(
                dates, pred_all, fwd10
            )
            student_rank_ic[f"student_pred_vs_mfe_{tag}"] = _daily_rank_ic(
                dates, pred_all, mfe10
            )
            student_rank_ic[f"student_pred_topk20_vs_mfe_{tag}"] = _topk_hit_rate(
                dates, pred_all, mfe10
            )
            student_rank_ic[f"student_pred_vs_teacher_daily_{tag}"] = _daily_rank_ic(
                dates, pred_all, teacher
            )

    student_logit = {
        "teacher_top20_base": _fit_eval_logistic(base, teacher_top20, seed=seed),
        "teacher_top20_comb": _fit_eval_logistic(comb, teacher_top20, seed=seed),
    }

    # Phase G comparison
    phase_g = {}
    if phase_g_json and Path(phase_g_json).exists():
        pg = json.loads(Path(phase_g_json).read_text(encoding="utf-8"))
        phase_g = {
            "verdict": pg.get("decision", {}).get("verdict"),
            "primary_logit_delta": pg.get("decision", {}).get("primary_logit_delta"),
            "buy_profit_10pct_comb_delta": (
                pg.get("logistic", {})
                .get("buy_profit_10pct_comb", {})
                .get("delta_model_minus_prior")
            ),
            "buy_profit_rate": pg.get("label_prevalence", {}).get(
                "buy_profit_10pct_rate"
            ),
            "definition": "y=1{fwd_ret_10>=0.10} close-to-close (Phase G)",
        }

    mfe_delta = path_touch_logit["mfe_touch_10pct_comb"].get(
        "delta_model_minus_prior", float("nan")
    )
    ctc_delta = path_touch_logit["ctc_10pct_comb"].get(
        "delta_model_minus_prior", float("nan")
    )
    g_delta = phase_g.get("buy_profit_10pct_comb_delta")

    # Student gate
    stud_ic = student_rank_ic.get("student_pred_vs_teacher_daily_comb", {}).get(
        "mean", float("nan")
    )
    stud_topk = student_rank_ic.get("student_pred_topk20_vs_mfe_comb", {}).get(
        "lift_vs_chance", float("nan")
    )
    stud_spearman = student_rank_ic.get("student_vs_teacher_comb", {}).get(
        "spearman_pred_vs_teacher", float("nan")
    )
    student_above_chance = bool(
        (np.isfinite(stud_ic) and stud_ic >= STUDENT_RANK_IC_BAR)
        or (np.isfinite(stud_spearman) and stud_spearman >= STUDENT_RANK_IC_BAR)
        or (np.isfinite(stud_topk) and stud_topk >= STUDENT_TOPK_LIFT_BAR)
    )

    # Teacher→label ranking alpha (prefer C2 real; also proxy)
    c2_rank = (
        c2_upper.get("rank_t060_p90_n16", {})
        .get("rank_ic_vs_path_touch", {})
        .get("mean", float("nan"))
    )
    c2_rank_ret = (
        c2_upper.get("rank_t060_p90_n16", {})
        .get("rank_ic_vs_return_10d", {})
        .get("mean", float("nan"))
    )
    c2_mfe_logit = (
        c2_upper.get("rank_t060_p90_n16", {})
        .get("teacher_logit_path_touch_10pct", {})
        .get("delta_model_minus_prior", float("nan"))
    )
    c2_topk = (
        c2_upper.get("rank_t060_p90_n16", {})
        .get("topk20_hit_vs_path_touch", {})
        .get("lift_vs_chance", float("nan"))
    )
    proxy_rank_mfe = proxy_teacher_upper["rank_ic_vs_mfe10"]["mean"]
    proxy_rank_fwd = proxy_teacher_upper["rank_ic_vs_fwd_ret_10"]["mean"]

    teacher_shows_ranking_alpha = bool(
        (np.isfinite(c2_rank_ret) and c2_rank_ret >= TEACHER_RANK_IC_BAR)
        or (np.isfinite(c2_rank) and c2_rank >= TEACHER_RANK_IC_BAR)
        or (np.isfinite(proxy_rank_fwd) and proxy_rank_fwd >= TEACHER_RANK_IC_BAR)
    )
    # Binary weak if logit Δ not past bar and/or pos pctile not clearly above 0.5
    c2_binary_weak = bool(
        (not np.isfinite(c2_mfe_logit) or c2_mfe_logit > LOGIT_DELTA_BAR)
        and (not np.isfinite(c2_topk) or c2_topk < STUDENT_TOPK_LIFT_BAR)
    )
    alpha_in_ranking_not_binary = bool(
        teacher_shows_ranking_alpha and c2_binary_weak
    )

    mfe_lifts = bool(np.isfinite(mfe_delta) and mfe_delta <= LOGIT_DELTA_BAR)
    clearly_better_than_g = bool(
        g_delta is not None
        and np.isfinite(mfe_delta)
        and np.isfinite(float(g_delta))
        and (mfe_delta <= float(g_delta) - 0.02)
    )

    if mfe_lifts or clearly_better_than_g:
        verdict = "path_touch_mfe10__lifts_enough_for_next_step"
        train_yes_no = "sidecar_only"
    elif student_above_chance and teacher_shows_ranking_alpha:
        verdict = "student_distills_proxy_teacher__ranking_alpha_present__no_binary_train"
        train_yes_no = "no"
    elif alpha_in_ranking_not_binary:
        verdict = "c2_teacher_shows_ranking_alpha_not_10pct_binary__do_not_train_binary"
        train_yes_no = "no"
    else:
        verdict = "path_touch_mfe10__no_lift__proxy_teacher_weak__do_not_train"
        train_yes_no = "no"

    decision = {
        "verdict": verdict,
        "train_yes_no": train_yes_no,
        "label_definition": (
            f"y=1{{mfe10>={threshold:.2f}}} where "
            "mfe10=max(high[T+1:T+10])/close[T]-1 (path-touch)"
        ),
        "mfe_touch_comb_delta": mfe_delta,
        "ctc_comb_delta_recomputed": ctc_delta,
        "phase_g_ctc_comb_delta": g_delta,
        "mfe_lifts_abs_bar": mfe_lifts,
        "clearly_better_than_phase_g": clearly_better_than_g,
        "student_above_chance": student_above_chance,
        "student_vs_teacher_daily_ic_comb": stud_ic,
        "student_spearman_vs_teacher_comb": stud_spearman,
        "student_topk_lift_vs_mfe_comb": stud_topk,
        "teacher_shows_ranking_alpha": teacher_shows_ranking_alpha,
        "alpha_in_ranking_not_binary": alpha_in_ranking_not_binary,
        "c2_daily_rank_ic_vs_return_10d": c2_rank_ret,
        "c2_daily_rank_ic_vs_path_touch": c2_rank,
        "c2_path_touch_logit_delta": c2_mfe_logit,
        "proxy_rank_ic_vs_mfe10": proxy_rank_mfe,
        "proxy_rank_ic_vs_fwd_ret_10": proxy_rank_fwd,
        "thresholds": {
            "logit_delta_bar": LOGIT_DELTA_BAR,
            "student_rank_ic_bar": STUDENT_RANK_IC_BAR,
            "student_topk_lift_bar": STUDENT_TOPK_LIFT_BAR,
            "teacher_rank_ic_bar": TEACHER_RANK_IC_BAR,
            "mfe_threshold": threshold,
        },
        "recommend_next": (
            "Path-touch MFE≥10% logistic did not clear Δ≤-0.04; C2 teacher on "
            "sealed OOS shows Rank IC on continuous returns but weak TopK/"
            "binary for +10% path-touch — alpha lives in ranking, not this "
            "binary. Do not open Kairos binary train. If pursuing ranking, "
            "need C2 scores on val dates or a true ranking objective — not "
            "more threshold knobs."
            if train_yes_no == "no"
            else "Path-touch label cleared a bar; only short sidecar probe, no R2."
        ),
    }

    return {
        "phase": "H_c2_teacher_distill_path_touch_mfe",
        "n_samples": int(len(base)),
        "base_dim": int(base.shape[1]),
        "xsec_dim": int(packed["xsec_dim"]),
        "xsec_cols": packed["xsec_cols"],
        "seed": seed,
        "threshold": float(threshold),
        "definition": {
            "primary_path_touch": (
                f"y=1{{mfe10>={threshold:.2f}}} "
                "mfe10=max(high[T+1:T+10])/close[T]-1"
            ),
            "phase_g_contrast": "y=1{fwd_ret_10>=0.10} close-to-close",
            "teacher_ideal": "C2 Best Seg@179 cross-sectional predicted_return_10d / rank",
            "teacher_fallback": (
                "held-out RS: 0.35*cs_rank(ret10)+0.35*cs_rank(ret60)+"
                "0.15*cs_rank(ind_rel_ret10)+0.15*cs_rank(ind_rel_ret60)"
            ),
        },
        "c2_availability": availability,
        "c2_teacher_upper_bound": c2_upper,
        "label_prevalence": {
            "mfe_touch_10pct_rate": float(np.mean(y_mfe)),
            "mfe_touch_10pct_n_pos": int(y_mfe.sum()),
            "ctc_10pct_rate": float(np.mean(y_ctc)),
            "ctc_10pct_n_pos": int(y_ctc.sum()),
            "mfe10_mean": float(np.mean(mfe10)),
            "mfe10_std": float(np.std(mfe10)),
            "fwd_ret_10_mean": float(np.mean(fwd10)),
            "overlap_mfe_and_ctc_pos": float(np.mean(y_mfe & y_ctc)),
            "precision_ctc_given_mfe": float(
                np.mean(y_ctc[y_mfe.astype(bool)]) if y_mfe.any() else float("nan")
            ),
            "recall_mfe_given_ctc": float(
                np.mean(y_mfe[y_ctc.astype(bool)]) if y_ctc.any() else float("nan")
            ),
        },
        "path_touch_logistic": path_touch_logit,
        "path_touch_regression": path_touch_reg,
        "proxy_teacher_upper_bound": proxy_teacher_upper,
        "student_ridge": student_ridge,
        "student_rank_ic": student_rank_ic,
        "student_logit_teacher_top20": student_logit,
        "phase_g_contrast": phase_g,
        "decision": decision,
    }


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
    dec = payload["decision"]
    avail = payload["c2_availability"]
    prev = payload["label_prevalence"]
    c2 = payload.get("c2_teacher_upper_bound", {}).get("rank_t060_p90_n16", {})
    logit = payload["path_touch_logistic"]
    pg = payload.get("phase_g_contrast", {})

    lines = [
        "# Kairos Phase H：C2 教师蒸馏 + 路径触及 MFE≥10%（廉价消融）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
        f"**{dec['verdict']}**；train_yes_no=`{dec['train_yes_no']}`。",
        "",
        f"- 建议：{dec['recommend_next']}",
        "",
        "## 标签澄清（相对 Phase G）",
        "",
        f"- **主标签（本 Phase）**：`{dec['label_definition']}`",
        f"- **Phase G**：`{pg.get('definition', 'y=1{fwd_ret_10>=0.10} close-to-close')}`",
        f"- 正类率 path-touch MFE≥10% = `{prev['mfe_touch_10pct_rate']:.4f}` "
        f"(`{prev['mfe_touch_10pct_n_pos']}` / `{payload['n_samples']}`)",
        f"- 正类率 close-to-close ≥10% = `{prev['ctc_10pct_rate']:.4f}`",
        f"- 二者同时为正率 = `{prev['overlap_mfe_and_ctc_pos']:.4f}`；"
        f"P(ctc|mfe)=`{prev['precision_ctc_given_mfe']:.4f}`；"
        f"P(mfe|ctc)=`{prev['recall_mfe_given_ctc']:.4f}`",
        "",
        "## C2 Best Seg@179 可用性",
        "",
        f"- checkpoint=`{avail['c2_checkpoint']}` segment=`{avail['c2_segment']}`",
        f"- OOS asof=`{avail['c2_oos_asof_start']}`..`{avail['c2_oos_asof_end']}`",
        f"- Val asof=`{avail['val_asof_start']}`..`{avail['val_asof_end']}`",
        f"- date_overlap=`{avail['date_overlap_with_val']}`",
        f"- **blocker**：{avail['blocker']}",
        f"- **fallback**：{avail['fallback']}",
        "",
        "## 1) C2 教师 → 标签上界（密封 18 日 OOS，真实分数）",
        "",
    ]
    if c2:
        ric = c2.get("rank_ic_vs_return_10d", {})
        ric_p = c2.get("rank_ic_vs_path_touch", {})
        topk = c2.get("topk20_hit_vs_path_touch", {})
        tl = c2.get("teacher_logit_path_touch_10pct", {})
        lines += [
            f"- 协议=`{c2.get('protocol')}` n=`{c2.get('n_rows')}`",
            f"- 路径触及近似：{c2.get('path_touch_note')}",
            f"- Daily Rank IC vs return_10d mean=`{ric.get('mean')}` "
            f"(n_days=`{ric.get('n_days')}`)",
            f"- Daily Rank IC vs path-touch approx mean=`{ric_p.get('mean')}`",
            f"- Top20% hit vs path-touch mean=`{topk.get('mean_hit')}` "
            f"chance=`{topk.get('chance')}` lift=`{topk.get('lift_vs_chance')}`",
            f"- Teacher→path-touch≥10% logistic Δ=`{tl.get('delta_model_minus_prior')}`",
            f"- buy_profit ctc pos teacher pctile=`{c2.get('buy_profit_ctc_10pct', {}).get('mean_teacher_pctile_pos')}` "
            f"vs neg=`{c2.get('buy_profit_ctc_10pct', {}).get('mean_teacher_pctile_neg')}`",
            f"- path-touch pos teacher pctile=`{c2.get('path_touch_mfe_approx_10pct', {}).get('mean_teacher_pctile_pos')}` "
            f"vs neg=`{c2.get('path_touch_mfe_approx_10pct', {}).get('mean_teacher_pctile_neg')}`",
            "",
        ]
    else:
        lines += ["- （无 C2 预测文件）", ""]

    proxy = payload["proxy_teacher_upper_bound"]
    lines += [
        "## 2) Val 面板：路径触及 logistic（特征 → 标签）",
        "",
        "| Head | prior LL | model LL | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in logit.items():
        if v.get("skipped"):
            continue
        lines.append(
            f"| {k} | {v['prior_log_loss']:.6f} | {v['model_log_loss']:.6f} | "
            f"{v['delta_model_minus_prior']:.6f} |"
        )
    lines += [
        "",
        f"- Phase G comb Δ（ctc）=`{pg.get('buy_profit_10pct_comb_delta')}`",
        f"- 本 Phase 重算 ctc comb Δ=`{dec['ctc_comb_delta_recomputed']}`",
        f"- 本 Phase mfe-touch comb Δ=`{dec['mfe_touch_comb_delta']}`",
        f"- 过绝对门槛 Δ≤{LOGIT_DELTA_BAR}：`{dec['mfe_lifts_abs_bar']}`；"
        f"明显优于 G：`{dec['clearly_better_than_phase_g']}`",
        "",
        "## 3) RS/配对代理教师（因 C2 无法对齐 val）",
        "",
        f"- 定义：`{payload['definition']['teacher_fallback']}`",
        f"- Rank IC vs fwd_ret_10=`{proxy['rank_ic_vs_fwd_ret_10']['mean']}`",
        f"- Rank IC vs mfe10=`{proxy['rank_ic_vs_mfe10']['mean']}`",
        f"- Top20 hit lift vs mfe=`{proxy['topk20_hit_vs_mfe10']['lift_vs_chance']}`",
        f"- Teacher→mfe-touch logistic Δ=`{proxy['teacher_logit_mfe_touch'].get('delta_model_minus_prior')}`",
        "",
        "## 4) Student 蒸馏（Base / Base+xsection → 代理教师）",
        "",
    ]
    for k, v in payload["student_ridge"].items():
        lines.append(
            f"- Ridge `{k}`: R²=`{v.get('ridge_r2')}` spearman=`{v.get('spearman_pred_vs_y')}`"
        )
    for k, v in payload["student_logit_teacher_top20"].items():
        if not v.get("skipped"):
            lines.append(
                f"- Logistic `{k}`: Δ=`{v.get('delta_model_minus_prior')}`"
            )
    sic = payload["student_rank_ic"]
    lines += [
        f"- Student pred daily IC vs teacher (comb)=`"
        f"{sic.get('student_pred_vs_teacher_daily_comb', {}).get('mean')}`",
        f"- Student pred daily IC vs mfe (comb)=`"
        f"{sic.get('student_pred_vs_mfe_comb', {}).get('mean')}`",
        f"- student_above_chance=`{dec['student_above_chance']}`",
        "",
        "## 5) 决策",
        "",
        f"- verdict = `{dec['verdict']}`",
        f"- train_yes_no = `{dec['train_yes_no']}`",
        f"- teacher_shows_ranking_alpha = `{dec['teacher_shows_ranking_alpha']}`",
        f"- alpha_in_ranking_not_binary = `{dec['alpha_in_ranking_not_binary']}`",
        f"- C2 Rank IC (return_10d)=`{dec['c2_daily_rank_ic_vs_return_10d']}`；"
        f"path-touch=`{dec['c2_daily_rank_ic_vs_path_touch']}`",
        "",
        "### 建议下一步",
        "",
        "1. **不要**基于 path-touch MFE≥10% 或 Phase G ctc 开 Kairos 二分类长训。",
        "2. C2 教师在密封窗证明**排序**有 alpha；+10% 路径/收盘二分类不是同一回事。",
        "3. 若要继续蒸馏，需把 C2 Seg@179 分数物化到 val 日期（推理成本高）或改用排序目标。",
        "4. **不要**重启同配置 R2 / 不要动同事 TPU WIP。",
        "",
        "## 用户要点（中文）",
        "",
        f"- **定义**：持有至多 10 日，路径最高价相对买入价涨幅 ≥10% 记正类（mfe10）。",
        f"- **正类率**：MFE=`{prev['mfe_touch_10pct_rate']:.4f}`；ctc=`{prev['ctc_10pct_rate']:.4f}`",
        f"- **特征→MFE logistic Δ**：`{dec['mfe_touch_comb_delta']}`（G 的 ctc Δ=`{pg.get('buy_profit_10pct_comb_delta')}`）",
        f"- **C2 上界 Rank IC**：`{dec['c2_daily_rank_ic_vs_return_10d']}`（排序有用，二分类弱）",
        f"- **开训**：`{'否' if dec['train_yes_no']=='no' else '仅短 sidecar'}`",
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
        "--phase-g-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_g_buy_profit_10pct.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("modernbert_finance/ablations/kairos_phase_h_teacher_distill.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_phase_h_teacher_distill_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold", type=float, default=MFE_TOUCH_THRESHOLD)
    args = parser.parse_args(argv)

    payload = run_phase_h(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        phase_g_json=args.phase_g_json,
        seed=args.seed,
        threshold=args.threshold,
    )
    # Drop non-serializable if any
    def _clean(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            x = float(o)
            if not np.isfinite(x):
                return None
            return x
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    payload = _clean(payload)
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
                "c2_availability": payload["c2_availability"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
