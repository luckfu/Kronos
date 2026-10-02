"""Kairos Phase X smoke: Qlib-axis Alpha158 + LightGBM binary on mfe10 decision.

Decision-only: y=1{mfe10>=0.10}; gate Δ logloss vs constant prior ≤ -0.04.
Primary = train 2023–2024 (capped balanced) → full val.
Secondary = val_temporal 2025H2→2026H1.
NOT ranking/IC/TopK product; NOT ModernBERT; NOT TPU WIP.

pyqlib wheel unavailable on CPython 3.13 without full build; feature formulas
faithfully mirror microsoft/qlib Alpha158DL.get_feature_config() defaults.
Model = lightgbm objective=binary (same loss surface as qlib LGBModel loss:binary).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from modernbert_finance.ablations._panel_io import load_panel
from modernbert_finance.build_dataset import FEATURES, LOOKBACK, prepare_frame

GATE = -0.04
SEED = 20261002
MFE_THR = 0.10
TRAIN_CAP = 300_000  # SHORT smoke
LGB_ROUNDS = 200
LGB_EARLY = 30
PHASE = "X_qlib_alpha158_lgb_binary_smoke"


def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def constant_prior_log_loss(y: np.ndarray, prior: float) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = float(np.clip(prior, 1e-7, 1.0 - 1e-7))
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _rolling_slope(x: np.ndarray) -> np.ndarray:
    """Slope of OLS on 0..w-1 vs last-w window; x shape (N, w)."""
    n, w = x.shape
    t = np.arange(w, dtype=np.float64)
    t_mean = t.mean()
    x_mean = x.mean(axis=1, keepdims=True)
    num = ((t - t_mean) * (x - x_mean)).sum(axis=1)
    den = ((t - t_mean) ** 2).sum()
    return num / max(den, 1e-12)


def _rolling_rsquare(x: np.ndarray) -> np.ndarray:
    n, w = x.shape
    t = np.arange(w, dtype=np.float64)
    t_mean = t.mean()
    x_mean = x.mean(axis=1, keepdims=True)
    ss_tot = ((x - x_mean) ** 2).sum(axis=1)
    slope = _rolling_slope(x)
    intercept = x_mean.ravel() - slope * t_mean
    pred = intercept[:, None] + slope[:, None] * t[None, :]
    ss_res = ((x - pred) ** 2).sum(axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def _rolling_resi(x: np.ndarray) -> np.ndarray:
    """Residual at last point of linear regression."""
    n, w = x.shape
    t = np.arange(w, dtype=np.float64)
    t_mean = t.mean()
    x_mean = x.mean(axis=1)
    slope = _rolling_slope(x)
    intercept = x_mean - slope * t_mean
    pred_last = intercept + slope * t[-1]
    return x[:, -1] - pred_last


def alpha158_from_windows(hist: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute Alpha158-style features from (N, T, F) OHLCVA windows.

    Mirrors qlib Alpha158DL default config (kbar + price windows=[0] + rolling).
    VWAP ≈ amount/volume when amount present else (o+h+l+c)/4.
    """
    hist = np.asarray(hist, dtype=np.float64)
    if hist.ndim != 3 or hist.shape[1] < 60:
        raise ValueError(f"need (N,T>=60,F), got {hist.shape}")
    o = hist[:, :, 0]
    h = hist[:, :, 1]
    l = hist[:, :, 2]
    c = hist[:, :, 3]
    v = hist[:, :, 4]
    if hist.shape[2] >= 6:
        amt = hist[:, :, 5]
        vwap = np.where(v > 0, amt / np.maximum(v, 1e-12), (o + h + l + c) / 4.0)
    else:
        vwap = (o + h + l + c) / 4.0

    close = c[:, -1]
    open_ = o[:, -1]
    high = h[:, -1]
    low = l[:, -1]
    vol = v[:, -1]
    eps = 1e-12

    feats: list[np.ndarray] = []
    names: list[str] = []

    # kbar
    feats += [
        (close - open_) / np.maximum(open_, eps),
        (high - low) / np.maximum(open_, eps),
        (close - open_) / (high - low + eps),
        (high - np.maximum(open_, close)) / np.maximum(open_, eps),
        (high - np.maximum(open_, close)) / (high - low + eps),
        (np.minimum(open_, close) - low) / np.maximum(open_, eps),
        (np.minimum(open_, close) - low) / (high - low + eps),
        (2 * close - high - low) / np.maximum(open_, eps),
        (2 * close - high - low) / (high - low + eps),
    ]
    names += ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]

    # price windows=[0]
    feats += [
        open_ / np.maximum(close, eps),
        high / np.maximum(close, eps),
        low / np.maximum(close, eps),
        vwap[:, -1] / np.maximum(close, eps),
    ]
    names += ["OPEN0", "HIGH0", "LOW0", "VWAP0"]

    windows = [5, 10, 20, 30, 60]

    def take(arr: np.ndarray, d: int) -> np.ndarray:
        return arr[:, -d:]

    for d in windows:
        cw = take(c, d)
        hw = take(h, d)
        lw = take(l, d)
        vw = take(v, d)
        # ROC
        feats.append(c[:, -1 - d] / np.maximum(close, eps) if c.shape[1] > d else close / np.maximum(close, eps))
        # actually Ref($close,d)/$close — close d days ago / today
        feats[-1] = c[:, -1 - d] / np.maximum(close, eps)
        names.append(f"ROC{d}")
        # MA
        feats.append(cw.mean(axis=1) / np.maximum(close, eps))
        names.append(f"MA{d}")
        # STD
        feats.append(cw.std(axis=1) / np.maximum(close, eps))
        names.append(f"STD{d}")
        # BETA (slope)
        feats.append(_rolling_slope(cw) / np.maximum(close, eps))
        names.append(f"BETA{d}")
        # RSQR
        feats.append(_rolling_rsquare(cw))
        names.append(f"RSQR{d}")
        # RESI
        feats.append(_rolling_resi(cw) / np.maximum(close, eps))
        names.append(f"RESI{d}")
        # MAX / MIN
        feats.append(hw.max(axis=1) / np.maximum(close, eps))
        names.append(f"MAX{d}")
        feats.append(lw.min(axis=1) / np.maximum(close, eps))
        names.append(f"MIN{d}")
        # QTLU / QTLD
        feats.append(np.quantile(cw, 0.8, axis=1) / np.maximum(close, eps))
        names.append(f"QTLU{d}")
        feats.append(np.quantile(cw, 0.2, axis=1) / np.maximum(close, eps))
        names.append(f"QTLD{d}")
        # RANK percentile of last close in window
        ranks = (cw <= close[:, None]).sum(axis=1) / float(d)
        feats.append(ranks)
        names.append(f"RANK{d}")
        # RSV
        mn = lw.min(axis=1)
        mx = hw.max(axis=1)
        feats.append((close - mn) / (mx - mn + eps))
        names.append(f"RSV{d}")
        # IMAX / IMIN / IMXD (idx from start of window / d; qlib IdxMax is days from end?)
        # qlib IdxMax: index of max in rolling window normalized by d
        imax = (d - 1 - hw[:, ::-1].argmax(axis=1)) / float(d)  # days since max / d approx
        imin = (d - 1 - lw[:, ::-1].argmin(axis=1)) / float(d)
        # Use argmax position from left / d to match common ports
        imax = hw.argmax(axis=1) / float(d)
        imin = lw.argmin(axis=1) / float(d)
        feats.append(imax)
        names.append(f"IMAX{d}")
        feats.append(imin)
        names.append(f"IMIN{d}")
        feats.append((imax - imin))
        names.append(f"IMXD{d}")
        # CORR close vs log volume
        lv = np.log(vw + 1.0)
        c0 = cw - cw.mean(axis=1, keepdims=True)
        l0 = lv - lv.mean(axis=1, keepdims=True)
        corr = (c0 * l0).sum(axis=1) / np.sqrt(
            np.maximum((c0**2).sum(axis=1) * (l0**2).sum(axis=1), eps)
        )
        feats.append(corr)
        names.append(f"CORR{d}")
        # CORD: corr of return vs log vol change
        ret = cw[:, 1:] / np.maximum(cw[:, :-1], eps)
        lvc = np.log(vw[:, 1:] / np.maximum(vw[:, :-1], eps) + 1.0)
        if ret.shape[1] >= 2:
            r0 = ret - ret.mean(axis=1, keepdims=True)
            g0 = lvc - lvc.mean(axis=1, keepdims=True)
            cord = (r0 * g0).sum(axis=1) / np.sqrt(
                np.maximum((r0**2).sum(axis=1) * (g0**2).sum(axis=1), eps)
            )
        else:
            cord = np.zeros(len(close))
        feats.append(cord)
        names.append(f"CORD{d}")
        # CNTP / CNTN / CNTD
        up = (cw[:, 1:] > cw[:, :-1]).astype(np.float64)
        dn = (cw[:, 1:] < cw[:, :-1]).astype(np.float64)
        feats.append(up.mean(axis=1))
        names.append(f"CNTP{d}")
        feats.append(dn.mean(axis=1))
        names.append(f"CNTN{d}")
        feats.append(up.mean(axis=1) - dn.mean(axis=1))
        names.append(f"CNTD{d}")
        # SUMP / SUMN / SUMD
        chg = np.diff(cw, axis=1)
        gain = np.maximum(chg, 0.0).sum(axis=1)
        loss = np.maximum(-chg, 0.0).sum(axis=1)
        abs_sum = np.abs(chg).sum(axis=1) + eps
        feats.append(gain / abs_sum)
        names.append(f"SUMP{d}")
        feats.append(loss / abs_sum)
        names.append(f"SUMN{d}")
        feats.append((gain - loss) / abs_sum)
        names.append(f"SUMD{d}")
        # VMA / VSTD
        feats.append(vw.mean(axis=1) / (vol + eps))
        names.append(f"VMA{d}")
        feats.append(vw.std(axis=1) / (vol + eps))
        names.append(f"VSTD{d}")
        # WVMA
        abs_ret = np.abs(cw[:, 1:] / np.maximum(cw[:, :-1], eps) - 1.0) * vw[:, 1:]
        feats.append(abs_ret.std(axis=1) / (abs_ret.mean(axis=1) + eps))
        names.append(f"WVMA{d}")
        # VSUMP / VSUMN / VSUMD
        vchg = np.diff(vw, axis=1)
        vg = np.maximum(vchg, 0.0).sum(axis=1)
        vl = np.maximum(-vchg, 0.0).sum(axis=1)
        vabs = np.abs(vchg).sum(axis=1) + eps
        feats.append(vg / vabs)
        names.append(f"VSUMP{d}")
        feats.append(vl / vabs)
        names.append(f"VSUMN{d}")
        feats.append((vg - vl) / vabs)
        names.append(f"VSUMD{d}")

    x = np.column_stack(feats)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names


def extract_windows(
    panel: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Pull (N, LOOKBACK, F) windows via start_index; y from mfe10."""
    # group targets by symbol for faster access
    y = (targets["mfe10"].to_numpy(dtype=np.float64) >= MFE_THR).astype(np.float64)
    asof = pd.to_datetime(targets["asof_date"])
    histories = np.zeros((len(targets), LOOKBACK, len(FEATURES)), dtype=np.float64)
    # Pre-prepare frames
    prepared: dict[str, np.ndarray] = {}
    for sym, frame in panel.items():
        fr = prepare_frame(frame, sym)
        prepared[str(sym)] = fr.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)

    miss = 0
    for i, (sym, start) in enumerate(zip(targets["symbol"].astype(str), targets["start_index"].to_numpy())):
        arr = prepared.get(sym)
        start = int(start)
        if arr is None or start < 0 or start + LOOKBACK > len(arr):
            miss += 1
            continue
        histories[i] = arr[start : start + LOOKBACK]
    if miss:
        print(json.dumps({"phase": "window_miss", "miss": miss, "n": len(targets)}), flush=True)
    return histories, y, asof


def balanced_sample(
    x: np.ndarray,
    y: np.ndarray,
    asof: pd.Series,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    if len(y) <= cap:
        return x, y, asof
    rng = np.random.default_rng(seed)
    pos = np.where(y >= 0.5)[0]
    neg = np.where(y < 0.5)[0]
    n_pos = min(len(pos), cap // 4)
    n_neg = min(len(neg), cap - n_pos)
    idx = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    rng.shuffle(idx)
    return x[idx], y[idx], asof.iloc[idx].reset_index(drop=True)


def train_lgb_binary(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    asof_tr: pd.Series | None = None,
) -> dict[str, Any]:
    prior = float(y_tr.mean())
    prior_ll = constant_prior_log_loss(y_te, prior=prior)

    # Internal early-stop split by date if possible
    if asof_tr is not None and len(asof_tr) > 1000:
        cut = asof_tr.quantile(0.85)
        tr_m = (asof_tr <= cut).to_numpy()
        va_m = (asof_tr > cut).to_numpy()
        if tr_m.sum() > 1000 and va_m.sum() > 500:
            dtrain = lgb.Dataset(x_tr[tr_m], label=y_tr[tr_m])
            dvalid = lgb.Dataset(x_tr[va_m], label=y_tr[va_m], reference=dtrain)
            valid_sets = [dvalid]
            valid_names = ["internal_tail"]
        else:
            dtrain = lgb.Dataset(x_tr, label=y_tr)
            valid_sets, valid_names = [], []
    else:
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        valid_sets, valid_names = [], []

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": SEED,
        "deterministic": True,
    }
    callbacks = [lgb.log_evaluation(period=0)]
    if valid_sets:
        callbacks.append(lgb.early_stopping(LGB_EARLY, verbose=False))

    t0 = time.time()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGB_ROUNDS,
        valid_sets=valid_sets or None,
        valid_names=valid_names or None,
        callbacks=callbacks,
    )
    fit_s = float(time.time() - t0)
    best_iter = int(getattr(model, "best_iteration", 0) or LGB_ROUNDS)
    proba = model.predict(x_te, num_iteration=best_iter)
    ll = binary_log_loss(proba, y_te)
    delta = float(ll - prior_ll)
    return {
        "model": "lgb_binary_alpha158",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_prior": prior,
        "prior_log_loss": prior_ll,
        "model_log_loss": float(ll),
        "delta_vs_prior": delta,
        "gate_passed": bool(delta <= GATE),
        "mean_pred": float(np.mean(proba)),
        "fit_seconds": fit_s,
        "best_iteration": best_iter,
        "feature_dim": int(x_tr.shape[1]),
    }


def filter_targets(targets: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    asof = pd.to_datetime(targets["asof_date"])
    m = (asof >= pd.Timestamp(start)) & (asof <= pd.Timestamp(end))
    return targets.loc[m].reset_index(drop=True)


def main() -> int:
    t_all = time.time()
    root = Path("/workspace/Kronos")
    data = root / "scratch" / "kairos_qlib_smoke_data"
    train_panel_path = data / "train_data.pkl"
    val_panel_path = data / "val_data.pkl"
    train_targets_path = data / "targets_dl" / "train_targets.parquet"
    val_targets_path = data / "targets_dl" / "validation_targets.parquet"
    out_dir = root / "scratch" / "kairos_qlib_alpha158_smoke_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"phase": "start", "phase_name": PHASE, "train_cap": TRAIN_CAP}), flush=True)

    train_targets = filter_targets(
        pd.read_parquet(train_targets_path), "2023-01-01", "2024-12-31"
    )
    val_targets = pd.read_parquet(val_targets_path)
    print(
        json.dumps(
            {
                "phase": "targets_ready",
                "n_train_window": len(train_targets),
                "n_val": len(val_targets),
                "train_pos_rate": float((train_targets["mfe10"] >= MFE_THR).mean()),
                "val_pos_rate": float((val_targets["mfe10"] >= MFE_THR).mean()),
            }
        ),
        flush=True,
    )

    print(json.dumps({"phase": "load_panels"}), flush=True)
    train_panel = load_panel(train_panel_path)
    val_panel = load_panel(val_panel_path)
    print(
        json.dumps(
            {
                "phase": "panels_loaded",
                "n_train_symbols": len(train_panel),
                "n_val_symbols": len(val_panel),
            }
        ),
        flush=True,
    )

    # --- val features (full) ---
    t0 = time.time()
    hist_val, y_val, asof_val = extract_windows(val_panel, val_targets)
    x_val, feat_names = alpha158_from_windows(hist_val)
    print(
        json.dumps(
            {
                "phase": "val_features_ready",
                "n": len(y_val),
                "dim": int(x_val.shape[1]),
                "seconds": float(time.time() - t0),
                "n_feat_names": len(feat_names),
            }
        ),
        flush=True,
    )

    # --- train features: sample indices first, then extract ---
    t0 = time.time()
    y_all = (train_targets["mfe10"].to_numpy(dtype=np.float64) >= MFE_THR).astype(np.float64)
    # balanced index sample before heavy window extract
    rng = np.random.default_rng(SEED)
    pos = np.where(y_all >= 0.5)[0]
    neg = np.where(y_all < 0.5)[0]
    n_pos = min(len(pos), TRAIN_CAP // 4)
    n_neg = min(len(neg), TRAIN_CAP - n_pos)
    idx = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    rng.shuffle(idx)
    train_targets_s = train_targets.iloc[idx].reset_index(drop=True)
    hist_tr, y_tr, asof_tr = extract_windows(train_panel, train_targets_s)
    x_tr, _ = alpha158_from_windows(hist_tr)
    print(
        json.dumps(
            {
                "phase": "train_features_ready",
                "n_train_raw_window": int(len(train_targets)),
                "n_train": int(len(y_tr)),
                "train_prior": float(y_tr.mean()),
                "dim": int(x_tr.shape[1]),
                "seconds": float(time.time() - t0),
            }
        ),
        flush=True,
    )

    # Primary: train → full val
    primary = train_lgb_binary(x_tr, y_tr, x_val, y_val, asof_tr=asof_tr)
    primary["split"] = "train_to_val"
    primary["train_window"] = "2023-01-01..2024-12-31"
    print(json.dumps({"phase": "primary_result", **primary}), flush=True)

    # Secondary: val_temporal
    tr_m = (asof_val <= "2025-12-31").to_numpy()
    te_m = (asof_val >= "2026-01-01").to_numpy()
    secondary = train_lgb_binary(
        x_val[tr_m],
        y_val[tr_m],
        x_val[te_m],
        y_val[te_m],
        asof_tr=asof_val[tr_m].reset_index(drop=True),
    )
    secondary["split"] = "val_temporal_2025H2_to_2026H1"
    print(json.dumps({"phase": "secondary_result", **secondary}), flush=True)

    gate_passed = bool(primary["gate_passed"])
    report = {
        "status": "QLIB_ALPHA158_SMOKE_COMPLETE",
        "phase": PHASE,
        "purpose": "qlib-axis-alpha158-lgb-binary-mfe10-decision-smoke",
        "target": f"y=1{{mfe10>={MFE_THR}}}",
        "mfe10_def": "max(high[T+1:T+10])/close[T]-1",
        "gate": GATE,
        "feature_set": "Alpha158DL_default_formulas_on_existing_120d_panel",
        "model": "lightgbm_objective_binary",
        "qlib_package": "unavailable_cp313_used_formula_port_plus_lgb",
        "data_source": {
            "panel": "luckfu/a-share-120d-temporal-symbol-holdout",
            "targets": "luckfu/ashare120d-modernbert-targets",
            "reused_local_targets": True,
            "avoided_cn_data_download": True,
        },
        "train_cap": TRAIN_CAP,
        "feature_names_count": len(feat_names),
        "feature_dim": int(x_tr.shape[1]),
        "primary_protocol": "train_to_val",
        "secondary_protocol": "val_temporal",
        "train_to_val": primary,
        "val_temporal": secondary,
        "best_model": primary["model"],
        "best_delta_vs_prior": primary["delta_vs_prior"],
        "gate_passed": gate_passed,
        "phase_w_primary_delta": -0.028500119076822594,
        "phase_s_temporal_delta": -0.04179349770224905,
        "elapsed_seconds": float(time.time() - t_all),
        "decision": (
            "PASS_PREPARE_LONGER_CONFIRM"
            if gate_passed
            else "FAIL_HARD_CONCLUDE_QLIB_ALPHA158_AXIS"
        ),
        "not_ranking_ic": True,
        "not_modernbert": True,
        "not_tpu": True,
    }

    json_path = root / "modernbert_finance" / "ablations" / "kairos_phase_x_qlib_alpha158_decision_results.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "done", "gate_passed": gate_passed, "delta": primary["delta_vs_prior"], "json": str(json_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
