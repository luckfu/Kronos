"""Kairos Phase Y smoke: CS top-quintile path-MFE label + S/T2 tabular stack.

Decision-only (still buy/not):
  y = 1{ daily cross-sectional percentile of mfe10 ≥ 0.80 }
  mfe10 = max(high[T+1:T+10]) / close[T] - 1   (unchanged)

Model: reuse S/T2 winners — logistic / enet / LR⊕shallow-MLP blend on Base+xsection.
Gate: Δ logloss vs constant prior ≤ -0.04 on PRIMARY train→val (2023–2024→val);
also report val_temporal.

Memory-safe train path: label CS on full window, balanced-sample indices, then
extract only capped hist windows (Phase X pattern). Light meta pass builds
full-day xsection ranks without stacking all histories.

NOT Alpha158/Qlib; NOT ModernBERT; NOT ranking-as-product; NOT TPU WIP.
Short smoke: train_cap=300k, MLP max_iter=60.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GATE = -0.04
SEED = 20261002
TRAIN_CAP = 300_000
MLP_MAX_ITER = 60
MLP_N_ITER_NO_CHANGE = 8
CS_PCT = 0.80
LABEL_MODE = "cs_top"
PHASE = "Y_mfe10_cs_top_tabular_smoke"
PHASE_S_BEST = -0.04179349770224905
PHASE_W_PRIMARY = -0.028500119076822594

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAL_PANEL = ROOT / "scratch/kairos_qlib_smoke_data/val_data.pkl"
DEFAULT_VAL_TARGETS = (
    ROOT / "scratch/kairos_ablation_data/targets_dl/validation_targets.parquet"
)
DEFAULT_TRAIN_PANEL = ROOT / "scratch/kairos_qlib_smoke_data/train_data.pkl"
DEFAULT_TRAIN_TARGETS = (
    ROOT / "scratch/kairos_ablation_data/targets_dl/train_targets.parquet"
)
OUT_DIR = ROOT / "scratch/kairos_phase_y_outputs"


def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def constant_prior_log_loss(y: np.ndarray, prior: float) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = float(np.clip(prior, 1e-7, 1.0 - 1e-7))
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def eval_split(
    name: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prior = float(y_tr.mean())
    prior_ll = constant_prior_log_loss(y_te, prior=prior)
    out: dict[str, Any] = {
        "split": name,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_prior": prior,
        "prior_log_loss": prior_ll,
        "pos_train": prior,
        "pos_test": float(y_te.mean()),
        "feature_dim": int(x_tr.shape[1]),
        "models": {},
        "budget": {
            "train_cap": TRAIN_CAP,
            "mlp_max_iter": MLP_MAX_ITER,
            "mlp_n_iter_no_change": MLP_N_ITER_NO_CHANGE,
        },
    }

    recipes: list[tuple[str, Any]] = [
        (
            "logistic_C0.01",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=SEED, C=0.01),
            ),
        ),
        (
            "enet_C0.01_l1_0.7",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    random_state=SEED,
                    C=0.01,
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.7,
                ),
            ),
        ),
        (
            "enet_C0.01_l1_0.5",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    random_state=SEED,
                    C=0.01,
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                ),
            ),
        ),
    ]

    for rname, clf in recipes:
        t1 = time.time()
        clf.fit(x_tr, y_tr.astype(int))
        proba = clf.predict_proba(x_te)[:, 1]
        ll = binary_log_loss(proba, y_te)
        delta = float(ll - prior_ll)
        rec = {
            "delta_vs_prior": delta,
            "model_log_loss": float(ll),
            "gate_passed": bool(delta <= GATE),
            "mean_pred": float(proba.mean()),
            "fit_seconds": float(time.time() - t1),
        }
        out["models"][rname] = rec
        print(
            json.dumps({"phase": "model_result", "split": name, "model": rname, **rec}),
            flush=True,
        )

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED, C=0.01),
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.1,
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=MLP_N_ITER_NO_CHANGE,
            random_state=SEED,
        ),
    )
    t1 = time.time()
    lr.fit(x_tr, y_tr.astype(int))
    mlp.fit(x_tr, y_tr.astype(int))
    p_lr = lr.predict_proba(x_te)[:, 1]
    p_mlp = mlp.predict_proba(x_te)[:, 1]
    blend_fit_s = float(time.time() - t1)
    mlp_n_iter = int(mlp.named_steps["mlpclassifier"].n_iter_)
    for w in (0.5, 0.7, 0.8, 0.9):
        proba = w * p_lr + (1.0 - w) * p_mlp
        ll = binary_log_loss(proba, y_te)
        delta = float(ll - prior_ll)
        rname = f"blend_lr{w}_mlp{1.0 - w:.1f}"
        rec = {
            "delta_vs_prior": delta,
            "model_log_loss": float(ll),
            "gate_passed": bool(delta <= GATE),
            "mean_pred": float(proba.mean()),
            "fit_seconds": blend_fit_s,
            "mlp_n_iter": mlp_n_iter,
            "mlp_max_iter": MLP_MAX_ITER,
        }
        out["models"][rname] = rec
        print(
            json.dumps({"phase": "model_result", "split": name, "model": rname, **rec}),
            flush=True,
        )

    ranked = sorted(
        ((k, v["delta_vs_prior"]) for k, v in out["models"].items()),
        key=lambda kv: kv[1],
    )
    out["best_model"] = ranked[0][0]
    out["best_delta_vs_prior"] = float(ranked[0][1])
    out["gate_passed"] = bool(ranked[0][1] <= GATE)
    out["ranked"] = [{"model": k, "delta": float(d)} for k, d in ranked]
    return out


def _pack_val(
    panel: Path,
    targets: Path,
) -> tuple[np.ndarray, np.ndarray, pd.Series, dict[str, Any]]:
    """Val is small (~124k); reuse full enriched builder."""
    from modernbert_finance.ablations.buy_profit_mfe_ablations import (
        build_mfe_buy_labels,
    )

    packed = build_mfe_buy_labels(
        panel,
        pd.read_parquet(targets),
        signal_start="2025-07-03",
        signal_end="2026-07-02",
        label_mode=LABEL_MODE,
        cs_pct=CS_PCT,
    )
    comb = np.nan_to_num(
        np.asarray(packed["combined_features"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y = np.asarray(packed["buy_labels"]["buy_worth_mfe10pct"], dtype=np.float64)
    asof = pd.to_datetime(packed["meta"]["asof_date"])
    meta = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "feature_dim": int(comb.shape[1]),
        "label_mode": packed["label_mode"],
        "cs_pct": packed["cs_pct"],
        "definition": packed["definition"]["primary"],
    }
    return comb, y, asof, meta


def _light_meta_xsec(
    panel: dict[str, pd.DataFrame],
    *,
    signal_start: str,
    signal_end: str,
) -> pd.DataFrame:
    """Vectorized per-symbol ret5/ret20/vol20 + full-day CS / ind-rel ranks."""
    from modernbert_finance.build_dataset import FEATURES, LOOKBACK, prepare_frame
    from modernbert_finance.build_targets import WINDOW

    start_ts = pd.Timestamp(signal_start)
    end_ts = pd.Timestamp(signal_end)
    close_idx = 3
    sym_l: list[str] = []
    start_l: list[int] = []
    asof_l: list[str] = []
    sector_l: list[str] = []
    ret5_l: list[np.ndarray] = []
    ret20_l: list[np.ndarray] = []
    vol20_l: list[np.ndarray] = []

    for symbol in sorted(panel):
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
        dates = pd.to_datetime(frame.index)
        sector_col = (
            frame["sector"].astype(str).to_numpy()
            if "sector" in frame.columns
            else np.full(len(frame), "UNK", dtype=object)
        )
        max_start = len(frame) - WINDOW + 1
        starts = np.arange(max_start, dtype=np.int64)
        asof_pos = starts + LOOKBACK - 1
        asof_dates = dates[asof_pos]
        keep = (asof_dates >= start_ts) & (asof_dates <= end_ts)
        if not np.any(keep):
            continue
        starts = starts[keep]
        asof_pos = asof_pos[keep]
        n = len(starts)
        # close at asof and lookbacks for ret5/ret20
        close = values[:, close_idx]
        c0 = close[asof_pos]
        c5 = close[np.maximum(asof_pos - 5, 0)]
        c20 = close[np.maximum(asof_pos - 20, 0)]
        ret5 = c0 / np.maximum(c5, 1e-8) - 1.0
        ret20 = c0 / np.maximum(c20, 1e-8) - 1.0
        # vol20: std of last 20 log-returns ending at asof
        log_c = np.log(np.maximum(close, 1e-8))
        log_ret = np.diff(log_c, prepend=log_c[0])
        # rolling std of 20 log_ret ending at asof_pos
        # use cumulative sum of squares trick
        lr2 = log_ret ** 2
        cs1 = np.cumsum(log_ret)
        cs2 = np.cumsum(lr2)
        end = asof_pos
        beg = asof_pos - 20
        # sum over (beg+1 .. end] = cs[end] - cs[beg]
        valid = beg >= 0
        s1 = np.zeros(n, dtype=np.float64)
        s2 = np.zeros(n, dtype=np.float64)
        s1[valid] = cs1[end[valid]] - cs1[beg[valid]]
        s2[valid] = cs2[end[valid]] - cs2[beg[valid]]
        mean = s1 / 20.0
        var = s2 / 20.0 - mean ** 2
        vol20 = np.sqrt(np.maximum(var, 0.0))
        vol20[~valid] = 0.0

        sym_l.extend([str(symbol)] * n)
        start_l.extend(starts.tolist())
        asof_l.extend(pd.Series(asof_dates).dt.strftime("%Y-%m-%d").tolist())
        sector_l.extend(sector_col[asof_pos].tolist())
        ret5_l.append(ret5)
        ret20_l.append(ret20)
        vol20_l.append(vol20)

    meta = pd.DataFrame(
        {
            "symbol": sym_l,
            "start_index": start_l,
            "asof_date": asof_l,
            "sector": sector_l,
            "ret5": np.concatenate(ret5_l) if ret5_l else np.array([], dtype=np.float64),
            "ret20": np.concatenate(ret20_l) if ret20_l else np.array([], dtype=np.float64),
            "vol20": np.concatenate(vol20_l) if vol20_l else np.array([], dtype=np.float64),
        }
    )
    for col in ("ret5", "ret20", "vol20"):
        meta[f"cs_rank_{col}"] = meta.groupby("asof_date")[col].rank(pct=True)
        sector_med = meta.groupby(["asof_date", "sector"])[col].transform("median")
        meta[f"ind_rel_{col}"] = meta[col] - sector_med
    return meta


def _extract_base_xsec_for_targets(
    panel: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    meta_xsec: pd.DataFrame | None = None,
) -> np.ndarray:
    """Build Base+xsection for given target rows via start_index (capped memory).

    If meta_xsec is None, compute ret5/ret20/vol20 from extracted windows and
    rank within the sample's asof_date (smoke-safe; full-day ranks used on val
    via build_enriched_matrix).
    """
    from modernbert_finance.ablations.simple_baseline import summarize_history_windows
    from modernbert_finance.build_dataset import FEATURES, LOOKBACK, prepare_frame

    n = len(targets)
    histories = np.zeros((n, LOOKBACK, len(FEATURES)), dtype=np.float32)
    prepared: dict[str, np.ndarray] = {}
    prepared_sector: dict[str, np.ndarray] = {}
    for sym, frame in panel.items():
        fr = prepare_frame(frame, sym)
        prepared[str(sym)] = fr.loc[:, list(FEATURES)].to_numpy(dtype=np.float32)
        if "sector" in fr.columns:
            prepared_sector[str(sym)] = fr["sector"].astype(str).to_numpy()
        else:
            prepared_sector[str(sym)] = np.full(len(fr), "UNK", dtype=object)

    miss = 0
    sectors = np.full(n, "UNK", dtype=object)
    for i, (sym, start) in enumerate(
        zip(targets["symbol"].astype(str), targets["start_index"].to_numpy())
    ):
        arr = prepared.get(sym)
        start_i = int(start)
        if arr is None or start_i < 0 or start_i + LOOKBACK > len(arr):
            miss += 1
            continue
        histories[i] = arr[start_i : start_i + LOOKBACK]
        asof_pos = start_i + LOOKBACK - 1
        sec = prepared_sector.get(sym)
        if sec is not None and asof_pos < len(sec):
            sectors[i] = sec[asof_pos]
    if miss:
        print(json.dumps({"phase": "window_miss", "miss": int(miss), "n": n}), flush=True)

    base = summarize_history_windows(histories.astype(np.float64))
    close_idx = 3
    # ret5 / ret20 / vol20 from window closes
    close = histories[:, :, close_idx].astype(np.float64)
    last = close[:, -1]
    c5 = close[:, -6] if close.shape[1] >= 6 else np.maximum(last, 1e-8)
    c20 = close[:, -21] if close.shape[1] >= 21 else np.maximum(last, 1e-8)
    ret5 = last / np.maximum(c5, 1e-8) - 1.0
    ret20 = last / np.maximum(c20, 1e-8) - 1.0
    log_c = np.log(np.maximum(close, 1e-8))
    log_ret = np.diff(log_c, axis=1)
    vol20 = np.std(log_ret[:, -20:], axis=1) if log_ret.shape[1] >= 20 else np.zeros(n)

    if meta_xsec is not None:
        key = ["symbol", "start_index"]
        merged = targets.loc[:, key].merge(
            meta_xsec.loc[
                :,
                key
                + [
                    "cs_rank_ret5",
                    "cs_rank_ret20",
                    "cs_rank_vol20",
                    "ind_rel_ret5",
                    "ind_rel_ret20",
                    "ind_rel_vol20",
                ],
            ],
            on=key,
            how="left",
        )
        xsec = merged[
            [
                "cs_rank_ret5",
                "cs_rank_ret20",
                "cs_rank_vol20",
                "ind_rel_ret5",
                "ind_rel_ret20",
                "ind_rel_vol20",
            ]
        ].to_numpy(dtype=np.float64)
    else:
        asof = pd.Series(pd.to_datetime(targets["asof_date"]).astype(str))
        tmp = pd.DataFrame(
            {
                "asof_date": asof,
                "sector": sectors,
                "ret5": ret5,
                "ret20": ret20,
                "vol20": vol20,
            }
        )
        for col in ("ret5", "ret20", "vol20"):
            tmp[f"cs_rank_{col}"] = tmp.groupby("asof_date")[col].rank(pct=True)
            med = tmp.groupby(["asof_date", "sector"])[col].transform("median")
            tmp[f"ind_rel_{col}"] = tmp[col] - med
        xsec = tmp[
            [
                "cs_rank_ret5",
                "cs_rank_ret20",
                "cs_rank_vol20",
                "ind_rel_ret5",
                "ind_rel_ret20",
                "ind_rel_vol20",
            ]
        ].to_numpy(dtype=np.float64)

    xsec = np.nan_to_num(xsec, nan=0.0)
    comb = np.concatenate([base, xsec], axis=1)
    return np.nan_to_num(comb, nan=0.0, posinf=0.0, neginf=0.0)


def _pack_train_sampled(
    panel_path: Path,
    targets_path: Path,
    *,
    signal_start: str = "2023-01-01",
    signal_end: str = "2024-12-31",
    cap: int = TRAIN_CAP,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Label full window CS-top, balanced-sample, extract capped Base+xsection."""
    from modernbert_finance.ablations._panel_io import load_panel
    from modernbert_finance.ablations.buy_profit_mfe_ablations import (
        mfe10_daily_cs_percentile,
    )

    t0 = time.time()
    targets = pd.read_parquet(targets_path)
    asof_ts = pd.to_datetime(targets["asof_date"])
    mask = (asof_ts >= pd.Timestamp(signal_start)) & (asof_ts <= pd.Timestamp(signal_end))
    targets = targets.loc[mask].reset_index(drop=True)
    mfe = targets["mfe10"].to_numpy(dtype=np.float64)
    mfe_cs = mfe10_daily_cs_percentile(mfe, targets["asof_date"])
    y_all = (mfe_cs >= float(CS_PCT)).astype(np.float64)
    raw_n = int(len(y_all))
    pos_rate_full = float(y_all.mean())
    print(
        json.dumps(
            {
                "phase": "train_labels_ready",
                "n_train_raw": raw_n,
                "pos_rate_full": pos_rate_full,
                "label_seconds": float(time.time() - t0),
            }
        ),
        flush=True,
    )

    rng = np.random.default_rng(seed)
    pos = np.where(y_all >= 0.5)[0]
    neg = np.where(y_all < 0.5)[0]
    n_pos = min(len(pos), cap // 4)
    n_neg = min(len(neg), cap - n_pos)
    idx = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    rng.shuffle(idx)
    targets_s = targets.iloc[idx].reset_index(drop=True)
    y = y_all[idx]

    print(json.dumps({"phase": "load_train_panel"}), flush=True)
    panel = load_panel(panel_path)
    print(json.dumps({"phase": "train_panel_loaded", "n_symbols": len(panel)}), flush=True)

    t_feat = time.time()
    # Smoke: sample-day xsection from capped windows (val still uses full-day builder).
    x = _extract_base_xsec_for_targets(panel, targets_s, meta_xsec=None)
    print(
        json.dumps(
            {
                "phase": "train_features_ready",
                "n_train": int(len(y)),
                "n_train_raw": raw_n,
                "feature_dim": int(x.shape[1]),
                "train_prior": float(y.mean()),
                "feat_seconds": float(time.time() - t_feat),
                "total_build_seconds": float(time.time() - t0),
            }
        ),
        flush=True,
    )
    meta = {
        "n": int(len(y)),
        "n_train_raw": raw_n,
        "pos_rate_full": pos_rate_full,
        "pos_rate_sampled": float(y.mean()),
        "feature_dim": int(x.shape[1]),
        "label_mode": LABEL_MODE,
        "cs_pct": CS_PCT,
        "train_window": f"{signal_start}..{signal_end}",
        "train_cap": cap,
    }
    return x, y, meta


def run_smoke(
    *,
    val_panel: Path = DEFAULT_VAL_PANEL,
    val_targets: Path = DEFAULT_VAL_TARGETS,
    train_panel: Path = DEFAULT_TRAIN_PANEL,
    train_targets: Path = DEFAULT_TRAIN_TARGETS,
    out_dir: Path = OUT_DIR,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "phase": "start",
                "label_mode": LABEL_MODE,
                "cs_pct": CS_PCT,
                "train_cap": TRAIN_CAP,
                "val_panel": str(val_panel),
                "train_panel": str(train_panel),
                "memory_safe_train": True,
            }
        ),
        flush=True,
    )

    t_val = time.time()
    val_x, val_y, val_asof, val_meta = _pack_val(val_panel, val_targets)
    val_build_s = float(time.time() - t_val)
    print(json.dumps({"phase": "val_ready", **val_meta, "build_s": val_build_s}), flush=True)

    tr_mask = (val_asof <= "2025-12-31").to_numpy()
    te_mask = (val_asof >= "2026-01-01").to_numpy()
    temporal = eval_split(
        "val_temporal_2025H2_to_2026H1",
        val_x[tr_mask],
        val_y[tr_mask],
        val_x[te_mask],
        val_y[te_mask],
    )

    train_to_val: dict[str, Any]
    try:
        tr_x, tr_y, tr_meta = _pack_train_sampled(train_panel, train_targets)
        train_to_val = eval_split("train_to_val", tr_x, tr_y, val_x, val_y)
        train_to_val["n_train_raw"] = int(tr_meta["n_train_raw"])
        train_to_val["train_window"] = tr_meta["train_window"]
        train_to_val["label_meta"] = tr_meta
    except Exception as exc:
        train_to_val = {
            "skipped": True,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
        print(json.dumps({"phase": "train_to_val_error", "error": str(exc)}), flush=True)

    if (
        isinstance(train_to_val, dict)
        and train_to_val.get("best_delta_vs_prior") is not None
        and not train_to_val.get("skipped")
    ):
        best_delta = float(train_to_val["best_delta_vs_prior"])
        best_model = train_to_val["best_model"]
        gate_passed = bool(train_to_val["gate_passed"])
        primary = "train_to_val"
        status = "PHASE_Y_SMOKE_COMPLETE"
    else:
        best_delta = float(temporal["best_delta_vs_prior"])
        best_model = temporal["best_model"]
        gate_passed = False
        primary = "train_to_val_MISSING_fallback_val_temporal"
        status = "PHASE_Y_SMOKE_PRIMARY_SKIPPED"

    temporal_gate = bool(temporal["gate_passed"])
    decision = (
        "PASS_LAUNCH_LONGER_CONFIRM"
        if gate_passed
        else "FAIL_TRY_ONE_ALTERNATE_OR_HARD_REPORT"
    )

    report = {
        "status": status,
        "phase": PHASE,
        "purpose": "phase-y-cs-top-quintile-mfe10-label-smoke-tabular-st2-stack",
        "target": f"y=1{{daily CS percentile(mfe10) >= {CS_PCT:.2f}}}",
        "mfe10_def": "max(high[T+1:T+10])/close[T]-1",
        "label_mode": LABEL_MODE,
        "cs_pct": CS_PCT,
        "gate": GATE,
        "phase_s_temporal_best_delta": PHASE_S_BEST,
        "phase_w_primary_best_delta": PHASE_W_PRIMARY,
        "budget": {
            "train_cap": TRAIN_CAP,
            "mlp_max_iter": MLP_MAX_ITER,
            "mlp_n_iter_no_change": MLP_N_ITER_NO_CHANGE,
            "memory_safe_train": True,
        },
        "primary_protocol": primary,
        "val_meta": val_meta,
        "val_build_seconds": val_build_s,
        "val_temporal": temporal,
        "val_temporal_gate_passed": temporal_gate,
        "train_to_val": train_to_val,
        "best_model": best_model,
        "best_delta_vs_prior": best_delta,
        "gate_passed": gate_passed,
        "decision": decision,
        "elapsed_seconds": float(time.time() - t0),
        "not_tokenizer_sequence": True,
        "not_ranking_ic": True,
        "not_qlib_alpha158": True,
        "not_tpu": True,
        "alternate_if_fail": [
            "y=1{mfe10>=0.08} soft absolute",
            "industry-neutral residual touch (if cheap)",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phase": "complete",
                "primary_protocol": primary,
                "best_model": best_model,
                "best_delta_vs_prior": best_delta,
                "gate_passed": gate_passed,
                "val_temporal_delta": temporal["best_delta_vs_prior"],
                "val_temporal_gate": temporal_gate,
                "decision": decision,
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run_smoke().get("status", "").startswith("PHASE_Y") else 1)
