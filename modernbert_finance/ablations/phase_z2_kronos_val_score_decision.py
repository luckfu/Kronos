#!/usr/bin/env python3
"""Phase Z2 decision eval: C2 Seg@179 prod scores on Kairos val → mfe10≥10%.

Primary honest protocol available without train scores: val_temporal
(early half of Kairos val dates → late half) with real path-touch mfe10.
train→val remains blocked until train-panel scores exist (multi-week GPU).
Gate: Δ log-loss vs prior ≤ −0.04. Decision-only (not ranking).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace/Kronos")
GATE = -0.04
SCORE_CANDIDATES = [
    Path("/kaggle/working/kronos_c2_kairos_val_prod_scores/predictions_prod_t065_p80_n5.csv.gz"),
    ROOT / "scratch/kairos_phase_z2_outputs/predictions_prod_t065_p80_n5.csv.gz",
    Path("/workspace/kaggle_c2_kairos_val_prod_scores/kronos_c2_kairos_val_prod_scores/predictions_prod_t065_p80_n5.csv.gz"),
]
VAL_TARGETS = ROOT / "scratch/kairos_ablation_data/targets_dl/validation_targets.parquet"
OUT_DIR = ROOT / "scratch/kairos_phase_z2_outputs"
RESULTS_JSON = ROOT / "modernbert_finance/ablations/kairos_phase_z2_kronos_val_score_decision_results.json"


def find_scores() -> Path:
    for p in SCORE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "C2 Kairos-val prod scores not found; download kernel output first. "
        f"tried={[str(p) for p in SCORE_CANDIDATES]}"
    )


def prior_log_loss(y: np.ndarray, prior: float) -> float:
    p = np.clip(prior, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _eval_preds(y_te: np.ndarray, p: np.ndarray, prior_ll: float) -> dict[str, Any]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-(y_te * np.log(p) + (1 - y_te) * np.log(1 - p)).mean())
    return {
        "delta_vs_prior": ll - prior_ll,
        "model_log_loss": ll,
        "prior_log_loss": prior_ll,
        "gate_passed": bool((ll - prior_ll) <= GATE),
        "mean_pred": float(p.mean()),
        "pos_rate_test": float(y_te.mean()),
    }


def fit_logistic(x_tr, y_tr, x_te, C=1.0):
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_tr)
    xte = scaler.transform(x_te)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    clf.fit(xtr, y_tr)
    return clf.predict_proba(xte)[:, 1], f"logistic_C{C}"


def fit_isotonic(score_tr, y_tr, score_te):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(score_tr, y_tr)
    return iso.predict(score_te), "isotonic_on_score"


def fit_threshold(score_tr, y_tr, score_te):
    thr = float(np.quantile(score_tr, 0.50))
    buy = score_tr >= thr
    p_buy = float(y_tr[buy].mean()) if buy.any() else float(y_tr.mean())
    p_skip = float(y_tr[~buy].mean()) if (~buy).any() else float(y_tr.mean())
    p_te = np.where(score_te >= thr, p_buy, p_skip)
    return p_te, "threshold_q0.50_calibrated", thr, p_buy, p_skip


def fit_mlp(x_tr, y_tr, x_te):
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_tr)
    xte = scaler.transform(x_te)
    t0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=0)
    clf.fit(xtr, y_tr)
    return clf.predict_proba(xte)[:, 1], "mlp_32_16", time.time() - t0


def eval_split(df: pd.DataFrame, tr_mask: np.ndarray, te_mask: np.ndarray, split_name: str) -> dict[str, Any]:
    y = (df["mfe10"].to_numpy(dtype=np.float64) >= 0.10).astype(np.int32)
    score = df["predicted_return_10d"].to_numpy(dtype=np.float64)
    pct = df.groupby("asof_date")["predicted_return_10d"].rank(pct=True).to_numpy(dtype=np.float64)
    # xsec: score + daily pctile + size_decile if present
    feats = [score, pct]
    if "size_decile" in df.columns:
        feats.append(df["size_decile"].to_numpy(dtype=np.float64))
    x = np.column_stack(feats)

    y_tr, y_te = y[tr_mask], y[te_mask]
    prior = float(y_tr.mean())
    prior_ll = prior_log_loss(y_te, prior)

    models: dict[str, Any] = {}
    p, name = fit_logistic(score[tr_mask].reshape(-1, 1), y_tr, score[te_mask].reshape(-1, 1), C=1.0)
    models["logistic_score"] = {**_eval_preds(y_te, p, prior_ll), "model": name}
    p, name = fit_logistic(score[tr_mask].reshape(-1, 1), y_tr, score[te_mask].reshape(-1, 1), C=0.01)
    models["logistic_score_C0.01"] = {**_eval_preds(y_te, p, prior_ll), "model": name}
    p, name = fit_logistic(pct[tr_mask].reshape(-1, 1), y_tr, pct[te_mask].reshape(-1, 1), C=1.0)
    models["logistic_cs_pctile"] = {**_eval_preds(y_te, p, prior_ll), "model": name}
    p, name = fit_isotonic(score[tr_mask], y_tr, score[te_mask])
    models["isotonic_score"] = {**_eval_preds(y_te, p, prior_ll), "model": name}
    p, name, thr, pb, ps = fit_threshold(score[tr_mask], y_tr, score[te_mask])
    models["threshold_calibrated"] = {
        **_eval_preds(y_te, p, prior_ll),
        "model": name,
        "threshold": thr,
        "p_buy": pb,
        "p_skip": ps,
    }
    p, name = fit_logistic(x[tr_mask], y_tr, x[te_mask], C=0.01)
    models["logistic_score_plus_xsec"] = {**_eval_preds(y_te, p, prior_ll), "model": name}
    p, name, secs = fit_mlp(x[tr_mask], y_tr, x[te_mask])
    models["mlp_score_plus_xsec"] = {**_eval_preds(y_te, p, prior_ll), "model": name, "fit_seconds": secs}

    ranked = sorted(
        ((k, v["delta_vs_prior"]) for k, v in models.items()),
        key=lambda kv: kv[1],
    )
    best_model, best_delta = ranked[0]
    return {
        "split": split_name,
        "n_train": int(tr_mask.sum()),
        "n_test": int(te_mask.sum()),
        "train_prior": prior,
        "prior_log_loss": prior_ll,
        "pos_train": prior,
        "pos_test": float(y_te.mean()),
        "models": models,
        "best_model": best_model,
        "best_delta_vs_prior": best_delta,
        "gate_passed": bool(best_delta <= GATE),
        "ranked": [{"model": k, "delta": d} for k, d in ranked],
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = find_scores()
    scores = pd.read_csv(score_path)
    scores["asof_date"] = scores["asof_date"].astype(str)
    vt = pd.read_parquet(VAL_TARGETS)
    vt["asof_date"] = vt["asof_date"].astype(str)
    df = scores.merge(vt[["symbol", "asof_date", "mfe10"]], on=["symbol", "asof_date"], how="inner")
    if len(df) == 0:
        raise RuntimeError("zero join between C2 scores and Kairos val mfe10")

    dates = sorted(df["asof_date"].unique())
    mid = len(dates) // 2
    early, late = dates[:mid], dates[mid:]
    asof = df["asof_date"]
    tr = asof.isin(early).to_numpy()
    te = asof.isin(late).to_numpy()
    val_temporal = eval_split(
        df,
        tr,
        te,
        split_name=f"kairos_val_temporal_{early[0]}_to_{early[-1]}__vs__{late[0]}_to_{late[-1]}",
    )
    val_temporal["early_dates"] = early
    val_temporal["late_dates"] = late

    report = {
        "status": "PHASE_Z2_DECISION_EVAL",
        "phase": "Z2_c2_seg179_scores_on_kairos_val",
        "score_path": str(score_path),
        "n_joined": int(len(df)),
        "n_dates": int(len(dates)),
        "asof_start": dates[0],
        "asof_end": dates[-1],
        "pos_rate": float((df["mfe10"] >= 0.10).mean()),
        "gate": GATE,
        "train_to_val": {
            "status": "BLOCKED",
            "reason": (
                "Train-panel C2 Seg@179 scores not materialized; "
                "full train rescore is multi-week GPU. Z2 delivered Kairos-val coverage only."
            ),
            "gate_passed": False,
            "best_delta_vs_prior": None,
        },
        "val_temporal": val_temporal,
        "best_delta_vs_prior": val_temporal["best_delta_vs_prior"],
        "gate_passed": val_temporal["gate_passed"],
        "decision": (
            "PASS_VAL_TEMPORAL"
            if val_temporal["gate_passed"]
            else "FAIL_VAL_TEMPORAL_or_await_scores"
        ),
        "kernel": "https://www.kaggle.com/code/user281434/kronos-c2-seg179-kairos-val-prod-scores",
        "elapsed_seconds": time.time() - t0,
        "not_ranking_ic": True,
        "not_tpu_wip": True,
    }
    OUT_DIR.joinpath("report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    RESULTS_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "joined": report["n_joined"],
        "best_model": val_temporal["best_model"],
        "best_delta": val_temporal["best_delta_vs_prior"],
        "gate_passed": val_temporal["gate_passed"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
