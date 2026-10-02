"""Phase Z: Kronos C2 Best@Seg179 production scores → P(mfe10≥10%) decision smoke.

DECISION buy/not (not ranking). Gate: Δ log loss vs prior ≤ −0.04 on train→val.
Report val_temporal analog where possible.

Score inventory (repo/artifacts):
  - C2 Best Seg@179 prod decode locked: T=0.65 top_p=0.8 N=5
  - Materialized OOS only: 2026-08-11..2026-09-03 (18d sealed)
  - Kairos train asof ..2024-12-31; val 2025-07-03..2026-07-02
  - date_overlap with production scores = False
  - Local cosine_c2_best dir has README/config only (no model.safetensors)
  → train→val with real Kronos features is BLOCKED without full C2 inference
    (skipped: no TPU WIP / long train / weights on box).

Runnable surface: sealed OOS calibrated decision rule on predicted_return_10d
(+ optional path_vol / dispersion / size). Label = path-touch approx
max(actual_return_d1..d10)≥0.10 (dump lacks high-price path).
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
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261002
GATE = -0.04
MFE_THR = 0.10
EPS = 1e-4
PHASE = "Z_kronos_score_decision_smoke"
OUT_DIR = ROOT / "scratch/kairos_phase_z_outputs"

C2_PROD = Path(
    "/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos/"
    "predictions_prod_t065_p80_n5.csv.gz"
)
C2_PROD_REPO = (
    ROOT / "finetune/kaggle_c2_top20_consensus_oos/predictions_prod_t065_p80_n5.csv.gz"
)
C2_RANK = Path(
    "/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos/"
    "predictions_rank_t060_p90_n16.csv.gz"
)
DECODE_LOCK = ROOT / "finetune/reports/c2_best_production_decode_lock.json"
C2_CKPT_META = ROOT / "scratch/kronos_small_0_1_cosine_c2_best"
VAL_TARGETS = ROOT / "scratch/kairos_ablation_data/targets_dl/validation_targets.parquet"
TRAIN_TARGETS = ROOT / "scratch/kairos_ablation_data/targets_dl/train_targets.parquet"


def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def constant_prior_log_loss(y_tr: np.ndarray, y_te: np.ndarray) -> float:
    prior = float(np.clip(np.mean(y_tr), EPS, 1.0 - EPS))
    return binary_log_loss(np.full(len(y_te), prior), y_te)


def _path_touch(df: pd.DataFrame) -> np.ndarray:
    cols = [f"actual_return_d{i}" for i in range(1, 11)]
    return df[cols].to_numpy(dtype=np.float64).max(axis=1)


def inventory_scores() -> dict[str, Any]:
    paths = {
        "c2_prod_18d_workspace": C2_PROD,
        "c2_prod_18d_repo_copy": C2_PROD_REPO,
        "c2_rank_18d_workspace": C2_RANK,
        "decode_lock": DECODE_LOCK,
        "c2_ckpt_meta_dir": C2_CKPT_META,
    }
    found: dict[str, Any] = {}
    for name, p in paths.items():
        exists = p.exists()
        entry: dict[str, Any] = {"path": str(p), "exists": bool(exists)}
        if exists and p.is_file() and str(p).endswith((".csv.gz", ".csv")):
            try:
                df = pd.read_csv(p, usecols=["asof_date", "symbol", "checkpoint"])
                entry.update(
                    {
                        "n_rows": int(len(df)),
                        "n_dates": int(df["asof_date"].nunique()),
                        "asof_start": str(df["asof_date"].min()),
                        "asof_end": str(df["asof_date"].max()),
                        "checkpoint_values": sorted(df["checkpoint"].astype(str).unique().tolist()),
                    }
                )
            except Exception as e:  # noqa: BLE001
                entry["read_error"] = str(e)
        if exists and p.is_dir():
            files = sorted(x.name for x in p.iterdir())
            entry["files"] = files
            entry["has_model_safetensors"] = "model.safetensors" in files
        found[name] = entry

    lock = {}
    if DECODE_LOCK.exists():
        lock = json.loads(DECODE_LOCK.read_text())

    # Coverage vs Kairos panels
    val_asof = None
    train_asof = None
    try:
        vt = pd.read_parquet(VAL_TARGETS, columns=["asof_date"])
        val_asof = (str(vt["asof_date"].min()), str(vt["asof_date"].max()), int(len(vt)))
        tt = pd.read_parquet(TRAIN_TARGETS, columns=["asof_date"])
        train_asof = (str(tt["asof_date"].min()), str(tt["asof_date"].max()), int(len(tt)))
    except Exception as e:  # noqa: BLE001
        found["targets_error"] = str(e)

    score_start = found.get("c2_prod_18d_workspace", {}).get("asof_start")
    score_end = found.get("c2_prod_18d_workspace", {}).get("asof_end")
    overlap = False
    if score_start and val_asof:
        # string ISO compare works for YYYY-MM-DD
        overlap = not (score_end < val_asof[0] or score_start > val_asof[1])

    return {
        "artifacts": found,
        "decode_lock_summary": {
            "checkpoint_label": lock.get("checkpoint", {}).get("label"),
            "segment": lock.get("checkpoint", {}).get("segment"),
            "production_decode": lock.get("production_decode"),
            "sealed_window": lock.get("sealed_window"),
        },
        "kairos_val_asof": (
            {"start": val_asof[0], "end": val_asof[1], "n": val_asof[2]}
            if val_asof
            else None
        ),
        "kairos_train_asof": (
            {"start": train_asof[0], "end": train_asof[1], "n": train_asof[2]}
            if train_asof
            else None
        ),
        "date_overlap_scores_vs_val": overlap,
        "train_to_val_kronos_features": "BLOCKED_no_score_coverage",
        "blocker": (
            "C2 Best Seg@179 production scores only on sealed OOS "
            f"{score_start}..{score_end}; Kairos val "
            f"{val_asof[0] if val_asof else '?'}..{val_asof[1] if val_asof else '?'}; "
            "no model.safetensors on box for reproducible local rescore; "
            "full C2 inference skipped (no TPU WIP / long train)."
        ),
    }


def _pack_oos(pred_path: Path) -> dict[str, Any]:
    usecols = [
        "symbol",
        "asof_date",
        "sector",
        "size_decile",
        "return_10d",
        "predicted_return_10d",
        "predicted_path_vol",
        "predicted_terminal_dispersion",
    ] + [f"actual_return_d{i}" for i in range(1, 11)]
    df = pd.read_csv(pred_path, usecols=usecols)
    path_touch = _path_touch(df)
    y = (path_touch >= MFE_THR).astype(np.float64)
    score = df["predicted_return_10d"].to_numpy(dtype=np.float64)
    pct = df.groupby("asof_date")["predicted_return_10d"].rank(pct=True).to_numpy(
        dtype=np.float64
    )
    vol = df["predicted_path_vol"].to_numpy(dtype=np.float64)
    disp = df["predicted_terminal_dispersion"].to_numpy(dtype=np.float64)
    size = df["size_decile"].to_numpy(dtype=np.float64)
    # optional few xsection from dump itself
    x_score = score.reshape(-1, 1)
    x_pct = pct.reshape(-1, 1)
    x_small = np.column_stack([score, pct, vol, disp, size])
    x_small = np.nan_to_num(x_small, nan=0.0, posinf=0.0, neginf=0.0)
    asof = pd.to_datetime(df["asof_date"])
    return {
        "df": df,
        "y": y,
        "path_touch": path_touch,
        "score": score,
        "pct": pct,
        "x_score": x_score,
        "x_pct": x_pct,
        "x_small": x_small,
        "asof": asof,
        "label_note": (
            "y=1{max(actual_return_d1..d10)>=0.10}; dump uses cumulative close "
            "returns — not high-price mfe10"
        ),
        "pos_rate": float(y.mean()),
        "n": int(len(y)),
        "n_dates": int(asof.nunique()),
        "asof_start": str(asof.min().date()),
        "asof_end": str(asof.max().date()),
    }


def _eval_preds(y_te: np.ndarray, p: np.ndarray, prior_ll: float) -> dict[str, Any]:
    mll = binary_log_loss(p, y_te)
    delta = mll - prior_ll
    return {
        "delta_vs_prior": float(delta),
        "model_log_loss": float(mll),
        "prior_log_loss": float(prior_ll),
        "gate_passed": bool(delta <= GATE),
        "mean_pred": float(np.mean(p)),
        "pos_rate_test": float(np.mean(y_te)),
    }


def fit_logistic(x_tr, y_tr, x_te, y_te, *, C: float = 1.0) -> dict[str, Any]:
    prior_ll = constant_prior_log_loss(y_tr, y_te)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=500,
            solver="lbfgs",
            random_state=SEED,
        ),
    )
    pipe.fit(x_tr, y_tr)
    p = pipe.predict_proba(x_te)[:, 1]
    out = _eval_preds(y_te, p, prior_ll)
    out["model"] = f"logistic_C{C}"
    return out


def fit_isotonic(score_tr, y_tr, score_te, y_te) -> dict[str, Any]:
    prior_ll = constant_prior_log_loss(y_tr, y_te)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
    # need both classes
    if len(np.unique(y_tr)) < 2:
        p = np.full(len(y_te), float(np.mean(y_tr)))
    else:
        iso.fit(score_tr, y_tr)
        p = iso.predict(score_te)
    out = _eval_preds(y_te, p, prior_ll)
    out["model"] = "isotonic_on_score"
    return out


def fit_threshold_calibrated(score_tr, y_tr, score_te, y_te) -> dict[str, Any]:
    """Daily-pctile threshold selected on train by log-loss of two-level calib."""
    prior_ll = constant_prior_log_loss(y_tr, y_te)
    # work in percentile space for stability
    # reconstruct pct on each split independently for threshold meaning
    # Here score_tr/te already may be raw; we use raw score quantiles on train.
    qs = np.linspace(0.5, 0.95, 10)
    best = None
    for q in qs:
        thr = float(np.quantile(score_tr, q))
        buy_tr = score_tr >= thr
        # two-level probabilities
        p_buy = float(np.clip(y_tr[buy_tr].mean() if buy_tr.any() else y_tr.mean(), EPS, 1 - EPS))
        p_skip = float(np.clip(y_tr[~buy_tr].mean() if (~buy_tr).any() else y_tr.mean(), EPS, 1 - EPS))
        p_tr = np.where(buy_tr, p_buy, p_skip)
        ll = binary_log_loss(p_tr, y_tr)
        cand = (ll, q, thr, p_buy, p_skip)
        if best is None or ll < best[0]:
            best = cand
    assert best is not None
    _, q, thr, p_buy, p_skip = best
    p_te = np.where(score_te >= thr, p_buy, p_skip)
    out = _eval_preds(y_te, p_te, prior_ll)
    out["model"] = f"threshold_q{q:.2f}_calibrated"
    out["threshold"] = float(thr)
    out["p_buy"] = float(p_buy)
    out["p_skip"] = float(p_skip)
    return out


def fit_mlp_small(x_tr, y_tr, x_te, y_te) -> dict[str, Any]:
    prior_ll = constant_prior_log_loss(y_tr, y_te)
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=80,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=SEED,
            learning_rate_init=1e-3,
        ),
    )
    t0 = time.time()
    pipe.fit(x_tr, y_tr)
    p = pipe.predict_proba(x_te)[:, 1]
    out = _eval_preds(y_te, p, prior_ll)
    out["model"] = "mlp_32_16"
    out["fit_seconds"] = float(time.time() - t0)
    return out


def eval_split(
    packed: dict[str, Any],
    tr_mask: np.ndarray,
    te_mask: np.ndarray,
    *,
    split_name: str,
) -> dict[str, Any]:
    y = packed["y"]
    y_tr, y_te = y[tr_mask], y[te_mask]
    models: dict[str, Any] = {}
    models["logistic_score"] = fit_logistic(
        packed["x_score"][tr_mask], y_tr, packed["x_score"][te_mask], y_te, C=1.0
    )
    models["logistic_score_C0.01"] = fit_logistic(
        packed["x_score"][tr_mask], y_tr, packed["x_score"][te_mask], y_te, C=0.01
    )
    models["logistic_cs_pctile"] = fit_logistic(
        packed["x_pct"][tr_mask], y_tr, packed["x_pct"][te_mask], y_te, C=1.0
    )
    models["isotonic_score"] = fit_isotonic(
        packed["score"][tr_mask], y_tr, packed["score"][te_mask], y_te
    )
    models["threshold_calibrated"] = fit_threshold_calibrated(
        packed["score"][tr_mask], y_tr, packed["score"][te_mask], y_te
    )
    models["logistic_score_plus_xsec"] = fit_logistic(
        packed["x_small"][tr_mask], y_tr, packed["x_small"][te_mask], y_te, C=0.01
    )
    models["mlp_score_plus_xsec"] = fit_mlp_small(
        packed["x_small"][tr_mask], y_tr, packed["x_small"][te_mask], y_te
    )
    ranked = sorted(
        ((k, v["delta_vs_prior"]) for k, v in models.items()),
        key=lambda kv: kv[1],
    )
    best_name, best_delta = ranked[0]
    return {
        "split": split_name,
        "n_train": int(tr_mask.sum()),
        "n_test": int(te_mask.sum()),
        "train_prior": float(y_tr.mean()),
        "prior_log_loss": float(constant_prior_log_loss(y_tr, y_te)),
        "pos_train": float(y_tr.mean()),
        "pos_test": float(y_te.mean()),
        "models": models,
        "best_model": best_name,
        "best_delta_vs_prior": float(best_delta),
        "gate_passed": bool(best_delta <= GATE),
        "ranked": [{"model": k, "delta": float(d)} for k, d in ranked],
    }


def run() -> dict[str, Any]:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = inventory_scores()
    print(json.dumps({"phase": "inventory", **{k: inv[k] for k in ("blocker", "date_overlap_scores_vs_val", "decode_lock_summary")}}, ensure_ascii=False), flush=True)

    pred_path = C2_PROD if C2_PROD.exists() else C2_PROD_REPO
    if not pred_path.exists():
        report = {
            "status": "PHASE_Z_BLOCKED",
            "phase": PHASE,
            "decision": "HARD_REPORT_NO_SCORES",
            "score_inventory": inv,
            "gate": GATE,
            "elapsed_seconds": float(time.time() - t0),
        }
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    packed = _pack_oos(pred_path)
    print(
        json.dumps(
            {
                "phase": "oos_packed",
                "n": packed["n"],
                "pos_rate": packed["pos_rate"],
                "asof": f"{packed['asof_start']}..{packed['asof_end']}",
                "label_note": packed["label_note"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Protocol A: stratified random (matches Phase H teacher logit)
    idx = np.arange(packed["n"])
    i_tr, i_te = train_test_split(
        idx, test_size=0.25, random_state=SEED, stratify=packed["y"]
    )
    tr_a = np.zeros(packed["n"], dtype=bool)
    te_a = np.zeros(packed["n"], dtype=bool)
    tr_a[i_tr] = True
    te_a[i_te] = True
    sealed_random = eval_split(packed, tr_a, te_a, split_name="sealed_oos_stratified_75_25")

    # Protocol B: temporal early→late dates (val_temporal analog on sealed window)
    dates = sorted(packed["asof"].dt.strftime("%Y-%m-%d").unique())
    mid = len(dates) // 2
    early = set(dates[:mid])
    late = set(dates[mid:])
    asof_s = packed["asof"].dt.strftime("%Y-%m-%d")
    tr_b = asof_s.isin(early).to_numpy()
    te_b = asof_s.isin(late).to_numpy()
    sealed_temporal = eval_split(
        packed,
        tr_b,
        te_b,
        split_name=f"sealed_oos_temporal_{dates[0]}_to_{dates[mid-1]}__vs__{dates[mid]}_to_{dates[-1]}",
    )
    sealed_temporal["early_dates"] = dates[:mid]
    sealed_temporal["late_dates"] = dates[mid:]

    # Primary train→val: blocked
    train_to_val = {
        "split": "train_to_val",
        "status": "BLOCKED",
        "reason": inv["blocker"],
        "gate_passed": False,
        "best_delta_vs_prior": None,
        "n_train": None,
        "n_test": None,
        "note": (
            "Cannot join C2 Seg@179 scores onto Kairos train/val panels; "
            "no local weights to rescore. Primary gate unevaluable with real "
            "Kronos features → treat as FAIL for Phase Z pass criteria."
        ),
    }

    # Decision: sealed OOS is the only runnable surface; still far from gate
    best_sealed = min(
        sealed_random["best_delta_vs_prior"],
        sealed_temporal["best_delta_vs_prior"],
    )
    any_gate = bool(sealed_random["gate_passed"] or sealed_temporal["gate_passed"])

    if train_to_val["status"] == "BLOCKED" and not any_gate:
        decision = "HARD_REPORT_FAIL"
        next_axis = [
            "STOP binary mfe10≥10% decision path (Y label churn + Z Kronos score both fail/blocked)",
            "OR last different axis: materialize C2 Seg@179 scores on Kairos val dates (full inference cost; not TPU WIP reuse) then re-run logistic head — only if parent explicitly budgets it",
            "OR abandon buy/not for this label family; ranking product already shown alive on return_10d but parent excluded IC/TopK as success",
        ]
    elif any_gate and train_to_val["status"] == "BLOCKED":
        decision = "FAIL_PRIMARY_BLOCKED_SEALED_PASS_ONLY"
        next_axis = [
            "Sealed OOS pass is not train→val; need score materialization before confirm/Kaggle"
        ]
    else:
        decision = "PASS_CONSIDER_CONFIRM"
        next_axis = ["longer confirm / Kaggle"]

    report: dict[str, Any] = {
        "status": "PHASE_Z_SMOKE_COMPLETE",
        "phase": PHASE,
        "purpose": "kronos-c2-best-seg179-score-as-calibrated-decision-rule-for-mfe10",
        "target": "y=1{mfe10>=0.10} (path-touch; OOS approx via max close path)",
        "mfe10_def": "ideal: max(high[T+1:T+10])/close[T]-1; OOS dump approx: max(actual_return_d1..d10)",
        "gate": GATE,
        "primary_protocol": "train_to_val",
        "score_source": {
            "checkpoint": "cosine_c2_best",
            "segment": 179,
            "decode": "prod_t065_p80_n5",
            "pred_path": str(pred_path),
        },
        "score_inventory": inv,
        "oos_meta": {
            "n": packed["n"],
            "pos_rate": packed["pos_rate"],
            "n_dates": packed["n_dates"],
            "asof_start": packed["asof_start"],
            "asof_end": packed["asof_end"],
            "label_note": packed["label_note"],
        },
        "train_to_val": train_to_val,
        "val_temporal": sealed_temporal,  # closest available temporal report
        "sealed_oos_random": sealed_random,
        "sealed_oos_temporal": sealed_temporal,
        "best_sealed_delta_vs_prior": float(best_sealed),
        "gate_passed_primary": False,
        "gate_passed_any_sealed": any_gate,
        "decision": decision,
        "next_axis_or_stop": next_axis,
        "not_ranking_ic": True,
        "not_tpu": True,
        "not_modernbert_long_train": True,
        "not_timeseries_library": True,
        "stop_label_family_churn": True,
        "elapsed_seconds": float(time.time() - t0),
        "phase_h_ref_teacher_logit_path_touch_delta": -0.003539513075496159,
        "phase_y_ref_train_to_val_best_delta": -0.031379848149291345,
        "phase_w_ref_train_to_val_best_delta": -0.028500119076822594,
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"phase": "done", "decision": decision, "best_sealed_delta": best_sealed, "elapsed": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    run()
