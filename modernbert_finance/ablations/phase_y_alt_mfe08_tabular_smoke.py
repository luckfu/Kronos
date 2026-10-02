"""Phase Y alternate (cheap): soft absolute y=1{mfe10>=0.08} + S/T2 tabular stack.

Only run after CS-top primary FAIL. Same protocols/gate. NOT churn forever.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse helpers from Phase Y primary smoke
from modernbert_finance.ablations.phase_y_mfe10_cs_top_tabular_smoke import (
    GATE,
    MLP_MAX_ITER,
    MLP_N_ITER_NO_CHANGE,
    ROOT,
    SEED,
    TRAIN_CAP,
    _extract_base_xsec_for_targets,
    binary_log_loss,
    constant_prior_log_loss,
    eval_split,
)

SOFT_THR = 0.08
PHASE = "Y_alt_mfe08_soft_absolute_tabular_smoke"
OUT_DIR = ROOT / "scratch/kairos_phase_y_alt_mfe08_outputs"
DEFAULT_VAL_PANEL = ROOT / "scratch/kairos_qlib_smoke_data/val_data.pkl"
DEFAULT_VAL_TARGETS = (
    ROOT / "scratch/kairos_ablation_data/targets_dl/validation_targets.parquet"
)
DEFAULT_TRAIN_PANEL = ROOT / "scratch/kairos_qlib_smoke_data/train_data.pkl"
DEFAULT_TRAIN_TARGETS = (
    ROOT / "scratch/kairos_ablation_data/targets_dl/train_targets.parquet"
)


def _pack_val(panel: Path, targets: Path):
    from modernbert_finance.ablations.buy_profit_mfe_ablations import (
        build_mfe_buy_labels,
    )

    packed = build_mfe_buy_labels(
        panel,
        pd.read_parquet(targets),
        signal_start="2025-07-03",
        signal_end="2026-07-02",
        label_mode="absolute",
        threshold=SOFT_THR,
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
        "label_mode": "absolute",
        "threshold": SOFT_THR,
        "definition": packed["definition"]["primary"],
    }
    return comb, y, asof, meta


def _pack_train_sampled(panel_path: Path, targets_path: Path):
    from modernbert_finance.ablations._panel_io import load_panel

    t0 = time.time()
    targets = pd.read_parquet(targets_path)
    asof_ts = pd.to_datetime(targets["asof_date"])
    mask = (asof_ts >= pd.Timestamp("2023-01-01")) & (
        asof_ts <= pd.Timestamp("2024-12-31")
    )
    targets = targets.loc[mask].reset_index(drop=True)
    mfe = targets["mfe10"].to_numpy(dtype=np.float64)
    y_all = (mfe >= float(SOFT_THR)).astype(np.float64)
    raw_n = int(len(y_all))
    print(
        json.dumps(
            {
                "phase": "train_labels_ready",
                "n_train_raw": raw_n,
                "pos_rate_full": float(y_all.mean()),
                "threshold": SOFT_THR,
            }
        ),
        flush=True,
    )
    rng = np.random.default_rng(SEED)
    pos = np.where(y_all >= 0.5)[0]
    neg = np.where(y_all < 0.5)[0]
    n_pos = min(len(pos), TRAIN_CAP // 4)
    n_neg = min(len(neg), TRAIN_CAP - n_pos)
    idx = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    rng.shuffle(idx)
    targets_s = targets.iloc[idx].reset_index(drop=True)
    y = y_all[idx]
    panel = load_panel(panel_path)
    x = _extract_base_xsec_for_targets(panel, targets_s, meta_xsec=None)
    meta = {
        "n": int(len(y)),
        "n_train_raw": raw_n,
        "pos_rate_full": float(y_all.mean()),
        "pos_rate_sampled": float(y.mean()),
        "feature_dim": int(x.shape[1]),
        "threshold": SOFT_THR,
        "build_seconds": float(time.time() - t0),
    }
    print(json.dumps({"phase": "train_features_ready", **meta}), flush=True)
    return x, y, meta


def run_smoke() -> dict[str, Any]:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"phase": "start", "phase_name": PHASE, "threshold": SOFT_THR}), flush=True)
    val_x, val_y, val_asof, val_meta = _pack_val(DEFAULT_VAL_PANEL, DEFAULT_VAL_TARGETS)
    print(json.dumps({"phase": "val_ready", **val_meta}), flush=True)
    tr_m = (val_asof <= "2025-12-31").to_numpy()
    te_m = (val_asof >= "2026-01-01").to_numpy()
    temporal = eval_split(
        "val_temporal_2025H2_to_2026H1",
        val_x[tr_m],
        val_y[tr_m],
        val_x[te_m],
        val_y[te_m],
    )
    try:
        tr_x, tr_y, tr_meta = _pack_train_sampled(
            DEFAULT_TRAIN_PANEL, DEFAULT_TRAIN_TARGETS
        )
        train_to_val = eval_split("train_to_val", tr_x, tr_y, val_x, val_y)
        train_to_val["n_train_raw"] = tr_meta["n_train_raw"]
        train_to_val["label_meta"] = tr_meta
    except Exception as exc:
        train_to_val = {
            "skipped": True,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
        print(json.dumps({"phase": "train_to_val_error", "error": str(exc)}), flush=True)

    if train_to_val.get("best_delta_vs_prior") is not None and not train_to_val.get(
        "skipped"
    ):
        best_delta = float(train_to_val["best_delta_vs_prior"])
        best_model = train_to_val["best_model"]
        gate_passed = bool(train_to_val["gate_passed"])
        primary = "train_to_val"
        status = "PHASE_Y_ALT_COMPLETE"
    else:
        best_delta = float(temporal["best_delta_vs_prior"])
        best_model = temporal["best_model"]
        gate_passed = False
        primary = "train_to_val_MISSING"
        status = "PHASE_Y_ALT_PRIMARY_SKIPPED"

    report = {
        "status": status,
        "phase": PHASE,
        "target": f"y=1{{mfe10>={SOFT_THR:.2f}}}",
        "mfe10_def": "max(high[T+1:T+10])/close[T]-1",
        "label_mode": "absolute_soft",
        "threshold": SOFT_THR,
        "gate": GATE,
        "primary_protocol": primary,
        "val_meta": val_meta,
        "val_temporal": temporal,
        "val_temporal_gate_passed": bool(temporal["gate_passed"]),
        "train_to_val": train_to_val,
        "best_model": best_model,
        "best_delta_vs_prior": best_delta,
        "gate_passed": gate_passed,
        "decision": (
            "PASS_UNEXPECTED_SOFT_ABS"
            if gate_passed
            else "FAIL_HARD_REPORT_LABEL_AXIS"
        ),
        "elapsed_seconds": float(time.time() - t0),
        "prior_phase_y_cs_top_primary_delta": -0.031379848149291345,
        "not_tokenizer_sequence": True,
        "not_ranking_ic": True,
        "not_tpu": True,
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phase": "complete",
                "best_delta_vs_prior": best_delta,
                "gate_passed": gate_passed,
                "val_temporal_delta": temporal["best_delta_vs_prior"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run_smoke()["status"].startswith("PHASE_Y") else 1)
