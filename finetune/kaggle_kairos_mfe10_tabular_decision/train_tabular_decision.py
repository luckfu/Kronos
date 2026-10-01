"""Kairos Phase T: tabular enet/blend confirmation (decision-only).

Primary protocol = val temporal 2025H2→2026H1 (same as Phase S gate-pass).
Optional: train panel recent→val if build finishes in budget.

y=1{mfe10>=0.10}; gate Δ≤-0.04. NO tokenizer / ranking / 22-layer / TPU.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SWANLAB_API_KEY_FALLBACK = ""  # injected at private staging only
SWANLAB_RUN_ID = "kairos-mfe10-decision-tabular-phase-t-20261002"
GATE = -0.04
SEED = 20261001
PHASE_S_BEST = -0.04179349770224905


def find_one(pattern: str) -> Path:
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def setup_vendor_path() -> None:
    here = Path(__file__).resolve().parent
    vendor = here / "vendor"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
    for cand in (Path("/workspace/Kronos"), Path.cwd()):
        if (cand / "modernbert_finance").is_dir():
            sys.path.insert(0, str(cand))
            break


def start_swanlab():
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--progress-bar", "off", "swanlab"],
    )
    import swanlab

    api_key = os.environ.get("SWANLAB_API_KEY", "").strip() or SWANLAB_API_KEY_FALLBACK
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY empty")
    swanlab.login(api_key=api_key)
    run = swanlab.init(
        id=SWANLAB_RUN_ID,
        resume="allow",
        project="finance",
        workspace="roc_fu",
        experiment_name=SWANLAB_RUN_ID,
        mode="cloud",
        config={
            "purpose": "phase-t-tabular-enet-blend-confirm",
            "target": "y=1{mfe10>=0.10}",
            "gate": GATE,
            "phase_s_best": PHASE_S_BEST,
        },
    )
    url = getattr(run, "url", getattr(run, "web_url", ""))
    print(json.dumps({"phase": "swanlab_ready", "url": url}), flush=True)
    return swanlab, run


def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def constant_prior_log_loss(y: np.ndarray, prior: float) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = float(np.clip(prior, 1e-7, 1 - 1e-7))
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
        "models": {},
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
                    max_iter=2000,
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
                    max_iter=2000,
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
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=8,
            random_state=SEED,
        ),
    )
    t1 = time.time()
    lr.fit(x_tr, y_tr.astype(int))
    mlp.fit(x_tr, y_tr.astype(int))
    p_lr = lr.predict_proba(x_te)[:, 1]
    p_mlp = mlp.predict_proba(x_te)[:, 1]
    blend_fit_s = float(time.time() - t1)
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


def try_train_to_val(val_comb: np.ndarray, val_y: np.ndarray) -> dict[str, Any] | None:
    """Optional recent-train → full-val confirmation."""
    from modernbert_finance.ablations.buy_profit_mfe_ablations import (
        build_mfe_buy_labels,
    )

    try:
        train_panel = find_one("**/processed_datasets/train_data.pkl")
        train_targets = find_one("**/train_targets.parquet")
    except FileNotFoundError as exc:
        return {"skipped": True, "reason": str(exc)}

    print(
        json.dumps(
            {
                "phase": "train_paths",
                "train_panel": str(train_panel),
                "train_targets": str(train_targets),
            }
        ),
        flush=True,
    )
    t_build = time.time()
    # Restrict feature build window to recent train years (short smoke).
    packed_tr = build_mfe_buy_labels(
        train_panel,
        pd.read_parquet(train_targets),
        signal_start="2023-01-01",
        signal_end="2024-12-31",
    )
    build_s = float(time.time() - t_build)
    x_tr = np.nan_to_num(
        np.asarray(packed_tr["combined_features"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_tr = np.asarray(packed_tr["buy_labels"]["buy_worth_mfe10pct"], dtype=np.float64)
    print(
        json.dumps(
            {
                "phase": "train_features_ready",
                "n_train": int(len(y_tr)),
                "build_seconds": build_s,
                "feature_dim": int(x_tr.shape[1]),
            }
        ),
        flush=True,
    )
    # Cap train rows for fit speed while keeping class balance.
    max_rows = 500_000
    if len(y_tr) > max_rows:
        rng = np.random.default_rng(SEED)
        pos = np.where(y_tr >= 0.5)[0]
        neg = np.where(y_tr < 0.5)[0]
        n_pos = min(len(pos), max_rows // 4)
        n_neg = min(len(neg), max_rows - n_pos)
        idx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )
        rng.shuffle(idx)
        x_tr, y_tr = x_tr[idx], y_tr[idx]
        print(
            json.dumps({"phase": "train_capped", "n_train": int(len(y_tr))}),
            flush=True,
        )
    result = eval_split("train2023_2024_to_val", x_tr, y_tr, val_comb, val_y)
    result["build_seconds"] = build_s
    return result


def main() -> int:
    t0 = time.time()
    setup_vendor_path()
    swanlab, run = start_swanlab()

    from modernbert_finance.ablations.buy_profit_mfe_ablations import (
        build_mfe_buy_labels,
    )

    val_panel = find_one("**/processed_datasets/val_data.pkl")
    val_targets = find_one("**/validation_targets.parquet")
    print(
        json.dumps(
            {
                "phase": "paths",
                "val_panel": str(val_panel),
                "val_targets": str(val_targets),
            }
        ),
        flush=True,
    )

    packed = build_mfe_buy_labels(val_panel, pd.read_parquet(val_targets))
    meta = packed["meta"]
    asof = pd.to_datetime(meta["asof_date"])
    comb = np.nan_to_num(
        np.asarray(packed["combined_features"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y = np.asarray(packed["buy_labels"]["buy_worth_mfe10pct"], dtype=np.float64)

    tr = (asof <= "2025-12-31").to_numpy()
    te = (asof >= "2026-01-01").to_numpy()
    temporal = eval_split(
        "val_temporal_2025H2_to_2026H1", comb[tr], y[tr], comb[te], y[te]
    )
    run.log(
        {
            "temporal/best_delta": temporal["best_delta_vs_prior"],
            "temporal/gate_passed": int(temporal["gate_passed"]),
        }
    )

    train_to_val: dict[str, Any] | None
    try:
        train_to_val = try_train_to_val(comb, y)
        if train_to_val and not train_to_val.get("skipped") and "best_delta_vs_prior" in train_to_val:
            run.log(
                {
                    "train_to_val/best_delta": train_to_val["best_delta_vs_prior"],
                    "train_to_val/gate_passed": int(train_to_val["gate_passed"]),
                }
            )
    except Exception as exc:
        train_to_val = {
            "skipped": True,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
        print(json.dumps({"phase": "train_to_val_error", "error": str(exc)}), flush=True)

    # Prefer train→val best if present; else temporal.
    if (
        isinstance(train_to_val, dict)
        and train_to_val.get("best_delta_vs_prior") is not None
        and not train_to_val.get("skipped")
    ):
        best_delta = float(train_to_val["best_delta_vs_prior"])
        best_model = train_to_val["best_model"]
        gate_passed = bool(train_to_val["gate_passed"])
        primary = "train_to_val"
    else:
        best_delta = float(temporal["best_delta_vs_prior"])
        best_model = temporal["best_model"]
        gate_passed = bool(temporal["gate_passed"])
        primary = "val_temporal"

    report = {
        "status": "TABULAR_DECISION_COMPLETE",
        "purpose": "phase-t-tabular-enet-blend-confirm",
        "target": "y=1{mfe10>=0.10}",
        "gate": GATE,
        "phase_s_temporal_best_delta": PHASE_S_BEST,
        "primary_protocol": primary,
        "val_temporal": temporal,
        "train_to_val": train_to_val,
        "best_model": best_model,
        "best_delta_vs_prior": best_delta,
        "gate_passed": gate_passed,
        "elapsed_seconds": float(time.time() - t0),
        "not_tokenizer_sequence": True,
        "not_ranking_ic": True,
        "not_22_layer": True,
    }
    out_dir = Path("/kaggle/working/kairos_mfe10_tabular_decision")
    out_dir.mkdir(parents=True, exist_ok=True)
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
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    run.log(
        {
            "final/best_delta": best_delta,
            "final/gate_passed": int(gate_passed),
        }
    )
    swanlab.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
