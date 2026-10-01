"""Phase B: structure ablations — single-head / coarsened vs 8-head logistic.

CPU-only. Builds summarized window features from the val panel without torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernbert_finance.ablations.label_time_diagnostics import BINARY_HEADS
from modernbert_finance.ablations.simple_baseline import summarize_history_windows
from modernbert_finance.build_dataset import FEATURES, LOOKBACK, prepare_frame
from modernbert_finance.build_targets import WINDOW
from modernbert_finance.ablations._panel_io import load_panel

EPS = 1e-4


def _prior_ll(y: np.ndarray) -> float:
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    probs = np.column_stack([np.full(len(y), 1.0 - p), np.full(len(y), p)])
    return float(log_loss(y, probs, labels=[0, 1]))


def build_feature_matrix(
    panel: dict[str, pd.DataFrame] | Path,
    targets: pd.DataFrame,
    *,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Enumerate windows in sorted(symbol)×start order; match sidecar length."""
    if isinstance(panel, (str, Path)):
        panel = load_panel(Path(panel))
    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None

    histories: list[np.ndarray] = []
    meta_asof: list[str] = []
    for symbol in sorted(panel):
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
        dates = frame.index
        max_start = len(frame) - WINDOW + 1
        for start in range(max_start):
            asof = pd.Timestamp(dates[start + LOOKBACK - 1]).date()
            if start_date and asof < start_date:
                continue
            if end_date and asof > end_date:
                continue
            histories.append(values[start : start + LOOKBACK])
            meta_asof.append(asof.isoformat())

    if len(histories) != len(targets):
        raise ValueError(
            f"feature windows {len(histories)} != targets {len(targets)}; "
            "check signal date bounds / panel"
        )
    hist = np.stack(histories, axis=0)
    features = summarize_history_windows(hist)
    arrays = {h: targets[h].to_numpy() for h in BINARY_HEADS}
    arrays["mfe10"] = targets["mfe10"].to_numpy(dtype=np.float64)
    arrays["mae10"] = targets["mae10"].to_numpy(dtype=np.float64)
    return features, arrays, np.asarray(meta_asof)


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


def run_structure_ablations(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    seed: int = 20261001,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)

    # 1) Per-head independent logistic (reference; same as cheap ablation)
    per_head = {}
    deltas = []
    for head in BINARY_HEADS:
        per_head[head] = _fit_eval_logistic(x, targets[head], seed=seed)
        if not per_head[head].get("skipped"):
            deltas.append(per_head[head]["delta_model_minus_prior"])
    independent_macro_delta = float(np.mean(deltas)) if deltas else float("nan")

    # 2) Coarsened labels
    any_up = (
        (targets["up_003"].astype(np.int64) + targets["up_005"].astype(np.int64)
         + targets["up_008"].astype(np.int64) + targets["up_012"].astype(np.int64))
        > 0
    ).astype(np.int64)
    # Nested: up_003 already implies weaker thresholds almost; use up_005 as "medium up"
    # Coarse directional: up_005 vs down_005 exclusive ternary collapsed to two binaries
    coarsened = {
        "any_up_threshold": any_up,
        "up_005_only": targets["up_005"].astype(np.int64),
        "down_005_only": targets["down_005"].astype(np.int64),
        "direction_up_minus_down": (
            (targets["up_005"].astype(np.int64) > targets["down_005"].astype(np.int64))
        ).astype(np.int64),
        # Mutual exclusive direction when one side dominates:
        "signed_move": np.where(
            targets["mfe10"] > -targets["mae10"],
            1,
            0,
        ).astype(np.int64),  # 1 if upside excursion magnitude > downside
    }
    coarse_results = {name: _fit_eval_logistic(x, y, seed=seed) for name, y in coarsened.items()}

    # 3) Multi-output logistic (shared features, separate heads) — effect size vs independent
    y_multi = np.column_stack([targets[h].astype(np.int64) for h in BINARY_HEADS])
    # stratify not available for multi; use same indices via first head
    idx = np.arange(len(x))
    _, _, _, y0_te, idx_tr, idx_te = train_test_split(
        x,
        y_multi[:, 0],
        idx,
        test_size=0.25,
        random_state=seed,
        stratify=y_multi[:, 0],
    )
    del y0_te
    multi = make_pipeline(
        StandardScaler(),
        MultiOutputClassifier(
            LogisticRegression(max_iter=400, random_state=seed),
            n_jobs=1,
        ),
    )
    multi.fit(x[idx_tr], y_multi[idx_tr])
    # predict_proba returns list of (n, 2) arrays
    probas = multi.predict_proba(x[idx_te])
    multi_head = {}
    multi_deltas = []
    for i, head in enumerate(BINARY_HEADS):
        y_te = y_multi[idx_te, i]
        proba = probas[i][:, 1]
        prior = _prior_ll(y_te)
        ll = float(log_loss(y_te, np.column_stack([1 - proba, proba]), labels=[0, 1]))
        delta = ll - prior
        multi_deltas.append(delta)
        multi_head[head] = {
            "prior_log_loss": prior,
            "model_log_loss": ll,
            "delta_model_minus_prior": delta,
            "beats_prior": bool(ll < prior - EPS),
        }

    # 4) Equal-weight macro loss comparison (what Kairos optimizes)
    # Simulate: average of 8 independent LLs vs average if only training up_005
    single_focus = per_head["up_005"]
    dilution = {
        "independent_8head_macro_delta": independent_macro_delta,
        "multioutput_8head_macro_delta": float(np.mean(multi_deltas)),
        "single_head_up_005_delta": single_focus.get("delta_model_minus_prior"),
        "coarsened_deltas": {
            k: v.get("delta_model_minus_prior")
            for k, v in coarse_results.items()
            if not v.get("skipped")
        },
        "verdict": (
            "If single-head / coarsened deltas are similar to 8-head macro, multi-head "
            "loss is NOT the main reason the deep model stuck at prior. "
            "If coarsened >> 8-head, prefer fewer heads."
        ),
    }

    return {
        "n_samples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "per_head_independent": per_head,
        "coarsened": coarse_results,
        "multioutput_shared_features": {
            "per_head": multi_head,
            "macro_delta": float(np.mean(multi_deltas)),
            "n_train": int(len(idx_tr)),
            "n_test": int(len(idx_te)),
        },
        "dilution_summary": dilution,
    }


def probe_frozen_embeddings_feasibility() -> dict[str, Any]:
    """Only run a linear probe if local ModernBERT/Kairos weights exist."""
    search_roots = [
        Path("/workspace/Kronos/scratch"),
        Path("/workspace/Kronos/finetune"),
    ]
    keywords = ("kairos", "modernbert", "answerdotai")
    found: list[str] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for pattern in ("*.safetensors", "*.bin", "pytorch_model.bin", "model.safetensors"):
            for hit in root.rglob(pattern):
                low = str(hit).lower()
                if any(k in low for k in keywords):
                    found.append(str(hit))
                if len(found) >= 20:
                    break
    has_transformers = False
    try:
        import transformers  # noqa: F401

        has_transformers = True
    except Exception:
        has_transformers = False

    if not found:
        reason = (
            "No local Kairos/ModernBERT checkpoint under scratch/finetune; "
            "skip freeze-backbone linear probe (would require HF/Kaggle download + train)."
        )
    else:
        reason = (
            "Kairos/ModernBERT-named weight files found locally but probe skipped "
            "to avoid long GPU work in this diagnostic pass."
        )
    return {
        "ran_probe": False,
        "reason": reason,
        "has_transformers": has_transformers,
        "candidate_hits": found[:20],
    }


def run_phase_b(
    *,
    val_panel: Path,
    val_targets: Path,
    out_json: Path | None = None,
    seed: int = 20261001,
) -> dict[str, Any]:
    targets = pd.read_parquet(val_targets)
    features, arrays, _ = build_feature_matrix(val_panel, targets)
    structure = run_structure_ablations(features, arrays, seed=seed)
    probe = probe_frozen_embeddings_feasibility()
    payload = {
        "phase": "B",
        "structure": structure,
        "frozen_embedding_probe": probe,
    }
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-panel", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv)
    payload = run_phase_b(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        out_json=args.out_json,
        seed=args.seed,
    )
    print(json.dumps(payload["structure"]["dilution_summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
