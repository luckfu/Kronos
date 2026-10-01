"""Helpers for Kairos short sidecar: y=1{mfe10>=0.10} (path MFE, not ctc).

Target definition (decision / Phase G2):
  mfe10 = max(high[T+1:T+10]) / close[T] - 1
  y     = 1{ mfe10 >= 0.10 }

This is NOT close-to-close fwd_ret_10, and NOT the old 8-head R2 recipe
(up_003/005/008/012 + down_*).
"""

from __future__ import annotations

from typing import Any

import numpy as np

MFE10_THRESHOLD = 0.10
MFE10_DEF = "max(high[T+1:T+10]) / close[T] - 1"
TARGET_NAME = "buy_worth_mfe10pct"
# Absolute gate vs constant prior (Phase G2 / handoff).
GATE_DELTA_VS_PRIOR = -0.04
# Early-stop: stuck within this band of prior for PATIENCE consecutive evals.
PRIOR_STUCK_TOL = 1e-3
PRIOR_STUCK_PATIENCE = 2
# Val positive rate observed on official holdout (~25%).
EXPECTED_VAL_POS_RATE = 0.2529


def binary_from_mfe10(
    mfe10: np.ndarray, threshold: float = MFE10_THRESHOLD
) -> np.ndarray:
    """Return float32 {0,1} labels from path MFE; never close-to-close."""
    arr = np.asarray(mfe10, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("mfe10 contains non-finite values")
    return (arr >= float(threshold)).astype(np.float32)


def constant_prior_log_loss(labels: np.ndarray, prior: float | None = None) -> float:
    """Bernoulli log loss of a constant probability prior on binary labels."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return float("nan")
    p = float(y.mean()) if prior is None else float(prior)
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch probs={p.shape} labels={y.shape}")
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def delta_vs_prior(model_ll: float, prior_ll: float) -> float:
    return float(model_ll - prior_ll)


def clears_gate(delta: float, gate: float = GATE_DELTA_VS_PRIOR) -> bool:
    return bool(delta <= gate)


def prior_stuck_stop(
    model_ll: float,
    prior_ll: float,
    stuck_streak: int,
    *,
    tol: float = PRIOR_STUCK_TOL,
    patience: int = PRIOR_STUCK_PATIENCE,
) -> tuple[str, int]:
    """Early-stop if |model_ll - prior_ll| <= tol for `patience` consecutive evals.

    Returns (reason, updated_stuck_streak). reason="" means continue.
    """
    if not np.isfinite(model_ll) or not np.isfinite(prior_ll):
        return "nonfinite_validation", stuck_streak + 1
    if abs(float(model_ll) - float(prior_ll)) <= float(tol):
        stuck_streak = stuck_streak + 1
    else:
        stuck_streak = 0
    if stuck_streak >= int(patience):
        return "stuck_at_constant_prior", stuck_streak
    return "", stuck_streak


def eval_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    train_prior: float,
) -> dict[str, Any]:
    """Single-head eval block with constant-prior baseline always present."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    prior = float(np.clip(train_prior, 1e-7, 1 - 1e-7))
    model_ll = binary_log_loss(p, y)
    prior_ll = constant_prior_log_loss(y, prior=prior)
    delta = delta_vs_prior(model_ll, prior_ll)
    return {
        "target": TARGET_NAME,
        "mfe10_def": MFE10_DEF,
        "threshold": MFE10_THRESHOLD,
        "samples": int(y.size),
        "positive_rate": float(y.mean()) if y.size else float("nan"),
        "train_prior": prior,
        "model_log_loss": model_ll,
        "constant_prior_log_loss": prior_ll,
        "delta_vs_prior": delta,
        "clears_gate_delta_le": GATE_DELTA_VS_PRIOR,
        "gate_passed": clears_gate(delta),
        "brier": float(np.square(np.clip(p, 0, 1) - y).mean()) if y.size else float("nan"),
    }


def target_contract() -> dict[str, Any]:
    return {
        "name": TARGET_NAME,
        "formula": f"y = 1{{mfe10 >= {MFE10_THRESHOLD}}} where mfe10 = {MFE10_DEF}",
        "not_close_to_close": True,
        "not_multi_head_r2": True,
        "expected_val_pos_rate": EXPECTED_VAL_POS_RATE,
        "gate_delta_vs_prior": GATE_DELTA_VS_PRIOR,
        "prior_stuck_tol": PRIOR_STUCK_TOL,
        "prior_stuck_patience": PRIOR_STUCK_PATIENCE,
    }
