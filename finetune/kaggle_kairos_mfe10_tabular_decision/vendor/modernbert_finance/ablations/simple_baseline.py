"""Simple logistic / linear baselines on ModernBERT window summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


UP_COLUMNS = ("up_003", "up_005", "up_008", "up_012")
DOWN_COLUMNS = ("down_003", "down_005", "down_008", "down_012")


@dataclass
class HeadBaselineResult:
    head: str
    kind: str
    n_train: int
    n_test: int
    prior_log_loss: float | None
    model_log_loss: float | None
    model_brier: float | None
    model_mse: float | None
    beats_prior: bool | None


@dataclass
class BaselineReport:
    ok: bool
    results: list[HeadBaselineResult]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def summarize_history_windows(history: np.ndarray) -> np.ndarray:
    """Compress (N, T, F) OHLCVA windows to (N, D) summary features.

    Uses last-bar levels, returns, and simple moments — enough for a cheap
    linear probe without re-implementing ModernBERT.
    """
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim != 3:
        raise ValueError("history must be (N, lookback, n_features)")
    n, t, f = hist.shape
    last = hist[:, -1, :]
    first = hist[:, 0, :]
    mean = hist.mean(axis=1)
    std = hist.std(axis=1)
    # Close return over the window (feature index 3 is close in FEATURES).
    close_idx = 3 if f >= 4 else f - 1
    ret = last[:, close_idx] / np.maximum(first[:, close_idx], 1e-8) - 1.0
    # Recent 5-bar momentum if long enough.
    if t >= 5:
        mom = last[:, close_idx] / np.maximum(hist[:, -5, close_idx], 1e-8) - 1.0
    else:
        mom = ret
    return np.concatenate(
        [last, mean, std, ret[:, None], mom[:, None]],
        axis=1,
    )


def _prior_log_loss(y: np.ndarray) -> float:
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    probs = np.full((len(y), 2), [1.0 - p, p], dtype=np.float64)
    return float(log_loss(y, probs, labels=[0, 1]))


def run_simple_baseline(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    heads: Sequence[str] | None = None,
    seed: int = 20261001,
    max_iter: int = 300,
) -> BaselineReport:
    """Train per-head logistic (binary) or linear (regression) baselines."""
    x = np.asarray(features, dtype=np.float64)
    if heads is None:
        heads = tuple(h for h in (*UP_COLUMNS, *DOWN_COLUMNS, "mfe10", "mae10") if h in targets)

    results: list[HeadBaselineResult] = []
    notes = [
        "Binary heads: LogisticRegression vs constant prevalence prior.",
        "mfe10/mae10: LinearRegression MSE only (no prior log loss).",
    ]
    for head in heads:
        y = np.asarray(targets[head])
        if len(y) != len(x):
            raise ValueError(f"{head}: target length {len(y)} != features {len(x)}")
        if head in UP_COLUMNS or head in DOWN_COLUMNS:
            y_bin = y.astype(np.int64).reshape(-1)
            if len(np.unique(y_bin)) < 2:
                results.append(
                    HeadBaselineResult(
                        head=head,
                        kind="logistic",
                        n_train=0,
                        n_test=0,
                        prior_log_loss=None,
                        model_log_loss=None,
                        model_brier=None,
                        model_mse=None,
                        beats_prior=None,
                    )
                )
                notes.append(f"{head}: skipped (single class)")
                continue
            x_tr, x_te, y_tr, y_te = train_test_split(
                x, y_bin, test_size=0.25, random_state=seed, stratify=y_bin
            )
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=max_iter, random_state=seed),
            )
            clf.fit(x_tr, y_tr)
            proba = clf.predict_proba(x_te)[:, 1]
            prior = _prior_log_loss(y_te)
            ll = float(log_loss(y_te, np.column_stack([1 - proba, proba]), labels=[0, 1]))
            brier = float(brier_score_loss(y_te, proba))
            results.append(
                HeadBaselineResult(
                    head=head,
                    kind="logistic",
                    n_train=int(len(y_tr)),
                    n_test=int(len(y_te)),
                    prior_log_loss=prior,
                    model_log_loss=ll,
                    model_brier=brier,
                    model_mse=None,
                    beats_prior=bool(ll < prior - 1e-4),
                )
            )
        else:
            y_reg = y.astype(np.float64).reshape(-1)
            x_tr, x_te, y_tr, y_te = train_test_split(
                x, y_reg, test_size=0.25, random_state=seed
            )
            reg = make_pipeline(StandardScaler(), LinearRegression())
            reg.fit(x_tr, y_tr)
            pred = reg.predict(x_te)
            mse = float(mean_squared_error(y_te, pred))
            results.append(
                HeadBaselineResult(
                    head=head,
                    kind="linear",
                    n_train=int(len(y_tr)),
                    n_test=int(len(y_te)),
                    prior_log_loss=None,
                    model_log_loss=None,
                    model_brier=None,
                    model_mse=mse,
                    beats_prior=None,
                )
            )

    ok = any(r.beats_prior for r in results if r.beats_prior is not None) or any(
        r.model_mse is not None for r in results
    )
    # ok means the baseline ran; signal vs prior is recorded per head.
    ok = len(results) > 0 and all(
        r.model_log_loss is not None or r.model_mse is not None or r.n_train == 0
        for r in results
    )
    return BaselineReport(ok=ok, results=results, notes=notes)
