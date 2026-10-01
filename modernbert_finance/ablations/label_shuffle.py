"""Label-shuffle sanity: same features, shuffled multi-head labels.

Expectation: a trivial linear/logistic fit on shuffled labels should not beat
the constant-prevalence prior by much (near-constant loss). If shuffled labels
fit much better than the prior, the evaluation harness is leaking or misaligned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ShuffleSanityReport:
    ok: bool
    n_samples: int
    head: str
    prior_log_loss: float
    shuffled_fit_log_loss: float
    true_fit_log_loss: float
    shuffle_minus_prior: float
    true_minus_prior: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _constant_prior_log_loss(y: np.ndarray) -> float:
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    probs = np.full((len(y), 2), [1.0 - p, p], dtype=np.float64)
    return float(log_loss(y, probs, labels=[0, 1]))


def run_label_shuffle_sanity(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    head: str = "up_005",
    seed: int = 20261001,
    max_iter: int = 200,
    shuffle_tol: float = 0.02,
) -> ShuffleSanityReport:
    """Fit logistic on true vs shuffled labels; compare to prevalence prior.

    Parameters
    ----------
    features:
        (N, D) float array — typically lookback window summary stats.
    labels:
        (N,) binary {0,1} for one head.
    shuffle_tol:
        Shuffled-fit log loss is allowed to undercut the prior by at most this
        amount before we flag the harness as suspicious.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be (N, D) aligned with labels (N,)")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels must be binary 0/1")
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes for log_loss sanity")

    prior = _constant_prior_log_loss(y)
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)

    x_tr, x_te, y_tr, y_te, ys_tr, ys_te = train_test_split(
        x, y, y_shuf, test_size=0.25, random_state=seed, stratify=y
    )

    def _fit_ll(y_train: np.ndarray, y_test: np.ndarray) -> float:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=seed),
        )
        clf.fit(x_tr, y_train)
        proba = clf.predict_proba(x_te)
        return float(log_loss(y_test, proba, labels=[0, 1]))

    shuf_ll = _fit_ll(ys_tr, ys_te)
    true_ll = _fit_ll(y_tr, y_te)

    # Prior on the same test fold prevalence for a fairer delta.
    prior_te = _constant_prior_log_loss(y_te)
    shuffle_gap = shuf_ll - prior_te
    true_gap = true_ll - prior_te
    notes = [
        "Shuffled labels should yield log loss ≈ constant prior (gap near 0).",
        "True labels may beat the prior if the summarized features carry signal.",
        f"test_prior_log_loss={prior_te:.6f}",
    ]
    ok = shuffle_gap >= -shuffle_tol
    if not ok:
        notes.append(
            f"FAIL: shuffled fit undercut prior by {-shuffle_gap:.4f} > tol {shuffle_tol}"
        )
    return ShuffleSanityReport(
        ok=ok,
        n_samples=int(len(y)),
        head=head,
        prior_log_loss=float(prior),
        shuffled_fit_log_loss=float(shuf_ll),
        true_fit_log_loss=float(true_ll),
        shuffle_minus_prior=float(shuffle_gap),
        true_minus_prior=float(true_gap),
        notes=notes,
    )
