"""CPU unit tests for Kairos cheap ablation scaffolding."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from modernbert_finance.ablations.alignment import verify_feature_label_alignment
from modernbert_finance.ablations.label_shuffle import run_label_shuffle_sanity
from modernbert_finance.ablations.run_cheap_ablations import main, run
from modernbert_finance.ablations.simple_baseline import (
    run_simple_baseline,
    summarize_history_windows,
)
from modernbert_finance.build_targets import build_split


def _tiny_panel(rows: int = 140) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 110.0, rows)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "amount": np.full(rows, 100000.0),
            "sector": ["J66货币金融服务"] * rows,
            "size_percentile": np.full(rows, 0.6),
        },
        index=dates,
    )
    return {"sh.600000": frame}


def test_alignment_matches_sidecar(tmp_path: Path) -> None:
    panel = _tiny_panel()
    targets = tmp_path / "train_targets.parquet"
    build_split(panel, targets, None, None, 0, 50)
    report = verify_feature_label_alignment(panel, targets, max_check=8)
    assert report.ok
    assert report.n_samples > 0
    assert report.mismatches == []


def test_label_shuffle_near_prior() -> None:
    rng = np.random.default_rng(0)
    n, d = 800, 12
    x = rng.normal(size=(n, d))
    # Weak signal on true labels.
    logits = 0.4 * x[:, 0] - 0.2 * x[:, 1]
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(np.int64)
    report = run_label_shuffle_sanity(x, y, head="synth", seed=1)
    assert report.ok
    assert report.shuffled_fit_log_loss >= report.prior_log_loss - 0.05


def test_simple_baseline_runs() -> None:
    rng = np.random.default_rng(1)
    hist = rng.normal(size=(200, 120, 6)) + 100
    features = summarize_history_windows(hist)
    assert features.shape[0] == 200
    y = (features[:, 3] > np.median(features[:, 3])).astype(np.int64)
    targets = {
        "up_005": y,
        "down_005": 1 - y,
        "mfe10": rng.normal(scale=0.02, size=200),
    }
    report = run_simple_baseline(features, targets, heads=("up_005", "mfe10"), seed=2)
    assert report.ok
    assert len(report.results) == 2


def test_smoke_cli(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    code = main(["--smoke", "--work-dir", str(tmp_path / "work"), "--out-json", str(out)])
    assert code in (0, 2)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["alignment"]["ok"] is True
    assert "label_shuffle" in payload
    assert "simple_baseline" in payload
