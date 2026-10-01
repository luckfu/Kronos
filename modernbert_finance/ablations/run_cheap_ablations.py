"""CLI: run alignment check + label-shuffle sanity + simple baseline.

Examples
--------
# Synthetic smoke (no Kaggle data needed):
python -m modernbert_finance.ablations.run_cheap_ablations --smoke

# Real panel + sidecar:
python -m modernbert_finance.ablations.run_cheap_ablations \\
  --panel /path/train_data.pkl \\
  --targets /path/train_targets.parquet \\
  --sector-vocab /path/sector_vocabulary.json \\
  --out-json /tmp/kairos_cheap_ablation.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modernbert_finance.ablations.alignment import verify_feature_label_alignment
from modernbert_finance.ablations.label_shuffle import run_label_shuffle_sanity
from modernbert_finance.ablations.simple_baseline import (
    run_simple_baseline,
    summarize_history_windows,
)
from modernbert_finance.build_dataset import LOOKBACK
from modernbert_finance.build_targets import WINDOW, build_split
from modernbert_finance.dataset import ModernBERTWindowDataset


def _synthetic_panel(n_symbols: int = 8, rows: int = 200, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        dates = pd.date_range("2020-01-01", periods=rows, freq="D")
        # Larger daily moves so exceedance heads are not single-class.
        close = 100 + np.cumsum(rng.normal(0.02 * (i - 3), 1.5, size=rows))
        close = np.maximum(close, 5.0)
        high = close * (1.0 + rng.uniform(0.005, 0.04, size=rows))
        low = close * (1.0 - rng.uniform(0.005, 0.04, size=rows))
        frame = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": rng.uniform(1e3, 5e3, size=rows),
                "amount": rng.uniform(1e5, 5e5, size=rows),
                "sector": ["J66货币金融服务"] * rows,
                "size_percentile": np.full(rows, 0.4 + 0.05 * i),
            },
            index=dates,
        )
        panel[f"sh.60000{i}"] = frame
    return panel


def _dataset_matrices(dataset: ModernBERTWindowDataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    histories = []
    targets: dict[str, list[float]] = {
        "mfe10": [],
        "mae10": [],
        "up_003": [],
        "up_005": [],
        "up_008": [],
        "up_012": [],
        "down_003": [],
        "down_005": [],
        "down_008": [],
        "down_012": [],
    }
    for i in range(len(dataset)):
        sample = dataset[i]
        histories.append(sample["history"].numpy())
        targets["mfe10"].append(float(sample["mfe10"]))
        targets["mae10"].append(float(sample["mae10"]))
        for j, name in enumerate(("up_003", "up_005", "up_008", "up_012")):
            targets[name].append(float(sample["up_target"][j]))
        for j, name in enumerate(("down_003", "down_005", "down_008", "down_012")):
            targets[name].append(float(sample["down_target"][j]))
    hist = np.stack(histories, axis=0)
    arrays = {k: np.asarray(v) for k, v in targets.items()}
    return hist, arrays


def _build_smoke_dataset(
    tmp: Path,
) -> tuple[ModernBERTWindowDataset, dict[str, pd.DataFrame], Path]:
    panel = _synthetic_panel()
    tmp.mkdir(parents=True, exist_ok=True)
    panel_path = tmp / "train_data.pkl"
    with panel_path.open("wb") as handle:
        pickle.dump(panel, handle)
    vocab_path = tmp / "sector_vocabulary.json"
    vocab_path.write_text(
        json.dumps({"sector_labels": ["J66货币金融服务"], "unknown_sector_id": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    targets_dir = tmp / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    targets_path = targets_dir / "train_targets.parquet"
    build_split(panel, targets_path, None, None, 0, 200)
    manifest = {
        "window": {"lookback": LOOKBACK, "source_window": WINDOW},
        "source": {"train_panel_sha256": "smoke"},
    }
    (targets_dir / "targets_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = ModernBERTWindowDataset(
        panel_path,
        targets_path,
        vocab_path,
        verify_source=False,
    )
    return dataset, panel, targets_path


def run(
    *,
    smoke: bool,
    panel: Path | None,
    targets: Path | None,
    sector_vocab: Path | None,
    out_json: Path | None,
    head: str,
    seed: int,
    work_dir: Path | None,
) -> dict[str, Any]:
    tmp = work_dir or Path("/tmp/kairos_cheap_ablations")
    tmp.mkdir(parents=True, exist_ok=True)

    if smoke:
        dataset, panel_obj, targets_path = _build_smoke_dataset(tmp / "smoke")
    else:
        if not (panel and targets and sector_vocab):
            raise ValueError("non-smoke mode requires --panel --targets --sector-vocab")
        dataset = ModernBERTWindowDataset(
            panel, targets, sector_vocab, verify_source=False
        )
        panel_obj = panel
        targets_path = targets

    alignment = verify_feature_label_alignment(panel_obj, targets_path, max_check=64)
    hist, target_arrays = _dataset_matrices(dataset)
    features = summarize_history_windows(hist)

    binary_heads = [
        name
        for name in (
            head,
            "up_003",
            "up_005",
            "up_008",
            "up_012",
            "down_003",
            "down_005",
            "down_008",
            "down_012",
        )
        if name in target_arrays and len(np.unique(target_arrays[name])) >= 2
    ]
    if not binary_heads:
        raise ValueError(
            f"no binary head with both classes; requested={head}; "
            f"available={sorted(target_arrays)}"
        )
    shuffle_head = binary_heads[0]
    shuffle_report = run_label_shuffle_sanity(
        features, target_arrays[shuffle_head], head=shuffle_head, seed=seed
    )
    baseline = run_simple_baseline(features, target_arrays, seed=seed)

    payload = {
        "mode": "smoke" if smoke else "real",
        "n_samples": int(len(dataset)),
        "feature_dim": int(features.shape[1]),
        "alignment": alignment.to_dict(),
        "label_shuffle": shuffle_report.to_dict(),
        "simple_baseline": baseline.to_dict(),
        "how_to_verify_alignment": [
            "Dataset enumerates sorted(symbol) × start_index; sidecar must match 1:1.",
            f"asof = start + lookback - 1 (lookback={LOOKBACK}); labels use only the next {10} bars.",
            "Run verify_feature_label_alignment(panel, targets_parquet) after any rebuild.",
            "Mismatch in row counts → wrong panel hash / date filter / sort order.",
        ],
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="CPU synthetic smoke run")
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--sector-vocab", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--head", default="up_005")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        payload = run(
            smoke=bool(args.smoke or not args.panel),
            panel=args.panel,
            targets=args.targets,
            sector_vocab=args.sector_vocab,
            out_json=args.out_json,
            head=args.head,
            seed=args.seed,
            work_dir=args.work_dir,
        )
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    shuffle_ok = payload["label_shuffle"]["ok"]
    align_ok = payload["alignment"]["ok"]
    return 0 if shuffle_ok and align_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
