#!/usr/bin/env python3
"""Equal-weight rank fusion of the frozen hidden-state head with C2's 16-path forecast.

Protocol: finetune/reports/20261112_head_c2_fusion_prereg_cn.md.  Nothing is
selected here: the head is the one T3 picked (ridge, alpha 1e5, fitted only on
validation-symbol windows that end 2026-07-02) and the fusion weight is 1:1.

Writes two shard sets for rolling_compare.py -- the head alone and the fusion --
on every signal date of the given C2 shard directory, and freezes the head
weights so the same rule can be scored once on the sealed window later.

    python3 finetune/c2_ceiling/fusion_head_c2.py --c2-shards <dir> \\
        --hidden-oos 'scratch/c2_ceiling/hidden/oos_hidden_*.npz' \\
        --hidden-val 'scratch/c2_ceiling/hidden/val_hidden_*.npz' \\
        --weights finetune/c2_ceiling/frozen/head_b_alpha1e5.npy --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_candidates as fc  # noqa: E402

ALPHA, SEAL_AFTER = 1e5, "2026-10-23"


def head_weights(path: Path, hidden_val: str) -> np.ndarray:
    """Load the frozen head, or fit it once from the validation-symbol windows."""
    if path.is_file():
        return np.load(path)
    x, dates, target = fc.load_hidden(hidden_val, None)
    if max(dates.tolist()) > "2026-07-02":
        raise SystemExit("Head training windows must end by 2026-07-02")
    weights = fc.ridge(fc.zscore_by_date(dates, x), target, ALPHA)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, weights)
    return weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-shards", required=True, help="16-path C2 shards (the baseline)")
    parser.add_argument("--hidden-oos", required=True)
    parser.add_argument("--hidden-val", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    weights = head_weights(Path(args.weights), args.hidden_val)
    frame = fc.load_shards(args.c2_shards, "2026-07-17", SEAL_AFTER)
    hidden = fc.zscore_by_date(frame.asof_date.to_numpy(), fc.load_hidden(args.hidden_oos, frame))
    frame["head"] = hidden @ weights
    ranks = frame.groupby("asof_date")[["head", "predicted_return_d10"]].rank(pct=True)
    fusion = 0.5 * ranks["head"] + 0.5 * ranks["predicted_return_d10"]

    out = Path(args.out_dir)
    fc.write_scores(frame, frame["head"].to_numpy(), "head", out / "head")
    fc.write_scores(frame, fusion.to_numpy(), "fusion", out / "fusion")
    summary = {"rows": int(len(frame)), "dates": int(frame.asof_date.nunique()),
               "first": frame.asof_date.min(), "last": frame.asof_date.max(), "alpha": ALPHA,
               "fusion": "0.5 * pct_rank(head) + 0.5 * pct_rank(C2 16-path median d10), within signal date",
               "weights_file": str(args.weights),
               "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()}
    (out / "fusion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
