"""Build the direction+amplitude report for epoch2 vs C2 from the new OOS shards.

The Kaggle kernel ``luckfu/kronos-small-clean-epoch2-vs-c2-full-oos`` writes
one shard per model per signal date.  This script concatenates them, scores
the same three layers as ``baseline_panel.py`` and adds the paired deltas.
It also reuses the archived C03 shards for a three-way view when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import panel as panel_module  # noqa: E402

DEFAULT_SHARDS = Path("artifacts/small_clean_epoch2_oos/shards")
DEFAULT_CLEAN_SHARDS = Path("artifacts/small_clean_oos_shards/small_clean_oos/shards")
DEFAULT_REPORT = Path("finetune/reports/small_clean_epoch2_vs_c2_full_oos_20261021.json")
BASELINE = "c2_best"
CANDIDATE = "epoch2_best"
CLEAN = "clean_best"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--clean-shards", type=Path, default=DEFAULT_CLEAN_SHARDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default=CANDIDATE)
    args = parser.parse_args()

    if not args.shards.exists():
        raise SystemExit(f"Missing epoch2 shards: {args.shards}")

    frames = {}
    dates = None
    for label in (args.baseline, args.candidate):
        frame = panel_module.load_shards(args.shards, label)
        found = sorted(frame.asof_date.unique())
        if dates is None:
            dates = found
        elif found != dates:
            raise SystemExit(f"Date mismatch between {args.baseline} and {label}")
        if len(frame) != 175428 or len(found) != 34:
            raise SystemExit(f"Incomplete {label}: {len(frame)} rows / {len(found)} dates")
        frames[label] = frame
        print(f"{label}: {len(frame)} rows, {len(found)} dates")

    if args.clean_shards.exists():
        clean = panel_module.load_shards(args.clean_shards, CLEAN)
        if sorted(clean.asof_date.unique()) != dates or len(clean) != len(frames[args.baseline]):
            raise SystemExit("Archived clean shards do not match the current OOS dates")
        frames[CLEAN] = clean
        print(f"{CLEAN}: {len(clean)} rows, {clean.asof_date.nunique()} dates (archived)")

    scores = {label: panel_module.panel(frame) for label, frame in frames.items()}
    pairs = [(args.baseline, args.candidate)]
    if CLEAN in frames:
        pairs.append((args.baseline, CLEAN))
    paired = {
        f"{candidate}_minus_{base}": panel_module.paired_deltas(
            frames[base], frames[candidate], dates)
        for base, candidate in pairs
    }

    document = dict(
        status="complete",
        purpose=("Direction + amplitude panel for Kronos-small clean epoch2 "
                 "(best segment 501) against the frozen C2 baseline, on the "
                 "34-date OOS."),
        baseline=args.baseline,
        candidate=args.candidate,
        scores=scores,
        paired_deltas=paired,
        uncertainty=dict(signal_dates=len(dates), block_length=panel_module.BLOCK,
                         replicates=panel_module.REPLICATES, seed=panel_module.RNG_SEED,
                         note="34 signal dates; overlapping labels; not a new blind set"),
        oos=dict(samples=int(len(frames[args.baseline])), dates=len(dates),
                 panel_sha256="be159cd89474a22dc000403137c7d7647cd2719eb325631a9f5ba701defa8118",
                 shard_source="luckfu/kronos-small-clean-epoch2-vs-c2-full-oos"),
        cost_per_trade=panel_module.COST,
        layer_definitions=dict(
            L1_money="sign(predicted_return_d10) * actual_return_d10, net of 0.4% round trip",
            L2_amplitude="rank correlation of |predicted| vs |realized| and the predicted-amplitude calibration curve",
            L3_ranking="cross-sectional d10 rank IC and top-bottom 10% spread",
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {args.out}")

    for name, entry in paired.items():
        print(f"\n{name}:")
        for key in ("direction", "return_mae", "return_rmse", "signed_return",
                    "net_pnl", "rank_ic", "amplitude_rank_ic", "spread"):
            item = entry.get(key)
            if not item:
                continue
            low, high = item["block_bootstrap_95ci"]
            flag = "*" if low > 0 or high < 0 else " "
            print(f"  {key:<18} delta={item['value']:+.5f}  "
                  f"95%CI [{low:+.5f},{high:+.5f}]{flag}")

    for label in (args.baseline, args.candidate):
        official = scores[label]["official"]
        money = scores[label]["L1_money"]
        amp = scores[label]["L2_amplitude"]
        print(f"\n{label}: d10_dir={scores[label]['official']['horizons']['10']['direction']:.4f} "
              f"path_dir={official['path_direction']:.4f} "
              f"mae={official['path_return_mae']:.4f} "
              f"rank_ic={scores[label]['L3_ranking']['d10_rank_ic_daily_mean']:.4f} "
              f"amp_ic={amp['amplitude_rank_ic_daily_mean']:.4f} "
              f"net_pnl={money['net_pnl']:+.4f} "
              f"top_decile_pnl={money['top_amplitude_decile']['net_pnl']:+.4f}")


if __name__ == "__main__":
    main()
