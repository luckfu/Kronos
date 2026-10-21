"""Build the direction+amplitude baseline panel from the published OOS shards.

Runs the same ``panel`` module the Kaggle evaluation uses, so the archived
C2/C03 numbers and any future epoch2 numbers are produced by identical code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import panel as panel_module  # noqa: E402

SHARDS = Path("artifacts/small_clean_oos_shards/small_clean_oos/shards")
REPORT = Path("finetune/reports/small_clean_direction_amplitude_baseline_20261021.json")
BASELINE = "c2_best"
CANDIDATE = "clean_best"


def main() -> None:
    if not SHARDS.exists():
        raise SystemExit(f"Missing shards: {SHARDS}")
    dates = None
    frames = {}
    for label in (BASELINE, CANDIDATE):
        frame = panel_module.load_shards(SHARDS, label)
        frames[label] = frame
        found = sorted(frame.asof_date.unique())
        dates = found if dates is None else dates
        if found != dates:
            raise SystemExit(f"Date mismatch between {BASELINE} and {label}")
        print(f"{label}: {len(frame)} rows, {len(found)} dates")

    scores = {label: panel_module.panel(frame) for label, frame in frames.items()}
    paired = panel_module.paired_deltas(frames[BASELINE], frames[CANDIDATE], dates)

    document = dict(
        status="complete",
        purpose=("Direction + amplitude panel for the online C2 model and the C03 "
                 "clean-coverage model, on the frozen 34-date OOS."),
        baseline=BASELINE,
        candidate=CANDIDATE,
        scores=scores,
        paired_deltas={f"{CANDIDATE}_minus_{BASELINE}": paired},
        uncertainty=dict(signal_dates=len(dates), block_length=panel_module.BLOCK,
                         replicates=panel_module.REPLICATES, seed=panel_module.RNG_SEED,
                         note="34 signal dates; overlapping labels; not a new blind set"),
        oos=dict(samples=int(len(frames[BASELINE])), dates=len(dates),
                 panel_sha256="be159cd89474a22dc000403137c7d7647cd2719eb325631a9f5ba701defa8118",
                 shard_source="luckfu/kronos-small-clean-vs-c2-full-oos"),
        cost_per_trade=panel_module.COST,
        layer_definitions=dict(
            L1_money="sign(predicted_return_d10) * actual_return_d10, net of 0.4% round trip",
            L2_amplitude="rank correlation of |predicted| vs |realized| and the predicted-amplitude calibration curve",
            L3_ranking="cross-sectional d10 rank IC and top-bottom 10% spread",
        ),
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {REPORT}")

    official = scores[BASELINE]["official"]
    print(f"sanity: c2 path_direction={official['path_direction']:.6f} "
          f"mae={official['path_return_mae']:.6f} rmse={official['path_return_rmse']:.6f}")
    print(f"        c3 path_direction={scores[CANDIDATE]['official']['path_direction']:.6f} "
          f"mae={scores[CANDIDATE]['official']['path_return_mae']:.6f} "
          f"rmse={scores[CANDIDATE]['official']['path_return_rmse']:.6f}")
    for key in ("direction", "return_mae", "return_rmse", "signed_return", "net_pnl",
                "rank_ic", "amplitude_rank_ic", "spread"):
        entry = paired.get(key)
        if entry is None:
            continue
        low, high = entry["block_bootstrap_95ci"]
        flag = "*" if low > 0 or high < 0 else " "
        print(f"  {key:<18} delta={entry['value']:+.5f}  95%CI [{low:+.5f},{high:+.5f}]{flag}")


if __name__ == "__main__":
    main()
