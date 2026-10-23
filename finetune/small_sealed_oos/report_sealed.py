"""Direction/amplitude panel for the sealed 42-day OOS (C2 vs bigbatch e03b).

The sealed kernel already writes summary.json with direction/MAE/IC.  This
script adds the three panel layers (cost-aware PnL, amplitude, ranking), using
the 0.3% double-sided cost fixed in the sealed-window plan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "finetune/small_clean_oos"))

import panel  # noqa: E402

SHARDS = ROOT / "artifacts/small_sealed_oos/small_sealed_oos/shards"
REPORT = ROOT / "finetune/reports/small_sealed_oos_20260803_20260930.json"
BASELINE = "c2_best"
CANDIDATE = "bigbatch_best"
FIRST_USE = "2026-09-18"
COST = 0.003


def main() -> None:
    if not SHARDS.exists():
        raise SystemExit(f"Missing sealed shards: {SHARDS}")
    panel.COST = COST

    frames = {label: panel.load_shards(SHARDS, label) for label in (BASELINE, CANDIDATE)}
    for label, frame in frames.items():
        print(f"{label}: {len(frame)} rows, {frame.asof_date.nunique()} dates")

    full_dates = sorted(frames[BASELINE].asof_date.unique())
    first_dates = [d for d in full_dates if d >= FIRST_USE]
    first = {label: frame[frame.asof_date >= FIRST_USE].reset_index(drop=True)
             for label, frame in frames.items()}
    print(f"first-use: {len(first[BASELINE])} rows, {len(first_dates)} dates "
          f"({first_dates[0]}..{first_dates[-1]})")

    def score_scope(scope_frames, dates, block):
        old_block = panel.BLOCK
        panel.BLOCK = block
        try:
            scores = {label: panel.panel(frame) for label, frame in scope_frames.items()}
            paired = panel.paired_deltas(
                scope_frames[BASELINE], scope_frames[CANDIDATE], dates)
        finally:
            panel.BLOCK = old_block
        return scores, paired

    full_scores, full_paired = score_scope(frames, full_dates, 10)
    first_scores, first_paired = score_scope(first, first_dates, 5)

    document = dict(
        status="complete",
        purpose="Sealed 42-day OOS panel for C2 best vs bigbatch e03b best.",
        oos=dict(dataset="wynstonliu/a-share-120d-oos-20260803-20260930",
                 kernel="wynstonliu/kronos-small-sealed-oos-20260803-20260930",
                 samples=int(len(frames[BASELINE])), dates=len(full_dates),
                 first_use_start=FIRST_USE, first_use_dates=len(first_dates),
                 first_use_samples=int(len(first[BASELINE])),
                 decode="sample_count=1, T=0.65, top_p=0.8, seed=20260906"),
        cost_per_trade=COST,
        full_42d=dict(scores=full_scores,
                      paired_deltas={f"{CANDIDATE}_minus_{BASELINE}": full_paired}),
        first_use_8d=dict(scores=first_scores,
                          paired_deltas={f"{CANDIDATE}_minus_{BASELINE}": first_paired}),
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {REPORT}")

    for name, scores, paired in (("full_42d", full_scores, full_paired),
                                 ("first_use_8d", first_scores, first_paired)):
        print(f"\n== {name}")
        for label in (BASELINE, CANDIDATE):
            s = scores[label]
            m = s["L1_money"]
            a = s["L2_amplitude"]
            print(f"  {label}: d10_dir={s['official']['horizons']['10']['direction']:.4f} "
                  f"rank_ic={s['L3_ranking']['d10_rank_ic_daily_mean']:.4f} "
                  f"amp_ic={a['amplitude_rank_ic_daily_mean']:.4f} "
                  f"net_pnl={m['net_pnl']:+.4f} "
                  f"top_decile={m['top_amplitude_decile']['net_pnl']:+.4f}")
        print(f"  delta: {json.dumps(paired, ensure_ascii=False)[:500]}")


if __name__ == "__main__":
    main()
