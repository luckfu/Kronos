"""Verify feature window / label sidecar temporal alignment.

Contract (same as ModernBERTWindowDataset):
- Panel windows are enumerated in sorted(symbol) then start_index order.
- Sidecar parquet rows must match that enumeration 1:1.
- Feature lookback uses indices [start, start+LOOKBACK); asof is start+LOOKBACK-1.
- Labels (mfe/mae/up_*/down_*) use only future bars after asof (HORIZON days).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from modernbert_finance.build_dataset import FEATURES, HORIZON, LOOKBACK, prepare_frame
from modernbert_finance.build_targets import WINDOW, target_row
from modernbert_finance.dataset import load_panel


@dataclass
class AlignmentReport:
    ok: bool
    n_samples: int
    n_checked: int
    mismatches: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_feature_label_alignment(
    panel: dict[str, pd.DataFrame] | Path,
    targets_path: Path | None = None,
    *,
    max_check: int = 64,
    signal_start: str | None = None,
    signal_end: str | None = None,
) -> AlignmentReport:
    """Recompute a sample of labels from the panel and compare to sidecar rows."""
    notes: list[str] = []
    if isinstance(panel, (str, Path)):
        panel = load_panel(Path(panel))

    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None

    expected_rows: list[dict[str, Any]] = []
    for symbol in sorted(panel):
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        max_start = len(frame) - WINDOW + 1
        for start in range(max_start):
            asof = pd.Timestamp(frame.index[start + LOOKBACK - 1]).date()
            if start_date and asof < start_date:
                continue
            if end_date and asof > end_date:
                continue
            expected_rows.append(target_row(symbol, frame, start))

    n_samples = len(expected_rows)
    mismatches: list[dict[str, Any]] = []

    if targets_path is not None:
        table = pq.read_table(Path(targets_path))
        if table.num_rows != n_samples:
            return AlignmentReport(
                ok=False,
                n_samples=n_samples,
                n_checked=0,
                mismatches=[
                    {
                        "reason": "row_count_mismatch",
                        "panel_windows": n_samples,
                        "sidecar_rows": table.num_rows,
                    }
                ],
                notes=[
                    "Sidecar length must equal filtered panel window enumeration "
                    "(sorted symbols, dense start indices)."
                ],
            )
        cols = set(table.column_names)
        check_cols = [
            c
            for c in (
                "symbol",
                "start_index",
                "asof_date",
                "mfe10",
                "mae10",
                "up_003",
                "up_005",
                "up_008",
                "up_012",
                "down_003",
                "down_005",
                "down_008",
                "down_012",
            )
            if c in cols
        ]
        sidecar = {c: np.asarray(table[c]) for c in check_cols}
        indices = np.linspace(0, n_samples - 1, num=min(max_check, n_samples), dtype=int)
        for idx in indices:
            exp = expected_rows[int(idx)]
            for col in check_cols:
                got = sidecar[col][int(idx)]
                want = exp[col]
                if col in ("mfe10", "mae10"):
                    if not np.isfinite(got) or abs(float(got) - float(want)) > 1e-5:
                        mismatches.append(
                            {
                                "row": int(idx),
                                "col": col,
                                "got": float(got) if np.isfinite(got) else None,
                                "want": float(want),
                            }
                        )
                else:
                    if str(got) != str(want) and got != want:
                        # numpy scalar vs python
                        try:
                            if int(got) == int(want):
                                continue
                        except Exception:
                            pass
                        if str(got) != str(want):
                            mismatches.append(
                                {
                                    "row": int(idx),
                                    "col": col,
                                    "got": got.item() if hasattr(got, "item") else got,
                                    "want": want,
                                }
                            )
        notes.append(
            f"Compared {len(indices)} rows against recomputed target_row(); "
            f"lookback={LOOKBACK}, horizon={HORIZON}, features={FEATURES}."
        )
    else:
        # No sidecar: still document causal contract on recomputed rows.
        indices = np.linspace(0, n_samples - 1, num=min(max_check, max(n_samples, 1)), dtype=int) if n_samples else []
        for idx in indices:
            row = expected_rows[int(idx)]
            symbol = row["symbol"]
            start = int(row["start_index"])
            frame = prepare_frame(panel[symbol], symbol)
            asof_pos = start + LOOKBACK - 1
            # Future-only labels: asof close must not use future high/low.
            future = frame.iloc[asof_pos + 1 : asof_pos + HORIZON + 1]
            if len(future) != HORIZON:
                mismatches.append({"row": int(idx), "reason": "short_future"})
                continue
            # Sanity: mfe from future highs only.
            close = float(frame.iloc[asof_pos]["close"])
            mfe = float(np.max(future["high"].to_numpy(dtype=np.float64)) / close - 1.0)
            if abs(mfe - float(row["mfe10"])) > 1e-5:
                mismatches.append(
                    {"row": int(idx), "reason": "mfe_recompute", "got": mfe, "want": float(row["mfe10"])}
                )
        notes.append(
            "No sidecar path given; verified causal recompute on sample rows only."
        )

    return AlignmentReport(
        ok=len(mismatches) == 0 and n_samples > 0,
        n_samples=n_samples,
        n_checked=min(max_check, n_samples),
        mismatches=mismatches[:20],
        notes=notes,
    )
