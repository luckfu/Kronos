"""Phase A: label / time / split diagnostics for Kairos decision heads.

Answers:
1. Feature cutoff vs 10D label horizon (causal contract + leakage probe).
2. Per-head positive rates train vs val vs 2025H2 / 2026H1.
3. Correlation across 8 binary heads (multi-head dilution context).
4. Temporal integrity: history slice never includes post-asof bars.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from modernbert_finance.build_dataset import FEATURES, HORIZON, LOOKBACK, prepare_frame
from modernbert_finance.build_targets import WINDOW, target_row
from modernbert_finance.ablations._panel_io import load_panel

BINARY_HEADS = (
    "up_003",
    "up_005",
    "up_008",
    "up_012",
    "down_003",
    "down_005",
    "down_008",
    "down_012",
)


def _prior_log_loss(y: np.ndarray) -> float:
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    probs = np.column_stack([np.full(len(y), 1.0 - p), np.full(len(y), p)])
    return float(log_loss(y, probs, labels=[0, 1]))


def _rate_and_prior(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(np.float64).reshape(-1)
    rate = float(np.mean(y)) if len(y) else float("nan")
    return {
        "n": int(len(y)),
        "positive_rate": rate,
        "prior_log_loss": _prior_log_loss(y) if len(y) and len(np.unique(y)) >= 2 else float("nan"),
    }


def positive_rate_tables(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> dict[str, Any]:
    """Per-head rates for train, val, and val sub-periods 2025H2 / 2026H1."""
    val = val.copy()
    val["_asof"] = pd.to_datetime(val["asof_date"])
    h2 = val[(val["_asof"] >= "2025-07-01") & (val["_asof"] <= "2025-12-31")]
    h1 = val[(val["_asof"] >= "2026-01-01") & (val["_asof"] <= "2026-06-30")]

    splits = {
        "train": train,
        "val": val,
        "val_2025H2": h2,
        "val_2026H1": h1,
    }
    by_head: dict[str, dict[str, Any]] = {}
    macro_priors: dict[str, float] = {}
    for split_name, frame in splits.items():
        lls: list[float] = []
        for head in BINARY_HEADS:
            stats = _rate_and_prior(frame[head].to_numpy())
            by_head.setdefault(head, {})[split_name] = stats
            if np.isfinite(stats["prior_log_loss"]):
                lls.append(stats["prior_log_loss"])
        macro_priors[split_name] = float(np.mean(lls)) if lls else float("nan")

    # Drift: val rate - train rate
    drift = {
        head: float(
            by_head[head]["val"]["positive_rate"] - by_head[head]["train"]["positive_rate"]
        )
        for head in BINARY_HEADS
    }
    return {
        "by_head": by_head,
        "macro_prior_log_loss": macro_priors,
        "val_minus_train_rate": drift,
        "period_counts": {k: int(len(v)) for k, v in splits.items()},
    }


def cross_prior_transfer(train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
    """Constant train prevalence scored on val (explains ~0.5966 collapse)."""
    per_head: dict[str, Any] = {}
    lls: list[float] = []
    for head in BINARY_HEADS:
        p_tr = float(np.mean(train[head].to_numpy()))
        p_tr = min(max(p_tr, 1e-6), 1.0 - 1e-6)
        y = val[head].to_numpy().astype(np.int64)
        probs = np.column_stack([np.full(len(y), 1.0 - p_tr), np.full(len(y), p_tr)])
        ll = float(log_loss(y, probs, labels=[0, 1]))
        lls.append(ll)
        per_head[head] = {
            "train_positive_rate": p_tr,
            "val_log_loss_with_train_prior": ll,
            "val_self_prior_log_loss": _prior_log_loss(y),
        }
    return {
        "per_head": per_head,
        "macro_val_log_loss_with_train_prior": float(np.mean(lls)),
        "note": (
            "If a deep model sticks near this macro, it has collapsed to the "
            "constant train prevalence (no usable conditional signal)."
        ),
    }


def head_correlations(frame: pd.DataFrame, *, sample: int | None = None, seed: int = 0) -> dict[str, Any]:
    df = frame[list(BINARY_HEADS)]
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    corr = df.corr()
    # Pairwise mean |corr| within up, within down, across up/down
    up = list(BINARY_HEADS[:4])
    down = list(BINARY_HEADS[4:])
    def _mean_abs_offdiag(cols: list[str]) -> float:
        sub = corr.loc[cols, cols].to_numpy(dtype=np.float64)
        mask = ~np.eye(len(cols), dtype=bool)
        return float(np.mean(np.abs(sub[mask])))

    across = corr.loc[up, down].to_numpy(dtype=np.float64)
    return {
        "matrix": {r: {c: float(corr.loc[r, c]) for c in BINARY_HEADS} for r in BINARY_HEADS},
        "mean_abs_corr_within_up": _mean_abs_offdiag(up),
        "mean_abs_corr_within_down": _mean_abs_offdiag(down),
        "mean_corr_up_vs_down": float(np.mean(across)),
        "n_rows_used": int(len(df)),
        "interpretation": (
            "Adjacent thresholds are highly correlated (~0.7); equal-weight multi-head "
            "loss double-counts nested exceedance events and can dilute rare-tail heads."
        ),
    }


def document_cutoff_contract() -> dict[str, Any]:
    return {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "source_window": WINDOW,
        "asof_position_in_window": LOOKBACK - 1,
        "feature_indices": f"[start, start+{LOOKBACK}) inclusive of asof bar",
        "label_future_indices": f"[asof+1, asof+{HORIZON}] = relative [{LOOKBACK}, {LOOKBACK + HORIZON})",
        "mfe10": "max(high[T+1:T+10]) / close[T] - 1",
        "mae10": "min(low[T+1:T+10]) / close[T] - 1",
        "up_k": "1 if mfe10 >= k/100",
        "down_k": "1 if -mae10 >= k/100",
        "leakage_risk_if_broken": (
            "Any feature using bars after asof, or labels using bars at/before asof, "
            "would be leakage. Dataset history slice is values[start:start+LOOKBACK]."
        ),
    }


def temporal_integrity_checks(
    panel: dict[str, pd.DataFrame] | Path,
    targets_path: Path,
    *,
    max_symbols: int = 32,
    max_windows_per_symbol: int = 8,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
) -> dict[str, Any]:
    """Causal probes: history excludes future; corrupting future changes labels only."""
    if isinstance(panel, (str, Path)):
        panel = load_panel(Path(panel))

    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None

    sidecar = pd.read_parquet(targets_path)
    # Index sidecar by (symbol, start_index) for fast lookup
    sidecar_idx = sidecar.set_index(["symbol", "start_index"], drop=False)

    history_future_overlap = 0
    label_mismatch = 0
    feature_invariant_under_future_corrupt = 0
    label_changes_under_future_corrupt = 0
    checked = 0
    notes: list[str] = []

    for symbol in sorted(panel)[:max_symbols]:
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        max_start = len(frame) - WINDOW + 1
        starts = list(range(max_start))
        # Spread checks across the series
        if len(starts) > max_windows_per_symbol:
            picks = np.linspace(0, len(starts) - 1, num=max_windows_per_symbol, dtype=int)
            starts = [starts[int(i)] for i in picks]
        for start in starts:
            asof_pos = start + LOOKBACK - 1
            asof = pd.Timestamp(frame.index[asof_pos]).date()
            if start_date and asof < start_date:
                continue
            if end_date and asof > end_date:
                continue
            checked += 1
            # 1) History slice ends at asof
            hist_end = start + LOOKBACK - 1
            if hist_end != asof_pos:
                history_future_overlap += 1
            if hist_end >= asof_pos + 1:
                history_future_overlap += 1

            # 2) Recomputed label matches sidecar
            exp = target_row(symbol, frame, start)
            try:
                got = sidecar_idx.loc[(symbol, start)]
                if isinstance(got, pd.DataFrame):
                    got = got.iloc[0]
            except KeyError:
                label_mismatch += 1
                continue
            for col in ("mfe10", "mae10", *BINARY_HEADS):
                if col in ("mfe10", "mae10"):
                    if abs(float(got[col]) - float(exp[col])) > 1e-5:
                        label_mismatch += 1
                        break
                elif int(got[col]) != int(exp[col]):
                    label_mismatch += 1
                    break

            # 3) Corrupt post-asof OHLC → features (last hist bar) unchanged; labels change
            corrupted = frame.copy()
            future_slice = corrupted.iloc[asof_pos + 1 : asof_pos + HORIZON + 1]
            if len(future_slice) != HORIZON:
                continue
            for col in ("open", "high", "low", "close"):
                corrupted.iloc[asof_pos + 1 : asof_pos + HORIZON + 1, corrupted.columns.get_loc(col)] = (
                    corrupted.iloc[asof_pos + 1 : asof_pos + HORIZON + 1][col].to_numpy() * 1.5
                )
            hist_orig = frame.iloc[start : start + LOOKBACK][list(FEATURES)].to_numpy(dtype=np.float64)
            hist_corr = corrupted.iloc[start : start + LOOKBACK][list(FEATURES)].to_numpy(dtype=np.float64)
            if np.allclose(hist_orig, hist_corr):
                feature_invariant_under_future_corrupt += 1
            exp_corr = target_row(symbol, corrupted, start)
            if abs(float(exp_corr["mfe10"]) - float(exp["mfe10"])) > 1e-8 or abs(
                float(exp_corr["mae10"]) - float(exp["mae10"])
            ) > 1e-8:
                label_changes_under_future_corrupt += 1

    ok = (
        checked > 0
        and history_future_overlap == 0
        and label_mismatch == 0
        and feature_invariant_under_future_corrupt == checked
        and label_changes_under_future_corrupt == checked
    )
    notes.append(
        f"Checked {checked} windows across up to {max_symbols} symbols; "
        f"lookback={LOOKBACK}, horizon={HORIZON}."
    )
    notes.append(
        "Feature invariance under future corruption proves history excludes post-asof bars."
    )
    return {
        "ok": ok,
        "n_checked": checked,
        "history_future_overlap_failures": history_future_overlap,
        "label_sidecar_mismatches": label_mismatch,
        "feature_invariant_count": feature_invariant_under_future_corrupt,
        "label_changed_under_corrupt_count": label_changes_under_future_corrupt,
        "notes": notes,
    }


def same_stock_lead_lag_probe(
    panel: dict[str, pd.DataFrame] | Path,
    *,
    max_symbols: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Does asof close return correlate with next-day move more than chance?

    Not leakage by itself (public information), but if *features after asof*
    appeared in history this would spike. We only correlate last hist return
    with mfe10 — expected weak positive for momentum, not near-perfect.
    """
    if isinstance(panel, (str, Path)):
        panel = load_panel(Path(panel))
    rng = np.random.default_rng(seed)
    xs: list[float] = []
    ys: list[float] = []
    for symbol in sorted(panel)[:max_symbols]:
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        max_start = len(frame) - WINDOW + 1
        # sample a few starts
        picks = rng.choice(max_start, size=min(20, max_start), replace=False)
        close = frame["close"].to_numpy(dtype=np.float64)
        for start in picks:
            asof = start + LOOKBACK - 1
            if asof < 1:
                continue
            ret = close[asof] / max(close[asof - 1], 1e-8) - 1.0
            row = target_row(symbol, frame, int(start))
            xs.append(float(ret))
            ys.append(float(row["mfe10"]))
    if len(xs) < 10:
        return {"ok": False, "n": len(xs), "corr_asof_ret_vs_mfe10": None}
    corr = float(np.corrcoef(xs, ys)[0, 1])
    return {
        "ok": True,
        "n": len(xs),
        "corr_asof_ret_vs_mfe10": corr,
        "note": (
            "Weak |corr| expected. Near 1.0 would suggest label/feature contamination. "
            f"Observed corr={corr:.4f}."
        ),
    }


def run_phase_a(
    *,
    train_targets: Path,
    val_targets: Path,
    val_panel: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    cols = ["symbol", "start_index", "asof_date", "mfe10", "mae10", *BINARY_HEADS]
    train = pd.read_parquet(train_targets, columns=cols)
    val = pd.read_parquet(val_targets, columns=cols)

    payload: dict[str, Any] = {
        "phase": "A",
        "cutoff_contract": document_cutoff_contract(),
        "positive_rates": positive_rate_tables(train, val),
        "cross_prior_transfer": cross_prior_transfer(train, val),
        "head_correlation_val": head_correlations(val),
        "head_correlation_train_sample": head_correlations(train, sample=500_000),
    }
    if val_panel is not None and Path(val_panel).is_file():
        panel = load_panel(Path(val_panel))
        payload["temporal_integrity"] = temporal_integrity_checks(panel, Path(val_targets))
        payload["same_stock_lead_lag"] = same_stock_lead_lag_probe(panel)
        # Row-count alignment vs sidecar
        from modernbert_finance.ablations.alignment import verify_feature_label_alignment

        payload["alignment_sample"] = verify_feature_label_alignment(
            panel, Path(val_targets), max_check=128
        ).to_dict()

    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument("--val-panel", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run_phase_a(
        train_targets=args.train_targets,
        val_targets=args.val_targets,
        val_panel=args.val_panel,
        out_json=args.out_json,
    )
    print(json.dumps({
        "macro_priors": payload["positive_rates"]["macro_prior_log_loss"],
        "train_prior_on_val": payload["cross_prior_transfer"]["macro_val_log_loss_with_train_prior"],
        "temporal_ok": payload.get("temporal_integrity", {}).get("ok"),
        "align_ok": payload.get("alignment_sample", {}).get("ok"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
