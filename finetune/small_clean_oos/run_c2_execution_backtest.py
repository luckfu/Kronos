"""Driver for the C2 step-8b execution-constrained backtest.

Reads a C2 prediction shard directory (or one CSV) and an OHLCV bar file.
Does not train, does not tune thresholds, and does not treat the consumed
sealed window or a synthetic smoke as the section 5.2 result.

When the prediction or bar paths are missing, writes a blocker document with
``metrics`` set to null. Full rules and the data gap are in
``docs/C2_EXECUTION_CONSTRAINED_BACKTEST_8B_CN.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune.small_clean_oos.execution_constrained import (  # noqa: E402
    FLAG_COLUMNS,
    attach_limit_flags,
    json_ready,
    normalize_bars,
    run_execution_backtest,
)
from finetune.small_clean_oos.panel import load_shards  # noqa: E402

DEFAULT_BLOCKER = ROOT / "finetune/reports/c2_execution_constrained_8b_blocker.json"


def load_predictions(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return load_shards(path, "c2_best")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def bars_with_unadjusted_flags(bars: pd.DataFrame, unadjusted: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Copy halt / one-word flags inferred from an unadjusted print onto ``bars``.

    Prices on ``bars`` stay as given (front-adjusted panels keep their
    returns). Rows with no unadjusted match are marked halted so they cannot
    fill, and the unmatched count is returned for the publication gate.
    """
    flagged, _mode = attach_limit_flags(normalize_bars(unadjusted))
    flags = flagged[["symbol", "date", *FLAG_COLUMNS]].copy()
    base = normalize_bars(bars).drop(columns=[column for column in FLAG_COLUMNS if column in bars.columns])
    merged = base.merge(flags, on=["symbol", "date"], how="left")
    missing = int(merged["halted"].isna().sum())
    merged["halted"] = merged["halted"].where(merged["halted"].notna(), True).astype(bool)
    for column in ("one_word_limit_up", "one_word_limit_down"):
        merged[column] = merged[column].where(merged[column].notna(), False).astype(bool)
    return merged, missing


def _refresh_role(result: dict) -> dict:
    publishable = not result["publication_blockers"]
    result["publishable_as_section_5_2"] = publishable
    result["metrics_are_section_5_2"] = publishable
    if publishable:
        result["role"] = "section_5_2"
    elif result.get("synthetic"):
        result["role"] = "interface_smoke"
    else:
        result["role"] = "interface_only"
    return result


def synthetic_smoke_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tiny unadjusted panel used only to prove the driver wiring.

    The dates are in 2024 on purpose. Do not quote the PnL.
    """
    dates = pd.bdate_range("2024-03-01", periods=36).strftime("%Y-%m-%d").tolist()
    signal = dates[20]
    entry = dates[21]
    rows = []
    for date in dates:
        price = 10.0
        volume = 1_000_000.0
        if date == entry:
            price = 11.0  # one-word limit-up versus the previous 10 close; buy is refused
        rows.append({
            "symbol": "sz.000001",
            "date": date,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "amount": 80_000_000.0,
            "volume": volume,
        })
        rows.append({
            "symbol": "sz.000002",
            "date": date,
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "amount": 80_000_000.0,
            "volume": 1_000_000.0,
        })
    labels = {f"label_date_d{horizon}": dates[20 + horizon] for horizon in range(1, 11)}
    predictions = pd.DataFrame([
        {"symbol": "sz.000001", "asof_date": signal, "predicted_return_d10": 0.08, "model": "c2_best", **labels},
        {"symbol": "sz.000002", "asof_date": signal, "predicted_return_d10": 0.06, "model": "c2_best", **labels},
    ])
    return predictions, pd.DataFrame(rows)


def blocker_document(reason: str, detail: str) -> dict:
    return {
        "role": "section_5_2_blocker",
        "publishable_as_section_5_2": False,
        "metrics_are_section_5_2": False,
        "metrics": None,
        "blocked": True,
        "reason": reason,
        "detail": detail,
        "target_signal_window": ["2026-10-09", "2026-10-23"],
        "label_maturity": "2026-11-06",
        "consumed_sealed_window_not_usable": ["2026-09-18", "2026-09-30"],
        "required_inputs": {
            "predictions": "C2 shards c2_best_YYYY-MM-DD.csv.gz or one CSV with symbol, asof_date, predicted_return_d10, label_date_d1..d10",
            "bars": "daily OHLCV: symbol, date, open, high, low, close, amount; halt and one-word flags, or a separate unadjusted file",
            "kaggle_c2_checkpoint": "wynstonliu/kronos-small-c2-best-checkpoint",
            "kaggle_old_panels_not_the_result": [
                "smmt315/a-share-120d-temporal-symbol-holdout",
                "wynstonliu/a-share-120d-oos-20260803-20260930",
            ],
        },
    }


def _write_result(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = json_ready(result, include_rows=False)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    stem = output.with_suffix("")
    if result.get("nav_curve"):
        pd.DataFrame(result["nav_curve"]).to_csv(stem.with_name(stem.name + "_nav.csv"), index=False)
    if result.get("trades"):
        pd.DataFrame(result["trades"]).to_csv(stem.with_name(stem.name + "_trades.csv"), index=False)
    if result.get("rejects"):
        pd.DataFrame(result["rejects"]).to_csv(stem.with_name(stem.name + "_rejects.csv"), index=False)


def _print_status(result: dict) -> None:
    print(json.dumps({
        "role": result["role"],
        "publishable_as_section_5_2": result["publishable_as_section_5_2"],
        "publication_blockers": result["publication_blockers"],
        "signal_start": result.get("signal_start"),
        "signal_end": result.get("signal_end"),
        "bar_date_max": result.get("bar_date_max"),
        "counts": result.get("counts"),
    }, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C2 execution-constrained backtest (plan step 8b)")
    parser.add_argument("--predictions", type=Path, help="C2 shard directory or CSV")
    parser.add_argument("--bars", type=Path, help="OHLCV CSV/parquet used for prices, amount, and optional flags")
    parser.add_argument("--unadjusted-bars", type=Path, default=None,
                        help="Unadjusted OHLCV used only to mark halts and one-word limit days")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Run the in-process 2024 fixture. Not a section 5.2 result.")
    args = parser.parse_args(argv)

    if args.smoke:
        if args.output and "smoke" not in args.output.name:
            print("refusing to write smoke output to a path whose name does not contain 'smoke'", file=sys.stderr)
            return 2
        predictions, bars = synthetic_smoke_inputs()
        result = run_execution_backtest(predictions, bars, synthetic=True)
        _print_status(result)
        if args.output:
            _write_result(result, args.output)
        return 0

    if args.predictions is None or args.bars is None:
        document = blocker_document(
            "market_data_or_c2_predictions_not_provided",
            "Pass --predictions and --bars. This checkout has no 2026-10-09..10-23 C2 shards and no bar file through 2026-11-06, so there is no section 5.2 PnL.",
        )
        destination = args.output or DEFAULT_BLOCKER
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"wrote": str(destination), "publishable_as_section_5_2": False, "metrics": None}, indent=2))
        return 0

    try:
        predictions = load_predictions(args.predictions)
        bars = load_bars(args.bars)
    except FileNotFoundError as exc:
        document = blocker_document("input_path_missing", str(exc))
        destination = args.output or DEFAULT_BLOCKER
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"wrote": str(destination), "publishable_as_section_5_2": False, "metrics": None}, indent=2))
        return 0

    override = None
    if args.unadjusted_bars is not None:
        if not args.unadjusted_bars.exists():
            document = blocker_document("unadjusted_bars_missing", str(args.unadjusted_bars))
            destination = args.output or DEFAULT_BLOCKER
            destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
            return 0
        bars, unmatched = bars_with_unadjusted_flags(bars, load_bars(args.unadjusted_bars))
        override = "unadjusted_file"
    else:
        unmatched = 0

    result = run_execution_backtest(predictions, bars, limit_flag_mode_override=override)
    if unmatched:
        result["publication_blockers"].append("unmatched_unadjusted_rows")
        result["unmatched_unadjusted_rows"] = unmatched
        _refresh_role(result)
    if args.output:
        _write_result(result, args.output)
    _print_status(result)
    if not result["publishable_as_section_5_2"]:
        print("not a section 5.2 result; see publication_blockers", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
