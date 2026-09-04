"""Fixed-capital Top-5 backtest with next-open entries and barrier exits."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)


def load_panel(path):
    with Path(path).open("rb") as handle:
        return NumpyCompatUnpickler(handle).load()


def candidate_path(row, panel, min_amount):
    frame = panel[row.symbol]
    try:
        position = frame.index.get_loc(pd.Timestamp(row.asof_date))
    except KeyError:
        return None
    if position + 10 >= len(frame):
        return None
    asof = frame.iloc[position]
    if float(asof.amount) < min_amount:
        return None
    future = frame.iloc[position + 1 : position + 11]
    entry = float(future.open.iloc[0])
    previous_close = float(asof.close)
    first = future.iloc[0]
    one_price_up = (
        float(first.high) == float(first.low)
        and entry / previous_close - 1 >= 0.095
    )
    if not np.isfinite(entry) or entry <= 0 or one_price_up:
        return None
    return {
        "entry_date": future.index[0],
        "entry_price": entry,
        "dates": list(future.index),
        "high": future.high.to_numpy(float),
        "low": future.low.to_numpy(float),
        "close": future.close.to_numpy(float),
        "asof_amount": float(asof.amount),
    }


def planned_exit(path, take_profit, stop_loss):
    entry = path["entry_price"]
    if take_profit is None or stop_loss is None:
        return (
            path["dates"][-1], 10,
            float(path["close"][-1] / entry - 1), "horizon",
        )
    for offset, (high, low) in enumerate(zip(path["high"], path["low"]), 1):
        hit_stop = low / entry - 1 <= -stop_loss
        hit_take = high / entry - 1 >= take_profit
        if hit_stop:
            return path["dates"][offset - 1], offset, -stop_loss, "stop_loss"
        if hit_take:
            return path["dates"][offset - 1], offset, take_profit, "take_profit"
    return path["dates"][-1], 10, float(path["close"][-1] / entry - 1), "horizon"


def simulate(frame, panel, take_profit, stop_loss, slots, cost_bps, min_amount):
    cost = cost_bps / 10000
    slot_state = [{"capital": 1.0 / slots, "available": pd.Timestamp.min} for _ in range(slots)]
    active_symbols = {}
    trades = []
    for asof_date, day in frame.groupby("asof_date", sort=True):
        signal_date = pd.Timestamp(asof_date)
        active_symbols = {
            symbol: exit_date for symbol, exit_date in active_symbols.items()
            if exit_date > signal_date
        }
        available = [state for state in slot_state if state["available"] <= signal_date]
        if not available:
            continue
        ranked = day.sort_values("predicted_return_d10", ascending=False)
        for row in ranked.itertuples(index=False):
            if not available:
                break
            if row.symbol in active_symbols:
                continue
            path = candidate_path(row, panel, min_amount)
            if path is None:
                continue
            exit_date, exit_day, gross, reason = planned_exit(path, take_profit, stop_loss)
            state = available.pop(0)
            net = (1 - cost) * (1 + gross) * (1 - cost) - 1
            state["capital"] *= 1 + net
            state["available"] = pd.Timestamp(exit_date)
            active_symbols[row.symbol] = pd.Timestamp(exit_date)
            trades.append({
                "model": row.model,
                "symbol": row.symbol,
                "asof_date": asof_date,
                "entry_date": str(path["entry_date"].date()),
                "exit_date": str(pd.Timestamp(exit_date).date()),
                "exit_day": exit_day,
                "exit_reason": reason,
                "gross_return": gross,
                "net_return": net,
                "asof_amount": path["asof_amount"],
            })
    trades = pd.DataFrame(trades)
    final_equity = sum(state["capital"] for state in slot_state)
    return trades, {
        "trades": int(len(trades)),
        "final_return": float(final_equity - 1),
        "mean_trade_return_net": float(trades.net_return.mean()) if len(trades) else None,
        "median_trade_return_net": float(trades.net_return.median()) if len(trades) else None,
        "win_rate_net": float((trades.net_return > 0).mean()) if len(trades) else None,
        "mean_exit_day": float(trades.exit_day.mean()) if len(trades) else None,
        "exit_reason_counts": trades.exit_reason.value_counts().to_dict() if len(trades) else {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--input", action="append", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slots", type=int, default=5)
    parser.add_argument("--cost-bps-per-side", type=float, default=20)
    parser.add_argument("--min-amount", type=float, default=50_000_000)
    args = parser.parse_args()
    panel = load_panel(args.panel)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rules = {
        "hold_d10": (None, None),
        "tp3_sl2": (0.03, 0.02),
        "tp5_sl3": (0.05, 0.03),
    }
    result = {}
    trade_frames = []
    for label, source_model, path in args.input:
        frame = pd.read_csv(path)
        frame = frame.loc[frame.model == source_model].copy()
        frame["model"] = label
        result[label] = {}
        for rule, (tp, sl) in rules.items():
            trades, metrics = simulate(
                frame, panel, tp, sl, args.slots, args.cost_bps_per_side, args.min_amount
            )
            trades["rule"] = rule
            result[label][rule] = metrics
            trade_frames.append(trades)
    pd.concat(trade_frames, ignore_index=True).to_csv(
        output / "august_fixed_top5_trades.csv", index=False
    )
    summary = {
        "selection": "highest predicted_return_d10 while a slot is available",
        "slots": args.slots,
        "entry": "next trading day open",
        "duplicate_positions": False,
        "minimum_asof_amount": args.min_amount,
        "cost_bps_per_side": args.cost_bps_per_side,
        "barrier_ordering": "stop first when both intraday barriers are touched",
        "metrics": result,
    }
    (output / "august_fixed_top5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
