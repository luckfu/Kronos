"""Cost-aware long-only backtest of frozen rank-fusion candidates (CPU only).

Candidates are equal-weight averages of same-day percentile ranks of
``predicted_return_d10`` (per model) and of the 10-row reversal ``-ret10``.
Nothing is tuned here: the book, the costs and the candidate list are fixed
before the run.

Book (declared up front, one primary and one secondary size):

- Each signal day T forms one sleeve with 1/10 of capital, held 10 sessions.
- Eligible names: traded on T and trailing 20-bar median amount >= 50M CNY
  (both known at the close of T).
- Long the top 10% of eligible names by score (primary) or the top 50
  (secondary), equal weight. No shorting.
- Execution reuses ``finetune.small_clean_oos.execution_constrained``:
  entry at the next session's open, one-word limit-up cannot be bought,
  halts defer, exit at the close of the 10th session (deferred to the next
  sellable open when blocked), slippage by amount tier, 0.3% per round trip.
  Unfilled weight stays in cash.
- Benchmark: every eligible name, same sleeves, same execution and costs.

Differences from the frozen 8b book, on purpose: scores are ranks, so there is
no ``|pred| >= 5%`` threshold; ``halt_gap`` is not applied because it reads
future label dates; an exit still blocked on the last bar is marked at that
bar's close and counted instead of being dropped.

Limit flags are inferred from front-adjusted bars (no unadjusted file, no ST
list), so one-word days around ex-rights dates and ST 5% limits can be
mis-labelled.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import fusion_analysis as fa  # noqa: E402
from finetune.small_clean_oos import execution_constrained as ec  # noqa: E402

CANDIDATES = {
    "c2": ["c2"],
    "c2+base": ["c2", "base"],
    "c2+rev": ["c2", "rev"],
    "c2+base+rev": ["c2", "base", "rev"],
    # single-signal references, not candidates
    "base": ["base"],
    "rev": ["rev"],
}
BOOKS = {"top_decile": ("fraction", 0.10), "top50": ("count", 50)}
BOOTSTRAP_BLOCK = 10
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261111


def infer_flags(bars: pd.DataFrame) -> pd.DataFrame:
    """Vectorised equivalent of ``ec.attach_limit_flags`` for bars without flag columns."""
    frame = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    prev = frame.groupby("symbol")["close"].shift(1)
    ratio = frame["symbol"].map(ec.limit_ratio).astype(float)
    prices = frame[["open", "high", "low", "close"]].to_numpy()
    flat = (prices.max(axis=1) - prices.min(axis=1)) <= 0.011
    has_prev = prev.notna().to_numpy() & (prev.fillna(0).to_numpy() > 0)
    up_px = np.floor(prev.to_numpy() * (1.0 + ratio.to_numpy()) * 100.0 + 0.5) / 100.0
    down_px = np.floor(prev.to_numpy() * (1.0 - ratio.to_numpy()) * 100.0 + 0.5) / 100.0
    close = frame["close"].to_numpy()
    frame["halted"] = False
    frame["one_word_limit_up"] = flat & has_prev & (np.abs(close - up_px) <= 0.011)
    frame["one_word_limit_down"] = flat & has_prev & (np.abs(close - down_px) <= 0.011)
    return frame


def resolve_long(tape: ec.SymbolTape, calendar: list[str], cal_index: dict, signal_index: int) -> dict:
    """One long round trip for a signal at ``calendar[signal_index]``."""
    entry = ec._resolve_entry(tape, calendar, signal_index, 1)
    if entry["status"] != "filled":
        return {"status": entry["status"]}
    exit_plan = ec._resolve_exit(tape, calendar, signal_index, 1, entry["entry_date"])
    marked = False
    if exit_plan["status"] == "unresolved":
        # Still unsellable on the last bar: mark at the last close instead of dropping the loser.
        last = tape.dates[-1]
        if last < entry["entry_date"]:
            return {"status": "unresolved"}
        exit_plan = {"exit_date": last, "exit_price": float(tape.close[-1]),
                     "exit_defer_sessions": exit_plan.get("exit_defer_sessions", 0)}
        marked = True
    elif exit_plan["status"] != "filled":
        return {"status": exit_plan["status"]}
    return {"status": "filled", "entry_date": entry["entry_date"], "entry_open": entry["entry_open"],
            "exit_date": exit_plan["exit_date"], "exit_price": exit_plan["exit_price"],
            "exit_deferred": int(exit_plan.get("exit_defer_sessions", 0)) > 0, "marked_at_end": marked}


def trade_path(tape: ec.SymbolTape, trade: dict, slip: float, cal_index: dict):
    """Daily PnL slices (same marking as ``ec._pnl_slices``) on calendar indices."""
    i0, i1 = tape.index[trade["entry_date"]], tape.index[trade["exit_date"]]
    fill_in = trade["entry_open"] * (1.0 + slip)
    fill_out = trade["exit_price"] * (1.0 - slip)
    marks = tape.close[i0:i1 + 1].astype(np.float64).copy()
    marks[-1] = fill_out
    refs = np.concatenate(([fill_in], marks[:-1]))
    pnl = (marks - refs) / fill_in
    days = np.array([cal_index[d] for d in tape.dates[i0:i1 + 1]], dtype=np.int64)
    return days, pnl


def block_bootstrap(values: np.ndarray, rng) -> list[float]:
    n = len(values)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, max(1, n - BOOTSTRAP_BLOCK + 1)))
            idx.extend(range(start, min(start + BOOTSTRAP_BLOCK, n)))
        draws.append(float(values[idx[:n]].mean()))
    return draws


def mean_ci(values: np.ndarray, seed_offset: int = 0) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = block_bootstrap(values, rng)
    return {"mean": float(values.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--c2", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    models = {"base": fa.load_model(Path(args.base)), "c2": fa.load_model(Path(args.c2))}
    frame = fa.align(models)
    for name in models:
        frame[f"r_{name}"] = fa.ranks(frame, f"p_{name}")
    frame["ret10"] = fa.reversal_10d(Path(args.raw), frame)
    frame["r_rev"] = fa.ranks(frame.assign(_n=-frame["ret10"]), "_n")
    for name, parts in CANDIDATES.items():
        frame[f"s_{name}"] = frame[[f"r_{p}" for p in parts]].mean(axis=1)

    bars = ec.normalize_bars(pd.read_csv(args.raw))
    bars = infer_flags(bars)
    bars["is_st"] = False
    tapes = ec._build_tapes(bars)
    calendar = sorted(set(bars["date"].tolist()))
    cal_index = {d: i for i, d in enumerate(calendar)}
    n_days = len(calendar)

    # Resolve every eligible (symbol, signal day) long trade once; books only select subsets.
    resolved: dict[tuple[str, str], dict] = {}
    eligible_rows = []
    reasons: dict[str, int] = {}
    for row in frame[["symbol", "asof_date"]].itertuples(index=True):
        tape = tapes.get(row.symbol)
        signal_index = cal_index.get(row.asof_date)
        if tape is None or signal_index is None or tape.loc(row.asof_date) is None:
            reasons["signal_day_no_bar"] = reasons.get("signal_day_no_bar", 0) + 1
            continue
        median_amount = ec._trailing_median_amount(tape, row.asof_date)
        slip = None if median_amount is None else ec.slippage_rate(median_amount)
        if slip is None:
            reasons["illiquid"] = reasons.get("illiquid", 0) + 1
            continue
        eligible_rows.append(row.Index)
        trade = resolve_long(tape, calendar, cal_index, signal_index)
        trade["slip"] = slip
        if trade["status"] == "filled":
            trade["days"], trade["pnl"] = trade_path(tape, trade, slip, cal_index)
            trade["net"] = float(trade["pnl"].sum()) - ec.ROUND_TRIP_COST
            trade["exit_index"] = cal_index[trade["exit_date"]]
        resolved[(row.symbol, row.asof_date)] = trade
    eligible = frame.loc[eligible_rows].copy()
    signal_days = sorted(eligible["asof_date"].unique())

    def run_book(selector) -> dict:
        gross = np.zeros(n_days)
        cost = np.zeros(n_days)
        exposure = np.zeros(n_days)
        sleeve_net, counts = [], {"selected": 0, "filled": 0, "exit_deferred": 0, "marked_at_end": 0}
        skipped: dict[str, int] = {}
        slips = []
        for asof, day in eligible.groupby("asof_date", sort=True):
            picks = selector(day)
            weight = ec.LAYER_CAPITAL / len(picks)
            sleeve = 0.0
            for symbol in picks:
                trade = resolved[(symbol, asof)]
                counts["selected"] += 1
                if trade["status"] != "filled":
                    skipped[trade["status"]] = skipped.get(trade["status"], 0) + 1
                    continue
                counts["filled"] += 1
                counts["exit_deferred"] += int(trade["exit_deferred"])
                counts["marked_at_end"] += int(trade["marked_at_end"])
                slips.append(trade["slip"])
                np.add.at(gross, trade["days"], weight * trade["pnl"])
                cost[trade["exit_index"]] += weight * ec.ROUND_TRIP_COST
                exposure[trade["days"]] += weight
                sleeve += trade["net"] / len(picks)
            sleeve_net.append(sleeve)
        return {"gross": gross, "net": gross - cost, "exposure": exposure,
                "sleeve_net": np.array(sleeve_net), "counts": counts, "skipped": skipped,
                "mean_slippage_per_side": float(np.mean(slips)) if slips else None}

    def selector_for(score: str, kind: str, size: float):
        def select(day: pd.DataFrame) -> list[str]:
            n = max(1, math.ceil(size * len(day))) if kind == "fraction" else int(min(size, len(day)))
            ordered = day.sort_values([score, "symbol"], ascending=[False, True])
            return ordered["symbol"].head(n).tolist()
        return select

    universe = run_book(lambda day: day["symbol"].tolist())
    active = np.flatnonzero(universe["exposure"] > 0)
    window = slice(int(active.min()), int(active.max()) + 1)

    def summarize(book: dict) -> dict:
        net = book["net"][window]
        nav = np.cumprod(1.0 + net)
        peak = np.maximum.accumulate(nav)
        curve = [{"net_return": float(r), "nav": float(v), "drawdown": float(v / p - 1.0)}
                 for r, v, p in zip(net, nav, peak)]
        metrics = ec._nav_metrics(curve)
        return {
            "total_net_return": metrics["total_return"],
            "total_gross_return": float(np.prod(1.0 + book["gross"][window]) - 1.0),
            "sharpe_annualized": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "nav_days": metrics["nav_days"],
            "mean_exposure": float(book["exposure"][window].mean()),
            "fill_rate": book["counts"]["filled"] / max(1, book["counts"]["selected"]),
            "counts": book["counts"], "skipped": book["skipped"],
            "mean_slippage_per_side": book["mean_slippage_per_side"],
            "sleeve_net_return_10d": {"mean": float(book["sleeve_net"].mean()),
                                      "win_rate": float((book["sleeve_net"] > 0).mean())},
        }

    report: dict = {
        "window": {"signal_start": signal_days[0], "signal_end": signal_days[-1],
                   "signal_days": len(signal_days), "nav_start": calendar[window.start],
                   "nav_end": calendar[window.stop - 1]},
        "rules": {"hold_sessions": ec.HOLD_SESSIONS, "sleeve_capital": ec.LAYER_CAPITAL,
                  "round_trip_cost": ec.ROUND_TRIP_COST,
                  "min_trailing_median_amount": ec.MIN_TRAILING_MEDIAN_AMOUNT,
                  "slippage_tiers": [list(t) for t in ec.SLIPPAGE_TIERS_CNY],
                  "entry": "next_open", "exit": "close_of_10th_session",
                  "limit_flags": "inferred_from_front_adjusted_bars", "short_side": "none",
                  "halt_gap_filter": "not_applied",
                  "bootstrap": {"block": BOOTSTRAP_BLOCK, "replicates": BOOTSTRAP_REPS,
                                "seed": BOOTSTRAP_SEED}},
        "eligibility": {"aligned_rows": int(len(frame)), "eligible_rows": int(len(eligible)),
                        "mean_eligible_per_day": float(eligible.groupby("asof_date").size().mean()),
                        "dropped": reasons},
        "universe_equal_weight": summarize(universe),
        "books": {},
    }

    daily: dict[tuple[str, str], np.ndarray] = {}
    sleeves: dict[tuple[str, str], np.ndarray] = {}
    for book_name, (kind, size) in BOOKS.items():
        report["books"][book_name] = {}
        for i, cand in enumerate(CANDIDATES):
            book = run_book(selector_for(f"s_{cand}", kind, size))
            daily[(book_name, cand)] = book["net"][window]
            sleeves[(book_name, cand)] = book["sleeve_net"]
            summary = summarize(book)
            summary["daily_excess_vs_universe"] = mean_ci(book["net"][window] - universe["net"][window], i)
            summary["sleeve_excess_vs_universe_10d"] = mean_ci(book["sleeve_net"] - universe["sleeve_net"], i)
            # How much of the next sleeve bought 10 sessions later is already held (netting potential).
            picks = {asof: set(selector_for(f"s_{cand}", kind, size)(day))
                     for asof, day in eligible.groupby("asof_date", sort=True)}
            keep1 = [len(picks[a] & picks[b]) / len(picks[b]) for a, b in zip(signal_days, signal_days[1:])]
            keep10 = [len(picks[a] & picks[b]) / len(picks[b]) for a, b in zip(signal_days, signal_days[10:])]
            summary["selection_overlap"] = {"next_day": float(np.mean(keep1)),
                                            "ten_sessions_later": float(np.mean(keep10))}
            report["books"][book_name][cand] = summary
        report["books"][book_name]["paired_vs_c2"] = {
            cand: {"daily_net": mean_ci(daily[(book_name, cand)] - daily[(book_name, "c2")], 100 + j),
                   "sleeve_net_10d": mean_ci(sleeves[(book_name, cand)] - sleeves[(book_name, "c2")], 200 + j)}
            for j, cand in enumerate(CANDIDATES) if cand != "c2"}

    # Label-based decile legs (close-to-close D10 label, no costs), relative to the same-day mean.
    # Shows how much of the rank signal sits in the short leg a long-only book cannot trade.
    def decile_legs(data: pd.DataFrame, score: str) -> dict:
        rows = []
        for _, day in data.groupby("asof_date"):
            n = max(1, math.ceil(0.10 * len(day)))
            ordered = day.sort_values(score)["actual_return_d10"]
            mean = float(day["actual_return_d10"].mean())
            rows.append((ordered.tail(n).mean() - mean, ordered.head(n).mean() - mean,
                         day[score].corr(day["actual_return_d10"], method="spearman")))
        top, bottom, ic = np.array(rows, dtype=np.float64).mean(axis=0)
        return {"top_decile_excess_10d": float(top), "bottom_decile_excess_10d": float(bottom),
                "rank_ic": float(ic)}

    report["label_decile_legs"] = {
        cand: {"all_names": decile_legs(frame, f"s_{cand}"),
               "eligible_names": decile_legs(eligible, f"s_{cand}")}
        for cand in CANDIDATES}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ec.json_ready(report, include_rows=True), indent=2,
                                         ensure_ascii=False) + "\n")
    print(json.dumps({"out": args.out, "eligible_rows": len(eligible)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
