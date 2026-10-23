"""C2 execution-constrained backtest (sealed-window plan step 8b).

Frozen book, not a sweep. Section 3.1 landed in the 0.05–0.10 band, so the
only authorized book is ``|predicted_return_d10| >= 5%`` after ex-ante
liquidity and halt filters. Full-sample top/bottom baskets are not built.

Execution, which the older overlay and August backtests do not model:

- Signal at the close of session T. The order goes out at the next session's
  open.
- A one-word limit-up day cannot be bought. A one-word limit-down day cannot
  be sold. An entry that would need that print is cancelled. An exit that
  would need it waits for the next open that can trade.
- A halt on the intended fill session waits for the resume session. The
  signal itself is dropped when the name is already halted on T.
- Slippage is a fixed tier of the trailing 20-session median amount known at
  the signal close. Tiers are not fit on an OOS window.
- Each signal day is one sleeve of a 10-session hold. Sleeve capital is
  ``1/10`` of the book (the same layering as the frozen overlay), held through
  the close of the 10th market session after T, which is the D10 label
  session when the name trades every session.
- Round-trip cost is 0.3% once per filled name (双边), not 0.3% per side.

PnL is marked from the bar panel. Prediction ``actual_return_*`` columns are
ignored, so a close-to-close label cannot stand in for the open fill.

``halt_gap`` is ``panel.halt_gap_mask``: the same label-date lag the panel
calls ex-ante. It uses the shard's label calendar, not the realized return.
Signal-day halts and the liquidity window additionally use only bars dated
on or before T.

A run is the section 5.2 result only when ``publishable_as_section_5_2`` is
true. That requires the signal dates 2026-10-09..2026-10-23, bars through
the planned exit (D10 of the last signal, 2026-11-06 on the weekday
calendar), explicit or unadjusted limit flags, and the halt-gap columns.
The consumed sealed window 2026-09-18..2026-09-30 and the 34-day panel are
rejected by that flag. Synthetic smoke sets it false as well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finetune.small_clean_oos.panel import HORIZONS, halt_gap_mask

PRED_ABS_MIN = 0.05
HOLD_SESSIONS = 10
LAYER_CAPITAL = 1.0 / HOLD_SESSIONS
ROUND_TRIP_COST = 0.003
LIQUIDITY_LOOKBACK = 20
MIN_TRAILING_MEDIAN_AMOUNT = 50_000_000.0
MAX_ENTRY_DEFER_SESSIONS = 10
MAX_EXIT_DEFER_SESSIONS = 20
TRADING_DAYS_PER_YEAR = 252
TARGET_SIGNAL_START = "2026-10-09"
TARGET_SIGNAL_END = "2026-10-23"
LABEL_MATURITY = "2026-11-06"
CONSUMED_SEALED_START = "2026-09-18"
CONSUMED_SEALED_END = "2026-09-30"
FLAG_COLUMNS = ("halted", "one_word_limit_up", "one_word_limit_down")
PUBLISHABLE_FLAG_MODES = ("explicit", "unadjusted_file")
# Fixed per-side slippage by trailing median amount (CNY). The lowest rung
# matches the liquidity floor, so every name that passes the filter has a tier.
SLIPPAGE_TIERS_CNY = (
    (500_000_000.0, 0.0005),
    (200_000_000.0, 0.0010),
    (100_000_000.0, 0.0015),
    (50_000_000.0, 0.0020),
)


def slippage_rate(median_amount: float) -> float | None:
    """Per-side slippage for a trailing median amount, or None if not tradable."""
    if median_amount is None or not math.isfinite(median_amount):
        return None
    for floor, rate in SLIPPAGE_TIERS_CNY:
        if median_amount >= floor:
            return rate
    return None


def limit_ratio(symbol: str, is_st: bool = False) -> float:
    """Exchange limit versus the previous close. ST 5%, ChiNext/STAR 20%, BSE 30%."""
    if is_st:
        return 0.05
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    if digits.startswith(("300", "301", "688", "689")):
        return 0.20
    if digits.startswith(("83", "87", "43", "92")):
        return 0.30
    return 0.10


def _tick_round(price: float) -> float:
    return math.floor(price * 100.0 + 0.5) / 100.0


def _date_str(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return ""
    return stamp.strftime("%Y-%m-%d")


def _one_word_at_limit(open_, high, low, close, prev_close, ratio: float, side: int) -> bool:
    prices = (float(open_), float(high), float(low), float(close))
    if not all(math.isfinite(px) and px > 0 for px in prices):
        return False
    if max(prices) - min(prices) > 0.011:
        return False
    if not math.isfinite(prev_close) or prev_close <= 0:
        return False
    limit_px = _tick_round(prev_close * (1.0 + side * ratio))
    return abs(prices[3] - limit_px) <= 0.011


@dataclass
class SymbolTape:
    dates: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    amount: np.ndarray
    halted: np.ndarray
    one_up: np.ndarray
    one_down: np.ndarray
    is_st: np.ndarray
    index: dict[str, int]

    def loc(self, date: str) -> int | None:
        return self.index.get(date)


def _infer_row_flags(open_, high, low, close, prev_close, ratio: float, volume, halted_hint) -> tuple[bool, bool, bool]:
    halted = bool(halted_hint) if halted_hint is not None and not (isinstance(halted_hint, float) and math.isnan(halted_hint)) else False
    if volume is not None and math.isfinite(volume) and volume <= 0:
        halted = True
    if halted or prev_close is None:
        return halted, False, False
    up = _one_word_at_limit(open_, high, low, close, prev_close, ratio, side=1)
    down = _one_word_at_limit(open_, high, low, close, prev_close, ratio, side=-1)
    return halted, up, down


def attach_limit_flags(bars: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Use explicit halt/limit columns when every row has them; otherwise infer.

    Inference compares each bar with the previous bar of the same symbol. It
    is the right test on unadjusted prints. Front-adjusted panels mis-label
    one-word days across an ex-rights gap, so inferred flags are not a
    publishable section 5.2 input.
    """
    frame = bars.copy()
    if all(column in frame.columns for column in FLAG_COLUMNS):
        present = frame.loc[:, FLAG_COLUMNS]
        if present.notna().all().all():
            for column in FLAG_COLUMNS:
                frame[column] = frame[column].astype(bool)
            return frame, "explicit"
    for column in FLAG_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    halted = np.zeros(len(frame), dtype=bool)
    one_up = np.zeros(len(frame), dtype=bool)
    one_down = np.zeros(len(frame), dtype=bool)
    prev_close: dict[str, float] = {}
    for i, row in frame.iterrows():
        symbol = row["symbol"]
        ratio = limit_ratio(symbol, bool(row["is_st"]) if "is_st" in frame.columns and pd.notna(row.get("is_st", np.nan)) else False)
        volume = float(row["volume"]) if "volume" in frame.columns and pd.notna(row.get("volume", np.nan)) else None
        hint = row["halted"] if pd.notna(row["halted"]) else None
        # An explicit flag on this row wins over the price test.
        if pd.notna(row["one_word_limit_up"]) or pd.notna(row["one_word_limit_down"]) or hint is not None:
            is_halt = bool(hint) if hint is not None else False
            if volume is not None and volume <= 0:
                is_halt = True
            halted[i] = is_halt
            one_up[i] = False if is_halt else bool(row["one_word_limit_up"]) if pd.notna(row["one_word_limit_up"]) else False
            one_down[i] = False if is_halt else bool(row["one_word_limit_down"]) if pd.notna(row["one_word_limit_down"]) else False
        else:
            halted[i], one_up[i], one_down[i] = _infer_row_flags(
                row["open"], row["high"], row["low"], row["close"], prev_close.get(symbol), ratio, volume, None,
            )
        if not halted[i] and math.isfinite(row["close"]) and row["close"] > 0:
            prev_close[symbol] = float(row["close"])
    frame["halted"] = halted
    frame["one_word_limit_up"] = one_up
    frame["one_word_limit_down"] = one_down
    return frame, "inferred_from_bars"


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    frame = bars.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["date"] = frame["date"].map(_date_str)
    frame = frame[frame["date"] != ""]
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[(frame["close"] > 0) & (frame["open"] > 0)]
    if "amount" in frame.columns:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    else:
        frame["amount"] = np.nan
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    if "is_st" in frame.columns:
        frame["is_st"] = frame["is_st"].fillna(False).astype(bool)
    frame = frame.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return frame.reset_index(drop=True)


def normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "predicted_return_d10" not in frame.columns:
        if "pred_d10" in frame.columns:
            frame["predicted_return_d10"] = frame["pred_d10"]
        else:
            raise ValueError("predictions need predicted_return_d10")
    for column in ("symbol", "asof_date"):
        if column not in frame.columns:
            raise ValueError(f"predictions missing {column}")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["asof_date"] = frame["asof_date"].map(_date_str)
    frame["predicted_return_d10"] = pd.to_numeric(frame["predicted_return_d10"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "asof_date", "predicted_return_d10"])
    frame = frame[frame["asof_date"] != ""]
    if "model" in frame.columns:
        labels = set(frame["model"].astype(str).str.lower())
        allowed = {"c2", "c2_best", "c2-best"}
        if not labels <= allowed:
            raise ValueError(f"8b is C2 only; found model labels {sorted(labels - allowed)}")
    if frame.duplicated(["symbol", "asof_date"]).any():
        raise ValueError("duplicate symbol/asof_date rows in predictions")
    return frame.reset_index(drop=True)


def _label_columns() -> list[str]:
    return [f"label_date_d{horizon}" for horizon in HORIZONS]


def apply_halt_gap(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Attach ``halt_gap`` via the panel helper when every label date exists."""
    columns = _label_columns()
    out = frame.copy()
    if not all(column in out.columns for column in columns):
        out["halt_gap"] = False
        out["bad_label"] = False
        return out, "unavailable"
    valid = np.ones(len(out), dtype=bool)
    for column in columns:
        parsed = out[column].map(_date_str)
        valid &= parsed.to_numpy() != ""
        out[column] = parsed.where(parsed.to_numpy() != "", np.nan)
    out["bad_label"] = ~valid
    mask = np.zeros(len(out), dtype=bool)
    if valid.any():
        sub = out.loc[valid, ["asof_date", *columns]].copy()
        mask[np.flatnonzero(valid)] = np.asarray(halt_gap_mask(sub), dtype=bool)
    out["halt_gap"] = mask
    return out, "applied"


def _build_tapes(bars: pd.DataFrame) -> dict[str, SymbolTape]:
    tapes = {}
    for symbol, group in bars.groupby("symbol", sort=False):
        group = group.sort_values("date")
        dates = group["date"].tolist()
        tapes[symbol] = SymbolTape(
            dates=dates,
            open=group["open"].to_numpy(dtype=np.float64),
            high=group["high"].to_numpy(dtype=np.float64),
            low=group["low"].to_numpy(dtype=np.float64),
            close=group["close"].to_numpy(dtype=np.float64),
            amount=group["amount"].to_numpy(dtype=np.float64),
            halted=group["halted"].to_numpy(dtype=bool),
            one_up=group["one_word_limit_up"].to_numpy(dtype=bool),
            one_down=group["one_word_limit_down"].to_numpy(dtype=bool),
            is_st=group["is_st"].to_numpy(dtype=bool) if "is_st" in group.columns else np.zeros(len(group), dtype=bool),
            index={date: i for i, date in enumerate(dates)},
        )
    return tapes


def _trailing_median_amount(tape: SymbolTape, asof: str) -> float | None:
    """Median amount of the last 20 traded bars on or before ``asof``."""
    cursor = tape.loc(asof)
    if cursor is None:
        return None
    amounts = []
    for j in range(cursor, -1, -1):
        if tape.halted[j]:
            continue
        amount = tape.amount[j]
        if math.isfinite(amount) and amount > 0:
            amounts.append(float(amount))
        if len(amounts) >= LIQUIDITY_LOOKBACK:
            break
    if len(amounts) < LIQUIDITY_LOOKBACK:
        return None
    median = float(np.median(np.asarray(amounts, dtype=np.float64)))
    if median < MIN_TRAILING_MEDIAN_AMOUNT:
        return None
    return median


def _entry_blocked(side: int, one_up: bool, one_down: bool) -> bool:
    if side > 0:
        return bool(one_up)
    return bool(one_down)


def _exit_blocked(side: int, one_up: bool, one_down: bool) -> bool:
    if side > 0:
        return bool(one_down)
    return bool(one_up)


def _resolve_entry(tape: SymbolTape, calendar: list[str], signal_index: int, side: int) -> dict:
    deferred = 0
    examined = 0
    for date in calendar[signal_index + 1:]:
        examined += 1
        if examined > MAX_ENTRY_DEFER_SESSIONS:
            break
        loc = tape.loc(date)
        if loc is None or tape.halted[loc]:
            deferred += 1
            continue
        if _entry_blocked(side, tape.one_up[loc], tape.one_down[loc]):
            reason = "skipped_one_word_limit_up" if side > 0 else "skipped_one_word_limit_down"
            return {"status": reason, "entry_defer_sessions": deferred}
        return {
            "status": "filled",
            "entry_date": date,
            "entry_open": float(tape.open[loc]),
            "entry_defer_sessions": deferred,
        }
    return {"status": "unfilled_halt", "entry_defer_sessions": deferred}


def _resolve_exit(tape: SymbolTape, calendar: list[str], signal_index: int, side: int, entry_date: str) -> dict:
    planned_index = signal_index + HOLD_SESSIONS
    if planned_index >= len(calendar):
        return {"status": "horizon_not_in_calendar"}
    planned = calendar[planned_index]
    if entry_date > planned:
        return {"status": "expired_before_fill"}
    # Walk from the planned session forward until a print can take the exit.
    deferred = 0
    for date in calendar[planned_index:]:
        if deferred > MAX_EXIT_DEFER_SESSIONS:
            return {"status": "unresolved", "exit_defer_sessions": deferred}
        loc = tape.loc(date)
        one_up = False if loc is None else bool(tape.one_up[loc])
        one_down = False if loc is None else bool(tape.one_down[loc])
        blocked = loc is None or bool(tape.halted[loc]) or _exit_blocked(side, one_up, one_down)
        if blocked:
            deferred += 1
            continue
        if date == planned:
            return {
                "status": "filled",
                "exit_date": date,
                "exit_price": float(tape.close[loc]),
                "exit_session": "planned_close",
                "exit_defer_sessions": 0,
            }
        return {
            "status": "filled",
            "exit_date": date,
            "exit_price": float(tape.open[loc]),
            "exit_session": "deferred_open",
            "exit_defer_sessions": deferred,
        }
    return {"status": "unresolved", "exit_defer_sessions": deferred}


def _pnl_slices(tape: SymbolTape, trade: dict, slip: float) -> list[tuple[str, float]]:
    side = trade["side"]
    entry_open = trade["entry_open"]
    exit_price = trade["exit_price"]
    if side > 0:
        fill_in = entry_open * (1.0 + slip)
        fill_out = exit_price * (1.0 - slip)
    else:
        fill_in = entry_open * (1.0 - slip)
        fill_out = exit_price * (1.0 + slip)
    if fill_in <= 0 or fill_out <= 0:
        return []
    path = []
    for date in tape.dates:
        if date < trade["entry_date"] or date > trade["exit_date"]:
            continue
        loc = tape.index[date]
        if tape.halted[loc]:
            continue
        path.append(date)
    if not path or path[0] != trade["entry_date"] or path[-1] != trade["exit_date"]:
        return []
    ref = fill_in
    slices = []
    for date in path:
        loc = tape.index[date]
        new = fill_out if date == trade["exit_date"] else float(tape.close[loc])
        if side > 0:
            pnl = (new - ref) / fill_in
        else:
            pnl = (ref - new) / fill_in
        slices.append((date, pnl))
        if date == trade["exit_date"]:
            break
        ref = float(tape.close[loc])
    return slices


def publication_blockers(
    signal_dates: list[str],
    bar_max: str | None,
    *,
    synthetic: bool,
    limit_flag_mode: str,
    halt_gap_filter: str,
    amount_missing: bool,
    unresolved: int,
    horizon_missing: int,
    bad_labels: int,
    planned_exit_max: str | None,
) -> list[str]:
    """Reasons this run must not be quoted as the section 5.2 number."""
    reasons = []
    if synthetic:
        reasons.append("synthetic_or_interface_smoke")
    if not signal_dates:
        reasons.append("no_signal_dates")
    else:
        if any(date < TARGET_SIGNAL_START or date > TARGET_SIGNAL_END for date in signal_dates):
            reasons.append("signal_window_is_not_2026-10-09_2026-10-23")
        if any(CONSUMED_SEALED_START <= date <= CONSUMED_SEALED_END for date in signal_dates):
            reasons.append("consumed_sealed_window_2026-09-18_2026-09-30")
    if bar_max is None or bar_max < LABEL_MATURITY:
        reasons.append("bars_do_not_reach_2026-11-06")
    if planned_exit_max and (bar_max is None or bar_max < planned_exit_max):
        reasons.append("bars_do_not_cover_planned_exit")
    if limit_flag_mode not in PUBLISHABLE_FLAG_MODES:
        reasons.append("limit_flags_not_explicit_or_unadjusted")
    if halt_gap_filter != "applied":
        reasons.append("halt_gap_filter_unavailable")
    if amount_missing:
        reasons.append("amount_column_missing")
    if unresolved:
        reasons.append("unresolved_exits")
    if horizon_missing:
        reasons.append("horizon_not_in_calendar")
    if bad_labels:
        reasons.append("incomplete_label_dates")
    return reasons


def _nav_metrics(curve: list[dict]) -> dict:
    if not curve:
        return {
            "nav_days": 0,
            "total_return": 0.0,
            "annualized_return": None,
            "daily_mean": None,
            "daily_std": None,
            "sharpe": None,
            "max_drawdown": 0.0,
            "calmar": None,
            "nav_end": 1.0,
        }
    returns = np.array([row["net_return"] for row in curve], dtype=np.float64)
    nav = float(curve[-1]["nav"])
    n = len(returns)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if n > 1 else float("nan")
    annualized = (nav ** (TRADING_DAYS_PER_YEAR / n) - 1.0) if nav > 0 else float("nan")
    sharpe = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR) if std and math.isfinite(std) else float("nan")
    max_dd = float(min(row["drawdown"] for row in curve))
    calmar = annualized / abs(max_dd) if max_dd < 0 and math.isfinite(annualized) else float("nan")
    return {
        "nav_days": n,
        "total_return": float(nav - 1.0),
        "annualized_return": None if not math.isfinite(annualized) else float(annualized),
        "daily_mean": mean,
        "daily_std": None if not math.isfinite(std) else std,
        "sharpe": None if not math.isfinite(sharpe) else float(sharpe),
        "max_drawdown": max_dd,
        "calmar": None if not math.isfinite(calmar) else float(calmar),
        "nav_end": nav,
    }


def _rule_echo() -> dict:
    return {
        "model": "C2",
        "score": "predicted_return_d10",
        "abs_threshold": PRED_ABS_MIN,
        "threshold_comparison": ">=",
        "full_sample_topk": False,
        "hold_sessions": HOLD_SESSIONS,
        "layer_capital": LAYER_CAPITAL,
        "round_trip_cost": ROUND_TRIP_COST,
        "round_trip_cost_meaning": "0.003 once per filled round trip, not per side",
        "entry": "next_session_open",
        "exit": "close_of_10th_market_session_after_signal",
        "deferred_exit": "next_eligible_open",
        "one_word_limit_up": "cannot_buy",
        "one_word_limit_down": "cannot_sell",
        "halt_on_fill_day": "defer_to_resume",
        "liquidity_lookback_sessions": LIQUIDITY_LOOKBACK,
        "min_trailing_median_amount": MIN_TRAILING_MEDIAN_AMOUNT,
        "slippage_tiers_cny": [
            {"min_median_amount": floor, "per_side": rate} for floor, rate in SLIPPAGE_TIERS_CNY
        ],
        "halt_gap": "finetune.small_clean_oos.panel.halt_gap_mask",
    }


def run_execution_backtest(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    synthetic: bool = False,
    limit_flag_mode_override: str | None = None,
) -> dict:
    """Run the frozen 8b book. See the module docstring for the rules."""
    preds = normalize_predictions(predictions)
    preds, halt_gap_filter = apply_halt_gap(preds)
    bar_frame = normalize_bars(bars)
    amount_missing = bool(bar_frame["amount"].isna().all())
    bar_frame, limit_flag_mode = attach_limit_flags(bar_frame)
    if limit_flag_mode_override:
        limit_flag_mode = limit_flag_mode_override
    if "is_st" not in bar_frame.columns:
        bar_frame["is_st"] = False
    tapes = _build_tapes(bar_frame)
    # Every date in the bar file is a market session. A symbol missing that
    # date, or flagged halted, defers. Weekends are simply absent.
    calendar = sorted(set(bar_frame["date"].tolist()))
    cal_index = {date: i for i, date in enumerate(calendar)}
    trades: list[dict] = []
    rejects: list[dict] = []
    daily_gross: dict[str, float] = {}
    daily_cost: dict[str, float] = {}
    daily_one_way: dict[str, float] = {}
    daily_exposure: dict[str, float] = {}
    bad_labels = int(preds["bad_label"].sum()) if "bad_label" in preds.columns else 0
    horizon_missing = 0

    for asof, day in preds.groupby("asof_date", sort=True):
        if asof not in cal_index:
            for row in day.itertuples(index=False):
                rejects.append({"symbol": row.symbol, "asof_date": asof, "reason": "signal_not_on_calendar"})
            continue
        candidates = []
        for row in day.itertuples(index=False):
            pred = float(row.predicted_return_d10)
            if abs(pred) < PRED_ABS_MIN:
                continue
            if bool(getattr(row, "bad_label", False)):
                rejects.append({"symbol": row.symbol, "asof_date": asof, "reason": "incomplete_label_dates"})
                continue
            if bool(row.halt_gap):
                rejects.append({"symbol": row.symbol, "asof_date": asof, "reason": "halt_gap"})
                continue
            tape = tapes.get(row.symbol)
            loc = None if tape is None else tape.loc(asof)
            if tape is None or loc is None or tape.halted[loc]:
                rejects.append({"symbol": row.symbol, "asof_date": asof, "reason": "signal_day_halt"})
                continue
            median_amount = _trailing_median_amount(tape, asof)
            slip = None if median_amount is None else slippage_rate(median_amount)
            if median_amount is None or slip is None:
                rejects.append({"symbol": row.symbol, "asof_date": asof, "reason": "illiquid"})
                continue
            candidates.append((row, pred, tape, median_amount, slip))
        if not candidates:
            continue
        weight = LAYER_CAPITAL / len(candidates)
        signal_index = cal_index[asof]
        for row, pred, tape, median_amount, slip in candidates:
            side = 1 if pred > 0 else -1
            entry = _resolve_entry(tape, calendar, signal_index, side)
            if entry["status"] != "filled":
                rejects.append({
                    "symbol": row.symbol,
                    "asof_date": asof,
                    "reason": entry["status"],
                    "entry_defer_sessions": entry.get("entry_defer_sessions", 0),
                })
                continue
            exit_plan = _resolve_exit(tape, calendar, signal_index, side, entry["entry_date"])
            if exit_plan["status"] != "filled":
                rejects.append({"symbol": row.symbol, "asof_date": asof, "reason": exit_plan["status"]})
                if exit_plan["status"] == "horizon_not_in_calendar":
                    horizon_missing += 1
                continue
            trade = {
                "symbol": row.symbol,
                "asof_date": asof,
                "side": side,
                "weight": weight,
                "predicted_return_d10": pred,
                "median_amount": median_amount,
                "slippage_per_side": slip,
                "entry_date": entry["entry_date"],
                "entry_open": entry["entry_open"],
                "entry_defer_sessions": int(entry["entry_defer_sessions"]),
                "exit_date": exit_plan["exit_date"],
                "exit_price": exit_plan["exit_price"],
                "exit_session": exit_plan["exit_session"],
                "exit_defer_sessions": int(exit_plan["exit_defer_sessions"]),
            }
            slices = _pnl_slices(tape, trade, slip)
            if not slices:
                rejects.append({"symbol": row.symbol, "asof_date": asof, "reason": "unresolved"})
                continue
            gross = float(sum(pnl for _, pnl in slices))
            trade["gross_return"] = gross
            trade["net_return"] = gross - ROUND_TRIP_COST
            trades.append(trade)
            for date, pnl in slices:
                daily_gross[date] = daily_gross.get(date, 0.0) + weight * pnl
            daily_cost[trade["exit_date"]] = daily_cost.get(trade["exit_date"], 0.0) + weight * ROUND_TRIP_COST
            daily_one_way[trade["entry_date"]] = daily_one_way.get(trade["entry_date"], 0.0) + weight
            daily_one_way[trade["exit_date"]] = daily_one_way.get(trade["exit_date"], 0.0) + weight
            for date in calendar:
                if trade["entry_date"] <= date <= trade["exit_date"]:
                    daily_exposure[date] = daily_exposure.get(date, 0.0) + weight

    curve = []
    nav = 1.0
    peak = 1.0
    if trades:
        start = min(trade["entry_date"] for trade in trades)
        end = max(trade["exit_date"] for trade in trades)
        for date in calendar:
            if date < start or date > end:
                continue
            gross = daily_gross.get(date, 0.0)
            cost = daily_cost.get(date, 0.0)
            net = gross - cost
            one_way = daily_one_way.get(date, 0.0)
            nav *= 1.0 + net
            peak = max(peak, nav)
            drawdown = nav / peak - 1.0
            curve.append({
                "date": date,
                "gross_return": gross,
                "cost": cost,
                "net_return": net,
                "one_way_turnover": one_way,
                "two_way_turnover": 0.5 * one_way,
                "gross_exposure": daily_exposure.get(date, 0.0),
                "nav": nav,
                "drawdown": drawdown,
            })

    signal_dates = sorted(preds["asof_date"].unique())
    bar_dates = sorted(bar_frame["date"].unique())
    bar_max = bar_dates[-1] if bar_dates else None
    planned_exits = []
    for asof in signal_dates:
        if asof in cal_index and cal_index[asof] + HOLD_SESSIONS < len(calendar):
            planned_exits.append(calendar[cal_index[asof] + HOLD_SESSIONS])
    planned_exit_max = max(planned_exits) if planned_exits else None
    unresolved = sum(1 for row in rejects if row["reason"] == "unresolved")
    blockers = publication_blockers(
        signal_dates,
        bar_max,
        synthetic=synthetic,
        limit_flag_mode=limit_flag_mode,
        halt_gap_filter=halt_gap_filter,
        amount_missing=amount_missing,
        unresolved=unresolved,
        horizon_missing=horizon_missing,
        bad_labels=bad_labels,
        planned_exit_max=planned_exit_max,
    )
    publishable = not blockers
    nets = [trade["net_return"] for trade in trades]
    metrics = _nav_metrics(curve)
    exposures = [row["gross_exposure"] for row in curve]
    turnovers = [row["two_way_turnover"] for row in curve]
    role = "section_5_2" if publishable else ("interface_smoke" if synthetic else "interface_only")
    return {
        "role": role,
        "publishable_as_section_5_2": publishable,
        "metrics_are_section_5_2": publishable,
        "publication_blockers": blockers,
        "rule": _rule_echo(),
        "limit_flag_mode": limit_flag_mode,
        "halt_gap_filter": halt_gap_filter,
        "synthetic": synthetic,
        "signal_start": signal_dates[0] if signal_dates else None,
        "signal_end": signal_dates[-1] if signal_dates else None,
        "signal_dates": len(signal_dates),
        "bar_date_min": bar_dates[0] if bar_dates else None,
        "bar_date_max": bar_max,
        "planned_exit_max": planned_exit_max,
        "target_signal_window": [TARGET_SIGNAL_START, TARGET_SIGNAL_END],
        "label_maturity": LABEL_MATURITY,
        "counts": _counts(preds, trades, rejects),
        "per_trade": {
            "n": len(trades),
            "mean_net_return": float(np.mean(nets)) if nets else None,
            "median_net_return": float(np.median(nets)) if nets else None,
            "win_rate": float(np.mean([net > 0 for net in nets])) if nets else None,
        },
        "book": {
            **metrics,
            "mean_two_way_turnover": float(np.mean(turnovers)) if turnovers else None,
            "mean_gross_exposure": float(np.mean(exposures)) if exposures else None,
            "sum_daily_net_return": float(sum(row["net_return"] for row in curve)) if curve else 0.0,
        },
        "nav_curve": curve,
        "trades": trades,
        "rejects": rejects,
    }


def _counts(preds: pd.DataFrame, trades: list[dict], rejects: list[dict]) -> dict:
    threshold = preds["predicted_return_d10"].abs() >= PRED_ABS_MIN
    reasons: dict[str, int] = {}
    for row in rejects:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    return {
        "prediction_rows": int(len(preds)),
        "threshold_rows": int(threshold.sum()),
        "filled": len(trades),
        "entry_deferrals": int(sum(1 for trade in trades if trade["entry_defer_sessions"] > 0)),
        "exit_deferrals": int(sum(1 for trade in trades if trade["exit_defer_sessions"] > 0)),
        "rejects": reasons,
    }


def json_ready(payload: dict, *, include_rows: bool = False) -> dict:
    """Drop trade rows by default and turn non-finite floats into None."""
    skip = set() if include_rows else {"trades", "rejects", "nav_curve"}
    return _json_ready({key: value for key, value in payload.items() if key not in skip})


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
