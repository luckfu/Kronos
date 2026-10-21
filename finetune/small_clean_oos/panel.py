"""Direction + amplitude evaluation panel.

The production criterion so far has been cross-sectional rank IC, but the
stated goal is to predict the *direction and the size* of the move.  Those are
different capabilities and a single number cannot score both, so this module
reports three layers and never collapses them into one:

L1 money    The P&L of actually following the signal: sign(prediction) times
            the realized return, net of the 0.4% round-trip cost the Beta
            utility contract already uses.  Because the cost is charged on
            every trade, a model only clears it where its predicted amplitude
            is large enough -- so this layer is reported *stratified by the
            model's own predicted amplitude*, never as a single average.
L2 amplitude  Whether the model can rank and calibrate the size of the move:
            rank correlation between |prediction| and |realized| across the
            cross-section, and the calibration curve (predicted amplitude
            decile -> realized amplitude).  The curve is what shows whether
            the predicted amplitude means anything at all.
L3 ranking  Cross-sectional rank IC, the criterion currently in production,
            so that a candidate can be required not to regress on it.

Ex-ante vs outcome-conditioned subsets
--------------------------------------
The amplitude buckets used by the earlier C04 comparison (``abs_ge_5pct``,
``amplitude_top_decile``) select rows by their *realized* amplitude.  They are
outcome-conditioned, so they are not tradable ex ante and are kept here only
as diagnostics.  The halt-gap subset below is different: halt status is known
before the trade, and the subset is derived from the label *dates* (the next
traded rows) rather than from realized returns, so it is ex ante.

Everything here is plain numpy/pandas so the same code scores the archived
C2/C03 shards locally and the new checkpoints inside the Kaggle kernel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

COST = 0.004
HORIZONS = list(range(1, 11))
BLOCK, REPLICATES, RNG_SEED = 10, 2000, 20261012
AMPLITUDE_DECILES = 10
AMPLITUDE_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08)
HALT_LAG_TOLERANCE_DAYS = 3


# --------------------------------------------------------------------- helpers
def spearman(x, y):
    """Spearman rank correlation of two 1-D arrays (NaN when undefined)."""
    import numpy as np
    import pandas as pd

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den else float("nan")


def _columns(frame):
    import numpy as np

    predicted = frame[[f"predicted_return_d{h}" for h in HORIZONS]].to_numpy(dtype=np.float64)
    actual = frame[[f"actual_return_d{h}" for h in HORIZONS]].to_numpy(dtype=np.float64)
    return predicted, actual


def dates_of(frame):
    return frame["asof_date"].to_numpy()


# ------------------------------------------------------------------- L1 money
def _money_by_decile(frame):
    """Direction accuracy and P&L per within-day decile of |predicted_return_d10|.

    The decile is an *ex-ante* split: it uses only the model's own prediction,
    so it can be applied before the trade.  A model that predicts well should
    show a monotonically rising direction accuracy and a rising net P&L, with
    only the high-amplitude deciles clearing the cost.
    """
    import numpy as np
    import pandas as pd

    work = frame[["asof_date", "predicted_return_d10", "actual_return_d10"]].copy()
    work["abs_pred"] = work["predicted_return_d10"].abs()
    work["decile"] = work.groupby("asof_date")["abs_pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), AMPLITUDE_DECILES,
                          labels=False, duplicates="drop"))
    work["signed"] = np.sign(work["predicted_return_d10"]) * work["actual_return_d10"]
    grouped = work.groupby("decile")
    rows = []
    for decile, part in grouped:
        signed = part["signed"].to_numpy()
        rows.append(dict(
            decile=int(decile),
            samples=int(len(part)),
            mean_abs_pred=float(part["abs_pred"].mean()),
            mean_abs_actual=float(part["actual_return_d10"].abs().mean()),
            direction=float((signed > 0).mean()),
            signed_return=float(signed.mean()),
            net_pnl=float(signed.mean() - COST),
        ))
    return rows


def _money_by_threshold(frame):
    """Coverage, direction accuracy and P&L at absolute |prediction| thresholds."""
    import numpy as np

    predicted = frame["predicted_return_d10"].to_numpy(dtype=np.float64)
    actual = frame["actual_return_d10"].to_numpy(dtype=np.float64)
    rows = []
    for threshold in AMPLITUDE_THRESHOLDS:
        selected = np.abs(predicted) > threshold
        if not selected.any():
            rows.append(dict(threshold=float(threshold), coverage=0.0, samples=0,
                             direction=None, signed_return=None, net_pnl=None))
            continue
        signed = np.sign(predicted[selected]) * actual[selected]
        rows.append(dict(
            threshold=float(threshold),
            coverage=float(selected.mean()),
            samples=int(selected.sum()),
            direction=float((signed > 0).mean()),
            signed_return=float(signed.mean()),
            net_pnl=float(signed.mean() - COST),
        ))
    return rows


def money_layer(frame):
    import numpy as np

    predicted = frame["predicted_return_d10"].to_numpy(dtype=np.float64)
    actual = frame["actual_return_d10"].to_numpy(dtype=np.float64)
    signed = np.sign(predicted) * actual
    daily = []
    days = dates_of(frame)
    for day in np.unique(days):
        selected = days == day
        daily.append(float(signed[selected].mean()))
    daily = np.asarray(daily, dtype=np.float64)
    daily_net = daily - COST
    std = float(daily_net.std(ddof=1)) if len(daily_net) > 1 else 0.0
    deciles = _money_by_decile(frame)
    top = deciles[-1] if deciles else None
    return dict(
        d10_direction=float((signed > 0).mean()),
        signed_return=float(signed.mean()),
        net_pnl=float(signed.mean() - COST),
        pnl_daily_mean=float(daily_net.mean()),
        pnl_daily_std=std,
        pnl_sharpe_annualized=float(daily_net.mean() / std * math.sqrt(252)) if std else None,
        pnl_positive_days=int((daily_net > 0).sum()),
        pnl_days=int(len(daily_net)),
        d10_prediction_actual_product=float((predicted * actual).mean()),
        by_predicted_amplitude_decile=deciles,
        by_predicted_amplitude_threshold=_money_by_threshold(frame),
        top_amplitude_decile=top,
    )


# ---------------------------------------------------------------- L2 amplitude
def amplitude_layer(frame):
    import numpy as np

    predicted = frame["predicted_return_d10"].to_numpy(dtype=np.float64)
    actual = frame["actual_return_d10"].to_numpy(dtype=np.float64)
    days = dates_of(frame)
    daily = []
    for day in np.unique(days):
        selected = days == day
        value = spearman(np.abs(predicted[selected]), np.abs(actual[selected]))
        if math.isfinite(value):
            daily.append(value)
    daily = np.asarray(daily, dtype=np.float64)
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    calibration = []
    for row in _money_by_decile(frame):
        ratio = row["mean_abs_actual"] / row["mean_abs_pred"] if row["mean_abs_pred"] else None
        calibration.append(dict(decile=row["decile"], samples=row["samples"],
                                 mean_abs_pred=row["mean_abs_pred"],
                                 mean_abs_actual=row["mean_abs_actual"],
                                 realized_over_predicted=ratio))
    return dict(
        amplitude_rank_ic_daily_mean=float(daily.mean()) if len(daily) else None,
        amplitude_rank_ic_daily_std=std,
        amplitude_rank_ic_icir=float(daily.mean() / std) if std else None,
        amplitude_rank_ic_positive_days=int((daily > 0).sum()) if len(daily) else 0,
        amplitude_rank_ic_pooled=spearman(np.abs(predicted), np.abs(actual)),
        mean_abs_pred_d10=float(np.abs(predicted).mean()),
        mean_abs_actual_d10=float(np.abs(actual).mean()),
        median_amplitude_ratio=float(np.median(
            np.abs(predicted) / np.maximum(np.abs(actual), 1e-9))),
        calibration_curve=calibration,
    )


# ------------------------------------------------------------------ L3 ranking
def ranking_layer(frame):
    import numpy as np

    predicted = frame["predicted_return_d10"].to_numpy(dtype=np.float64)
    actual = frame["actual_return_d10"].to_numpy(dtype=np.float64)
    days = dates_of(frame)
    daily_ic, daily_spread = [], []
    for day in np.unique(days):
        selected = days == day
        value = spearman(predicted[selected], actual[selected])
        if math.isfinite(value):
            daily_ic.append(value)
        order = np.argsort(predicted[selected])
        sorted_actual = actual[selected][order]
        keep = max(1, int(math.ceil(len(sorted_actual) * 0.10)))
        daily_spread.append(float(sorted_actual[-keep:].mean() - sorted_actual[:keep].mean()))
    daily_ic = np.asarray(daily_ic, dtype=np.float64)
    daily_spread = np.asarray(daily_spread, dtype=np.float64)
    ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else 0.0
    spread_std = float(daily_spread.std(ddof=1)) if len(daily_spread) > 1 else 0.0
    return dict(
        d10_rank_ic_daily_mean=float(daily_ic.mean()) if len(daily_ic) else None,
        d10_rank_ic_daily_std=ic_std,
        d10_rank_ic_icir=float(daily_ic.mean() / ic_std) if ic_std else None,
        d10_rank_ic_positive_days=int((daily_ic > 0).sum()) if len(daily_ic) else 0,
        d10_rank_ic_pooled=spearman(predicted, actual),
        top_bottom_10pct_spread_mean=float(daily_spread.mean()) if len(daily_spread) else None,
        top_bottom_10pct_spread_std=spread_std,
        top_bottom_10pct_spread_sharpe=(
            float(daily_spread.mean() / spread_std * math.sqrt(252)) if spread_std else None),
        top_bottom_10pct_positive_days=int((daily_spread > 0).sum()) if len(daily_spread) else 0,
    )


# ------------------------------------------------------- official (archive-compatible)
def official_stats(frame):
    """Exactly the semantics of evaluate.score, so archives stay comparable."""
    import numpy as np

    predicted, actual = _columns(frame)
    error = predicted - actual
    direction = np.sign(predicted) == np.sign(actual)
    result = dict(
        samples=int(len(frame)),
        path_direction=float(direction.mean()),
        path_return_mae=float(np.abs(error).mean()),
        path_return_rmse=float(math.sqrt(float(np.square(error).mean()))),
        horizons={},
    )
    for index in range(10):
        result["horizons"][str(index + 1)] = dict(
            direction=float(direction[:, index].mean()),
            return_mae=float(np.abs(error[:, index]).mean()),
            return_rmse=float(math.sqrt(float(np.square(error[:, index]).mean()))),
        )
    ics = []
    for day in frame.groupby("asof_date"):
        if len(day[1]) < 3:
            continue
        value = day[1].predicted_return_d10.corr(day[1].actual_return_d10, method="spearman")
        if np.isfinite(value):
            ics.append(float(value))
    result["d10_daily_rank_ic"] = float(np.mean(ics)) if ics else None
    return result


# ------------------------------------------------------------------- halt subset
def halt_gap_mask(frame):
    """Rows whose label horizon is pushed out by a suspension.

    The OOS label is the next ten *actually traded* rows, so a suspended stock
    has its label dates pushed out relative to the market.  Halt status is
    public before the trade and the subset is built from label dates rather
    than from realized returns, so it is ex ante, unlike the outcome-
    conditioned amplitude buckets.
    """
    import numpy as np
    import pandas as pd

    work = frame[["asof_date"] + [f"label_date_d{h}" for h in HORIZONS]].copy()
    for h in HORIZONS:
        work[f"lag{h}"] = (pd.to_datetime(work[f"label_date_d{h}"])
                           - pd.to_datetime(work["asof_date"])).dt.days
    lags = [f"lag{h}" for h in HORIZONS]
    median = work.groupby("asof_date")[lags].transform("median")
    flagged = np.zeros(len(work), dtype=bool)
    for column in lags:
        flagged |= work[column].to_numpy() > median[column].to_numpy() + HALT_LAG_TOLERANCE_DAYS
    return flagged


def halt_subset_stats(frame):
    import numpy as np

    flagged = halt_gap_mask(frame)
    predicted = frame["predicted_return_d10"].to_numpy(dtype=np.float64)
    actual = frame["actual_return_d10"].to_numpy(dtype=np.float64)
    out = {"rows": int(flagged.sum()), "fraction": float(flagged.mean())}
    for tag, selected in (("halt", flagged), ("normal", ~flagged)):
        if not selected.any():
            out[tag] = None
            continue
        signed = np.sign(predicted[selected]) * actual[selected]
        out[tag] = dict(
            samples=int(selected.sum()),
            d10_direction=float((signed > 0).mean()),
            signed_return=float(signed.mean()),
            net_pnl=float(signed.mean() - COST),
            mean_abs_actual_d10=float(np.abs(actual[selected]).mean()),
            d10_rmse=float(math.sqrt(float(((predicted[selected] - actual[selected]) ** 2).mean()))),
        )
    return out


# ------------------------------------------------------------------ panel entry
def panel(frame, with_subsets: bool = True) -> dict:
    """The full three-layer panel for one model's OOS frame."""
    result = dict(
        official=official_stats(frame),
        L1_money=money_layer(frame),
        L2_amplitude=amplitude_layer(frame),
        L3_ranking=ranking_layer(frame),
    )
    if with_subsets:
        result["halt_gap"] = halt_subset_stats(frame)
    return result


# --------------------------------------------------------- paired block bootstrap
def block_indices(count: int, dates: list[str]):
    """Moving block bootstrap index sets over the signal dates."""
    import numpy as np

    rng = np.random.default_rng(RNG_SEED)
    samples = []
    for _ in range(REPLICATES):
        picked: list[int] = []
        while len(picked) < count:
            start = int(rng.integers(0, max(1, count - BLOCK + 1)))
            picked.extend(range(start, start + BLOCK))
        samples.append([dates[i] for i in picked[:count]])
    return samples


def daily_ratio_stats(frame):
    """Per-date sufficient statistics for the ratio-of-sums metrics."""
    import numpy as np

    days = dates_of(frame)
    predicted = frame["predicted_return_d10"].to_numpy(dtype=np.float64)
    actual = frame["actual_return_d10"].to_numpy(dtype=np.float64)
    signed = np.sign(predicted) * actual
    error = predicted - actual
    stats = {}
    for day in np.unique(days):
        selected = days == day
        stats[str(day)] = (
            int(selected.sum()),
            int((signed[selected] > 0).sum()),
            float(np.abs(error[selected]).sum()),
            float(np.square(error[selected]).sum()),
            float(signed[selected].sum()),
        )
    return stats


def aggregate_ratio(stats, days):
    import numpy as np

    total = sum(stats[day][0] for day in days if day in stats)
    if total == 0:
        return None
    direction = sum(stats[day][1] for day in days if day in stats) / total
    mae = sum(stats[day][2] for day in days if day in stats) / total
    rmse = math.sqrt(sum(stats[day][3] for day in days if day in stats) / total)
    signed = sum(stats[day][4] for day in days if day in stats) / total
    return dict(direction=float(direction), return_mae=float(mae),
                return_rmse=float(rmse), signed_return=float(signed),
                net_pnl=float(signed - COST), samples=int(total))


def daily_rank_values(frame):
    """Per-date rank IC, amplitude rank IC and top-bottom spread."""
    import numpy as np

    predicted = frame["predicted_return_d10"].to_numpy(dtype=np.float64)
    actual = frame["actual_return_d10"].to_numpy(dtype=np.float64)
    days = dates_of(frame)
    values = {}
    for day in np.unique(days):
        selected = days == day
        p, a = predicted[selected], actual[selected]
        order = np.argsort(p)
        sorted_actual = a[order]
        keep = max(1, int(math.ceil(len(sorted_actual) * 0.10)))
        values[str(day)] = dict(
            rank_ic=spearman(p, a),
            amplitude_rank_ic=spearman(np.abs(p), np.abs(a)),
            spread=float(sorted_actual[-keep:].mean() - sorted_actual[:keep].mean()),
        )
    return values


def _mean(values):
    import numpy as np

    array = np.asarray([v for v in values if v is not None and math.isfinite(v)],
                       dtype=np.float64)
    return float(array.mean()) if len(array) else None


def paired_deltas(base_frame, candidate_frame, dates: list[str],
                  blocks=None) -> dict:
    """Paired candidate-minus-base deltas with a moving block bootstrap CI.

    Every layer gets a confidence interval, including the rank IC that the
    earlier comparisons left without one.  Ratio metrics aggregate as
    ratio-of-sums over the resampled dates; rank metrics average the
    per-date values.
    """
    import numpy as np

    blocks = blocks if blocks is not None else block_indices(len(dates), dates)
    base_ratio = daily_ratio_stats(base_frame)
    candidate_ratio = daily_ratio_stats(candidate_frame)
    base_rank = daily_rank_values(base_frame)
    candidate_rank = daily_rank_values(candidate_frame)

    base_point = aggregate_ratio(base_ratio, dates)
    candidate_point = aggregate_ratio(candidate_ratio, dates)
    result = {"samples": base_point["samples"] if base_point else 0}

    ratio_keys = ("direction", "return_mae", "return_rmse", "signed_return", "net_pnl")
    draws = {key: [] for key in ratio_keys}
    for sample in blocks:
        left = aggregate_ratio(base_ratio, sample)
        right = aggregate_ratio(candidate_ratio, sample)
        if left is None or right is None:
            continue
        for key in ratio_keys:
            draws[key].append(right[key] - left[key])
    for key in ratio_keys:
        if not draws[key]:
            result[key] = None
            continue
        values = np.asarray(draws[key], dtype=np.float64)
        result[key] = dict(
            value=float(candidate_point[key] - base_point[key]),
            block_bootstrap_95ci=[float(np.quantile(values, 0.025)),
                                  float(np.quantile(values, 0.975))],
            bootstrap_mean=float(values.mean()),
        )

    rank_keys = ("rank_ic", "amplitude_rank_ic", "spread")
    rank_draws = {key: [] for key in rank_keys}
    for sample in blocks:
        for key in rank_keys:
            left = [base_rank[day][key] for day in sample if day in base_rank]
            right = [candidate_rank[day][key] for day in sample if day in candidate_rank]
            left_mean, right_mean = _mean(left), _mean(right)
            if left_mean is None or right_mean is None:
                continue
            rank_draws[key].append(right_mean - left_mean)
    for key in rank_keys:
        if not rank_draws[key]:
            result[key] = None
            continue
        values = np.asarray(rank_draws[key], dtype=np.float64)
        base_mean = _mean([base_rank[day][key] for day in dates if day in base_rank])
        candidate_mean = _mean([candidate_rank[day][key] for day in dates if day in candidate_rank])
        result[key] = dict(
            value=float(candidate_mean - base_mean),
            block_bootstrap_95ci=[float(np.quantile(values, 0.025)),
                                  float(np.quantile(values, 0.975))],
            bootstrap_mean=float(values.mean()),
        )
    return result


# ------------------------------------------------------------------ shard loading
def load_shards(directory, label: str):
    """Concatenate the per-date shards of one model."""
    import pandas as pd

    parts = sorted(Path(directory).glob(f"{label}_*.csv.gz"))
    if not parts:
        raise RuntimeError(f"No shards for {label} in {directory}")
    frame = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
    return frame.sort_values("identity").reset_index(drop=True)
