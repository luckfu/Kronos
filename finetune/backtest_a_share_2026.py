"""Pilot out-of-sample backtest for the A-share size-conditioned predictor."""

import argparse
import gc
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference


FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']
PRED_LEN = 10
LOOKBACK = 90


def time_features(index):
    index = pd.DatetimeIndex(index)
    return np.column_stack([
        index.minute,
        index.hour,
        index.weekday,
        index.day,
        index.month,
    ]).astype(np.float32)


def load_panel(dataset_dir):
    with open(os.path.join(dataset_dir, 'train_data.pkl'), 'rb') as handle:
        train = pickle.load(handle)
    with open(os.path.join(dataset_dir, 'val_data.pkl'), 'rb') as handle:
        validation = pickle.load(handle)

    panel = {}
    for symbol in sorted(set(train) | set(validation)):
        parts = []
        if symbol in train:
            parts.append(train[symbol])
        if symbol in validation:
            parts.append(validation[symbol])
        frame = pd.concat(parts).sort_index()
        frame = frame[~frame.index.duplicated(keep='last')]
        panel[symbol] = frame
    calendar = pd.DatetimeIndex(sorted(set().union(*(set(frame.index) for frame in validation.values()))))
    return panel, calendar


def build_periods(panel, calendar, universe_size, universe_selection='liquidity'):
    periods = []
    for period_index, signal_position in enumerate(range(0, len(calendar) - PRED_LEN, PRED_LEN)):
        signal_date = pd.Timestamp(calendar[signal_position])
        future_dates = pd.DatetimeIndex(
            calendar[signal_position + 1:signal_position + PRED_LEN + 1]
        )
        candidates = []
        for symbol, frame in panel.items():
            if signal_date not in frame.index:
                continue
            if future_dates[0] not in frame.index or future_dates[-1] not in frame.index:
                continue
            context = frame.loc[:signal_date].tail(LOOKBACK)
            if len(context) != LOOKBACK:
                continue
            if context.index[-1] != signal_date:
                continue
            if (signal_date - pd.Timestamp(context.index[0])).days > 150:
                continue
            required = FEATURE_COLUMNS + ['size_bucket']
            if context[required].isnull().values.any():
                continue
            liquidity = float(context['amount'].tail(20).median())
            if not np.isfinite(liquidity) or liquidity <= 0:
                continue
            entry_open = float(frame.loc[future_dates[0], 'open'])
            exit_close = float(frame.loc[future_dates[-1], 'close'])
            signal_close = float(context['close'].iloc[-1])
            if min(entry_open, exit_close, signal_close) <= 0:
                continue
            candidates.append({
                'symbol': symbol,
                'context': context,
                'future_dates': future_dates,
                'liquidity': liquidity,
                'size_bucket': int(context['size_bucket'].iloc[-1]),
                'size_percentile': float(
                    context['size_percentile'].iloc[-1]
                    if 'size_percentile' in context.columns
                    else (int(context['size_bucket'].iloc[-1]) + 0.5) / 10.0
                ),
                'last_close': signal_close,
                'actual_close_return': exit_close / signal_close - 1,
                'realized_return': exit_close / entry_open - 1,
            })

        if universe_selection == 'smallest_market_cap':
            # size_bucket is a point-in-time cross-sectional market-cap proxy:
            # 0 is the smallest bucket and 9 is the largest.
            candidates.sort(
                key=lambda item: (item['size_bucket'], -item['liquidity'], item['symbol'])
            )
        else:
            candidates.sort(key=lambda item: (-item['liquidity'], item['symbol']))
        selected = candidates[:universe_size]
        if len(selected) < universe_size:
            raise RuntimeError(
                f'{signal_date:%Y-%m-%d}: only {len(selected)} eligible symbols; '
                f'{universe_size} requested'
            )
        periods.append({
            'period': period_index,
            'signal_date': signal_date,
            'entry_date': future_dates[0],
            'exit_date': future_dates[-1],
            'records': selected,
        })
    return periods


def prepare_batch(records):
    normalized = []
    x_stamps = []
    y_stamps = []
    means = []
    stds = []
    size_buckets = []
    size_percentiles = []
    for record in records:
        context = record['context']
        values = context[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        normalized.append(np.clip((values - mean) / (std + 1e-5), -5, 5))
        x_stamps.append(time_features(context.index))
        y_stamps.append(time_features(record['future_dates']))
        means.append(mean)
        stds.append(std)
        size_buckets.append(record['size_bucket'])
        size_percentiles.append(record['size_percentile'])
    return {
        'x': np.stack(normalized).astype(np.float32),
        'x_stamp': np.stack(x_stamps).astype(np.float32),
        'y_stamp': np.stack(y_stamps).astype(np.float32),
        'means': np.stack(means).astype(np.float32),
        'stds': np.stack(stds).astype(np.float32),
        'size_buckets': np.asarray(size_buckets, dtype=np.int64),
        'size_percentiles': np.asarray(size_percentiles, dtype=np.float32),
    }


def load_model(path, device, conditioned):
    kwargs = {
        'num_sectors': 0,
        'num_size_buckets': 10,
        'context_layer': 10,
    } if conditioned else {}
    return Kronos.from_pretrained(path, **kwargs).to(device).eval()


def run_inference(
    label,
    model_path,
    conditioned,
    tokenizer,
    periods,
    device,
    batch_size,
    sample_count,
    seed,
):
    print(f'\nLoading {label}: {model_path}')
    model = load_model(model_path, device, conditioned)
    rows = []
    start_time = time.time()
    with torch.no_grad():
        for period in periods:
            torch.manual_seed(seed + period['period'])
            np.random.seed(seed + period['period'])
            records = period['records']
            period_start = time.time()
            for batch_start in range(0, len(records), batch_size):
                batch_records = records[batch_start:batch_start + batch_size]
                batch = prepare_batch(batch_records)
                predictions = auto_regressive_inference(
                    tokenizer,
                    model,
                    torch.as_tensor(batch['x'], device=device),
                    torch.as_tensor(batch['x_stamp'], device=device),
                    torch.as_tensor(batch['y_stamp'], device=device),
                    max_context=512,
                    pred_len=PRED_LEN,
                    clip=5,
                    T=0.6,
                    top_k=0,
                    top_p=0.9,
                    sample_count=sample_count,
                    verbose=False,
                    size_bucket=(
                        torch.as_tensor(batch['size_buckets'], device=device)
                        if conditioned else None
                    ),
                    size_percentile=(
                        torch.as_tensor(batch['size_percentiles'], device=device)
                        if conditioned and model.use_size_percentile else None
                    ),
                )
                forecast = predictions[:, -PRED_LEN:, :]
                close_index = FEATURE_COLUMNS.index('close')
                predicted_close = (
                    forecast[:, -1, close_index]
                    * (batch['stds'][:, close_index] + 1e-5)
                    + batch['means'][:, close_index]
                )
                for record, final_close in zip(batch_records, predicted_close):
                    rows.append({
                        'model': label,
                        'period': period['period'],
                        'signal_date': period['signal_date'],
                        'entry_date': period['entry_date'],
                        'exit_date': period['exit_date'],
                        'symbol': record['symbol'],
                        'size_bucket': record['size_bucket'],
                        'size_percentile': record['size_percentile'],
                        'score': float(final_close / record['last_close'] - 1),
                        'actual_close_return': record['actual_close_return'],
                        'realized_return': record['realized_return'],
                        'liquidity': record['liquidity'],
                    })
            if device.type == 'mps':
                torch.mps.synchronize()
            print(
                f"{label} period {period['period'] + 1}/{len(periods)} "
                f"{period['signal_date']:%Y-%m-%d}: {time.time() - period_start:.1f}s"
            )
    print(f'{label} inference: {(time.time() - start_time) / 60:.1f} min')
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


def max_drawdown(period_returns):
    equity = (1 + pd.Series(period_returns)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min()) if len(drawdown) else 0.0


def summarize(predictions, top_k, transaction_cost, positive_only=False):
    period_rows = []
    previous_top = None
    for period, frame in predictions.groupby('period', sort=True):
        ranked = frame.sort_values(['score', 'symbol'], ascending=[False, True])
        top = ranked[ranked['score'] > 0].head(top_k) if positive_only else ranked.head(top_k)
        bottom = ranked.tail(top_k)
        top_symbols = set(top['symbol'])
        turnover_denominator = max(len(previous_top or set()), len(top_symbols), 1)
        turnover = (
            1.0
            if previous_top is None
            else 1 - len(top_symbols & previous_top) / turnover_denominator
        )
        previous_top = top_symbols
        ic = frame['score'].corr(frame['actual_close_return'], method='spearman')
        pearson_ic = frame['score'].corr(frame['actual_close_return'], method='pearson')
        direction = np.mean(
            np.sign(frame['score'].to_numpy())
            == np.sign(frame['actual_close_return'].to_numpy())
        )
        top_return = float(top['realized_return'].mean()) if len(top) else 0.0
        bottom_return = float(bottom['realized_return'].mean())
        benchmark_return = float(frame['realized_return'].mean())
        period_cost = transaction_cost if len(top) else 0.0
        period_rows.append({
            'model': frame['model'].iloc[0],
            'period': int(period),
            'signal_date': frame['signal_date'].iloc[0],
            'entry_date': frame['entry_date'].iloc[0],
            'exit_date': frame['exit_date'].iloc[0],
            'rank_ic': float(ic),
            'pearson_ic': float(pearson_ic),
            'direction_accuracy': float(direction),
            'return_mae': float(np.mean(np.abs(frame['score'] - frame['actual_close_return']))),
            'top_return_gross': top_return,
            'top_return_net': top_return - period_cost,
            'bottom_return': bottom_return,
            'long_short_return': top_return - bottom_return,
            'benchmark_return': benchmark_return,
            'excess_return_net': top_return - period_cost - benchmark_return,
            'turnover': turnover,
            'holding_count': int(len(top)),
            'top_symbols': ','.join(sorted(top_symbols)),
        })
    period_metrics = pd.DataFrame(period_rows)
    net_returns = period_metrics['top_return_net']
    gross_returns = period_metrics['top_return_gross']
    benchmark_returns = period_metrics['benchmark_return']
    periods_per_year = 252 / PRED_LEN
    summary = {
        'periods': int(len(period_metrics)),
        'observations': int(len(predictions)),
        'mean_rank_ic': float(period_metrics['rank_ic'].mean()),
        'rank_ic_positive_rate': float((period_metrics['rank_ic'] > 0).mean()),
        'mean_pearson_ic': float(period_metrics['pearson_ic'].mean()),
        'direction_accuracy': float(predictions.assign(
            correct=np.sign(predictions['score']) == np.sign(predictions['actual_close_return'])
        )['correct'].mean()),
        'return_mae': float(np.mean(np.abs(predictions['score'] - predictions['actual_close_return']))),
        'top_cumulative_gross': float((1 + gross_returns).prod() - 1),
        'top_cumulative_net': float((1 + net_returns).prod() - 1),
        'benchmark_cumulative': float((1 + benchmark_returns).prod() - 1),
        'excess_cumulative_net': float(
            (1 + net_returns).prod() / (1 + benchmark_returns).prod() - 1
        ),
        'long_short_cumulative': float((1 + period_metrics['long_short_return']).prod() - 1),
        'mean_period_turnover': float(period_metrics['turnover'].mean()),
        'mean_holding_count': float(period_metrics['holding_count'].mean()),
        'net_annualized_sharpe': float(
            net_returns.mean() / net_returns.std(ddof=1) * np.sqrt(periods_per_year)
            if net_returns.std(ddof=1) > 0 else 0
        ),
        'net_max_drawdown': max_drawdown(net_returns),
    }
    return period_metrics, summary


def paired_bootstrap(period_metrics, metric, seed=100, draws=5000):
    pivot = period_metrics.pivot(index='period', columns='model', values=metric).dropna()
    differences = (pivot['finetuned'] - pivot['baseline']).to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    return {
        'mean_difference': float(differences.mean()),
        'ci_95_low': float(np.quantile(samples, 0.025)),
        'ci_95_high': float(np.quantile(samples, 0.975)),
        'probability_positive': float((samples > 0).mean()),
    }


def plot_results(period_metrics, output_path):
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, frame in period_metrics.groupby('model'):
        frame = frame.sort_values('period')
        axes[0].plot(
            frame['exit_date'],
            (1 + frame['top_return_net']).cumprod() - 1,
            marker='o',
            label=f'{label} Top portfolio (net)',
        )
        axes[1].plot(frame['exit_date'], frame['rank_ic'], marker='o', label=label)
    benchmark = period_metrics[period_metrics['model'] == 'baseline'].sort_values('period')
    axes[0].plot(
        benchmark['exit_date'],
        (1 + benchmark['benchmark_return']).cumprod() - 1,
        linestyle='--',
        color='black',
        label='Equal-weight universe',
    )
    axes[0].set_title('2026 Pilot Backtest: Cumulative Return')
    axes[0].set_ylabel('Return')
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Cross-sectional Rank IC by Period')
    axes[1].set_ylabel('Rank IC')
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='A-share 2026 pilot backtest')
    parser.add_argument('--dataset-dir', default='./data/a_share/processed_datasets')
    parser.add_argument('--base-model', default='./Kronos-base')
    parser.add_argument(
        '--finetuned-model',
        default='./outputs/models/a_share_size_kronos_base_earlystop50/checkpoints/best_model',
    )
    parser.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    parser.add_argument('--output-dir', default='./outputs/backtest_results/a_share_2026_pilot')
    parser.add_argument('--universe-size', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument(
        '--universe-selection',
        choices=['liquidity', 'smallest_market_cap'],
        default='liquidity',
        help='Build the common stock pool by liquidity or smallest size buckets.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-count', type=int, default=1)
    parser.add_argument('--transaction-cost-bps', type=float, default=25.0)
    parser.add_argument(
        '--predictions-input',
        default=None,
        help='Reuse a predictions.csv file and recompute portfolio rules without model inference.',
    )
    parser.add_argument(
        '--positive-only',
        action='store_true',
        help='Only hold stocks with a positive predicted return; otherwise hold cash.',
    )
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Device: {device}')
    if args.predictions_input:
        predictions = pd.read_csv(args.predictions_input)
        print(f'Reusing predictions: {args.predictions_input}')
    else:
        panel, calendar = load_panel(args.dataset_dir)
        periods = build_periods(
            panel, calendar, args.universe_size, universe_selection=args.universe_selection
        )
        print(
            f'Periods: {len(periods)}, universe/period: {args.universe_size}, '
            f'range: {periods[0]["signal_date"]:%Y-%m-%d} to {periods[-1]["exit_date"]:%Y-%m-%d}'
        )
        tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
        baseline = run_inference(
            'baseline', args.base_model, False, tokenizer, periods, device,
            args.batch_size, args.sample_count, args.seed,
        )
        finetuned = run_inference(
            'finetuned', args.finetuned_model, True, tokenizer, periods, device,
            args.batch_size, args.sample_count, args.seed,
        )
        predictions = pd.concat([baseline, finetuned], ignore_index=True)
    transaction_cost = args.transaction_cost_bps / 10_000

    metric_frames = []
    summaries = {}
    for label, frame in predictions.groupby('model'):
        period_metrics, summary = summarize(
            frame, args.top_k, transaction_cost, positive_only=args.positive_only
        )
        metric_frames.append(period_metrics)
        summaries[label] = summary
    period_metrics = pd.concat(metric_frames, ignore_index=True)
    bootstrap = {
        'rank_ic': paired_bootstrap(period_metrics, 'rank_ic', args.seed),
        'top_return_net': paired_bootstrap(period_metrics, 'top_return_net', args.seed),
        'excess_return_net': paired_bootstrap(period_metrics, 'excess_return_net', args.seed),
    }
    result = {
        'configuration': {
            'lookback': LOOKBACK,
            'holding_days': PRED_LEN,
            'universe_selection': (
                f'top {args.universe_size} by smallest point-in-time size buckets '
                '(same-bucket tie-break: trailing-20-day median amount)'
                if args.universe_selection == 'smallest_market_cap'
                else f'top {args.universe_size} by trailing-20-day median amount'
            ),
            'top_k': args.top_k,
            'sample_count': args.sample_count,
            'positive_score_only': args.positive_only,
            'sell_rule': 'sell when predicted return <= 0' if args.positive_only else 'none',
            'transaction_cost_bps_per_period': args.transaction_cost_bps,
            'execution': 'signal after close; enter next trading-day open; exit day-10 close',
            'device': str(device),
        },
        'summary': summaries,
        'paired_bootstrap_finetuned_minus_baseline': bootstrap,
        'limitations': [
            'Pilot backtest uses the historical CSI800 constituent union, not point-in-time membership.',
            (
                f'Universe is restricted to the {args.universe_size} smallest eligible '
                'size-bucket stocks per period; raw market cap is not retained in the '
                'processed panel.'
                if args.universe_selection == 'smallest_market_cap'
                else f'Universe is restricted to the {args.universe_size} most liquid eligible stocks per period.'
            ),
            'Single sampled path is used for computational tractability.',
            'Transaction cost is a conservative fixed 25 bps per 10-day holding period.',
            'No limit-up/down, suspension execution, impact, or short-sale constraints are modeled.',
        ],
    }

    predictions.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    period_metrics.to_csv(os.path.join(args.output_dir, 'period_metrics.csv'), index=False)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    plot_results(period_metrics, os.path.join(args.output_dir, 'backtest.png'))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
