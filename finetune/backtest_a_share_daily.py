"""Daily rolling backtest for the small-cap, positive-signal strategy."""

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

from finetune.backtest_a_share_2026 import (
    FEATURE_COLUMNS,
    LOOKBACK,
    PRED_LEN,
    prepare_batch,
    time_features,
)


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
        panel[symbol] = frame[~frame.index.duplicated(keep='last')]
    calendar = pd.DatetimeIndex(
        sorted(set().union(*(set(frame.index) for frame in validation.values())))
    )
    return panel, calendar


def build_daily_periods(panel, calendar, universe_size, max_signal_days=None):
    """Build one point-in-time small-cap universe for every signal date."""
    last_position = len(calendar) - PRED_LEN
    signal_positions = range(max(0, last_position))
    if max_signal_days is not None:
        signal_positions = list(signal_positions)[:max_signal_days]

    periods = []
    for period_index, signal_position in enumerate(signal_positions):
        signal_date = pd.Timestamp(calendar[signal_position])
        future_dates = pd.DatetimeIndex(
            calendar[signal_position + 1:signal_position + PRED_LEN + 1]
        )
        next_date = future_dates[0]
        candidates = []
        for symbol, frame in panel.items():
            if (
                signal_date not in frame.index
                or next_date not in frame.index
                or future_dates[-1] not in frame.index
            ):
                continue
            context = frame.loc[:signal_date].tail(LOOKBACK)
            if len(context) != LOOKBACK or context.index[-1] != signal_date:
                continue
            if (signal_date - pd.Timestamp(context.index[0])).days > 150:
                continue
            required = FEATURE_COLUMNS + ['size_bucket']
            if context[required].isnull().values.any():
                continue
            liquidity = float(context['amount'].tail(20).median())
            if not np.isfinite(liquidity) or liquidity <= 0:
                continue
            next_open = float(frame.loc[next_date, 'open'])
            next_close = float(frame.loc[next_date, 'close'])
            signal_close = float(context['close'].iloc[-1])
            if min(next_open, next_close, signal_close) <= 0:
                continue
            exit_close = float(frame.loc[future_dates[-1], 'close'])
            candidates.append({
                'symbol': symbol,
                'context': context,
                'future_dates': future_dates,
                'size_bucket': int(context['size_bucket'].iloc[-1]),
                'size_percentile': float(
                    context['size_percentile'].iloc[-1]
                    if 'size_percentile' in context.columns
                    else (int(context['size_bucket'].iloc[-1]) + 0.5) / 10.0
                ),
                'liquidity': liquidity,
                'last_close': signal_close,
                'actual_close_return': exit_close / signal_close - 1,
                'realized_return': exit_close / next_open - 1,
                'next_day_return': next_close / next_open - 1,
            })

        # Bucket 0 is the smallest point-in-time market-cap decile. The
        # liquidity tie-break keeps the simulated universe tradable.
        candidates.sort(key=lambda item: (item['size_bucket'], -item['liquidity'], item['symbol']))
        selected = candidates[:universe_size]
        if len(selected) < universe_size:
            raise RuntimeError(
                f'{signal_date:%Y-%m-%d}: only {len(selected)} eligible symbols; '
                f'{universe_size} requested'
            )
        periods.append({
            'period': period_index,
            'signal_date': signal_date,
            'entry_date': next_date,
            'exit_date': next_date,
            'records': selected,
        })
    return periods


def load_model(path, device, conditioned):
    kwargs = {'num_sectors': 0, 'num_size_buckets': 10, 'context_layer': 10} if conditioned else {}
    return Kronos.from_pretrained(path, **kwargs).to(device).eval()


def run_inference(label, model_path, conditioned, tokenizer, periods, device, batch_size, sample_count, seed):
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
                        'symbol': record['symbol'],
                        'size_bucket': record['size_bucket'],
                        'size_percentile': record['size_percentile'],
                        'score': float(final_close / record['last_close'] - 1),
                        'actual_close_return': record['actual_close_return'],
                        'next_day_return': record['next_day_return'],
                        'liquidity': record['liquidity'],
                    })
            if device.type == 'mps':
                torch.mps.synchronize()
            print(
                f"{label} day {period['period'] + 1}/{len(periods)} "
                f"{period['signal_date']:%Y-%m-%d}: {time.time() - period_start:.1f}s"
            )
    print(f'{label} inference: {(time.time() - start_time) / 60:.1f} min')
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


def max_drawdown(returns):
    equity = (1 + pd.Series(returns)).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def simulate(model_predictions, top_k, transaction_cost):
    rows = []
    holdings = set()
    equity = 1.0
    for period, frame in model_predictions.groupby('period', sort=True):
        ranked = frame.sort_values(['score', 'symbol'], ascending=[False, True])
        universe = set(frame['symbol'])
        positive = set(frame.loc[frame['score'] > 0, 'symbol'])
        negative_exits = (holdings & universe) - positive
        universe_exits = holdings - universe
        survivors = holdings & positive
        additions = ranked[
            (ranked['score'] > 0) & ~ranked['symbol'].isin(survivors)
        ].head(top_k - len(survivors))
        target = survivors | set(additions['symbol'])
        target_frame = frame[frame['symbol'].isin(target)]
        turnover_units = len(holdings - target) + len(target - holdings)
        turnover = turnover_units / top_k
        cost = transaction_cost * turnover
        gross_return = float(target_frame['next_day_return'].mean()) if len(target_frame) else 0.0
        net_return = (1 + gross_return) * (1 - cost) - 1
        equity *= 1 + net_return
        rank_ic = frame['score'].corr(frame['actual_close_return'], method='spearman')
        rows.append({
            'model': frame['model'].iloc[0],
            'period': int(period),
            'signal_date': frame['signal_date'].iloc[0],
            'rank_ic': float(rank_ic) if pd.notna(rank_ic) else 0.0,
            'direction_accuracy': float(np.mean(
                np.sign(frame['score'].to_numpy()) == np.sign(frame['actual_close_return'].to_numpy())
            )),
            'gross_return': gross_return,
            'net_return': net_return,
            'benchmark_return': float(frame['next_day_return'].mean()),
            'excess_return': net_return - float(frame['next_day_return'].mean()),
            'turnover': turnover,
            'holding_count': int(len(target)),
            'cash': int(len(target) == 0),
            'negative_signal_exits': int(len(negative_exits)),
            'universe_exits': int(len(universe_exits)),
            'buys': int(len(target - holdings)),
            'equity': equity,
            'symbols': ','.join(sorted(target)),
        })
        holdings = target

    metrics = pd.DataFrame(rows)
    net = metrics['net_return']
    benchmark = metrics['benchmark_return']
    summary = {
        'signal_days': int(len(metrics)),
        'observations': int(len(model_predictions)),
        'mean_rank_ic': float(metrics['rank_ic'].mean()),
        'rank_ic_positive_rate': float((metrics['rank_ic'] > 0).mean()),
        'direction_accuracy_10d': float(model_predictions.assign(
            correct=np.sign(model_predictions['score'])
            == np.sign(model_predictions['actual_close_return'])
        )['correct'].mean()),
        'top_cumulative_gross': float((1 + metrics['gross_return']).prod() - 1),
        'top_cumulative_net': float((1 + net).prod() - 1),
        'benchmark_cumulative': float((1 + benchmark).prod() - 1),
        'excess_cumulative_net': float((1 + net).prod() / (1 + benchmark).prod() - 1),
        'mean_period_turnover': float(metrics['turnover'].mean()),
        'mean_holding_count': float(metrics['holding_count'].mean()),
        'cash_days': int(metrics['cash'].sum()),
        'negative_signal_exits': int(metrics['negative_signal_exits'].sum()),
        'universe_exits': int(metrics['universe_exits'].sum()),
        'net_annualized_sharpe': float(
            net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) > 0 else 0
        ),
        'net_max_drawdown': max_drawdown(net),
    }
    return metrics, summary


def paired_bootstrap(metrics, metric, seed=100, draws=5000):
    pivot = metrics.pivot(index='period', columns='model', values=metric).dropna()
    differences = (pivot['finetuned'] - pivot['baseline']).to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    return {
        'mean_difference': float(differences.mean()),
        'ci_95_low': float(np.quantile(samples, 0.025)),
        'ci_95_high': float(np.quantile(samples, 0.975)),
        'probability_positive': float((samples > 0).mean()),
    }


def plot_results(metrics, output_path):
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, frame in metrics.groupby('model'):
        frame = frame.sort_values('period')
        dates = pd.to_datetime(frame['signal_date'])
        axes[0].plot(dates, frame['equity'] - 1, marker='.', label=f'{label} net')
        axes[1].plot(dates, frame['rank_ic'], marker='.', label=label)
    benchmark = metrics[metrics['model'] == 'baseline'].sort_values('period')
    axes[0].plot(
        pd.to_datetime(benchmark['signal_date']),
        (1 + benchmark['benchmark_return']).cumprod() - 1,
        linestyle='--', color='black', label='small-cap universe',
    )
    axes[0].set_title('2026 Daily Rolling Small-Cap Backtest')
    axes[0].set_ylabel('Cumulative return')
    axes[1].set_title('10-day forecast Rank IC')
    axes[1].set_ylabel('Rank IC')
    axes[1].axhline(0, color='black', linewidth=0.8)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Daily rolling A-share small-cap backtest')
    parser.add_argument('--dataset-dir', default='./data/a_share/processed_datasets')
    parser.add_argument('--base-model', default='./Kronos-base')
    parser.add_argument(
        '--finetuned-model',
        default='./outputs/models/a_share_size_kronos_base_earlystop50/checkpoints/best_model',
    )
    parser.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    parser.add_argument('--output-dir', default='./outputs/backtest_results/a_share_2026_daily_smallcap')
    parser.add_argument('--universe-size', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sample-count', type=int, default=1)
    parser.add_argument('--transaction-cost-bps', type=float, default=25.0)
    parser.add_argument(
        '--predictions-input',
        default=None,
        help='Reuse daily predictions and rerun only the portfolio simulation.',
    )
    parser.add_argument(
        '--baseline-predictions-input',
        default=None,
        help='Reuse baseline rows from an earlier predictions.csv and infer only the fine-tuned model.',
    )
    parser.add_argument('--max-signal-days', type=int, default=None)
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
        periods = build_daily_periods(panel, calendar, args.universe_size, args.max_signal_days)
        print(
            f'Signal days: {len(periods)}, universe/day: {args.universe_size}, '
            f'range: {periods[0]["signal_date"]:%Y-%m-%d} to {periods[-1]["entry_date"]:%Y-%m-%d}'
        )
        tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
        if args.baseline_predictions_input:
            previous = pd.read_csv(args.baseline_predictions_input)
            baseline = previous.loc[previous['model'] == 'baseline'].copy()
            expected_rows = len(periods) * args.universe_size
            if len(baseline) != expected_rows:
                raise ValueError(
                    f'Reused baseline has {len(baseline)} rows; expected {expected_rows}'
                )
            print(f'Reusing baseline predictions: {args.baseline_predictions_input}')
        else:
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
        metrics, summary = simulate(frame, args.top_k, transaction_cost)
        metric_frames.append(metrics)
        summaries[label] = summary
    metrics = pd.concat(metric_frames, ignore_index=True)
    result = {
        'configuration': {
            'lookback': LOOKBACK,
            'forecast_horizon_days': PRED_LEN,
            'signal_frequency': 'daily after close',
            'execution': 'rebalance at next trading-day open; mark at close',
            'universe_selection': (
                f'{args.universe_size} smallest point-in-time size-bucket stocks '
                '(same-bucket tie-break: trailing-20-day median amount)'
            ),
            'top_k': args.top_k,
            'rule': (
                'keep positive holdings; sell non-positive or out-of-universe holdings; '
                'fill vacancies with top positive predictions; cash otherwise'
            ),
            'transaction_cost_bps_per_full_turnover': args.transaction_cost_bps,
            'sample_count': args.sample_count,
            'device': str(device),
        },
        'summary': summaries,
        'paired_bootstrap_finetuned_minus_baseline': {
            'net_return': paired_bootstrap(metrics, 'net_return', args.seed),
            'rank_ic': paired_bootstrap(metrics, 'rank_ic', args.seed),
        },
        'limitations': [
            'Historical CSI800 constituent union is used, not point-in-time membership.',
            'Raw market cap is not retained; size_bucket is the point-in-time market-cap proxy.',
            'One sampled forecast path is used per signal date.',
            'No limit-up/down, suspension execution, market impact, or short selling is modeled.',
            'A 25 bps cost is charged proportional to daily portfolio turnover.',
        ],
    }
    predictions.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    metrics.to_csv(os.path.join(args.output_dir, 'daily_metrics.csv'), index=False)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    plot_results(metrics, os.path.join(args.output_dir, 'backtest.png'))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
