"""Compare A-share checkpoints on stocks absent from the training panel."""

import argparse
import gc
import json
import os
import pickle
import random
import re
import sys
import time

try:
    import baostock as bs
except ModuleNotFoundError:  # Offline holdout evaluation does not need Baostock.
    bs = None
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference


FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']
LOOKBACK = 90
PRED_LEN = 10
STOCK_CODE = re.compile(r'^(sh\.(?:60[0135]\d{3}|68\d{4})|sz\.(?:00[0123]\d{3}|30[01]\d{3}))$')


def read_result(result):
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock query failed: {result.error_code} {result.error_msg}')
    return rows


def load_training_symbols(dataset_dir):
    symbols = set()
    for split in ('train', 'val', 'test'):
        path = os.path.join(dataset_dir, f'{split}_data.pkl')
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as handle:
            symbols.update(pickle.load(handle))
    return symbols


def query_csi800(asof):
    symbols = set()
    for query in (bs.query_hs300_stocks, bs.query_zz500_stocks):
        for row in read_result(query(asof)):
            symbols.add(row[1])
    return symbols


def query_unseen_universe(training_symbols, current_csi800, asof):
    rows = read_result(bs.query_all_stock(day=asof))
    return [
        {'symbol': row[0], 'name': row[2]}
        for row in rows
        if len(row) >= 3
        and row[1] == '1'
        and STOCK_CODE.match(row[0])
        and row[0] not in training_symbols
        and row[0] not in current_csi800
        and 'ST' not in row[2].upper()
        and '退' not in row[2]
    ]


def download_history(symbol, start, end):
    fields = 'date,code,open,high,low,close,volume,amount,turn,pctChg'
    rows = read_result(bs.query_history_k_data_plus(
        symbol, fields, start_date=start, end_date=end, frequency='d', adjustflag='2'
    ))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=fields.split(','))
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    for column in FEATURE_COLUMNS + ['turn', 'pctChg']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['date', *FEATURE_COLUMNS, 'turn'])
    frame = frame[(frame['close'] > 0) & (frame['volume'] > 0) & (frame['amount'] > 0)]
    return frame.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)


def select_candidates(universe, cache_path, start, end, pool_size, stock_count, seed):
    cached = pd.read_csv(cache_path, parse_dates=['date']) if os.path.exists(cache_path) else pd.DataFrame()
    cached_symbols = set(cached['symbol']) if not cached.empty else set()
    rng = random.Random(seed)
    shuffled = list(universe)
    rng.shuffle(shuffled)
    names = {item['symbol']: item['name'] for item in universe}

    frames = [cached] if not cached.empty else []
    attempted = 0
    valid = set(cached_symbols)
    for item in shuffled:
        if len(valid) >= pool_size:
            break
        symbol = item['symbol']
        if symbol in cached_symbols:
            continue
        attempted += 1
        try:
            frame = download_history(symbol, start, end)
        except Exception as exc:
            print(f'warning: {symbol} download failed: {exc}')
            continue
        if len(frame) < 220 or frame['date'].max() < pd.Timestamp(end) - pd.Timedelta(days=10):
            continue
        frame['symbol'] = symbol
        frame['name'] = item['name']
        frames.append(frame)
        valid.add(symbol)
        if len(valid) % 10 == 0:
            print(f'Candidate histories: {len(valid)}/{pool_size}')
        time.sleep(0.02)

    if not frames:
        raise RuntimeError('No eligible unseen-stock history was downloaded')
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(['symbol', 'date'], keep='last')
    panel.to_csv(cache_path, index=False)

    latest = panel.sort_values('date').groupby('symbol', as_index=False).tail(1).copy()
    latest = latest[(latest['turn'] > 0) & (latest['volume'] > 0)]
    latest['market_cap'] = latest['close'] * latest['volume'] / (latest['turn'] / 100.0)
    latest = latest.replace([np.inf, -np.inf], np.nan).dropna(subset=['market_cap'])
    latest = latest.sort_values(['market_cap', 'symbol']).reset_index(drop=True)
    if len(latest) < stock_count:
        raise RuntimeError(f'Only {len(latest)} eligible unseen stocks; {stock_count} requested')

    positions = np.linspace(0, len(latest) - 1, stock_count).round().astype(int)
    selected = latest.iloc[positions].drop_duplicates('symbol')
    if len(selected) < stock_count:
        remaining = latest[~latest['symbol'].isin(selected['symbol'])]
        selected = pd.concat([selected, remaining.head(stock_count - len(selected))])
    selected = selected.sort_values(['market_cap', 'symbol']).reset_index(drop=True)
    selected['name'] = selected['symbol'].map(names).fillna(selected['name'])
    print(f'Universe={len(universe)}, attempted={attempted}, cached panel={panel.symbol.nunique()}')
    return panel[panel['symbol'].isin(selected['symbol'])].copy(), selected


def load_size_references(raw_path):
    reference = pd.read_csv(raw_path, usecols=['date', 'market_cap'], parse_dates=['date'])
    reference['market_cap'] = pd.to_numeric(reference['market_cap'], errors='coerce')
    reference = reference[(reference['market_cap'] > 0) & reference['market_cap'].notna()]
    return {
        pd.Timestamp(date): np.sort(group['market_cap'].to_numpy(dtype=np.float64))
        for date, group in reference.groupby('date')
        if len(group) >= 100
    }


def time_features(index):
    index = pd.DatetimeIndex(index)
    return np.column_stack([
        index.minute, index.hour, index.weekday, index.day, index.month,
    ]).astype(np.float32)


def build_periods(panel, references, period_count, eval_start, eval_end):
    frames = {
        symbol: rows.set_index('date').sort_index()
        for symbol, rows in panel.groupby('symbol')
    }
    eval_start = pd.Timestamp(eval_start)
    eval_end = pd.Timestamp(eval_end)
    calendar = pd.DatetimeIndex(sorted(
        date for date in references if eval_start <= date <= eval_end
    ))
    if len(calendar) <= PRED_LEN:
        raise RuntimeError(
            f'Only {len(calendar)} reference dates are available from '
            f'{eval_start:%Y-%m-%d} to {eval_end:%Y-%m-%d}'
        )
    valid_positions = np.arange(0, len(calendar) - PRED_LEN)
    positions = np.linspace(valid_positions[0], valid_positions[-1], period_count).round().astype(int)
    periods = []
    for period_index, position in enumerate(positions):
        signal_date = pd.Timestamp(calendar[position])
        future_dates = pd.DatetimeIndex(calendar[position + 1:position + PRED_LEN + 1])
        caps = references[signal_date]
        records = []
        for symbol, frame in frames.items():
            if signal_date not in frame.index or not set(future_dates).issubset(frame.index):
                continue
            context = frame.loc[:signal_date].tail(LOOKBACK)
            if len(context) != LOOKBACK or context.index[-1] != signal_date:
                continue
            latest = context.iloc[-1]
            if latest['turn'] <= 0 or latest['volume'] <= 0:
                continue
            market_cap = float(latest['close'] * latest['volume'] / (latest['turn'] / 100.0))
            percentile = float(np.searchsorted(caps, market_cap, side='right') / len(caps))
            percentile = float(np.clip(percentile, 0.0, 1.0))
            size_bucket = min(int(np.floor(percentile * 10)), 9)
            signal_close = float(latest['close'])
            exit_close = float(frame.loc[future_dates[-1], 'close'])
            average_close = float(frame.loc[future_dates, 'close'].mean())
            entry_open = float(frame.loc[future_dates[0], 'open'])
            records.append({
                'symbol': symbol,
                'name': str(latest['name']),
                'context': context,
                'future_dates': future_dates,
                'size_bucket': size_bucket,
                'size_percentile': percentile,
                'last_close': signal_close,
                'actual_close_return': average_close / signal_close - 1,
                'realized_return': exit_close / entry_open - 1,
            })
        if len(records) < 8:
            raise RuntimeError(f'{signal_date:%Y-%m-%d}: only {len(records)} eligible stocks')
        periods.append({
            'period': period_index,
            'signal_date': signal_date,
            'entry_date': future_dates[0],
            'exit_date': future_dates[-1],
            'records': records,
        })
    return periods


def prepare_batch(records):
    xs, x_stamps, y_stamps, means, stds, buckets = [], [], [], [], [], []
    for record in records:
        values = record['context'][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        mean, std = values.mean(axis=0), values.std(axis=0)
        xs.append(np.clip((values - mean) / (std + 1e-5), -5, 5))
        x_stamps.append(time_features(record['context'].index))
        y_stamps.append(time_features(record['future_dates']))
        means.append(mean)
        stds.append(std)
        buckets.append(record['size_bucket'])
    return {
        'x': np.stack(xs).astype(np.float32),
        'x_stamp': np.stack(x_stamps).astype(np.float32),
        'y_stamp': np.stack(y_stamps).astype(np.float32),
        'means': np.stack(means).astype(np.float32),
        'stds': np.stack(stds).astype(np.float32),
        'buckets': np.asarray(buckets, dtype=np.int64),
    }


def run_model(
    label,
    path,
    tokenizer,
    periods,
    device,
    batch_size,
    sample_count,
    seed,
    temperature=0.6,
    top_p=0.9,
):
    model = Kronos.from_pretrained(
        path, num_sectors=0, num_size_buckets=10, context_layer=10,
        use_size_percentile=False, size_mlp_hidden_dim=64,
    ).to(device).eval()
    rows = []
    for period in periods:
        torch.manual_seed(seed + period['period'])
        np.random.seed(seed + period['period'])
        for offset in range(0, len(period['records']), batch_size):
            records = period['records'][offset:offset + batch_size]
            batch = prepare_batch(records)
            forecast = auto_regressive_inference(
                tokenizer, model,
                torch.as_tensor(batch['x'], device=device),
                torch.as_tensor(batch['x_stamp'], device=device),
                torch.as_tensor(batch['y_stamp'], device=device),
                max_context=512, pred_len=PRED_LEN, clip=5, T=temperature,
                top_k=0, top_p=top_p, sample_count=sample_count, verbose=False,
                size_bucket=torch.as_tensor(batch['buckets'], device=device),
            )[:, -PRED_LEN:, :]
            close_index = FEATURE_COLUMNS.index('close')
            predicted_close = (
                forecast[:, :, close_index].mean(axis=1)
                * (batch['stds'][:, close_index] + 1e-5)
                + batch['means'][:, close_index]
            )
            for record, final_close in zip(records, predicted_close):
                rows.append({
                    'model': label,
                    'period': period['period'],
                    'signal_date': period['signal_date'],
                    'entry_date': period['entry_date'],
                    'exit_date': period['exit_date'],
                    'symbol': record['symbol'],
                    'name': record['name'],
                    'size_bucket': record['size_bucket'],
                    'size_percentile': record['size_percentile'],
                    'score': float(final_close / record['last_close'] - 1),
                    'actual_close_return': record['actual_close_return'],
                    'realized_return': record['realized_return'],
                })
        print(f'{label}: period {period["period"] + 1}/{len(periods)} {period["signal_date"]:%Y-%m-%d}')
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


def summarize(frame):
    per_period = []
    for period, rows in frame.groupby('period'):
        per_period.append({
            'period': int(period),
            'rank_ic': float(rows['score'].corr(rows['actual_close_return'], method='spearman')),
            'direction_accuracy': float((np.sign(rows['score']) == np.sign(rows['actual_close_return'])).mean()),
            'return_mae': float(np.abs(rows['score'] - rows['actual_close_return']).mean()),
        })
    metrics = pd.DataFrame(per_period)
    return {
        'observations': int(len(frame)),
        'stocks': int(frame['symbol'].nunique()),
        'periods': int(frame['period'].nunique()),
        'mean_rank_ic': float(metrics['rank_ic'].mean()),
        'rank_ic_positive_rate': float((metrics['rank_ic'] > 0).mean()),
        'direction_accuracy': float((np.sign(frame['score']) == np.sign(frame['actual_close_return'])).mean()),
        'return_mae': float(np.abs(frame['score'] - frame['actual_close_return']).mean()),
    }, metrics


def paired_bootstrap(
    predictions, metric, draws=5000, seed=20260803, higher_is_better=True
):
    summaries = []
    for label, frame in predictions.groupby('model'):
        _, metrics = summarize(frame)
        summaries.append(metrics.assign(model=label))
    table = pd.concat(summaries).pivot(index='period', columns='model', values=metric).dropna()
    difference = table['latest'].to_numpy() - table['best'].to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(difference, size=(draws, len(difference)), replace=True).mean(axis=1)
    return {
        'latest_minus_best': float(difference.mean()),
        'ci95': [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        'probability_latest_better': float(
            ((sampled > 0) if higher_is_better else (sampled < 0)).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints on unseen A-shares')
    parser.add_argument('--dataset-dir', default='./data/a_share/processed_datasets')
    parser.add_argument('--raw-panel', default='./data/a_share/a_share_daily.csv')
    parser.add_argument('--best-model', default='./outputs/models/a_share_size_full_coverage_colab_bs32_best/checkpoints/best_model')
    parser.add_argument('--latest-model', default='./outputs/models/a_share_size_full_coverage_colab_bs32_latest/checkpoints/best_model')
    parser.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    parser.add_argument('--output-dir', default='./outputs/backtest_results/unseen_a_share_best_vs_latest')
    parser.add_argument('--start', default='2025-07-01')
    parser.add_argument('--end', default='2026-07-31')
    parser.add_argument(
        '--universe-asof', default=None,
        help='Constituent snapshot used to exclude the current CSI800; defaults to --end.',
    )
    parser.add_argument('--eval-start', default='2026-01-01')
    parser.add_argument('--eval-end', default='2026-07-31')
    parser.add_argument('--candidate-pool-size', type=int, default=80)
    parser.add_argument('--stock-count', type=int, default=24)
    parser.add_argument('--period-count', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--sample-count', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260803)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, 'candidate_history.csv')
    training_symbols = load_training_symbols(args.dataset_dir)
    universe_asof = args.universe_asof or args.end
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f'BaoStock login failed: {login.error_code} {login.error_msg}')
    try:
        current_csi800 = query_csi800(universe_asof)
        universe = query_unseen_universe(training_symbols, current_csi800, universe_asof)
        panel, selected = select_candidates(
            universe, cache_path, args.start, args.end,
            args.candidate_pool_size, args.stock_count, args.seed,
        )
    finally:
        bs.logout()

    selected.to_csv(os.path.join(args.output_dir, 'selected_stocks.csv'), index=False)
    periods = build_periods(
        panel, load_size_references(args.raw_panel), args.period_count,
        args.eval_start, args.eval_end,
    )
    device = torch.device(
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device={device}, unseen stocks={len(selected)}, periods={len(periods)}')
    print(selected[['symbol', 'name', 'market_cap']].to_string(index=False))
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    best = run_model('best', args.best_model, tokenizer, periods, device, args.batch_size, args.sample_count, args.seed)
    latest = run_model('latest', args.latest_model, tokenizer, periods, device, args.batch_size, args.sample_count, args.seed)
    predictions = pd.concat([best, latest], ignore_index=True)
    predictions.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    summaries = {label: summarize(frame)[0] for label, frame in predictions.groupby('model')}
    result = {
        'configuration': {
            'training_symbol_count': len(training_symbols),
            'current_csi800_count': len(current_csi800),
            'universe_asof': universe_asof,
            'unseen_universe_count': len(universe),
            'selection': 'fixed-seed random candidate pool, then evenly spaced by latest float-cap proxy',
            'lookback': LOOKBACK,
            'forecast_days': PRED_LEN,
            'sample_count': args.sample_count,
            'seed': args.seed,
            'signal_date_range': [
                f'{periods[0]["signal_date"]:%Y-%m-%d}',
                f'{periods[-1]["signal_date"]:%Y-%m-%d}',
            ],
            'exit_date': f'{periods[-1]["exit_date"]:%Y-%m-%d}',
        },
        'summary': summaries,
        'paired_bootstrap': {
            'rank_ic': paired_bootstrap(predictions, 'rank_ic', seed=args.seed),
            'direction_accuracy': paired_bootstrap(predictions, 'direction_accuracy', seed=args.seed),
            'return_mae': paired_bootstrap(
                predictions, 'return_mae', seed=args.seed, higher_is_better=False
            ),
        },
        'note': 'For return_mae, a negative latest-minus-best difference favors latest.',
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
