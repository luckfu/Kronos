"""Prepare an A-share daily panel for Kronos predictor fine-tuning.

Input is a long CSV (or a directory of CSV files) with at least:
symbol,date,open,high,low,close,volume,sector,market_cap

The script creates train_data.pkl, val_data.pkl, test_data.pkl and a
point-in-time asset_metadata.csv compatible with finetune/dataset.py.
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'}


def load_input(path: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(path, '*.csv'))) if os.path.isdir(path) else [path]
    if not paths:
        raise FileNotFoundError(f'No CSV files found under {path}')
    frames = [pd.read_csv(item) for item in paths]
    frame = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame['symbol'] = frame['symbol'].astype(str)
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    frame = frame.dropna(subset=['symbol', 'date'])
    numeric = ['open', 'high', 'low', 'close', 'volume']
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=numeric)
    frame = frame[(frame['close'] > 0) & (frame['volume'] >= 0)]
    if 'amount' not in frame.columns:
        frame['amount'] = frame[['open', 'high', 'low', 'close']].mean(axis=1) * frame['volume']
    else:
        frame['amount'] = pd.to_numeric(frame['amount'], errors='coerce')
        frame['amount'] = frame['amount'].fillna(
            frame[['open', 'high', 'low', 'close']].mean(axis=1) * frame['volume']
        )
    if 'market_cap' in frame.columns:
        frame['market_cap'] = pd.to_numeric(frame['market_cap'], errors='coerce')
        frame['market_cap'] = (
            frame.sort_values(['symbol', 'date'])
            .groupby('symbol', group_keys=False)['market_cap']
            .transform(lambda values: values.ffill().bfill())
        )
    frame = frame.sort_values(['symbol', 'date']).drop_duplicates(['symbol', 'date'], keep='last')
    return frame.reset_index(drop=True)


def add_size_buckets(frame: pd.DataFrame, bucket_count: int) -> pd.DataFrame:
    if 'size_percentile' in frame.columns:
        ranks = pd.to_numeric(frame['size_percentile'], errors='coerce')
    elif 'market_cap' in frame.columns:
        frame['market_cap'] = pd.to_numeric(frame['market_cap'], errors='coerce')
        # Point-in-time cross-sectional ranks preserve continuous size
        # information while remaining comparable across calendar years.
        ranks = frame.groupby('date')['market_cap'].rank(method='first', pct=True)
    elif 'size_bucket' in frame.columns:
        bucket = pd.to_numeric(frame['size_bucket'], errors='coerce')
        ranks = (bucket + 0.5) / bucket_count
    else:
        raise ValueError(
            'Input must contain market_cap, size_percentile, or size_bucket for size conditioning'
        )

    frame['size_percentile'] = ranks.clip(0.0, 1.0)
    if 'size_bucket' in frame.columns:
        frame['size_bucket'] = pd.to_numeric(frame['size_bucket'], errors='coerce')
    else:
        bucket = np.floor(frame['size_percentile'] * bucket_count)
        frame['size_bucket'] = bucket.clip(0, bucket_count - 1)
    frame.loc[frame['size_percentile'].isna(), 'size_bucket'] = np.nan
    return frame


def split_data(
    frame: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    val_context_start: str = None,
):
    train_end = pd.Timestamp(train_end)
    val_start, val_end = pd.Timestamp(val_start), pd.Timestamp(val_end)
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    val_context_start = pd.Timestamp(val_context_start or val_start)
    masks = {
        'train': frame['date'] <= train_end,
        # Keep a history prefix in the pickle. QlibDataset applies the
        # point-in-time val_signal_start/end filter to the as-of date, so
        # these rows provide context without becoming validation targets.
        'val': frame['date'].between(val_context_start, val_end),
        'test': frame['date'].between(test_start, test_end),
    }
    result = {}
    for split, mask in masks.items():
        split_dict = {}
        for symbol, rows in frame.loc[mask].groupby('symbol', sort=False):
            rows = rows.sort_values('date').set_index('date')
            rows.index.name = 'datetime'
            columns = [
                'open', 'high', 'low', 'close', 'volume', 'amount',
                'size_bucket', 'size_percentile',
            ]
            if 'sector' in rows.columns:
                columns.append('sector')
            rows = rows[columns]
            if len(rows) > 0:
                split_dict[symbol] = rows
        result[split] = split_dict
    return result


def main():
    parser = argparse.ArgumentParser(description='Prepare A-share panel data for Kronos fine-tuning')
    parser.add_argument('--input', required=True, help='Long CSV or directory of per-symbol CSV files')
    parser.add_argument('--output-dir', default='./data/a_share/processed_datasets')
    parser.add_argument('--metadata-out', default='./data/a_share/asset_metadata.csv')
    parser.add_argument(
        '--universe-manifest', default=None,
        help='Optional CSV with symbol and split=train/holdout columns.',
    )
    parser.add_argument(
        '--holdout-output', default=None,
        help='Optional pickle containing every row for symbols held out from training.',
    )
    parser.add_argument('--universe-label', default='configured A-share universe')
    parser.add_argument(
        '--size-reference-out',
        default='./webui/size_reference.json',
        help='Portable latest market-cap cross-section used for unseen-stock sizing.',
    )
    parser.add_argument('--num-size-buckets', type=int, default=10)
    parser.add_argument('--train-end', default='2025-12-31')
    parser.add_argument('--val-start', default='2026-01-01')
    parser.add_argument('--val-end', default='2026-12-31')
    parser.add_argument(
        '--val-context-start', default='2025-07-01',
        help='Earliest date retained in val_data.pkl for the 120-day lookback history.',
    )
    parser.add_argument('--test-start', default='2027-01-01')
    parser.add_argument('--test-end', default='2027-12-31')
    args = parser.parse_args()

    frame = add_size_buckets(load_input(args.input), args.num_size_buckets)
    reference_frame = frame
    holdout_frame = frame.iloc[0:0].copy()
    if args.universe_manifest:
        manifest = pd.read_csv(args.universe_manifest, usecols=['symbol', 'split'])
        manifest['symbol'] = manifest['symbol'].astype(str)
        split_by_symbol = manifest.drop_duplicates('symbol').set_index('symbol')['split']
        frame['universe_split'] = frame['symbol'].map(split_by_symbol)
        missing = frame['universe_split'].isna()
        if missing.any():
            print(f'warning: dropping {missing.sum()} rows absent from universe manifest')
            frame = frame.loc[~missing].copy()
        holdout_frame = frame[frame['universe_split'] == 'holdout'].copy()
        frame = frame[frame['universe_split'] == 'train'].copy()
    if 'sector' in frame.columns:
        frame['sector'] = frame['sector'].astype(str)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_out) or '.', exist_ok=True)
    metadata_columns = ['symbol', 'date', 'size_bucket', 'size_percentile']
    if 'sector' in frame.columns:
        metadata_columns.insert(2, 'sector')
    metadata = frame[metadata_columns].copy()
    metadata['date'] = metadata['date'].dt.strftime('%Y-%m-%d')
    metadata.drop_duplicates(['symbol', 'date'], keep='last').to_csv(args.metadata_out, index=False)

    if 'market_cap' in frame.columns:
        valid_caps = reference_frame.dropna(subset=['market_cap'])
        valid_caps = valid_caps[valid_caps['market_cap'] > 0]
        if not valid_caps.empty:
            reference_date = pd.Timestamp(valid_caps['date'].max())
            market_caps = sorted(
                float(value)
                for value in valid_caps.loc[
                    valid_caps['date'] == reference_date, 'market_cap'
                ]
            )
            os.makedirs(os.path.dirname(args.size_reference_out) or '.', exist_ok=True)
            with open(args.size_reference_out, 'w') as handle:
                json.dump({
                    'reference_date': reference_date.strftime('%Y-%m-%d'),
                    'market_caps': market_caps,
                    'count': len(market_caps),
                    'method': 'amount / (turnover_pct / 100)',
                    'universe': args.universe_label,
                }, handle, separators=(',', ':'))
            print(f'size reference: {args.size_reference_out} ({len(market_caps)} stocks)')

    splits = split_data(
        frame, args.train_end, args.val_start, args.val_end,
        args.test_start, args.test_end, args.val_context_start,
    )
    for split, data in splits.items():
        with open(os.path.join(args.output_dir, f'{split}_data.pkl'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{split}: {len(data)} symbols, {sum(len(item) for item in data.values())} rows')
    if args.holdout_output:
        os.makedirs(os.path.dirname(args.holdout_output) or '.', exist_ok=True)
        holdout = split_data(
            holdout_frame,
            train_end=args.test_end,
            val_start='1900-01-01', val_end='1900-01-01',
            test_start='1900-01-01', test_end='1900-01-01',
        )['train']
        with open(args.holdout_output, 'wb') as handle:
            pickle.dump(holdout, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f'holdout: {len(holdout)} symbols, '
            f'{sum(len(item) for item in holdout.values())} rows -> {args.holdout_output}'
        )
    print(f'metadata: {args.metadata_out}')


if __name__ == '__main__':
    main()
