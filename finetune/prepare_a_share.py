"""Prepare an A-share daily panel for Kronos predictor fine-tuning.

Input is a long CSV (or a directory of CSV files) with at least:
symbol,date,open,high,low,close,volume,sector,market_cap

The script creates train_data.pkl, val_data.pkl, test_data.pkl and a
point-in-time asset_metadata.csv compatible with finetune/dataset.py.
"""

import argparse
import glob
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
    frame = frame.sort_values(['symbol', 'date']).drop_duplicates(['symbol', 'date'], keep='last')
    return frame.reset_index(drop=True)


def add_size_buckets(frame: pd.DataFrame, bucket_count: int) -> pd.DataFrame:
    if 'size_bucket' in frame.columns:
        frame['size_bucket'] = pd.to_numeric(frame['size_bucket'], errors='coerce')
        return frame
    if 'market_cap' not in frame.columns:
        raise ValueError('Input must contain market_cap or size_bucket for size conditioning')
    frame['market_cap'] = pd.to_numeric(frame['market_cap'], errors='coerce')
    # Rank within each date, so the bucket is comparable across time and does
    # not expose the absolute level of the market.
    ranks = frame.groupby('date')['market_cap'].rank(method='first', pct=True)
    frame['size_bucket'] = np.minimum(
        (ranks * bucket_count).fillna(-1).astype(int), bucket_count - 1
    )
    return frame


def split_data(frame: pd.DataFrame, train_end: str, val_start: str, val_end: str, test_start: str, test_end: str):
    train_end = pd.Timestamp(train_end)
    val_start, val_end = pd.Timestamp(val_start), pd.Timestamp(val_end)
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    masks = {
        'train': frame['date'] <= train_end,
        'val': frame['date'].between(val_start, val_end),
        'test': frame['date'].between(test_start, test_end),
    }
    result = {}
    for split, mask in masks.items():
        split_dict = {}
        for symbol, rows in frame.loc[mask].groupby('symbol', sort=False):
            rows = rows.sort_values('date').set_index('date')
            rows.index.name = 'datetime'
            columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'size_bucket']
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
    parser.add_argument('--num-size-buckets', type=int, default=10)
    parser.add_argument('--train-end', default='2025-12-31')
    parser.add_argument('--val-start', default='2026-01-01')
    parser.add_argument('--val-end', default='2026-12-31')
    parser.add_argument('--test-start', default='2027-01-01')
    parser.add_argument('--test-end', default='2027-12-31')
    args = parser.parse_args()

    frame = add_size_buckets(load_input(args.input), args.num_size_buckets)
    if 'sector' in frame.columns:
        frame['sector'] = frame['sector'].astype(str)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_out) or '.', exist_ok=True)
    metadata_columns = ['symbol', 'date', 'size_bucket']
    if 'sector' in frame.columns:
        metadata_columns.insert(2, 'sector')
    metadata = frame[metadata_columns].copy()
    metadata['date'] = metadata['date'].dt.strftime('%Y-%m-%d')
    metadata.drop_duplicates(['symbol', 'date'], keep='last').to_csv(args.metadata_out, index=False)

    splits = split_data(frame, args.train_end, args.val_start, args.val_end, args.test_start, args.test_end)
    for split, data in splits.items():
        with open(os.path.join(args.output_dir, f'{split}_data.pkl'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{split}: {len(data)} symbols, {sum(len(item) for item in data.values())} rows')
    print(f'metadata: {args.metadata_out}')


if __name__ == '__main__':
    main()
