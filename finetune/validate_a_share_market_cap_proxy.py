"""Validate the front-adjusted BaoStock panel and turnover-derived size proxy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, parse_dates=['date'])
    required = {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'market_cap'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'missing required columns: {missing}')
    if frame.duplicated(['symbol', 'date']).any():
        raise ValueError('duplicate symbol/date rows found; resume is not safe')

    numeric = ['open', 'high', 'low', 'close', 'volume', 'amount', 'market_cap']
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='coerce')
    valid_price = (frame[['open', 'high', 'low', 'close']] > 0).all(axis=1)
    valid_proxy = np.isfinite(frame['market_cap']) & (frame['market_cap'] > 0)
    frame['old_proxy'] = frame['close'] * frame['volume']
    ratio = (frame['market_cap'] / frame['old_proxy']).replace([np.inf, -np.inf], np.nan)

    # The raw input only has 20 stocks, but this confirms that downstream
    # daily ranking produces deterministic integer buckets without gaps.
    ranks = frame.loc[valid_proxy].groupby('date')['market_cap'].rank(method='first', pct=True)
    buckets = np.floor(ranks * 10).clip(upper=9).astype(int)
    bucket_counts = buckets.value_counts().sort_index().to_dict()

    report = {
        'input': str(Path(args.input).resolve()),
        'rows': int(len(frame)),
        'symbols': int(frame['symbol'].nunique()),
        'date_range': [str(frame['date'].min().date()), str(frame['date'].max().date())],
        'duplicate_symbol_date_rows': int(frame.duplicated(['symbol', 'date']).sum()),
        'invalid_ohlc_rows': int((~valid_price).sum()),
        'invalid_size_proxy_rows': int((~valid_proxy).sum()),
        'front_adjusted_price_jump_p99': float(
            frame.sort_values(['symbol', 'date']).groupby('symbol')['close'].pct_change().abs().quantile(0.99)
        ),
        'new_to_old_proxy_ratio': {
            'p01': float(ratio.quantile(0.01)),
            'median': float(ratio.median()),
            'p99': float(ratio.quantile(0.99)),
        },
        'daily_bucket_counts': {str(key): int(value) for key, value in bucket_counts.items()},
    }
    if report['symbols'] == 0 or report['rows'] == 0:
        raise ValueError('download returned no usable rows')
    # A small number of suspended/listing-edge rows can have zero turnover;
    # these are reported but are handled by the preparation step's per-symbol
    # forward/backward fill rather than treated as corrupt OHLC observations.
    if report['invalid_ohlc_rows']:
        raise ValueError(json.dumps(report, ensure_ascii=True))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
