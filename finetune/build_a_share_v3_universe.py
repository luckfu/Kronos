"""Build a point-in-time, size-stratified A-share universe for V3 training."""

import argparse
import json
import os
import re
import time

import baostock as bs
import numpy as np
import pandas as pd


CSI_SNAPSHOT_DATES = [
    '2015-12-31', '2016-12-30', '2017-12-29', '2018-12-28',
    '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-30',
    '2023-12-29', '2024-12-31', '2025-12-31',
]
STOCK_CODE = re.compile(
    r'^(sh\.(?:60[0135]\d{3}|68\d{4})|sz\.(?:00[0123]\d{3}|30[01]\d{3}))$'
)
STRATA = (
    ('micro', 0.0, 0.10),
    ('small', 0.10, 0.30),
    ('small_mid', 0.30, 0.60),
)


def read_result(result):
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock query failed: {result.error_code} {result.error_msg}')
    return rows


def login():
    result = bs.login()
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock login failed: {result.error_code} {result.error_msg}')


def reconnect():
    bs.logout()
    time.sleep(0.5)
    login()


def query_csi800_history():
    symbols = set()
    for date in CSI_SNAPSHOT_DATES:
        for query in (bs.query_hs300_stocks, bs.query_zz500_stocks):
            for row in read_result(query(date)):
                symbols.add(row[1])
    return symbols


def query_active_a_shares(asof):
    rows = read_result(bs.query_all_stock(day=asof))
    return {
        row[0]: row[2]
        for row in rows
        if len(row) >= 3
        and row[1] == '1'
        and STOCK_CODE.match(row[0])
        and 'ST' not in row[2].upper()
        and '退' not in row[2]
    }


def query_snapshot_metrics(symbol, asof, lookback_days):
    end = pd.Timestamp(asof)
    start = end - pd.Timedelta(days=lookback_days)
    fields = 'date,code,close,volume,amount,turn'
    rows = read_result(bs.query_history_k_data_plus(
        symbol,
        fields,
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
        frequency='d',
        adjustflag='2',
    ))
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=fields.split(','))
    for column in ('close', 'volume', 'amount', 'turn'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    usable = frame[
        (frame['close'] > 0)
        & (frame['volume'] > 0)
        & (frame['amount'] > 0)
        & (frame['turn'] > 0)
    ]
    if usable.empty:
        return None
    latest = usable.iloc[-1]
    market_cap = float(latest['close'] * latest['volume'] / (latest['turn'] / 100.0))
    median_amount = float(usable['amount'].tail(20).median())
    if not np.isfinite(market_cap) or market_cap <= 0:
        return None
    return {
        'market_cap': market_cap,
        'median_amount_20d': median_amount,
        'market_cap_asof': latest['date'],
    }


def query_ipo_date(symbol):
    rows = read_result(bs.query_stock_basic(code=symbol))
    if not rows:
        return None
    value = pd.to_datetime(rows[0][2], errors='coerce')
    return None if pd.isna(value) else pd.Timestamp(value)


def choose_stratum(rows, count, preferred_cutoff, latest_cutoff, rng, ipo_cache):
    eligible = []
    order = rng.permutation(len(rows))
    for random_order, position in enumerate(order):
        row = rows.iloc[int(position)]
        symbol = row['symbol']
        if symbol not in ipo_cache:
            ipo_cache[symbol] = query_ipo_date(symbol)
        ipo_date = ipo_cache[symbol]
        if ipo_date is None or ipo_date > latest_cutoff:
            continue
        item = row.to_dict()
        item['ipo_date'] = ipo_date.strftime('%Y-%m-%d')
        item['_priority'] = 0 if ipo_date <= preferred_cutoff else 1
        item['_random_order'] = random_order
        eligible.append(item)
    eligible.sort(key=lambda item: (item['_priority'], item['_random_order']))
    chosen = eligible[:count]
    if len(chosen) < count:
        raise RuntimeError(
            f'Only {len(chosen)} stocks in stratum satisfy latest IPO cutoff; {count} requested'
        )
    for item in chosen:
        item.pop('_priority', None)
        item.pop('_random_order', None)
    return chosen


def main():
    parser = argparse.ArgumentParser(description='Build the A-share V3 universe manifest')
    parser.add_argument('--asof', default='2025-12-31')
    parser.add_argument('--listed-before', default='2016-01-01')
    parser.add_argument(
        '--latest-ipo-date', default='2025-06-30',
        help='Fallback IPO cutoff; older listings are always selected first.',
    )
    parser.add_argument('--train-per-stratum', type=int, default=300)
    parser.add_argument('--holdout-per-stratum', type=int, default=80)
    parser.add_argument('--min-median-amount', type=float, default=5_000_000)
    parser.add_argument('--snapshot-lookback-days', type=int, default=45)
    parser.add_argument('--seed', type=int, default=20260803)
    parser.add_argument('--output', default='./data/a_share_v3/universe_manifest.csv')
    parser.add_argument('--summary', default='./data/a_share_v3/universe_summary.json')
    parser.add_argument(
        '--snapshot-cache', default='./data/a_share_v3/full_market_snapshot.csv',
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.summary) or '.', exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cutoff = pd.Timestamp(args.listed_before)
    latest_cutoff = pd.Timestamp(args.latest_ipo_date)

    login()
    try:
        core_symbols = query_csi800_history()
        active = query_active_a_shares(args.asof)
        print(f'CSI800 historical union: {len(core_symbols)}')
        print(f'Active Shanghai/Shenzhen A-shares on {args.asof}: {len(active)}')

        if os.path.exists(args.snapshot_cache):
            cached_metrics = pd.read_csv(args.snapshot_cache)
            metric_rows = cached_metrics.to_dict('records')
            completed = set(cached_metrics['symbol'].astype(str))
            print(f'Resuming market-cap snapshot cache: {len(completed)} symbols')
        else:
            metric_rows = []
            completed = set()
        for index, (symbol, name) in enumerate(sorted(active.items()), 1):
            if symbol in completed:
                continue
            if index > 1 and index % 100 == 1:
                reconnect()
            try:
                metrics = query_snapshot_metrics(
                    symbol, args.asof, args.snapshot_lookback_days
                )
            except Exception as exc:
                print(f'warning: metrics {symbol}: {exc}')
                continue
            if metrics and metrics['median_amount_20d'] >= args.min_median_amount:
                metric_rows.append({'symbol': symbol, 'name': name, **metrics})
            if index % 250 == 0:
                print(f'market-cap scan: {index}/{len(active)}, usable={len(metric_rows)}')
                pd.DataFrame(metric_rows).to_csv(args.snapshot_cache, index=False)

        metrics = pd.DataFrame(metric_rows).sort_values(['market_cap', 'symbol'])
        if metrics.empty:
            raise RuntimeError('No usable full-market snapshot metrics')
        metrics.to_csv(args.snapshot_cache, index=False)
        metrics['full_market_percentile'] = metrics['market_cap'].rank(
            method='first', pct=True
        )
        candidates = metrics[~metrics['symbol'].isin(core_symbols)].copy()
        ipo_cache = {}
        selected = []
        requested = args.train_per_stratum + args.holdout_per_stratum
        for label, lower, upper in STRATA:
            rows = candidates[
                (candidates['full_market_percentile'] > lower)
                & (candidates['full_market_percentile'] <= upper)
            ].reset_index(drop=True)
            chosen = choose_stratum(
                rows, requested, cutoff, latest_cutoff, rng, ipo_cache
            )
            for position, item in enumerate(chosen):
                item['cohort'] = label
                item['split'] = (
                    'train' if position < args.train_per_stratum else 'holdout'
                )
                selected.append(item)
            print(
                f'{label}: candidates={len(rows)}, train={args.train_per_stratum}, '
                f'holdout={args.holdout_per_stratum}'
            )

        core_rows = []
        metric_lookup = metrics.set_index('symbol').to_dict('index')
        for symbol in sorted(core_symbols):
            item = metric_lookup.get(symbol, {})
            core_rows.append({
                'symbol': symbol,
                'name': item.get('name', active.get(symbol, '')),
                'market_cap': item.get('market_cap', np.nan),
                'median_amount_20d': item.get('median_amount_20d', np.nan),
                'market_cap_asof': item.get('market_cap_asof', ''),
                'full_market_percentile': item.get('full_market_percentile', np.nan),
                'ipo_date': '',
                'cohort': 'csi800_historical_union',
                'split': 'train',
            })

        manifest = pd.DataFrame(core_rows + selected)
        manifest = manifest.drop_duplicates('symbol', keep='first').sort_values(
            ['split', 'cohort', 'symbol']
        )
        manifest.to_csv(args.output, index=False)
        summary = {
            'asof': args.asof,
            'listed_before': args.listed_before,
            'latest_ipo_date': args.latest_ipo_date,
            'seed': args.seed,
            'full_market_usable': int(len(metrics)),
            'symbols_total': int(len(manifest)),
            'train_symbols': int((manifest['split'] == 'train').sum()),
            'holdout_symbols': int((manifest['split'] == 'holdout').sum()),
            'counts': {
                f'{split_name}:{cohort}': int(len(rows))
                for (split_name, cohort), rows in manifest.groupby(['split', 'cohort'])
            },
            'market_cap_method': 'close * volume / (turnover_pct / 100)',
            'limitations': [
                'Extra-stock stratification uses the 2025-12-31 active universe.',
                'CSI800 core uses annual historical constituent snapshots from 2015-2025.',
                'IPO cutoff ensures extra stocks existed by the beginning of 2016.',
            ],
        }
        with open(args.summary, 'w') as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'written: {args.output}')
    finally:
        bs.logout()


if __name__ == '__main__':
    main()
