"""Download an A-share CSI300/CSI800 daily panel through BaoStock.

The output is a long CSV accepted by prepare_a_share.py. BaoStock does not
expose a reliable point-in-time free-float market-cap field, so this script
uses the turnover-derived float-cap proxy only for size ranking and records
the approximation in the output metadata workflow.
"""

import argparse
import os
import time

import baostock as bs
import numpy as np
import pandas as pd


SNAPSHOT_DATES = ['2020-12-31', '2021-12-31', '2022-12-30', '2023-12-29', '2024-12-31', '2025-12-31']


def read_result(result):
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != '0':
        raise RuntimeError(f'BaoStock query failed: {result.error_code} {result.error_msg}')
    return rows


def query_universe(universe: str):
    functions = ['query_hs300_stocks'] if universe == 'csi300' else ['query_hs300_stocks', 'query_zz500_stocks']
    symbols = set()
    for date in SNAPSHOT_DATES:
        for function in functions:
            for row in read_result(getattr(bs, function)(date)):
                symbols.add(row[1])
    return sorted(symbols)


def query_industry_history():
    history = {}
    for date in SNAPSHOT_DATES:
        rows = read_result(bs.query_stock_industry(date=date))
        history[pd.Timestamp(date)] = {
            row[1]: (row[3] or 'unknown') for row in rows if len(row) >= 4
        }
    return history


def sector_asof(history, symbol, date):
    values = [key for key in history if key <= date]
    if not values:
        values = [min(history)]
    return history[max(values)].get(symbol, 'unknown')


def download_symbol(symbol, start_date, end_date, industry_history=None):
    fields = 'date,code,open,high,low,close,volume,amount,turn,pctChg'
    result = bs.query_history_k_data_plus(
        symbol, fields, start_date, end_date, frequency='d', adjustflag='2'
    )
    rows = read_result(result)
    if not rows:
        return []
    frame = pd.DataFrame(rows, columns=fields.split(','))
    for column in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    frame = frame.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
    frame = frame[(frame['close'] > 0) & (frame['volume'] >= 0)]
    # volume / turnover approximates float shares; use it only for cross-sectional size ranking.
    turn_fraction = frame['turn'].replace(0, np.nan) / 100.0
    frame['market_cap'] = frame['close'] * frame['volume'] / turn_fraction
    frame['market_cap'] = frame['market_cap'].replace([np.inf, -np.inf], np.nan)
    frame['market_cap'] = frame['market_cap'].ffill()
    frame['symbol'] = symbol
    columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'market_cap']
    if industry_history is not None:
        frame['sector'] = [sector_asof(industry_history, symbol, date) for date in frame['date']]
        columns.insert(-1, 'sector')
    return frame[columns].to_dict('records')


def main():
    parser = argparse.ArgumentParser(description='Download A-share daily data with BaoStock')
    parser.add_argument('--universe', choices=['csi300', 'csi800'], default='csi800')
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default='2026-07-31')
    parser.add_argument('--output', default='./data/a_share/a_share_daily.csv')
    parser.add_argument('--resume', action='store_true', help='Skip symbols already present in output')
    parser.add_argument('--with-sector', action='store_true', help='Also query BaoStock industry labels; slower and optional')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    existing = pd.read_csv(args.output, usecols=['symbol']) if args.resume and os.path.exists(args.output) else None
    completed = set(existing['symbol'].astype(str)) if existing is not None else set()
    wrote_output = bool(completed)
    if not args.resume and os.path.exists(args.output):
        os.remove(args.output)
        wrote_output = False

    def login_baostock():
        login = bs.login()
        if login.error_code != '0':
            raise RuntimeError(f'BaoStock login failed: {login.error_code} {login.error_msg}')

    login_baostock()
    try:
        symbols = query_universe(args.universe)
        industry_history = query_industry_history() if args.with_sector else None
        print(f'universe={args.universe}, symbols={len(symbols)}, skip={len(completed)}')
        rows = []
        for idx, symbol in enumerate(symbols, 1):
            if symbol in completed:
                continue
            if idx > 1 and idx % 100 == 1:
                bs.logout()
                time.sleep(0.5)
                login_baostock()
            try:
                rows.extend(download_symbol(symbol, args.start, args.end, industry_history))
            except Exception as exc:
                if '10001001' in str(exc):
                    bs.logout()
                    time.sleep(0.5)
                    login_baostock()
                    try:
                        rows.extend(download_symbol(symbol, args.start, args.end, industry_history))
                    except Exception as retry_exc:
                        print(f'warning: {symbol}: {retry_exc}')
                else:
                    print(f'warning: {symbol}: {exc}')
            if idx % 25 == 0:
                print(f'{idx}/{len(symbols)} symbols, rows={len(rows)}')
                if rows:
                    pd.DataFrame(rows).to_csv(args.output, mode='a' if wrote_output else 'w', index=False, header=not wrote_output)
                    wrote_output = True
                    rows.clear()
            time.sleep(0.02)
        if rows:
            pd.DataFrame(rows).to_csv(args.output, mode='a' if wrote_output else 'w', index=False, header=not wrote_output)
        print(f'written: {args.output}')
    finally:
        bs.logout()


if __name__ == '__main__':
    main()
