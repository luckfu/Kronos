"""Parallel BaoStock downloader for a fixed V3 symbol manifest."""

import argparse
import multiprocessing as mp
import os
import time

import baostock as bs
import pandas as pd

from download_a_share_baostock import download_symbol, read_result


def worker(worker_id, symbols, start, end, chunk_dir):
    output = os.path.join(chunk_dir, f'chunk_{worker_id:02d}.csv')
    completed = set()
    if os.path.exists(output):
        try:
            completed = set(pd.read_csv(output, usecols=['symbol'])['symbol'].astype(str))
        except Exception:
            completed = set()
    def worker_login():
        login_result = bs.login()
        if login_result.error_code != '0':
            raise RuntimeError(
                f'worker {worker_id} login failed: {login_result.error_msg}'
            )

    worker_login()
    wrote = bool(completed)
    rows = []
    try:
        for index, symbol in enumerate(symbols, 1):
            if symbol in completed:
                continue
            if index > 1 and index % 100 == 1:
                bs.logout()
                time.sleep(0.5)
                worker_login()
            try:
                rows.extend(download_symbol(symbol, start, end))
            except Exception as exc:
                if '10001001' in str(exc):
                    bs.logout()
                    time.sleep(0.5)
                    worker_login()
                    try:
                        rows.extend(download_symbol(symbol, start, end))
                    except Exception as retry_exc:
                        print(
                            f'[worker {worker_id}] warning {symbol}: {retry_exc}',
                            flush=True,
                        )
                else:
                    print(f'[worker {worker_id}] warning {symbol}: {exc}', flush=True)
            if index % 10 == 0:
                print(
                    f'[worker {worker_id}] {index}/{len(symbols)} symbols, '
                    f'rows={len(rows)}', flush=True
                )
            if index % 25 == 0 and rows:
                pd.DataFrame(rows).to_csv(
                    output, mode='a' if wrote else 'w', index=False,
                    header=not wrote,
                )
                wrote = True
                rows.clear()
            time.sleep(0.02)
        if rows:
            pd.DataFrame(rows).to_csv(
                output, mode='a' if wrote else 'w', index=False, header=not wrote
            )
    finally:
        bs.logout()


def merge_chunks(chunk_dir, output):
    paths = sorted(
        os.path.join(chunk_dir, item)
        for item in os.listdir(chunk_dir)
        if item.startswith('chunk_') and item.endswith('.csv')
    )
    if not paths:
        raise RuntimeError('No completed download chunks found')
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    wrote = False
    for path in paths:
        for frame in pd.read_csv(path, chunksize=50_000):
            frame.to_csv(output, mode='a' if wrote else 'w', index=False, header=not wrote)
            wrote = True
    merged = pd.read_csv(output, usecols=['symbol'])
    print(f'merged {merged.symbol.nunique()} symbols -> {output}')


def main():
    parser = argparse.ArgumentParser(description='Parallel A-share BaoStock downloader')
    parser.add_argument('--symbols-file', required=True)
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2026-07-31')
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk-dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    manifest = pd.read_csv(args.symbols_file, usecols=['symbol'])
    symbols = sorted(set(manifest['symbol'].astype(str)))
    os.makedirs(args.chunk_dir, exist_ok=True)
    chunks = [symbols[index::args.workers] for index in range(args.workers)]
    context = mp.get_context('spawn')
    processes = [
        context.Process(
            target=worker,
            args=(index, chunk, args.start, args.end, args.chunk_dir),
        )
        for index, chunk in enumerate(chunks)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise SystemExit(f'Workers failed: {failed}')
    merge_chunks(args.chunk_dir, args.output)


if __name__ == '__main__':
    main()
