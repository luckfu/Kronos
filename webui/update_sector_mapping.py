"""Refresh the A-share symbol-to-sector mapping without changing model IDs."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import tempfile


DEFAULT_VOCABULARY_PATH = Path(__file__).with_name('sector_vocabulary.json')
DEFAULT_OUTPUT_PATH = Path(__file__).with_name('symbol_sector_map.json')
A_SHARE_PREFIXES = (
    'sh.600', 'sh.601', 'sh.603', 'sh.605', 'sh.688', 'sh.689',
    'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.300', 'sz.301',
)


def read_baostock_result(result):
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != '0':
        raise RuntimeError(
            f'BaoStock query failed: {result.error_code} {result.error_msg}'
        )
    return rows


def load_vocabulary(path):
    with Path(path).open(encoding='utf-8') as handle:
        vocabulary = json.load(handle)
    labels = vocabulary.get('sector_labels') or []
    num_sectors = int(vocabulary.get('num_sectors', -1))
    unknown_sector_id = int(vocabulary.get('unknown_sector_id', -1))
    if len(labels) != num_sectors:
        raise ValueError(
            f'Vocabulary contains {len(labels)} labels; num_sectors={num_sectors}'
        )
    if len(set(labels)) != len(labels):
        raise ValueError('Vocabulary contains duplicate sector labels')
    if unknown_sector_id != num_sectors:
        raise ValueError('unknown_sector_id must equal num_sectors')
    if not vocabulary.get('vocabulary_id'):
        raise ValueError('Vocabulary is missing vocabulary_id')
    return vocabulary


def is_supported_a_share(symbol):
    value = str(symbol or '').strip().lower()
    return len(value) == 9 and value.startswith(A_SHARE_PREFIXES) and value[3:].isdigit()


def build_mapping_payload(rows, vocabulary, requested_date=None, source=None):
    labels = [str(value) for value in vocabulary['sector_labels']]
    label_to_id = {label: index for index, label in enumerate(labels)}
    unknown_sector_id = int(vocabulary['unknown_sector_id'])
    symbols = {}

    for row in rows:
        if len(row) < 4:
            continue
        reference_date = str(row[0] or requested_date or dt.date.today().isoformat())
        try:
            reference_date = dt.date.fromisoformat(reference_date[:10]).isoformat()
        except ValueError:
            continue
        symbol = str(row[1] or '').strip().lower()
        if not is_supported_a_share(symbol):
            continue
        sector_label = str(row[3] or 'unknown').strip() or 'unknown'
        record = {
            'sector_id': int(label_to_id.get(sector_label, unknown_sector_id)),
            'sector_label': sector_label,
            'reference_date': reference_date,
        }
        previous = symbols.get(symbol)
        if previous is None or record['reference_date'] >= previous['reference_date']:
            symbols[symbol] = record

    if not symbols:
        raise ValueError('BaoStock returned no supported Shanghai or Shenzhen A-share rows')
    reference_date = max(record['reference_date'] for record in symbols.values())
    unknown_count = sum(
        record['sector_id'] == unknown_sector_id for record in symbols.values()
    )
    return {
        'schema_version': 1,
        'vocabulary_id': vocabulary['vocabulary_id'],
        'reference_date': reference_date,
        'source': source or 'baostock.query_stock_industry',
        'requested_date': requested_date,
        'symbol_count': len(symbols),
        'unknown_sector_count': unknown_count,
        'symbols': dict(sorted(symbols.items())),
    }


def payload_from_legacy(path, vocabulary):
    with Path(path).open(encoding='utf-8') as handle:
        legacy = json.load(handle)
    if legacy.get('sector_labels') != vocabulary['sector_labels']:
        raise ValueError('Legacy sector labels do not match the fixed model vocabulary')
    symbols = legacy.get('symbols') or {}
    rows = [
        [
            record.get('reference_date') or legacy.get('reference_date'),
            symbol,
            '',
            record.get('sector_label') or 'unknown',
        ]
        for symbol, record in symbols.items()
    ]
    return build_mapping_payload(
        rows,
        vocabulary,
        requested_date=legacy.get('reference_date'),
        source=legacy.get('source') or 'legacy_sector_reference',
    )


def query_baostock_industries(query_date=None):
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError('baostock is required; install webui/requirements.txt') from exc

    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f'BaoStock login failed: {login.error_code} {login.error_msg}')
    try:
        result = (
            bs.query_stock_industry(date=query_date)
            if query_date else bs.query_stock_industry()
        )
        return read_baostock_result(result)
    finally:
        bs.logout()


def validate_replacement(payload, output_path, minimum_symbols, minimum_retention, force):
    new_count = len(payload['symbols'])
    if new_count < minimum_symbols and not force:
        raise ValueError(
            f'Refusing to write only {new_count} symbols; minimum is {minimum_symbols}. '
            'Use --force only after checking the provider response.'
        )
    output_path = Path(output_path)
    if not output_path.exists() or force:
        return
    with output_path.open(encoding='utf-8') as handle:
        previous = json.load(handle)
    previous_count = len(previous.get('symbols') or {})
    required_count = int(previous_count * minimum_retention)
    if previous_count and new_count < required_count:
        raise ValueError(
            f'Refusing to shrink mapping from {previous_count} to {new_count} symbols; '
            f'minimum retained count is {required_count}. Use --force only after review.'
        )


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, Path(f'{path}.bak'))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Update the A-share industry mapping while preserving Beta V1.2 IDs.'
    )
    parser.add_argument('--date', help='Optional BaoStock snapshot date (YYYY-MM-DD)')
    parser.add_argument('--vocabulary', type=Path, default=DEFAULT_VOCABULARY_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--minimum-symbols', type=int, default=1000)
    parser.add_argument('--minimum-retention', type=float, default=0.8)
    parser.add_argument(
        '--legacy-reference', type=Path,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.date:
        try:
            args.date = dt.date.fromisoformat(args.date).isoformat()
        except ValueError:
            parser.error('--date must use YYYY-MM-DD')
    if args.minimum_symbols < 1:
        parser.error('--minimum-symbols must be positive')
    if not 0 < args.minimum_retention <= 1:
        parser.error('--minimum-retention must be in (0, 1]')
    return args


def main():
    args = parse_args()
    vocabulary = load_vocabulary(args.vocabulary)
    if args.legacy_reference:
        payload = payload_from_legacy(args.legacy_reference, vocabulary)
    else:
        rows = query_baostock_industries(args.date)
        payload = build_mapping_payload(rows, vocabulary, requested_date=args.date)
    validate_replacement(
        payload,
        args.output,
        minimum_symbols=args.minimum_symbols,
        minimum_retention=args.minimum_retention,
        force=args.force,
    )
    action = 'checked' if args.dry_run else 'written'
    if not args.dry_run:
        write_json_atomic(args.output, payload)
    print(
        f'{action}: {args.output} | date={payload["reference_date"]} '
        f'| symbols={payload["symbol_count"]} '
        f'| unknown={payload["unknown_sector_count"]} '
        f'| vocabulary={payload["vocabulary_id"]}'
    )


if __name__ == '__main__':
    main()
