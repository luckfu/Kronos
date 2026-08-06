"""Verify continuous-context signal windows for the A-share V4 experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .dataset import load_merged_panels
except ImportError:
    from dataset import load_merged_panels


LOOKBACK = 90
PREDICT = 10
WINDOW = LOOKBACK + PREDICT + 1


def summarize(panel, signal_start, signal_end):
    start = np.datetime64(signal_start, 'D')
    end = np.datetime64(signal_end, 'D')
    returns = []
    signal_days = set()
    symbols = 0
    for frame in panel.values():
        sample_count = len(frame) - WINDOW + 1
        if sample_count <= 0:
            continue
        closes = pd.to_numeric(frame['close'], errors='coerce').to_numpy(float)
        if not np.isfinite(closes).all() or (closes <= 0).any():
            raise ValueError('Panel contains an invalid close price')
        asof_positions = np.arange(sample_count) + LOOKBACK - 1
        dates = frame.index.to_numpy(dtype='datetime64[D]')[asof_positions]
        eligible = (dates >= start) & (dates <= end)
        if not eligible.any():
            continue
        asof_positions = asof_positions[eligible]
        dates = dates[eligible]
        cumulative = np.concatenate(([0.0], np.cumsum(closes)))
        future_mean = (
            cumulative[asof_positions + PREDICT + 1]
            - cumulative[asof_positions + 1]
        ) / PREDICT
        returns.append(future_mean / closes[asof_positions] - 1.0)
        signal_days.update(dates.tolist())
        symbols += 1
    values = np.concatenate(returns) if returns else np.empty(0)
    return {
        'symbols': symbols,
        'windows': int(len(values)),
        'signal_days': int(len(signal_days)),
        'signal_start': str(min(signal_days)) if signal_days else None,
        'signal_end': str(max(signal_days)) if signal_days else None,
        'down_rate': float((values < 0).mean()) if len(values) else None,
        'mean_return': float(values.mean()) if len(values) else None,
        'median_return': float(np.median(values)) if len(values) else None,
    }


def require_equal(label, actual, expected):
    if actual != expected:
        raise SystemExit(f'{label}: expected {expected!r}, found {actual!r}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data/a_share_v3')
    parser.add_argument('--train-start', default='2026-01-01')
    parser.add_argument('--train-end', default='2026-06-17')
    parser.add_argument('--val-start', default='2026-06-18')
    parser.add_argument('--val-end', default='2026-07-16')
    parser.add_argument('--replay-ratio', type=float, default=0.20)
    parser.add_argument('--output', default='')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    root = Path(args.data_root)
    processed = root / 'processed_datasets'
    paths = [processed / 'train_data.pkl', processed / 'val_data.pkl']
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f'Missing continuous-context panel source: {missing}')
    panel = load_merged_panels(paths)
    train = summarize(panel, args.train_start, args.train_end)
    validation = summarize(panel, args.val_start, args.val_end)
    replay_windows = round(
        train['windows'] * args.replay_ratio / (1.0 - args.replay_ratio)
    )
    report = {
        'data_root': str(root),
        'lookback': LOOKBACK,
        'predict': PREDICT,
        'train': train,
        'temporal_validation': validation,
        'replay_ratio': args.replay_ratio,
        'replay_windows': replay_windows,
        'mixed_train_windows': train['windows'] + replay_windows,
    }
    if args.strict:
        require_equal('train.windows', train['windows'], 249_280)
        require_equal('train.signal_days', train['signal_days'], 108)
        require_equal('train.signal_start', train['signal_start'], '2026-01-05')
        require_equal('train.signal_end', train['signal_end'], '2026-06-17')
        require_equal('temporal_validation.windows', validation['windows'], 46_129)
        require_equal('temporal_validation.signal_days', validation['signal_days'], 20)
        require_equal(
            'temporal_validation.signal_start',
            validation['signal_start'],
            '2026-06-18',
        )
        require_equal(
            'temporal_validation.signal_end',
            validation['signal_end'],
            '2026-07-16',
        )
        require_equal('replay_windows', replay_windows, 62_320)
    document = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    print(document, end='')
    if args.output:
        Path(args.output).write_text(document, encoding='utf-8')


if __name__ == '__main__':
    main()
