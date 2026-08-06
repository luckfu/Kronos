"""Build compact continuous-context panels for the V4 A/B experiment."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

try:
    from .dataset import load_merged_panels
except ImportError:
    from dataset import load_merged_panels


LOOKBACK = 90
PREDICT = 10


def crop_for_signal_range(panel, signal_start, signal_end):
    start = pd.Timestamp(signal_start)
    end = pd.Timestamp(signal_end)
    cropped = {}
    for symbol, frame in panel.items():
        dates = pd.DatetimeIndex(frame.index)
        first_signal = int(dates.searchsorted(start, side='left'))
        last_signal_exclusive = int(dates.searchsorted(end, side='right'))
        context_start = max(0, first_signal - LOOKBACK)
        target_end = min(len(frame), last_signal_exclusive + PREDICT + 1)
        selected = frame.iloc[context_start:target_end].copy()
        if len(selected) >= LOOKBACK + PREDICT + 1:
            cropped[symbol] = selected
    return cropped


def describe(panel):
    frames = list(panel.values())
    return {
        'symbols': len(panel),
        'rows': sum(len(frame) for frame in frames),
        'start': str(min(frame.index.min() for frame in frames).date()),
        'end': str(max(frame.index.max() for frame in frames).date()),
    }


def save_panel(path, panel):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--train-start', default='2026-01-01')
    parser.add_argument('--train-end', default='2026-06-17')
    parser.add_argument('--val-start', default='2026-06-18')
    parser.add_argument('--val-end', default='2026-07-16')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    source = data_root / 'processed_datasets'
    panel = load_merged_panels([
        source / 'train_data.pkl',
        source / 'val_data.pkl',
    ])
    recent = crop_for_signal_range(panel, args.train_start, args.train_end)
    temporal_validation = crop_for_signal_range(
        panel, args.val_start, args.val_end
    )
    recent_path = output_root / 'processed_datasets/recent_context_data.pkl'
    val_path = output_root / 'processed_datasets/temporal_val_data.pkl'
    save_panel(recent_path, recent)
    save_panel(val_path, temporal_validation)
    report = {
        'train_signal_range': [args.train_start, args.train_end],
        'validation_signal_range': [args.val_start, args.val_end],
        'recent_context': describe(recent),
        'temporal_validation': describe(temporal_validation),
        'recent_context_path': str(recent_path),
        'temporal_validation_path': str(val_path),
    }
    report_path = output_root / 'v4_runtime_summary.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
