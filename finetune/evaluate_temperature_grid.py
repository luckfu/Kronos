"""Calibrate sampling temperature on a fixed A-share time holdout."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune.compare_kaggle_best_last import (
    build_signal_periods,
    load_holdout,
)
from finetune.evaluate_unseen_a_share import run_model, summarize
from model import KronosTokenizer


def summarize_temperature(frame):
    summary, per_period = summarize(frame)
    spreads = []
    predicted_down = []
    for _, rows in frame.groupby('period'):
        ordered = rows.sort_values('score', ascending=False)
        n = max(1, len(ordered) // 5)
        spreads.append(float(
            ordered.head(n)['actual_close_return'].mean()
            - ordered.tail(n)['actual_close_return'].mean()
        ))
        predicted_down.append(float((rows['score'] < 0).mean()))
    summary.update({
        'mean_top_bottom_actual_return_spread': float(np.mean(spreads)),
        'predicted_down_rate': float(np.mean(predicted_down)),
    })
    return summary, per_period


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='./outputs/models/a_share_v4_corrected_2026_replay20_latest/checkpoints/last_model',
    )
    parser.add_argument('--holdout', default='./data/a_share_v3/processed_datasets/symbol_holdout_data.pkl')
    parser.add_argument('--manifest', default='./data/a_share_v3/universe_manifest.csv')
    parser.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    parser.add_argument('--output-dir', default='./outputs/backtest_results/v4b_temperature_grid')
    parser.add_argument('--signal-start', default='2026-06-18')
    parser.add_argument('--signal-end', default='2026-07-16')
    parser.add_argument('--period-count', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sample-count', type=int, default=5)
    parser.add_argument('--lookback', type=int, default=90)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=20260807)
    parser.add_argument('--temperatures', default='0.4,0.6,0.8,1.0')
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    temperatures = [float(value) for value in args.temperatures.split(',') if value.strip()]
    if not temperatures or any(value <= 0 for value in temperatures):
        parser.error('--temperatures must contain positive comma-separated values')
    if args.lookback < 1:
        parser.error('--lookback must be positive')
    if not 0 < args.top_p <= 1:
        parser.error('--top-p must be in (0, 1]')

    panel, calendar = load_holdout(args.holdout, args.manifest)
    periods = build_signal_periods(
        panel,
        calendar,
        args.signal_start,
        args.signal_end,
        args.period_count,
        lookback=args.lookback,
    )
    device = torch.device(
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    print(
        f'holdout_symbols={len(panel)}, periods={len(periods)}, '
        f'dates={periods[0]["signal_date"]:%Y-%m-%d}..{periods[-1]["signal_date"]:%Y-%m-%d}, '
        f'Device={device}'
    )
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    results = {}
    all_predictions = []
    for temperature in temperatures:
        label = f'T={temperature:g}'
        print(f'\n=== {label} ===')
        predictions = run_model(
            label,
            args.model,
            tokenizer,
            periods,
            device,
            args.batch_size,
            args.sample_count,
            args.seed,
            temperature=temperature,
            top_p=args.top_p,
        )
        predictions['temperature'] = temperature
        summary, per_period = summarize_temperature(predictions)
        results[label] = {
            'temperature': temperature,
            'summary': summary,
            'per_period': per_period.to_dict(orient='records'),
        }
        all_predictions.append(predictions)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    payload = {
        'configuration': {
            'model': args.model,
            'holdout': args.holdout,
            'lookback': args.lookback,
            'forecast_days': 10,
            'top_p': args.top_p,
            'signal_start': args.signal_start,
            'signal_end': args.signal_end,
            'period_count': len(periods),
            'sample_count': args.sample_count,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'temperatures': temperatures,
            'device': str(device),
            'note': 'Calibration holdout; temperature was selected on this fixed window.',
        },
        'results': results,
    }
    with open(output / 'summary.json', 'w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f'RESULT_PATH {output}')


if __name__ == '__main__':
    main()
