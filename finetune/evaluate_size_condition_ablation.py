"""Measure whether the production model uses market-cap bucket conditioning."""

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from finetune.compare_kaggle_best_last import build_signal_periods, load_holdout
from finetune.evaluate_unseen_a_share import (
    FEATURE_COLUMNS,
    PRED_LEN,
    prepare_batch,
    summarize,
    time_features,
)
from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference


def distribution_metrics(counts, reference_counts=None):
    """Return entropy and optional Jensen-Shannon divergence for token counts."""
    keys = sorted(set(counts) | set(reference_counts or {}))
    values = np.asarray([counts.get(key, 0) for key in keys], dtype=np.float64)
    probabilities = values / values.sum() if values.sum() else values
    nonzero = probabilities > 0
    result = {
        'unique_tokens': int(nonzero.sum()),
        'entropy': float(-(probabilities[nonzero] * np.log2(probabilities[nonzero])).sum()),
    }
    if reference_counts is not None:
        reference = np.asarray(
            [reference_counts.get(key, 0) for key in keys], dtype=np.float64
        )
        reference = reference / reference.sum() if reference.sum() else reference
        midpoint = (probabilities + reference) / 2

        def kl_divergence(left, right):
            valid = left > 0
            return float((left[valid] * np.log2(left[valid] / right[valid])).sum())

        result['js_divergence_vs_bucket_0'] = (
            kl_divergence(probabilities, midpoint)
            + kl_divergence(reference, midpoint)
        ) / 2
    return result


def probability_shift_metrics(logits, reference_logits, temperature=1.0):
    """Compare full categorical distributions instead of sparse sampled tokens."""
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    reference = torch.softmax(reference_logits.float() / temperature, dim=-1)
    midpoint = (probabilities + reference) / 2
    epsilon = torch.finfo(probabilities.dtype).tiny
    js = 0.5 * (
        (probabilities * torch.log2((probabilities + epsilon) / (midpoint + epsilon))).sum(-1)
        + (reference * torch.log2((reference + epsilon) / (midpoint + epsilon))).sum(-1)
    )
    entropy = -(probabilities * torch.log2(probabilities + epsilon)).sum(-1)
    return {
        'mean_entropy': float(entropy.mean().item()),
        'mean_js_divergence_vs_bucket_0': float(js.mean().item()),
        'mean_total_variation_vs_bucket_0': float(
            (probabilities - reference).abs().sum(-1).div(2).mean().item()
        ),
        'top1_change_rate_vs_bucket_0': float(
            (probabilities.argmax(-1) != reference.argmax(-1)).float().mean().item()
        ),
    }


def load_model(path, device):
    return Kronos.from_pretrained(
        path,
        num_sectors=0,
        num_size_buckets=10,
        context_layer=10,
        use_size_percentile=False,
        size_mlp_hidden_dim=64,
    ).to(device).eval()


def actual_window(panel, record):
    future = panel[record['symbol']].loc[record['future_dates'], FEATURE_COLUMNS]
    values = np.concatenate([
        record['context'][FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        future.to_numpy(dtype=np.float32),
    ])
    past = values[:-PRED_LEN]
    mean = past.mean(axis=0)
    std = past.std(axis=0)
    values = np.clip((values - mean) / (std + 1e-5), -5, 5)
    dates = record['context'].index.append(record['future_dates'])
    return values.astype(np.float32), time_features(dates)


def evaluate_token_loss(model, tokenizer, panel, periods, device, batch_size, mode):
    losses = []
    for period in periods:
        records = period['records']
        for offset in range(0, len(records), batch_size):
            batch_records = records[offset:offset + batch_size]
            windows = [actual_window(panel, record) for record in batch_records]
            x = torch.as_tensor(np.stack([item[0] for item in windows]), device=device)
            stamp = torch.as_tensor(np.stack([item[1] for item in windows]), device=device)
            true_buckets = torch.as_tensor(
                [record['size_bucket'] for record in batch_records], device=device
            )
            buckets = (
                true_buckets if mode == 'true_bucket'
                else torch.full_like(true_buckets, model.num_size_buckets)
            )
            with torch.no_grad():
                s1, s2 = tokenizer.encode(x, half=True)
                s1_logits, s2_logits = model(
                    s1[:, :-1], s2[:, :-1], stamp[:, :-1],
                    size_bucket=buckets,
                    use_teacher_forcing=True,
                    s1_targets=s1[:, 1:],
                )
                sample_loss = (
                    F.cross_entropy(
                        s1_logits.transpose(1, 2), s1[:, 1:], reduction='none'
                    ).mean(1)
                    + F.cross_entropy(
                        s2_logits.transpose(1, 2), s2[:, 1:], reduction='none'
                    ).mean(1)
                ) / 2
            for record, loss in zip(batch_records, sample_loss.cpu().tolist()):
                losses.append({
                    'mode': mode,
                    'period': period['period'],
                    'symbol': record['symbol'],
                    'true_bucket': record['size_bucket'],
                    'token_loss': float(loss),
                })
        print(f'{mode} loss: period {period["period"] + 1}/{len(periods)}')
    return pd.DataFrame(losses)


def evaluate_predictions(
    model, tokenizer, periods, device, batch_size, sample_count, seed,
    temperature, mode,
):
    rows = []
    for period in periods:
        torch.manual_seed(seed + period['period'])
        np.random.seed(seed + period['period'])
        records = period['records']
        for offset in range(0, len(records), batch_size):
            batch_records = records[offset:offset + batch_size]
            batch = prepare_batch(batch_records)
            buckets = torch.as_tensor(batch['buckets'], device=device)
            if mode == 'unknown_bucket':
                buckets = torch.full_like(buckets, model.num_size_buckets)
            forecast = auto_regressive_inference(
                tokenizer, model,
                torch.as_tensor(batch['x'], device=device),
                torch.as_tensor(batch['x_stamp'], device=device),
                torch.as_tensor(batch['y_stamp'], device=device),
                max_context=512, pred_len=PRED_LEN, clip=5, T=temperature,
                top_k=0, top_p=0.9, sample_count=sample_count, verbose=False,
                size_bucket=buckets,
            )[:, -PRED_LEN:, :]
            close_index = FEATURE_COLUMNS.index('close')
            predicted_close = (
                forecast[:, :, close_index].mean(axis=1)
                * (batch['stds'][:, close_index] + 1e-5)
                + batch['means'][:, close_index]
            )
            for record, close in zip(batch_records, predicted_close):
                rows.append({
                    'model': mode,
                    'period': period['period'],
                    'signal_date': period['signal_date'],
                    'symbol': record['symbol'],
                    'true_bucket': record['size_bucket'],
                    'score': float(close / record['last_close'] - 1),
                    'actual_close_return': record['actual_close_return'],
                })
        print(f'{mode} forecast: period {period["period"] + 1}/{len(periods)}')
    return pd.DataFrame(rows)


def paired_period_differences(predictions):
    metrics = []
    for (mode, period), rows in predictions.groupby(['model', 'period']):
        metrics.append({
            'mode': mode,
            'period': period,
            'rank_ic': rows['score'].corr(
                rows['actual_close_return'], method='spearman'
            ),
            'direction_accuracy': (
                np.sign(rows['score']) == np.sign(rows['actual_close_return'])
            ).mean(),
        })
    metrics = pd.DataFrame(metrics)
    result = {}
    rng = np.random.default_rng(20260807)
    for metric in ('rank_ic', 'direction_accuracy'):
        table = metrics.pivot(index='period', columns='mode', values=metric).dropna()
        difference = table['unknown_bucket'] - table['true_bucket']
        bootstrap = rng.choice(
            difference.to_numpy(), size=(5000, len(difference)), replace=True
        ).mean(axis=1)
        result[f'unknown_minus_true_{metric}'] = {
            'mean': float(difference.mean()),
            'ci95': [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
    return result


def balance_period_records(periods, stocks_per_bucket, seed):
    """Select the same maximum count from every available true bucket."""
    if stocks_per_bucket <= 0:
        return periods
    balanced = []
    for period in periods:
        selected = []
        for bucket in range(10):
            candidates = [
                record for record in period['records']
                if record['size_bucket'] == bucket
            ]
            candidates.sort(key=lambda record: record['symbol'])
            rng = np.random.default_rng(seed + period['period'] * 100 + bucket)
            count = min(stocks_per_bucket, len(candidates))
            if count:
                positions = rng.choice(len(candidates), size=count, replace=False)
                selected.extend(candidates[position] for position in sorted(positions))
        if len(selected) < 100:
            raise RuntimeError(
                f'Balanced period {period["period"]} has only {len(selected)} records'
            )
        balanced.append({**period, 'records': selected})
    return balanced


def choose_counterfactual_records(period, count):
    records = sorted(
        period['records'], key=lambda item: (item['size_bucket'], item['symbol'])
    )
    selected = []
    for bucket in range(10):
        candidates = [record for record in records if record['size_bucket'] == bucket]
        if candidates:
            selected.append(candidates[len(candidates) // 2])
    if len(selected) >= count:
        return selected[:count]
    used = {record['symbol'] for record in selected}
    remaining = [record for record in records if record['symbol'] not in used]
    positions = np.linspace(0, len(remaining) - 1, count - len(selected)).round().astype(int)
    return selected + [remaining[position] for position in positions]


def counterfactual_sweep(
    model, tokenizer, period, device, count, sample_count, seed, temperature,
):
    records = choose_counterfactual_records(period, count)
    batch = prepare_batch(records)
    close_index = FEATURE_COLUMNS.index('close')
    rows = []
    token_arrays = {}
    token_counts = {}
    probability_metrics = {}
    context_x = torch.as_tensor(batch['x'], device=device)
    context_stamp = torch.as_tensor(batch['x_stamp'], device=device)
    with torch.no_grad():
        context_s1, context_s2 = tokenizer.encode(context_x, half=True)
    reference_logits = None
    for bucket in range(10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        buckets = torch.full((len(records),), bucket, device=device)
        with torch.no_grad():
            s1_logits, s2_logits = model(
                context_s1[:, :-1], context_s2[:, :-1], context_stamp[:, :-1],
                size_bucket=buckets,
                use_teacher_forcing=True,
                s1_targets=context_s1[:, 1:],
            )
        if reference_logits is None:
            reference_logits = (s1_logits.detach(), s2_logits.detach())
        probability_metrics[bucket] = {
            's1': probability_shift_metrics(
                s1_logits, reference_logits[0], temperature
            ),
            's2': probability_shift_metrics(
                s2_logits, reference_logits[1], temperature
            ),
        }
        forecast, tokens = auto_regressive_inference(
            tokenizer, model,
            context_x,
            context_stamp,
            torch.as_tensor(batch['y_stamp'], device=device),
            max_context=512, pred_len=PRED_LEN, clip=5, T=temperature,
            top_k=0, top_p=0.9, sample_count=sample_count, verbose=False,
            size_bucket=buckets,
            return_samples=True, return_generated_tokens=True,
        )
        token_arrays[bucket] = tokens
        token_counts[bucket] = {
            level: Counter(values.reshape(-1).tolist())
            for level, values in tokens.items()
        }
        close = (
            forecast[:, :, -PRED_LEN:, close_index]
            * (batch['stds'][:, None, None, close_index] + 1e-5)
            + batch['means'][:, None, None, close_index]
        )
        mean_paths = close.mean(axis=2)
        daily_returns = np.diff(close, axis=2) / close[:, :, :-1]
        for index, record in enumerate(records):
            path_returns = mean_paths[index] / record['last_close'] - 1
            rows.append({
                'symbol': record['symbol'],
                'actual_bucket': record['size_bucket'],
                'counterfactual_bucket': bucket,
                'predicted_return': float(path_returns.mean()),
                'path_return_std': float(path_returns.std()),
                'forecast_daily_volatility': float(daily_returns[index].std()),
            })
        print(f'counterfactual bucket {bucket}/9')

    frame = pd.DataFrame(rows)
    baseline_tokens = token_arrays[0]
    bucket_summary = []
    for bucket, bucket_rows in frame.groupby('counterfactual_bucket'):
        item = {
            'bucket': int(bucket),
            'mean_predicted_return': float(bucket_rows['predicted_return'].mean()),
            'mean_forecast_daily_volatility': float(
                bucket_rows['forecast_daily_volatility'].mean()
            ),
        }
        for level in ('s1', 's2'):
            metrics = distribution_metrics(
                token_counts[bucket][level], token_counts[0][level]
            )
            metrics['change_rate_vs_bucket_0'] = float(
                (token_arrays[bucket][level] != baseline_tokens[level]).mean()
            )
            item[f'{level}_sampled_tokens'] = metrics
            item[f'{level}_probability_distribution'] = probability_metrics[bucket][level]
        bucket_summary.append(item)
    sensitivity = frame.groupby('symbol')['predicted_return'].agg(
        lambda values: float(values.max() - values.min())
    )
    summary = {
        'records': len(records),
        'median_return_range_across_buckets': float(sensitivity.median()),
        'mean_return_range_across_buckets': float(sensitivity.mean()),
        'max_return_range_across_buckets': float(sensitivity.max()),
        'bucket_summary': bucket_summary,
    }
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        default='./outputs/models/a_share_v4_corrected_2026_replay20_latest/checkpoints/last_model',
    )
    parser.add_argument(
        '--holdout', default='./data/a_share_v3/processed_datasets/symbol_holdout_data.pkl'
    )
    parser.add_argument('--manifest', default='./data/a_share_v3/universe_manifest.csv')
    parser.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    parser.add_argument(
        '--output-dir', default='./outputs/backtest_results/v4b_size_condition_ablation'
    )
    parser.add_argument('--signal-start', default='2026-06-18')
    parser.add_argument('--signal-end', default='2026-07-16')
    parser.add_argument('--period-count', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sample-count', type=int, default=1)
    parser.add_argument('--counterfactual-count', type=int, default=10)
    parser.add_argument('--counterfactual-sample-count', type=int, default=3)
    parser.add_argument(
        '--stocks-per-bucket', type=int, default=0,
        help='Deterministically cap every true bucket to this many stocks per date.',
    )
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=20260807)
    args = parser.parse_args()

    if args.period_count <= 0 or args.sample_count <= 0:
        parser.error('period and sample counts must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel, calendar = load_holdout(args.holdout, args.manifest)
    periods = build_signal_periods(
        panel, calendar, args.signal_start, args.signal_end, args.period_count
    )
    periods = balance_period_records(periods, args.stocks_per_bucket, args.seed)
    device = torch.device(
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device={device}, periods={len(periods)}, holdout_symbols={len(panel)}')
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    model = load_model(args.model, device)

    loss_frames = [
        evaluate_token_loss(
            model, tokenizer, panel, periods, device, args.batch_size, mode
        )
        for mode in ('true_bucket', 'unknown_bucket')
    ]
    losses = pd.concat(loss_frames, ignore_index=True)
    losses.to_csv(output / 'token_losses.csv', index=False)

    prediction_frames = [
        evaluate_predictions(
            model, tokenizer, periods, device, args.batch_size,
            args.sample_count, args.seed, args.temperature, mode,
        )
        for mode in ('true_bucket', 'unknown_bucket')
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / 'ablation_predictions.csv', index=False)

    counterfactuals, counterfactual_summary = counterfactual_sweep(
        model, tokenizer, periods[-1], device, args.counterfactual_count,
        args.counterfactual_sample_count, args.seed, args.temperature,
    )
    counterfactuals.to_csv(output / 'counterfactual_predictions.csv', index=False)

    loss_summary = losses.groupby('mode')['token_loss'].mean().to_dict()
    loss_pairs = losses.pivot_table(
        index=['period', 'symbol'], columns='mode', values='token_loss'
    ).dropna()
    period_loss_difference = (
        loss_pairs['unknown_bucket'] - loss_pairs['true_bucket']
    ).groupby('period').mean().to_numpy()
    rng = np.random.default_rng(args.seed)
    loss_bootstrap = rng.choice(
        period_loss_difference,
        size=(5000, len(period_loss_difference)),
        replace=True,
    ).mean(axis=1)
    prediction_summary = {
        mode: summarize(frame)[0]
        for mode, frame in predictions.groupby('model')
    }
    summary = {
        'configuration': {
            'model': args.model,
            'holdout': args.holdout,
            'periods': len(periods),
            'signal_start': args.signal_start,
            'signal_end': args.signal_end,
            'sample_count': args.sample_count,
            'stocks_per_bucket': args.stocks_per_bucket,
            'observations_by_true_bucket': {
                str(int(bucket)): int(count)
                for bucket, count in losses[losses['mode'] == 'true_bucket'][
                    'true_bucket'
                ].value_counts().sort_index().items()
            },
            'temperature': args.temperature,
            'unknown_bucket': model.num_size_buckets,
            'device': str(device),
            'seed': args.seed,
        },
        'condition_ablation': {
            'mean_token_loss': loss_summary,
            'unknown_minus_true_token_loss': float(
                loss_summary['unknown_bucket'] - loss_summary['true_bucket']
            ),
            'unknown_minus_true_token_loss_ci95': [
                float(np.quantile(loss_bootstrap, 0.025)),
                float(np.quantile(loss_bootstrap, 0.975)),
            ],
            'prediction_metrics': prediction_summary,
            'paired_period_differences': paired_period_differences(predictions),
        },
        'counterfactual_bucket_sweep': counterfactual_summary,
        'size_embedding': {
            'row_norms': [
                float(value) for value in model.size_emb.weight.detach().float().norm(dim=1)
            ],
            'unknown_row': model.num_size_buckets,
        },
        'interpretation': {
            'ablation': (
                'Positive unknown-minus-true loss and degraded return metrics indicate '
                'that the trained model uses market-cap conditioning.'
            ),
            'counterfactual': (
                'Near-zero return range and token divergence indicate ignored conditioning; '
                'large changes relative to normal forecast dispersion indicate over-reliance.'
            ),
        },
    }
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'RESULT_PATH {output}')

    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()


if __name__ == '__main__':
    main()
