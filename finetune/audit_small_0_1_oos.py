"""Re-score saved predictions against raw panel closes; never run a model."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rank_ic(a, b):
    return float(a.rank(method='average').corr(b.rank(method='average')))


def restore_returns(frame, package):
    """Recover the old denominator exactly, then rebase onto the raw close."""
    manifest = json.loads((package / 'evaluation_manifest.json').read_text())
    panel_path = package / manifest['artifacts']['panel_file']
    with panel_path.open('rb') as f:
        panel = pickle.load(f)  # Trusted, locally built project dataset.
    restored = []
    for row in frame.drop_duplicates('identity').itertuples():
        symbol, start, asof, target = row.identity.split('|')
        window = panel[symbol].iloc[int(start):int(start) + 131]
        assert len(window) == 131
        assert str(window.index[119].date()) == asof
        assert str(window.index[129].date()) == target
        values = window[FEATURES].to_numpy(dtype=np.float32)
        mean, std = values[:120].mean(0), values[:120].std(0)
        normalized = np.clip((values - mean) / (std + 1e-5), -5, 5).astype(np.float32)
        old_close = float(normalized[119, 3] * (std[3] + 1e-5) + mean[3])
        closes = window['close'].to_numpy(dtype=np.float64)
        actual = closes[120:130] / closes[119] - 1
        assert abs(actual[-1] - row.return_10d) < 1e-8, row.identity
        restored.append({'identity': row.identity, 'denominator_ratio': old_close / closes[119],
                         **{f'raw_actual_d{h+1}': float(actual[h]) for h in range(10)}})
    result = frame.merge(pd.DataFrame(restored), on='identity', how='left', validate='many_to_one')
    assert np.isfinite(result.denominator_ratio).all()
    for h in range(1, 11):
        result[f'raw_predicted_d{h}'] = (result[f'predicted_return_d{h}'] + 1) * result.denominator_ratio - 1
    return result, {'manifest_sha256': sha(package / 'evaluation_manifest.json'),
                    'panel_sha256': sha(panel_path), 'identities': len(restored),
                    'old_denominator_relative_change_max': float((result.denominator_ratio - 1).abs().max())}


def metrics(frame):
    output = {}
    for model, group in frame.groupby('model'):
        horizons = []
        for h in range(1, 11):
            pred, actual = f'raw_predicted_d{h}', f'raw_actual_d{h}'
            daily = []
            for date, g in group.groupby('asof_date'):
                ranked = g.sort_values([pred, 'identity'], kind='stable')
                k = max(1, int(np.ceil(len(g) * .1)))
                daily.append({'date': date, 'n': len(g), 'rank_ic': rank_ic(g[pred], g[actual]),
                              'top_bottom_spread': float(ranked.tail(k)[actual].mean() - ranked.head(k)[actual].mean())})
            ic = pd.Series([r['rank_ic'] for r in daily])
            horizons.append({'horizon': h, 'direction_accuracy': float(((group[pred] > 0) == (group[actual] > 0)).mean()),
                             'pooled_rank_ic': rank_ic(group[pred], group[actual]),
                             'daily_rank_ic': float(ic.mean()), 'ic_std': float(ic.std(ddof=1)),
                             'icir': float(ic.mean() / ic.std(ddof=1)), 'positive_ic_dates': int((ic > 0).sum()),
                             'return_mae': float((group[pred] - group[actual]).abs().mean()),
                             'top_bottom_spread': float(np.mean([r['top_bottom_spread'] for r in daily])),
                             'daily': daily})
        output[model] = {'windows': len(group), 'dates': group.asof_date.nunique(), 'horizons': horizons}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    specs = [
        ('first_6', Path('artifacts/kronos_small_0_1_c2_alpha_oos_output/kronos_small_0_1_stage2_oos/predictions.csv.gz'),
         Path('data/a_share_v1_beta_eval_20260826/package'), 30930),
        ('next_13', Path('artifacts/kronos_small_0_1_c2_alpha_oos_output_smmt315_20260914/kronos_small_0_1_stage2_oos/predictions.csv.gz'),
         Path('data/a_share_v1_beta_eval_20260914_incremental/package'), 66986),
    ]
    parts, provenance, results = [], {}, {}
    for period, path, package, count in specs:
        print(f'auditing {period}', flush=True)
        frame = pd.read_csv(path)
        frame['model'] = frame.model.replace({'best_segment_530': 'c2_best_segment_179', 'last_segment_534': 'c2_last_segment_267'})
        assert not frame.duplicated(['model', 'identity']).any()
        ids = [set(g.identity) for _, g in frame.groupby('model')]
        assert len(ids) == 3 and all(len(s) == count and s == ids[0] for s in ids)
        frame, info = restore_returns(frame, package)
        info.update(predictions_path=str(path), predictions_sha256=sha(path))
        provenance[period] = info
        results[period] = metrics(frame)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    assert not combined.duplicated(['model', 'identity']).any()
    assert combined.asof_date.nunique() == 19
    results['combined_19'] = metrics(combined)
    report = {'method': {'model_rerun': False, 'target': 'raw panel close_h / raw signal close - 1',
                        'prediction': '(saved_return+1)*reconstructed_old_denominator/raw_signal_close-1',
                        'direction': '(pred > 0) == (actual > 0); zero is nonpositive',
                        'ranking': 'Spearman average tie ranks; daily dates equal weighted',
                        'spread': 'daily top/bottom ceil(0.1*N), raw cumulative horizon return, no trading costs',
                        'icir': 'daily mean/sample std, not annualized'},
              'provenance': provenance, 'periods': results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for period, models in results.items():
        for name, result in models.items():
            print(period, name, {k:v for k,v in result['horizons'][-1].items() if k != 'daily'})


if __name__ == '__main__':
    main()
