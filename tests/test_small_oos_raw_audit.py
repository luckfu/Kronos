import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from finetune.audit_small_0_1_oos import FEATURES, metrics, restore_returns


def test_raw_prices_and_saved_prediction_denominator(tmp_path):
    dates = pd.bdate_range('2026-01-01', periods=131)
    close = np.linspace(10., 11., 131)
    close[119] = 20.
    close[120:130] = np.linspace(21., 40., 10)
    panel = pd.DataFrame({k: close.copy() for k in FEATURES}, index=dates)
    payload = pickle.dumps({'stock': panel})
    (tmp_path / 'panel.pkl').write_bytes(payload)
    manifest = {'artifacts': {'panel_file': 'panel.pkl', 'panel_sha256': hashlib.sha256(payload).hexdigest()}}
    (tmp_path / 'evaluation_manifest.json').write_text(json.dumps(manifest))
    x = panel[FEATURES].to_numpy(dtype=np.float32)
    mean, std = x[:120].mean(0), x[:120].std(0)
    z = np.clip((x - mean) / (std + 1e-5), -5, 5)
    old = float(z[119, 3] * (std[3] + 1e-5) + mean[3])
    assert abs(old - close[119]) > 1
    identity = f'stock|0|{dates[119].date()}|{dates[129].date()}'
    record = {'identity': identity, 'return_10d': close[129] / close[119] - 1,
              **{f'predicted_return_d{h}': 30. / old - 1 for h in range(1, 11)}}
    result, _ = restore_returns(pd.DataFrame([record]), tmp_path)
    for h in range(1, 11):
        assert result[f'raw_predicted_d{h}'].iloc[0] == pytest.approx(.5)
        assert result[f'raw_actual_d{h}'].iloc[0] == pytest.approx(close[119+h] / 20 - 1)


def test_direction_zero_nonpositive_and_daily_weighting():
    rows = []
    for d, n in [('a', 4), ('b', 6)]:
        for i in range(n):
            row = {'identity': f'{d}/{i}', 'asof_date': d, 'model': 'test'}
            for h in range(1, 11):
                row[f'raw_actual_d{h}'] = float(i)
                row[f'raw_predicted_d{h}'] = float(i) if d == 'a' else float(n-1-i)
            rows.append(row)
    report = metrics(pd.DataFrame(rows))['test']['horizons'][0]
    assert report['direction_accuracy'] == .8
    assert report['daily_rank_ic'] == pytest.approx(0.)
    assert report['positive_ic_dates'] == 1
