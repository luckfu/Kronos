import numpy as np
import pandas as pd
import json

import webui.app as web_app


def test_remote_context_maps_market_cap_to_reference(monkeypatch):
    monkeypatch.setattr(web_app, 'size_reference_cache', {
        'date': pd.Timestamp('2026-07-31'),
        'market_caps': np.asarray([10.0, 20.0, 30.0, 40.0]),
        'count': 4,
    })
    context = pd.DataFrame({
        'timestamps': pd.to_datetime(['2026-07-30', '2026-07-31']),
        'close': [1.0, 2.0],
        'volume': [10.0, 10.0],
        'turn': [100.0, 100.0],
    })

    result = web_app.size_condition_from_remote_context(context)

    assert result['estimated_market_cap'] == 20.0
    assert result['size_percentile'] == 0.5
    assert result['size_bucket'] == 5
    assert result['size_reference_date'] == pd.Timestamp('2026-07-31')


def test_remote_context_uses_latest_non_suspended_turnover(monkeypatch):
    monkeypatch.setattr(web_app, 'size_reference_cache', {
        'date': pd.Timestamp('2026-07-31'),
        'market_caps': np.asarray([10.0, 20.0, 30.0, 40.0]),
        'count': 4,
    })
    context = pd.DataFrame({
        'timestamps': pd.to_datetime(['2026-07-30', '2026-07-31']),
        'close': [2.0, 2.0],
        'volume': [10.0, 0.0],
        'turn': [100.0, 0.0],
    })

    result = web_app.size_condition_from_remote_context(context)

    assert result['estimated_market_cap'] == 20.0
    assert result['size_bucket_asof'] == pd.Timestamp('2026-07-30')


def test_local_context_uses_percentile_or_bucket_midpoint():
    with_percentile = pd.DataFrame({
        'timestamps': pd.to_datetime(['2026-07-31']),
        'size_bucket': [3],
        'size_percentile': [0.373],
    })
    without_percentile = with_percentile.drop(columns='size_percentile')

    assert web_app.size_condition_from_local_frame(with_percentile)['size_percentile'] == 0.373
    assert web_app.size_condition_from_local_frame(without_percentile)['size_percentile'] == 0.35


def test_portable_size_reference_does_not_require_raw_panel(monkeypatch, tmp_path):
    reference_path = tmp_path / 'size_reference.json'
    reference_path.write_text(json.dumps({
        'reference_date': '2026-07-31',
        'market_caps': [float(value) for value in range(1, 101)],
        'count': 100,
    }))
    monkeypatch.setattr(web_app, 'size_reference_cache', None)
    monkeypatch.setattr(web_app, 'A_SHARE_SIZE_REFERENCE_PATH', str(reference_path))
    monkeypatch.setattr(web_app, 'A_SHARE_RAW_DATA_PATH', str(tmp_path / 'missing.csv'))

    result = web_app.load_latest_size_reference()

    assert result['date'] == pd.Timestamp('2026-07-31')
    assert result['count'] == 100
    assert result['market_caps'].tolist() == [float(value) for value in range(1, 101)]


def test_unknown_symbol_can_be_confirmed_without_local_panel(monkeypatch):
    monkeypatch.setattr(
        web_app,
        'resolve_request_data',
        lambda data: (None, 'Unknown A-share symbol: sh.600012', None, None),
    )
    client = web_app.app.test_client()

    response = client.post('/api/load-data', json={'symbol': '600012'})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['data_info']['symbol'] == 'sh.600012'
    assert payload['data_info']['remote_only'] is True


def test_symbol_lookup_degrades_cleanly_without_local_dataset(monkeypatch):
    monkeypatch.setattr(
        web_app,
        'load_a_share_splits',
        lambda: (_ for _ in ()).throw(FileNotFoundError('panel missing')),
    )

    frame, error = web_app.get_a_share_symbol_frame('600012')

    assert frame is None
    assert error == 'panel missing'
