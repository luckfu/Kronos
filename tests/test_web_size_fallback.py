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


def test_portfolio_ranking_accepts_remote_only_symbols(monkeypatch):
    monkeypatch.setattr(web_app, 'size_reference_cache', {
        'date': pd.Timestamp('2026-08-06'),
        'market_caps': np.linspace(1_000_000, 1_000_000_000, 1000),
        'count': 1000,
    })

    def remote_inputs(symbol, lookback, pred_len, history_rows=None):
        end = pd.Timestamp('2026-08-06' if symbol == 'sz.000430' else '2026-08-05')
        dates = pd.bdate_range(end=end, periods=history_rows)
        values = np.arange(len(dates), dtype=np.float64) + 10
        context = pd.DataFrame({
            'timestamps': dates,
            'open': values,
            'high': values + 1,
            'low': values - 1,
            'close': values + 0.5,
            'volume': np.full(len(dates), 1_000_000.0),
            'amount': np.full(len(dates), 10_000_000.0),
            'turn': np.full(len(dates), 1.0),
        })
        return {
            'context': context,
            'future_dates': pd.Series(
                pd.bdate_range(end + pd.Timedelta(days=1), periods=pred_len),
                name='timestamps',
            ),
            'size_bucket': 5,
            'size_percentile': 0.55,
            'size_source': 'baostock_proxy_vs_local_cross_section',
            'in_local_panel': False,
            'data_source': 'baostock',
        }

    monkeypatch.setattr(web_app, 'latest_prediction_inputs', remote_inputs)

    batch = web_app.portfolio_ranking_batch(
        ['sz.000430', 'sz.300886'],
        lookback=90,
        pred_len=10,
    )

    assert batch['as_of_date'] == pd.Timestamp('2026-08-05')
    assert batch['x'].shape == (2, 90, 6)
    assert batch['y_stamp'].shape == (2, 10, 5)
    assert all(not item['in_local_panel'] for item in batch['records'])


def test_forecast_return_summary_uses_horizon_average_per_path():
    summary = web_app.forecast_return_summary(
        np.asarray([
            [100.0, 110.0],
            [90.0, 100.0],
            [120.0, 100.0],
        ]),
        latest_close=100.0,
    )

    assert summary['predicted_average_close_p50'] == 105.0
    assert np.isclose(summary['predicted_return_p50'], 0.05)
    assert np.isclose(summary['positive_path_rate'], 2 / 3)
