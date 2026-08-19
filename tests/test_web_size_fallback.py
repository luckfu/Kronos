import numpy as np
import pandas as pd
import json
import pytest

import webui.app as web_app


def test_production_model_uses_v6_segment542_checkpoint():
    config = web_app.AVAILABLE_MODELS['a-share-size-kronos-base']

    assert config['default_lookback'] == 120
    assert 'a_share_v6_segment542_latest' in config['model_id']
    assert 'V6' in config['name']


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
        'latest_prediction_inputs',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('offline')),
    )
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


def test_symbol_list_identifies_training_panel_without_limiting_market(monkeypatch):
    rows = pd.DataFrame(
        {'size_bucket': [4], 'close': [10.5]},
        index=pd.to_datetime(['2026-08-07']),
    )
    monkeypatch.setattr(web_app, 'load_a_share_splits', lambda: {
        'train': {'sh.600000': rows},
        'val': {'sz.000001': rows},
    })
    client = web_app.app.test_client()

    response = client.get('/api/a-share/symbols')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['count'] == 2
    assert payload['training_panel_count'] == 2
    assert payload['market_scope'] == 'all-a-share'


def test_homepage_describes_symbols_as_autocomplete_training_panel():
    response = web_app.app.test_client().get('/')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert '支持全市场 A 股' in page
    assert '只股票可用' not in page
    assert 'Modal 批量排名' not in page
    assert 'id="batch-input"' not in page
    assert '一个代码显示详情，2–12 个代码自动排名' in page


def test_result_views_are_switched_exclusively():
    response = web_app.app.test_client().get('/')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "function showOnlyResultContent(activeId)" in page
    assert "['result-content', 'batch-results', 'history-results']" in page
    assert "showOnlyResultContent('batch-results')" in page
    assert "showOnlyResultContent('history-results')" in page
    assert "showOnlyResultContent('result-content')" in page

def test_history_summary_drops_symbol_repeated_as_stock_name(monkeypatch):
    web_app.stock_name_cache.clear()
    monkeypatch.setattr(web_app, 'query_remote_stock_name', lambda symbol: None)
    summary = web_app.prediction_record_summary({
        'record_id': 'record-1',
        'symbol': 'sz.000001',
        'name': 'SZ.000001',
        'prediction_results': [],
    })

    assert summary['symbol'] == 'sz.000001'
    assert summary['name'] is None


def test_history_summary_recovers_missing_stock_name(monkeypatch):
    monkeypatch.setattr(web_app, 'query_remote_stock_name', lambda symbol: {
        'sz.000001': '平安银行',
        'sh.600000': '浦发银行',
    }.get(symbol))

    summary = web_app.prediction_record_summary({
        'record_id': 'record-2',
        'symbol': 'sz.000001',
        'name': 'sz.000001',
        'prediction_results': [],
    })

    assert summary['name'] == '平安银行'


def test_history_summary_distinguishes_single_and_cross_section_profiles():
    single = web_app.prediction_record_summary({
        'record_id': 'single-123', 'symbol': 'sz.000001',
        'prediction_results': [],
    })
    batch = web_app.prediction_record_summary({
        'record_id': 'batch-123', 'symbol': 'sz.000001',
        'batch_id': 'batch_20260819_1', 'prediction_results': [],
    })

    assert single['inference_profile'] == 'single_stock_entry'
    assert single['inference_profile_label'] == '单股买点判断'
    assert batch['inference_profile'] == 'cross_section_ranking'
    assert batch['inference_profile_label'] == '横截面排序'


def test_history_summary_preserves_ranking_group_fields():
    summary = web_app.prediction_record_summary({
        'record_id': 'batch-123', 'batch_id': 'batch_1', 'symbol': 'sz.000001',
        'prediction_results': [], 'positive_path_rate': 0.6,
        'timing_signal': {'key': 'bullish', 'label': '偏多'},
    })

    assert summary['batch_id'] == 'batch_1'
    assert summary['positive_path_rate'] == 0.6
    assert summary['timing_signal']['label'] == '偏多'


def test_homepage_exposes_profile_specific_parameters():
    response = web_app.app.test_client().get('/')
    page = response.get_data(as_text=True)

    assert 'top_p: 1.0' in page
    assert 'top_p: 0.8' in page
    assert 'history-single-tab' in page
    assert 'history-ranking-tab' in page


def test_prediction_history_can_delete_only_the_requested_record(monkeypatch, tmp_path):
    record_path = tmp_path / 'record-123.json'
    other_path = tmp_path / 'record-456.json'
    record_path.write_text(json.dumps({'record_id': 'record-123'}), encoding='utf-8')
    other_path.write_text(json.dumps({'record_id': 'record-456'}), encoding='utf-8')
    monkeypatch.setattr(web_app, 'PREDICTION_RESULTS_DIR', str(tmp_path))

    response = web_app.app.test_client().delete('/api/prediction-history/record-123')

    assert response.status_code == 200
    assert response.get_json()['deleted'] is True
    assert not record_path.exists()
    assert other_path.exists()


def test_prediction_history_rejects_invalid_delete_id(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, 'PREDICTION_RESULTS_DIR', str(tmp_path))

    response = web_app.app.test_client().delete('/api/prediction-history/../secret')

    assert response.status_code in (400, 404)


def test_latest_inputs_prefers_eastmoney_over_baostock(monkeypatch):
    dates = pd.bdate_range(end='2026-08-07', periods=120)
    remote = pd.DataFrame({
        'timestamps': dates,
        'open': np.full(120, 10.0),
        'high': np.full(120, 11.0),
        'low': np.full(120, 9.0),
        'close': np.full(120, 10.0),
        'volume': np.full(120, 1_000_000.0),
        'amount': np.full(120, 10_000_000.0),
        'turn': np.full(120, 1.0),
        'pctChg': np.zeros(120),
    })
    local = remote.assign(size_bucket=4, size_percentile=0.45)
    monkeypatch.setattr(web_app, 'get_a_share_symbol_frame', lambda symbol: (local, None))
    monkeypatch.setattr(
        web_app,
        'query_eastmoney_daily_data',
        lambda *args, **kwargs: (
            remote,
            pd.Series(pd.bdate_range('2026-08-10', periods=10)),
        ),
    )
    monkeypatch.setattr(
        web_app,
        'query_latest_daily_data',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('BaoStock should not be queried after Eastmoney succeeds')
        ),
    )
    monkeypatch.setattr(web_app, 'size_reference_cache', {
        'date': pd.Timestamp('2026-07-31'),
        'market_caps': np.linspace(1_000_000, 2_000_000_000, 1000),
        'count': 1000,
    })

    result = web_app.latest_prediction_inputs('600000', 120, 10)

    assert result['data_source'] == 'eastmoney'
    assert result['context']['timestamps'].iloc[-1] == pd.Timestamp('2026-08-07')
    assert result['refresh_error'] is None


def test_market_data_cache_merges_incremental_rows_and_replaces_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, 'MARKET_DATA_CACHE_DIR', str(tmp_path))
    original = pd.DataFrame([{
        'timestamps': pd.Timestamp('2026-08-06'), 'open': 10.0, 'high': 11.0,
        'low': 9.0, 'close': 10.0, 'volume': 100.0, 'amount': 1000.0,
        'turn': 1.0, 'pctChg': 0.0,
    }])
    revised_and_new = pd.DataFrame([
        {
            'timestamps': pd.Timestamp('2026-08-06'), 'open': 10.1, 'high': 11.1,
            'low': 9.1, 'close': 10.1, 'volume': 110.0, 'amount': 1100.0,
            'turn': 1.1, 'pctChg': 1.0,
        },
        {
            'timestamps': pd.Timestamp('2026-08-07'), 'open': 10.2, 'high': 11.2,
            'low': 9.2, 'close': 10.2, 'volume': 120.0, 'amount': 1200.0,
            'turn': 1.2, 'pctChg': 1.0,
        },
    ])

    web_app._merge_market_data_cache('600000', original)
    merged = web_app._merge_market_data_cache('sh.600000', revised_and_new)

    assert len(merged) == 2
    assert merged['timestamps'].tolist() == [pd.Timestamp('2026-08-06'), pd.Timestamp('2026-08-07')]
    assert merged.iloc[0]['close'] == pytest.approx(10.1)
    assert (tmp_path / 'sh_600000.csv').exists()


def test_market_data_cache_is_ignored_by_git():
    assert web_app.MARKET_DATA_CACHE_DIR.endswith('market_data_cache')


def test_load_data_confirms_stock_with_live_context(monkeypatch):
    dates = pd.bdate_range(end='2026-08-07', periods=120)
    context = pd.DataFrame({
        'timestamps': dates,
        'open': np.full(120, 10.0),
        'high': np.full(120, 11.0),
        'low': np.full(120, 9.0),
        'close': np.full(120, 10.5),
        'volume': np.full(120, 1_000_000.0),
        'amount': np.full(120, 10_000_000.0),
        'turn': np.full(120, 1.0),
    })
    monkeypatch.setattr(web_app, 'latest_prediction_inputs', lambda *args, **kwargs: {
        'context': context,
        'size_bucket': 5,
        'size_percentile': 0.55,
        'in_local_panel': True,
        'data_source': 'eastmoney',
    })
    client = web_app.app.test_client()

    response = client.post('/api/load-data', json={'symbol': '600000'})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['data_info']['end_date'].startswith('2026-08-07')
    assert payload['data_info']['data_source'] == 'eastmoney'


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


def test_rankings_support_local_backend_and_return_clickable_details(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, 'PREDICTION_RESULTS_DIR', str(tmp_path))
    dates = pd.bdate_range(end='2026-08-07', periods=120)
    future_dates = pd.Series(pd.bdate_range('2026-08-10', periods=10), name='timestamps')

    def make_record(symbol):
        context = pd.DataFrame({
            'timestamps': dates,
            'open': np.full(120, 10.0), 'high': np.full(120, 11.0),
            'low': np.full(120, 9.0), 'close': np.full(120, 10.0),
            'volume': np.full(120, 1_000_000.0),
            'amount': np.full(120, 10_000_000.0),
        })
        return {
            'symbol': symbol, 'context': context, 'future_dates': future_dates,
            'stock_name': symbol, 'size_bucket': 5, 'size_percentile': 0.55,
            'data_source': 'eastmoney', 'in_local_panel': False,
        }

    records = [make_record('sh.600000'), make_record('sz.000001')]
    monkeypatch.setattr(web_app, 'portfolio_ranking_batch', lambda *args, **kwargs: {
        'as_of_date': pd.Timestamp('2026-08-07'), 'records': records,
    })

    def local_result(*args, **kwargs):
        predictions = [{
            'timestamp': timestamp.isoformat(), 'open': 10.0, 'high': 11.0,
            'low': 9.0, 'close': 10.5, 'volume': 1_000_000.0,
            'amount': 10_000_000.0, 'close_p10': 9.5,
            'close_p50': 10.5, 'close_p90': 11.5,
        } for timestamp in future_dates]
        pred_df = pd.DataFrame(predictions).set_index(pd.DatetimeIndex(future_dates))
        interval = pred_df[['close_p10', 'close_p50', 'close_p90']]
        return predictions, pred_df, interval, np.full((10, 10), 10.5), 'cpu'

    monkeypatch.setattr(web_app, 'local_inference', local_result)
    monkeypatch.setattr(
        web_app, 'call_remote_inference',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Modal must not be called')),
    )
    monkeypatch.setattr(web_app, 'create_operational_chart', lambda *args: '{"data":[],"layout":{}}')

    response = web_app.app.test_client().post('/api/a-share/rankings', json={
        'symbols': ['600000', '000001'], 'backend': 'local', 'sample_count': 10,
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['backend'] == 'local'
    assert payload['model_device'] == 'cpu'
    assert len(payload['rankings']) == 2
    assert len(payload['rankings'][0]['prediction_results']) == 10
    assert payload['rankings'][0]['chart'].startswith('{')
    assert len(list(tmp_path.glob('*.json'))) == 2


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
