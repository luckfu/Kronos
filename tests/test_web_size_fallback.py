import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import webui.app as web_app


def test_frontend_model_uses_beta_v1_2_best_checkpoint():
    config = web_app.AVAILABLE_MODELS[web_app.KRONOS_MODEL_KEY]

    assert config['default_lookback'] == 120
    assert 'beta_v1.2' in config['model_id']
    assert config['checkpoint'] == 'Best@871'
    assert config['num_sectors'] == 86
    assert config['num_size_buckets'] == 0
    assert config['use_size_percentile'] is True


def test_remote_context_maps_market_cap_to_reference(monkeypatch):
    monkeypatch.setattr(web_app, 'size_reference_cache', {
        'date': pd.Timestamp('2026-07-31'),
        'market_caps': np.asarray([10.0, 20.0, 30.0, 40.0]),
        'count': 4,
    })
    context = pd.DataFrame({
        'timestamps': pd.to_datetime(['2026-07-30', '2026-07-31']),
        'amount': [10.0, 20.0],
        'turn': [100.0, 100.0],
    })

    result = web_app.size_condition_from_remote_context(context)

    assert result['estimated_market_cap'] == 20.0
    assert result['size_percentile'] == 0.5
    assert result['size_reference_date'] == pd.Timestamp('2026-07-31')


def test_remote_context_uses_latest_non_suspended_turnover(monkeypatch):
    monkeypatch.setattr(web_app, 'size_reference_cache', {
        'date': pd.Timestamp('2026-07-31'),
        'market_caps': np.asarray([10.0, 20.0, 30.0, 40.0]),
        'count': 4,
    })
    context = pd.DataFrame({
        'timestamps': pd.to_datetime(['2026-07-30', '2026-07-31']),
        'amount': [20.0, 0.0],
        'turn': [100.0, 0.0],
    })

    result = web_app.size_condition_from_remote_context(context)

    assert result['estimated_market_cap'] == 20.0
    assert result['size_percentile_asof'] == pd.Timestamp('2026-07-30')


def test_local_context_requires_continuous_percentile():
    with_percentile = pd.DataFrame({
        'timestamps': pd.to_datetime(['2026-07-31']),
        'size_bucket': [3],
        'size_percentile': [0.373],
    })
    without_percentile = with_percentile.drop(columns='size_percentile')

    assert web_app.size_condition_from_local_frame(with_percentile)['size_percentile'] == 0.373
    assert web_app.size_condition_from_local_frame(without_percentile) is None


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


def test_sector_vocabulary_and_symbol_mapping_are_loaded_separately(monkeypatch, tmp_path):
    vocabulary = json.loads(
        Path(web_app.A_SHARE_SECTOR_VOCABULARY_PATH).read_text(encoding='utf-8')
    )
    mapping_path = tmp_path / 'symbol_sector_map.json'
    mapping_path.write_text(json.dumps({
        'schema_version': 1,
        'vocabulary_id': vocabulary['vocabulary_id'],
        'reference_date': '2026-08-28',
        'symbols': {
            'sh.600000': {
                'sector_id': 63,
                'sector_label': 'J66货币金融服务',
                'reference_date': '2026-08-28',
            },
        },
    }), encoding='utf-8')
    monkeypatch.setattr(web_app, 'sector_reference_cache', None)
    monkeypatch.setattr(web_app, 'A_SHARE_SYMBOL_SECTOR_MAP_PATH', str(mapping_path))

    result = web_app.load_sector_reference()

    assert result['vocabulary_id'] == 'beta-v1.2-csrc-86-v1'
    assert result['labels'][63] == 'J66货币金融服务'
    assert result['symbols']['sh.600000']['sector_id'] == 63
    assert result['date'] == pd.Timestamp('2026-08-28')


def test_sector_mapping_rejects_id_that_disagrees_with_vocabulary(monkeypatch, tmp_path):
    vocabulary = json.loads(
        Path(web_app.A_SHARE_SECTOR_VOCABULARY_PATH).read_text(encoding='utf-8')
    )
    mapping_path = tmp_path / 'symbol_sector_map.json'
    mapping_path.write_text(json.dumps({
        'vocabulary_id': vocabulary['vocabulary_id'],
        'reference_date': '2026-08-28',
        'symbols': {
            'sh.600000': {
                'sector_id': 1,
                'sector_label': 'J66货币金融服务',
                'reference_date': '2026-08-28',
            },
        },
    }), encoding='utf-8')
    monkeypatch.setattr(web_app, 'sector_reference_cache', None)
    monkeypatch.setattr(web_app, 'A_SHARE_SYMBOL_SECTOR_MAP_PATH', str(mapping_path))

    with pytest.raises(ValueError, match='invalid ID'):
        web_app.load_sector_reference()


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
    monkeypatch.setattr(web_app, 'sector_reference_cache', {
        'date': pd.Timestamp('2026-07-31'),
        'vocabulary_id': 'beta-v1.2-csrc-86-v1',
        'labels': ('A01农业',),
        'label_to_id': {'A01农业': 0},
        'unknown_sector_id': 86,
        'symbols': {
            'sh.600000': {'sector_id': 0, 'sector_label': 'A01农业', 'reference_date': '2026-07-31'},
            'sz.000001': {'sector_id': 63, 'sector_label': 'J66货币金融服务', 'reference_date': '2026-07-31'},
        },
    })
    client = web_app.app.test_client()

    response = client.get('/api/a-share/symbols')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['count'] == 2
    assert payload['mapping_count'] == 2
    assert payload['mapping_reference_date'] == '2026-07-31'
    assert payload['sector_vocabulary_id'] == 'beta-v1.2-csrc-86-v1'
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


def test_history_ranking_exposes_per_stock_delete_action():
    response = web_app.app.test_client().get('/')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'id="delete-history-group"' not in page
    assert 'id="batch-action-heading"' in page
    assert 'aria-label="从截面删除个股"' in page
    assert 'signal_date: signalDate' in page
    assert 'records,' in page
    assert 'function deleteHistoryStock(record)' in page
    assert '/symbols/${encodeURIComponent(record.symbol)}' in page
    assert "style.display = fromHistory ? 'table-cell' : 'none'" in page


def test_history_group_can_be_rerun_with_latest_market_data():
    response = web_app.app.test_client().get('/')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'id="rerun-history-group"' in page
    assert '↻ 再测一次' in page
    assert 'function rerunHistoryGroup()' in page
    assert '(group?.rankings || []).map(item => item.symbol)' in page
    assert 'await runPrediction(symbols[0])' in page
    assert 'await runBatchPrediction(symbols)' in page


def test_ranking_table_headers_support_sorting():
    response = web_app.app.test_client().get('/')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert page.count('class="batch-sort-button"') == 7
    assert 'data-sort-key="predicted_return_p50"' in page
    assert "batchSort: {key: 'predicted_return_p50', direction: 'desc'}" in page
    assert 'function sortedBatchRankings(rankings)' in page
    assert 'function setBatchSort(key)' in page
    assert "heading.setAttribute('aria-sort', ascending ? 'ascending' : 'descending')" in page
    assert 'renderBatchRankingTable(sortedBatchRankings(result.rankings), fromHistory)' in page


def test_prediction_detail_shows_ten_day_high_and_low():
    response = web_app.app.test_client().get('/')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert '10 日预测最高 / 最低' in page
    assert 'id="metric-high-low"' in page
    assert 'Math.max(...predictedHighs)' in page
    assert 'Math.min(...predictedLows)' in page
    assert 'grid-template-columns: repeat(6, minmax(0, 1fr))' in page

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


def test_history_summary_uses_market_data_cutoff_as_signal_date():
    summary = web_app.prediction_record_summary({
        'record_id': 'record-123', 'symbol': 'sz.000001',
        'latest_data_date': '2026-08-28T00:00:00',
        'created_at': '2026-08-30T12:00:00+00:00',
        'prediction_results': [], 'positive_path_rate': 0.6,
        'timing_signal': {'key': 'bullish', 'label': '偏多'},
        'sector_id': 63, 'sector_label': 'J66货币金融服务',
        'size_percentile': 0.99,
    })

    assert summary['signal_date'] == '2026-08-28'
    assert summary['sector_id'] == 63
    assert summary['size_percentile'] == 0.99
    assert summary['positive_path_rate'] == 0.6
    assert summary['timing_signal']['label'] == '偏多'


def test_homepage_uses_one_sampling_contract_for_all_entry_points():
    response = web_app.app.test_client().get('/')
    page = response.get_data(as_text=True)

    assert 'const PRODUCTION_SAMPLING = Object.freeze({' in page
    assert page.count('...PRODUCTION_SAMPLING') == 2
    assert 'temperature: 0.65' in page
    assert 'top_p: 0.8' in page
    assert 'top_p: 1.0' not in page
    assert 'history-single-tab' not in page
    assert 'history-ranking-tab' not in page
    assert '同一天分开提交或批量提交的股票进入同一个截面' in page


def test_model_sampling_defaults_match_frontend_contract():
    config = web_app.AVAILABLE_MODELS[web_app.KRONOS_MODEL_KEY]

    assert config['default_temperature'] == 0.65
    assert config['default_top_p'] == 0.8


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


def test_prediction_history_can_delete_a_complete_signal_date(monkeypatch, tmp_path):
    grouped_first = tmp_path / 'record-001.json'
    grouped_second = tmp_path / 'record-002.json'
    unrelated = tmp_path / 'record-003.json'
    grouped_first.write_text(json.dumps({
        'record_id': 'record-001', 'latest_data_date': '2026-08-20T00:00:00',
    }), encoding='utf-8')
    grouped_second.write_text(json.dumps({
        'record_id': 'record-002', 'latest_data_date': '2026-08-20T00:00:00',
    }), encoding='utf-8')
    unrelated.write_text(json.dumps({
        'record_id': 'record-003', 'latest_data_date': '2026-08-21T00:00:00',
    }), encoding='utf-8')
    monkeypatch.setattr(web_app, 'PREDICTION_RESULTS_DIR', str(tmp_path))

    response = web_app.app.test_client().delete(
        '/api/prediction-history/dates/2026-08-20'
    )

    assert response.status_code == 200
    assert response.get_json()['records_deleted'] == 2
    assert response.get_json()['signal_date'] == '2026-08-20'
    assert not grouped_first.exists()
    assert not grouped_second.exists()
    assert unrelated.exists()


def test_prediction_history_can_delete_one_stock_from_signal_date(monkeypatch, tmp_path):
    duplicate_new = tmp_path / 'record-001.json'
    duplicate_old = tmp_path / 'record-002.json'
    same_date_other_stock = tmp_path / 'record-003.json'
    same_stock_other_date = tmp_path / 'record-004.json'
    fixtures = {
        duplicate_new: ('2026-08-20T00:00:00', 'sh.600000'),
        duplicate_old: ('2026-08-20T00:00:00', 'sh.600000'),
        same_date_other_stock: ('2026-08-20T00:00:00', 'sz.000001'),
        same_stock_other_date: ('2026-08-21T00:00:00', 'sh.600000'),
    }
    for path, (latest_data_date, symbol) in fixtures.items():
        path.write_text(json.dumps({
            'record_id': path.stem,
            'latest_data_date': latest_data_date,
            'symbol': symbol,
        }), encoding='utf-8')
    monkeypatch.setattr(web_app, 'PREDICTION_RESULTS_DIR', str(tmp_path))

    response = web_app.app.test_client().delete(
        '/api/prediction-history/dates/2026-08-20/symbols/600000'
    )

    assert response.status_code == 200
    assert response.get_json()['records_deleted'] == 2
    assert response.get_json()['symbol'] == 'sh.600000'
    assert not duplicate_new.exists()
    assert not duplicate_old.exists()
    assert same_date_other_stock.exists()
    assert same_stock_other_date.exists()


def test_prediction_history_deduplicates_symbol_within_signal_date(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, 'PREDICTION_RESULTS_DIR', str(tmp_path))
    monkeypatch.setattr(web_app, 'query_remote_stock_name', lambda symbol: None)
    records = [
        {
            'record_id': '20260830_new', 'created_at': '2026-08-30T12:00:00+00:00',
            'latest_data_date': '2026-08-28T00:00:00', 'symbol': 'sh.600000',
            'name': '浦发银行', 'latest_close': 12.0, 'predicted_return_p50': 0.02,
            'prediction_results': [], 'chart': '{}',
        },
        {
            'record_id': '20260830_old', 'created_at': '2026-08-30T10:00:00+00:00',
            'latest_data_date': '2026-08-28T00:00:00', 'symbol': 'sh.600000',
            'name': '浦发银行', 'latest_close': 11.0, 'predicted_return_p50': 0.01,
            'prediction_results': [], 'chart': '{}',
        },
        {
            'record_id': '20260830_other', 'created_at': '2026-08-30T11:00:00+00:00',
            'latest_data_date': '2026-08-28T00:00:00', 'symbol': 'sz.000001',
            'name': '平安银行', 'latest_close': 10.0, 'predicted_return_p50': -0.01,
            'prediction_results': [], 'chart': '{}',
        },
    ]
    for record in records:
        (tmp_path / f'{record["record_id"]}.json').write_text(
            json.dumps(record), encoding='utf-8'
        )

    response = web_app.app.test_client().get('/api/prediction-history')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['total'] == 2
    assert {record['symbol'] for record in payload['records']} == {
        'sh.600000', 'sz.000001',
    }
    latest = next(record for record in payload['records'] if record['symbol'] == 'sh.600000')
    assert latest['record_id'] == '20260830_new'
    assert latest['signal_date'] == '2026-08-28'


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
        'size_percentile': 0.55,
        'size_reference_date': pd.Timestamp('2026-07-31'),
        'sector_id': 63,
        'sector_label': 'J66货币金融服务',
        'sector_reference_date': pd.Timestamp('2026-07-31'),
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
            'size_percentile': 0.55,
            'size_source': 'amount_turnover_proxy_vs_full_market_cross_section',
            'sector_id': 86,
            'sector_label': 'unknown',
            'sector_reference_date': None,
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
            'stock_name': symbol, 'size_percentile': 0.55,
            'size_reference_date': pd.Timestamp('2026-07-31'),
            'sector_id': 63, 'sector_label': 'J66货币金融服务',
            'sector_reference_date': pd.Timestamp('2026-07-31'),
            'data_source': 'eastmoney', 'in_local_panel': False,
        }

    records = [make_record('sh.600000'), make_record('sz.000001')]
    monkeypatch.setattr(web_app, 'portfolio_ranking_batch', lambda *args, **kwargs: {
        'as_of_date': pd.Timestamp('2026-08-07'), 'records': records,
    })

    sampling_calls = []

    def local_result(*args, **kwargs):
        sampling_calls.append((args[3], args[4]))
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
        'temperature': 9.0, 'top_p': 1.0,
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['backend'] == 'local'
    assert payload['model_device'] == 'cpu'
    assert len(payload['rankings']) == 2
    assert len(payload['rankings'][0]['prediction_results']) == 10
    assert payload['rankings'][0]['chart'].startswith('{')
    assert sampling_calls == [(9.0, 1.0), (9.0, 1.0)]
    assert len(list(tmp_path.glob('*.json'))) == 2


def test_single_prediction_keeps_gateway_sampling_parameters_configurable(monkeypatch):
    dates = pd.bdate_range(end='2026-08-28', periods=120)
    future_dates = pd.Series(pd.bdate_range('2026-08-31', periods=10), name='timestamps')
    context = pd.DataFrame({
        'timestamps': dates,
        'open': np.full(120, 10.0), 'high': np.full(120, 11.0),
        'low': np.full(120, 9.0), 'close': np.full(120, 10.0),
        'volume': np.full(120, 1_000_000.0),
        'amount': np.full(120, 10_000_000.0),
    })
    monkeypatch.setattr(web_app, 'latest_prediction_inputs', lambda *args, **kwargs: {
        'context': context,
        'future_dates': future_dates,
        'stock_name': '浦发银行',
        'size_percentile': 0.99,
        'size_percentile_asof': pd.Timestamp('2026-08-28'),
        'size_reference_date': pd.Timestamp('2026-07-31'),
        'size_source': 'test',
        'estimated_market_cap': 1_000_000_000.0,
        'sector_id': 63,
        'sector_label': 'J66货币金融服务',
        'sector_reference_date': pd.Timestamp('2026-08-24'),
        'sector_source': 'test',
        'in_local_panel': False,
        'data_source': 'eastmoney',
        'calendar_source': 'test',
        'refresh_error': None,
    })
    captured = {}

    def remote_inference(payload, **kwargs):
        captured.update(payload)
        predictions = [{
            'timestamp': timestamp.isoformat(),
            'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.5,
            'volume': 1_000_000.0, 'amount': 10_000_000.0,
            'close_p10': 9.5, 'close_p50': 10.5, 'close_p90': 11.5,
        } for timestamp in future_dates]
        return {
            'predictions': predictions,
            'samples': {'close': np.full((50, 10), 10.5).tolist()},
            'meta': {'model_device': 'cuda:0'},
        }

    monkeypatch.setattr(web_app, 'call_remote_inference', remote_inference)
    monkeypatch.setattr(web_app, 'create_operational_chart', lambda *args: '{"data":[],"layout":{}}')
    monkeypatch.setattr(web_app, 'save_prediction_record', lambda record: {'record_id': 'fixed'})

    response = web_app.app.test_client().post('/api/predict', json={
        'symbol': '600000', 'backend': 'remote', 'sample_count': 50,
        'temperature': 9.0, 'top_p': 1.0,
    })

    assert response.status_code == 200
    assert captured['temperature'] == 9.0
    assert captured['top_p'] == 1.0


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
