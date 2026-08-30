import os
import pickle

import numpy as np
import pandas as pd

from finetune.dataset import (
    QlibDataset,
    balanced_stratum_quotas,
    build_balanced_coverage_order,
    select_stratified_replay,
)
from finetune.prepare_a_share_v4_runtime import crop_for_signal_range


def test_balanced_coverage_uses_every_position_once():
    bucket_ids = np.repeat(np.arange(4, dtype=np.uint8), [11, 9, 7, 5])
    order = build_balanced_coverage_order(bucket_ids, segment_size=8, seed=100)

    assert sorted(order.tolist()) == list(range(len(bucket_ids)))
    first_segment = bucket_ids[order[:8]]
    counts = np.bincount(first_segment, minlength=4)
    assert counts.tolist() == [2, 2, 2, 2]


def test_balanced_coverage_finishes_small_buckets_without_duplicates():
    bucket_ids = np.repeat(np.arange(3, dtype=np.uint8), [3, 7, 11])
    order = build_balanced_coverage_order(bucket_ids, segment_size=6, seed=5)

    assert len(np.unique(order)) == len(bucket_ids)
    assert set(order.tolist()) == set(range(len(bucket_ids)))


def test_unknown_bucket_positions_are_covered_last():
    bucket_ids = np.asarray([0, 1, 0, 1, 255, 255], dtype=np.uint8)
    order = build_balanced_coverage_order(bucket_ids, segment_size=4, seed=7)

    assert set(bucket_ids[order[:4]].tolist()) == {0, 1}
    assert bucket_ids[order[-2:]].tolist() == [255, 255]


def test_stratified_replay_is_deterministic_and_balanced():
    strata = np.repeat(np.asarray([2015 * 256, 2015 * 256 + 1, 2025 * 256]), [9, 20, 30])
    unique, quotas = balanced_stratum_quotas(strata, target=15)
    first = select_stratified_replay(strata, target=15, seed=7)
    second = select_stratified_replay(strata, target=15, seed=7)

    assert quotas.sum() == 15
    assert dict(zip(unique.tolist(), quotas.tolist())) == {
        2015 * 256: 5,
        2015 * 256 + 1: 5,
        2025 * 256: 5,
    }
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 15


def _panel(dates):
    values = np.arange(len(dates), dtype=np.float64) + 10.0
    return pd.DataFrame(
        {
            'open': values,
            'high': values + 1,
            'low': values - 1,
            'close': values + 0.25,
            'volume': values * 100,
            'amount': values * 1000,
            'size_bucket': np.full(len(dates), 3),
            'size_percentile': np.full(len(dates), 0.35),
        },
        index=pd.DatetimeIndex(dates, name='datetime'),
    )


def test_signal_filter_keeps_preperiod_rows_as_context(tmp_path, monkeypatch):
    history_dates = pd.bdate_range('2025-08-01', '2025-12-31')
    recent_dates = pd.bdate_range('2026-01-01', '2026-07-31')
    history_path = tmp_path / 'history.pkl'
    recent_path = tmp_path / 'recent.pkl'
    with history_path.open('wb') as handle:
        pickle.dump({'sh.600001': _panel(history_dates)}, handle)
    with recent_path.open('wb') as handle:
        pickle.dump({'sh.600001': _panel(recent_dates)}, handle)

    monkeypatch.setenv(
        'KRONOS_TRAIN_DATA_PATHS',
        os.pathsep.join([str(history_path), str(recent_path)]),
    )
    monkeypatch.setenv('KRONOS_TRAIN_SIGNAL_START', '2026-01-01')
    monkeypatch.setenv('KRONOS_TRAIN_SIGNAL_END', '2026-06-17')
    monkeypatch.setenv('KRONOS_TRAIN_SAMPLES_PER_SEGMENT', '0')
    monkeypatch.setenv('KRONOS_HISTORY_REPLAY_RATIO', '0')
    monkeypatch.setenv('KRONOS_USE_SECTOR_FEATURES', '0')

    dataset = QlibDataset('train')
    signal_dates = {
        dataset.timestamps_by_symbol[str(symbol)].iloc[
            start + dataset.config.lookback_window - 1
        ]
        for symbol, start in dataset.indices
    }

    assert min(signal_dates) == pd.Timestamp('2026-01-01')
    assert max(signal_dates) == pd.Timestamp('2026-06-17')
    assert dataset.selection_report['signal_days'] == len(signal_dates)
    assert dataset.selection_report['replay_samples'] == 0


def test_runtime_crop_keeps_context_and_future_targets():
    dates = pd.bdate_range('2025-08-01', '2026-07-31')
    panel = {'sh.600001': _panel(dates)}
    cropped = crop_for_signal_range(
        panel, signal_start='2026-01-01', signal_end='2026-06-17'
    )['sh.600001']

    first_signal = cropped.index.get_loc(pd.Timestamp('2026-01-01'))
    last_signal = cropped.index.get_loc(pd.Timestamp('2026-06-17'))
    assert first_signal >= 90
    assert len(cropped) - last_signal - 1 >= 11
