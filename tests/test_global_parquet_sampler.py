import numpy as np
import pytest

from finetune.global_parquet_sampler import GlobalParquetSampler


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return list(self.rows)


class FakeParquet:
    def __init__(self, rows):
        self.rows = rows

    def read(self, columns):
        return FakeTable([{column: row[column] for column in columns} for row in self.rows])


def test_global_sampler_is_exactly_once_and_seeded():
    rows = [{'id': index} for index in range(101)]
    first = GlobalParquetSampler(FakeParquet(rows), ['id'], seed=20260927)
    output = []
    while first.cursor < first.total:
        output.extend(first.take(13 if first.total - first.cursor >= 13 else first.total - first.cursor)[0])
    assert sorted(row['id'] for row in output) == list(range(101))
    second = GlobalParquetSampler(FakeParquet(rows), ['id'], seed=20260927)
    third = GlobalParquetSampler(FakeParquet(rows), ['id'], seed=20260928)
    assert np.array_equal(second.permutation, GlobalParquetSampler(FakeParquet(rows), ['id'], 20260927).permutation)
    assert not np.array_equal(second.permutation, third.permutation)


def test_resume_matches_uninterrupted_sequence():
    rows = [{'id': index} for index in range(37)]
    uninterrupted = GlobalParquetSampler(FakeParquet(rows), ['id'], seed=7)
    expected = uninterrupted.take(37)[0]
    interrupted = GlobalParquetSampler(FakeParquet(rows), ['id'], seed=7)
    prefix = interrupted.take(11)[0]
    resumed = GlobalParquetSampler.from_state(FakeParquet(rows), ['id'], 7, interrupted.state_dict())
    assert prefix + resumed.take(26)[0] == expected


def test_exhaustion_and_tampered_state_are_rejected():
    rows = [{'id': index} for index in range(3)]
    sampler = GlobalParquetSampler(FakeParquet(rows), ['id'], seed=1)
    sampler.take(3)
    with pytest.raises(RuntimeError):
        sampler.take(1)
    with pytest.raises(RuntimeError):
        GlobalParquetSampler.from_state(FakeParquet(rows), ['id'], 1, {'cursor': 0, 'total': 4})
