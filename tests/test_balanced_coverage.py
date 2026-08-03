import numpy as np

from finetune.dataset import build_balanced_coverage_order


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
