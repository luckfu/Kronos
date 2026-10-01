from finetune.kairos_mixed_sampler import MixedCursor, cursor_from_sizes, take_round_robin


def test_round_robin_mixes_groups_and_covers_exactly_once():
    groups = [["a0", "a1"], ["b0", "b1", "b2"], ["c0", "c1"]]
    cursor = cursor_from_sizes([0, 1, 2], [len(rows) for rows in groups])
    first, cursor = take_round_robin(groups, cursor, 5)
    second, _ = take_round_robin(groups, cursor, 2)
    assert first == ["a0", "b0", "c0", "a1", "b1"]
    assert second == ["c1", "b2"]
    assert sorted(first + second) == sorted(sum(groups, []))


def test_cursor_round_trip_preserves_resume_position():
    groups = [[1, 2], [3, 4, 5]]
    cursor = cursor_from_sizes([0, 1], [2, 3])
    _, cursor = take_round_robin(groups, cursor, 2)
    restored = MixedCursor.from_dict(cursor.as_dict())
    left, _ = take_round_robin(groups, cursor, 3)
    right, _ = take_round_robin(groups, restored, 3)
    assert left == right
