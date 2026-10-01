"""Deterministic, resumable round-robin sampling across row groups."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class MixedCursor:
    group_order: list[int]
    group_offsets: list[int]
    next_group_index: int = 0

    def as_dict(self) -> dict:
        return {
            "group_order": list(self.group_order),
            "group_offsets": list(self.group_offsets),
            "next_group_index": int(self.next_group_index),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "MixedCursor":
        return cls(
            group_order=[int(x) for x in state["group_order"]],
            group_offsets=[int(x) for x in state["group_offsets"]],
            next_group_index=int(state["next_group_index"]),
        )


def cursor_from_sizes(group_order: Sequence[int], sizes: Sequence[int]) -> MixedCursor:
    if len(group_order) != len(sizes):
        raise ValueError("group order and sizes must have equal length")
    return MixedCursor(list(group_order), [0] * len(sizes), 0)


def take_round_robin(
    groups: Sequence[Sequence[object]], cursor: MixedCursor, count: int
) -> tuple[list[object], MixedCursor]:
    """Take rows across groups without repetition and return a resume cursor."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if len(groups) != len(cursor.group_offsets):
        raise ValueError("cursor does not match groups")
    output: list[object] = []
    offsets = list(cursor.group_offsets)
    index = cursor.next_group_index % max(len(groups), 1)
    while len(output) < count:
        if not any(offsets[i] < len(rows) for i, rows in enumerate(groups)):
            raise RuntimeError("mixed sampler exhausted before requested count")
        for _ in range(len(groups)):
            group_id = index % len(groups)
            index += 1
            if offsets[group_id] >= len(groups[group_id]):
                continue
            output.append(groups[group_id][offsets[group_id]])
            offsets[group_id] += 1
            if len(output) == count:
                break
    return output, MixedCursor(list(cursor.group_order), offsets, index % len(groups))
