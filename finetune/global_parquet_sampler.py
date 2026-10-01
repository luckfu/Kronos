"""Deterministic global sample permutation for resumable Parquet training."""

from typing import Any

import numpy as np

SAMPLER_VERSION = 'global_sample_permutation_v1'


class GlobalParquetSampler:
    """Materialize sample rows and consume one deterministic global permutation."""

    def __init__(self, parquet: Any, columns: list[str], seed: int,
                 cursor: int = 0) -> None:
        self.rows = parquet.read(columns=columns).to_pylist()
        self.total = len(self.rows)
        self.permutation = np.random.default_rng(seed).permutation(self.total)
        self.cursor = int(cursor)
        if not 0 <= self.cursor <= self.total:
            raise RuntimeError('global sampler cursor out of range')

    @classmethod
    def from_state(cls, parquet: Any, columns: list[str], seed: int,
                   state: dict) -> 'GlobalParquetSampler':
        sampler = cls(parquet, columns, seed, int(state['cursor']))
        if int(state.get('total', -1)) != sampler.total:
            raise RuntimeError('global sampler sample count mismatch')
        return sampler

    def state_dict(self) -> dict[str, Any]:
        return {'sampler_version': SAMPLER_VERSION, 'cursor': self.cursor,
                'total': self.total}

    def take(self, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if count < 0 or self.cursor + count > self.total:
            raise RuntimeError('global sampler exhausted or invalid request')
        positions = self.permutation[self.cursor:self.cursor + count]
        rows = [self.rows[int(position)] for position in positions]
        self.cursor += count
        return rows, {'cursor': self.cursor}
