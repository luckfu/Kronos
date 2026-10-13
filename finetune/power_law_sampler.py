"""Deterministic capped power-weighted mixture over eligible training windows."""

import hashlib

import numpy as np


SAMPLER_VERSION = 'capped_power_mixture_v1'


def window_amplitudes(closes, starts, lookback=120, horizon=10):
    closes = np.asarray(closes, dtype=np.float64)
    anchors = np.asarray(starts, dtype=np.int64) + lookback - 1
    baseline = closes[anchors]
    amplitudes = np.zeros(len(anchors), dtype=np.float64)
    for offset in range(1, horizon + 1):
        amplitudes = np.maximum(
            amplitudes, np.abs(closes[anchors + offset] / baseline - 1.0)
        )
    if not np.isfinite(amplitudes).all() or np.any(baseline <= 0):
        raise ValueError('Invalid closes in training amplitude index')
    return amplitudes


class PowerLawPlan:
    def __init__(self, amplitudes, samples_per_segment, segments, seed,
                 ratio=0.6, reference=0.05, exponent=1.0, cap=4.0):
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        if (amplitudes.ndim != 1 or not len(amplitudes)
                or not np.isfinite(amplitudes).all() or np.any(amplitudes < 0)):
            raise ValueError('Invalid training amplitudes')
        if not 0 <= ratio <= 1 or reference <= 0 or exponent <= 0 or cap < 1:
            raise ValueError('Invalid power mixture parameters')
        if samples_per_segment <= 0 or segments <= 0:
            raise ValueError('Sampling plan must have a positive budget')
        weights = np.minimum(
            np.maximum(amplitudes / reference, 1.0) ** exponent, cap
        )
        probabilities = (1.0 - ratio) / len(weights) + ratio * weights / weights.sum()
        self.order = np.random.default_rng(seed).choice(
            len(weights), size=samples_per_segment * segments,
            replace=True, p=probabilities,
        )
        self.samples_per_segment = int(samples_per_segment)
        self.segments = int(segments)
        self.sha256 = hashlib.sha256(self.order.astype('<i8').tobytes()).hexdigest()
        first_seen = np.full(len(weights), len(self.order), dtype=np.int64)
        np.minimum.at(first_seen, self.order, np.arange(len(self.order)))
        first_segments = first_seen[first_seen < len(self.order)] // samples_per_segment
        self.unique_counts = np.cumsum(np.bincount(first_segments, minlength=segments))
        selected = amplitudes[self.order]
        self.report = {
            'sampler_version': SAMPLER_VERSION, 'plan_sha256': self.sha256,
            'seed': int(seed), 'weighted_ratio': ratio, 'reference': reference,
            'exponent': exponent, 'weight_cap': cap, 'replacement': True,
            'eligible_windows': len(weights), 'sample_draws': len(self.order),
            'unique_windows': int(self.unique_counts[-1]),
            'natural_amplitude_mean': float(amplitudes.mean()),
            'sampled_amplitude_mean': float(selected.mean()),
            'natural_amplitude_ge_5pct': float(np.mean(amplitudes >= 0.05)),
            'sampled_amplitude_ge_5pct': float(np.mean(selected >= 0.05)),
        }

    def take_segment(self, segment):
        if not 0 <= segment < self.segments:
            raise ValueError('Sampling segment outside frozen plan')
        start = segment * self.samples_per_segment
        return self.order[start:start + self.samples_per_segment]

    def unique_through(self, completed_segments):
        if completed_segments <= 0:
            return 0
        return int(self.unique_counts[min(completed_segments, self.segments) - 1])
