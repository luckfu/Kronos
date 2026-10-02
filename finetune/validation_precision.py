"""Precision-safe validation loss aggregation.

Validation losses are computed from float32 logits, accumulated as float64
sums weighted by sample count, and reduced across ranks without routing the
sums through a low-precision device float.  The TPU trainer no longer sets
``XLA_USE_BF16`` (fp32 master weights + bf16 autocast), but the cross-rank
reduction still uses an exact integer fixed-point encoding: it is exact
regardless of device float precision, TPU has no native float64, and it
survives ``XLA_USE_32BIT_LONG``.

``torch`` is imported lazily so the pure-python helpers stay testable without
an accelerator stack.
"""

from __future__ import annotations

import math

FIXED_POINT_FRACTION_BITS = 20
FIXED_POINT_SCALE = 1 << FIXED_POINT_FRACTION_BITS
VALIDATION_DECIMALS = 8


def to_float32(value):
    """Cast a tensor (or list/tuple/dict of tensors) to float32 for losses."""
    if isinstance(value, dict):
        return {key: to_float32(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_float32(item) for item in value)
    if hasattr(value, 'is_floating_point') and value.is_floating_point():
        return value.float()
    return value


def encode_fixed_point(values):
    """Split floats into exact integer (whole, fraction) parts.

    Both parts stay small enough for int32 even after summing eight ranks,
    so the encoding survives ``XLA_USE_32BIT_LONG`` as well as bf16 floats.
    """
    whole, fraction = [], []
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'Cannot reduce non-finite validation sum: {value}')
        integer = math.floor(value)
        remainder = int(round((value - integer) * FIXED_POINT_SCALE))
        if remainder >= FIXED_POINT_SCALE:
            integer += 1
            remainder -= FIXED_POINT_SCALE
        whole.append(int(integer))
        fraction.append(remainder)
    return whole, fraction


def decode_fixed_point(whole, fraction):
    return [
        float(integer) + float(remainder) / FIXED_POINT_SCALE
        for integer, remainder in zip(whole, fraction)
    ]


def weighted_means(keys, totals, samples):
    """Global mean = sum(loss_i * n_i) / sum(n_i), computed in float64."""
    divisor = max(1, int(samples))
    return {key: float(total) / divisor for key, total in zip(keys, totals)}


class ValidationLossAccumulator:
    """Accumulate sample-weighted fp32 losses into float64 sums.

    On CUDA/CPU the sums stay on the device in float64 (no per-batch host
    sync).  XLA (bf16 floats under XLA_USE_BF16) and MPS (no float64) copy
    one stacked fp32 vector per batch to the host and sum it in Python
    floats, replacing the previous one ``.item()`` sync per loss term.
    """

    def __init__(self, keys, device, host_accumulate=None):
        self.keys = tuple(keys)
        self.index = {key: position for position, key in enumerate(self.keys)}
        self.device = device
        if host_accumulate is None:
            host_accumulate = getattr(device, 'type', 'cpu') in {'xla', 'mps'}
        self.host_accumulate = bool(host_accumulate)
        self._host = [0.0] * len(self.keys)
        self._device_sums = None

    def add(self, values, weight):
        import torch

        weight = int(weight)
        if weight <= 0 or not values:
            return
        names = list(values)
        stacked = torch.stack([
            torch.as_tensor(values[name]).detach().reshape(()).float()
            for name in names
        ])
        if self.host_accumulate:
            for name, value in zip(names, stacked.cpu().tolist()):
                self._host[self.index[name]] += float(value) * weight
            return
        if self._device_sums is None:
            self._device_sums = torch.zeros(
                len(self.keys), dtype=torch.float64, device=stacked.device
            )
        positions = torch.tensor(
            [self.index[name] for name in names], device=stacked.device
        )
        self._device_sums.index_add_(0, positions, stacked.double() * weight)

    def totals(self):
        totals = list(self._host)
        if self._device_sums is not None:
            for position, value in enumerate(self._device_sums.cpu().tolist()):
                totals[position] += float(value)
        return totals


def all_reduce_validation_sums(totals, counts, device, xm=None, xla_world_size=1):
    """Sum float64 loss totals and integer counts across ranks.

    Returns ``(totals, counts)`` as Python float/int lists.  torch.distributed
    uses a float64 all-reduce; multi-worker XLA reduces an exact int64
    fixed-point encoding with ``xm.all_reduce`` (float tensors would be bf16).
    """
    import torch
    import torch.distributed as dist

    totals = [float(value) for value in totals]
    counts = [int(value) for value in counts]
    if dist.is_available() and dist.is_initialized():
        reduce_device = device if getattr(device, 'type', 'cpu') != 'mps' else 'cpu'
        total_tensor = torch.tensor(totals, dtype=torch.float64, device=reduce_device)
        count_tensor = torch.tensor(counts, dtype=torch.long, device=reduce_device)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        return (
            [float(value) for value in total_tensor.cpu().tolist()],
            [int(value) for value in count_tensor.cpu().tolist()],
        )
    if (
        getattr(device, 'type', 'cpu') == 'xla'
        and xm is not None
        and int(xla_world_size) > 1
    ):
        whole, fraction = encode_fixed_point(totals)
        packed = torch.tensor(whole + fraction + counts, dtype=torch.long, device=device)
        packed = xm.all_reduce(xm.REDUCE_SUM, packed)
        values = [int(value) for value in packed.cpu().tolist()]
        size = len(totals)
        return (
            decode_fixed_point(values[:size], values[size:2 * size]),
            values[2 * size:],
        )
    return totals, counts
