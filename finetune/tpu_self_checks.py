"""Fail-fast self checks for multi-core torch_xla training.

These checks exist because an earlier Kaggle TPU launcher (``pjrt.spawn_threads``)
left ``torch_xla.runtime.world_size() == 1``.  ``xm.optimizer_step`` then
silently skipped the gradient all-reduce, every core trained its own replica,
and ``XLA_USE_BF16=1`` rounded ~97% of AdamW updates away.  Every check below
raises ``RuntimeError`` so a broken topology or precision setup stops the run
in minutes instead of burning a full Kaggle session.

Collectives are injected (``all_reduce_sum`` / ``all_gather``) so the logic is
unit-testable without an accelerator.  All ranks must call each collective
check (they are collectives); only the log printing is rank-0 gated by callers.
"""

from __future__ import annotations

import json
import math
import os

TRUE_VALUES = {"1", "true", "yes", "on"}
BF16_ENV_FLAGS = ("XLA_USE_BF16", "XLA_DOWNCAST_BF16")
PROBE_ELEMENTS_PER_PARAMETER = 64
# A fresh fp32 parent is essentially never bf16-representable (measured 0.0%
# on Kronos-A-Share-Beta-V2-1); XLA_USE_BF16 storage makes it 100%.
MIN_NON_BF16_FRACTION = 0.5
# After one AdamW step every element with a non-zero update moves in fp32.
# Under bf16 storage with lr=3e-6 almost nothing moves (<1%).
MIN_CHANGED_FRACTION = 0.05


class SelfCheckError(RuntimeError):
    """Raised when a TPU topology / precision invariant is violated."""


def bf16_env_flags(environ=None):
    environ = os.environ if environ is None else environ
    return {
        name: environ.get(name)
        for name in BF16_ENV_FLAGS
        if str(environ.get(name, "")).strip().lower() in TRUE_VALUES
    }


def assert_no_bf16_storage_env(environ=None):
    """fp32 master weights require that torch_xla does not downcast storage."""
    flags = bf16_env_flags(environ)
    if flags:
        raise SelfCheckError(
            "fp32_master_weight_check_failed: "
            f"{flags} makes torch_xla store every fp32 tensor (weights and "
            "AdamW moments) as bfloat16. Unset it and use bf16 autocast."
        )


def check_world_size(runtime_world_size, expected_world_size, context=""):
    runtime_world_size = int(runtime_world_size)
    expected_world_size = int(expected_world_size)
    if runtime_world_size != expected_world_size:
        raise SelfCheckError(
            "xla_world_size_check_failed: torch_xla.runtime.world_size()="
            f"{runtime_world_size} but expected {expected_world_size} {context}. "
            "xm.optimizer_step would skip the gradient all-reduce."
        )
    return (
        f"xla_world_size_check_passed runtime.world_size={runtime_world_size} "
        f"expected={expected_world_size} {context}".strip()
    )


def check_collectives(ordinal, world_size, all_reduce_sum, all_gather):
    """Verify a real cross-replica SUM and GATHER (all ranks must call)."""
    total = float(all_reduce_sum(1.0))
    ordinal_sum = float(all_reduce_sum(float(ordinal)))
    gathered = [int(round(float(value))) for value in all_gather(float(ordinal))]
    expected_ordinal_sum = world_size * (world_size - 1) / 2
    ok = (
        math.isclose(total, float(world_size))
        and math.isclose(ordinal_sum, expected_ordinal_sum)
        and sorted(gathered) == list(range(world_size))
    )
    if not ok:
        raise SelfCheckError(
            "xla_collective_check_failed: "
            f"all_reduce(1)={total} (expected {world_size}), "
            f"all_reduce(ordinal)={ordinal_sum} (expected {expected_ordinal_sum}), "
            f"all_gather(ordinal)={gathered} (expected 0..{world_size - 1})"
        )
    return (
        f"xla_collective_check_passed all_reduce_sum={total:g} "
        f"ordinal_sum={ordinal_sum:g} all_gather={gathered}"
    )


def parameter_probe(parameters, elements_per_parameter=PROBE_ELEMENTS_PER_PARAMETER):
    """Concatenate the first elements of every parameter (fp32, 1-D)."""
    import torch

    pieces = [
        parameter.detach().reshape(-1)[:elements_per_parameter].float()
        for parameter in parameters
    ]
    if not pieces:
        raise SelfCheckError("parameter probe found no parameters")
    return torch.cat(pieces)


def gradient_probe(parameters, elements_per_parameter=PROBE_ELEMENTS_PER_PARAMETER):
    import torch

    pieces = []
    for parameter in parameters:
        flat = parameter.reshape(-1)[:elements_per_parameter]
        if parameter.grad is None:
            pieces.append(torch.zeros_like(flat, dtype=torch.float32))
        else:
            pieces.append(parameter.grad.detach().reshape(-1)[:elements_per_parameter].float())
    return torch.cat(pieces)


def non_bf16_representable_fraction(values):
    """Fraction of fp32 values whose low 16 mantissa bits are non-zero."""
    import torch

    values = values.detach().float().cpu().contiguous()
    low_bits = values.view(torch.int32) & 0xFFFF
    return float((low_bits != 0).float().mean().item())


def check_fp32_storage(probe_values, parameter_dtypes, allow_bf16_parent=False):
    """Startup check: parameters are fp32 and not silently bf16-rounded."""
    dtypes = sorted({str(dtype).removeprefix("torch.") for dtype in parameter_dtypes})
    if dtypes != ["float32"]:
        raise SelfCheckError(
            f"fp32_master_weight_check_failed: parameter dtypes {dtypes}, expected float32"
        )
    fraction = non_bf16_representable_fraction(probe_values)
    if fraction < MIN_NON_BF16_FRACTION and not allow_bf16_parent:
        raise SelfCheckError(
            "fp32_master_weight_check_failed: only "
            f"{fraction:.4f} of probed device weights carry fp32 precision "
            "(bf16-rounded storage). Set KRONOS_ALLOW_BF16_PARENT=1 only if the "
            "parent checkpoint itself is bf16."
        )
    return {
        "param_dtype": dtypes[0],
        "probe_non_bf16_fraction": round(fraction, 6),
    }


def check_optimizer_state_dtypes(optimizer):
    dtypes = set()
    states = 0
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                dtypes.add(str(state[key].dtype).removeprefix("torch."))
                states += 1
    if not states:
        raise SelfCheckError("optimizer_state_check_failed: no AdamW state after step 1")
    if dtypes != {"float32"}:
        raise SelfCheckError(
            f"optimizer_state_check_failed: AdamW state dtypes {sorted(dtypes)}, expected float32"
        )
    return {"optimizer_state_dtype": "float32", "optimizer_state_tensors": states}


def summarize_replica_rows(rows):
    """rows: list (one per rank) of equal-length float lists."""
    if not rows:
        raise SelfCheckError("replica comparison received no rows")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise SelfCheckError("replica rows have different lengths")
    max_diff = 0.0
    for column in range(width):
        values = [row[column] for row in rows]
        max_diff = max(max_diff, max(values) - min(values))
    return max_diff


def evaluate_first_step(
    *,
    world_size,
    gathered_params_after,
    gathered_grads_before_reduce,
    gathered_grads_after_reduce,
    changed_fraction,
    global_batch_rows,
    local_batch_rows,
    optimizer_state_info,
):
    """Decide pass/fail for the first-optimizer-step synchronization check.

    ``gathered_*`` are per-rank lists of probe values (len == world_size).
    """
    if len(gathered_params_after) != world_size:
        raise SelfCheckError(
            "grad_sync_check_failed: all_gather returned "
            f"{len(gathered_params_after)} rows, expected {world_size}"
        )
    param_diff = summarize_replica_rows(gathered_params_after)
    grad_after_diff = summarize_replica_rows(gathered_grads_after_reduce)
    grad_before_diff = summarize_replica_rows(gathered_grads_before_reduce)
    result = {
        "world_size": world_size,
        "param_probe_max_abs_diff_across_ranks": param_diff,
        "reduced_grad_probe_max_abs_diff_across_ranks": grad_after_diff,
        "local_grad_probe_max_abs_diff_across_ranks": grad_before_diff,
        "param_probe_fraction_changed_by_step1": round(float(changed_fraction), 6),
        "rank0_param_probe_head": [round(v, 9) for v in gathered_params_after[0][:3]],
        "rank_last_param_probe_head": [round(v, 9) for v in gathered_params_after[-1][:3]],
        "local_batch_rows": int(local_batch_rows),
        "effective_global_batch": int(global_batch_rows),
        **optimizer_state_info,
    }
    failures = []
    if param_diff != 0.0:
        failures.append("parameters differ across ranks after step 1")
    if grad_after_diff != 0.0:
        failures.append("gradients differ across ranks after xm.reduce_gradients all-reduce")
    if world_size > 1 and grad_before_diff == 0.0:
        failures.append(
            "local gradients identical on every rank before reduction "
            "(all ranks appear to read the same samples)"
        )
    if changed_fraction < MIN_CHANGED_FRACTION:
        failures.append(
            f"only {changed_fraction:.4f} of probed weights changed after step 1 "
            "(updates rounded away: bf16 storage?)"
        )
    if int(global_batch_rows) != int(local_batch_rows) * world_size:
        failures.append(
            f"global batch {global_batch_rows} != {local_batch_rows} x {world_size}"
        )
    if failures:
        raise SelfCheckError(
            "grad_sync_check_failed: " + "; ".join(failures)
            + " " + json.dumps(result, sort_keys=True)
        )
    return result


def format_passed(result):
    return "grad_sync_check_passed " + json.dumps(result, sort_keys=True)


def one_hot_row(ordinal, world_size):
    """Host-side one-hot used to gather via SUM all-reduce.

    Transferred to the device as *data* (not a graph constant) so every rank
    compiles an identical program; torch_xla's pin-layout all_gather bakes the
    ordinal into the HLO instead.
    """
    return [1.0 if index == int(ordinal) else 0.0 for index in range(int(world_size))]


def gather_rows_via_all_reduce(value, ordinal, world_size, device, all_reduce_sum_tensor):
    """Return a (world_size, n) tensor whose row r is rank r's 1-D ``value``.

    Exact for floats: every element is ``x + 0 + ... + 0``.
    """
    import torch

    mask = torch.tensor(one_hot_row(ordinal, world_size), dtype=torch.float32).to(device)
    rows = mask.unsqueeze(1) * value.float().reshape(1, -1)
    return all_reduce_sum_tensor(rows)


class Watchdog:
    """Hard-exit the process if a collective check hangs (fail fast on Kaggle)."""

    def __init__(self, seconds, label, exit_fn=None):
        self.seconds = float(seconds)
        self.label = label
        self.exit_fn = exit_fn or (lambda: os._exit(3))
        self._timer = None

    def _fire(self):
        print(
            f"{self.label}_failed: watchdog timeout after {self.seconds:.0f}s "
            "(collective hang: replicas are not all participating). Exiting.",
            flush=True,
        )
        self.exit_fn()

    def __enter__(self):
        import threading

        if self.seconds > 0:
            self._timer = threading.Timer(self.seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, *exc):
        if self._timer is not None:
            self._timer.cancel()
        return False
