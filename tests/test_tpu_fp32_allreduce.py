"""fp32 master weights + real 8-core gradient all-reduce on Kaggle TPU."""

import importlib.util
import json
import os
import runpy
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
for _path in (str(FINETUNE), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tpu_self_checks as checks  # noqa: E402

TPU_ENTRY = FINETUNE / "tpu_train_entry.py"
TPU_RUNNER = FINETUNE / "kaggle_beta_v21_c1_tpu.py"
TRAINER = FINETUNE / "train_predictor.py"


class FakeDevice:
    def __init__(self, kind):
        self.type = kind


def load_trainer():
    import train_predictor
    return train_predictor


def load_runner():
    spec = importlib.util.spec_from_file_location("kronos_tpu_runner_fp32", TPU_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# AMP dtype resolution
# --------------------------------------------------------------------------

def test_resolve_amp_dtype_is_bfloat16_autocast_on_xla():
    trainer = load_trainer()
    xla = FakeDevice("xla")
    assert trainer.resolve_amp_dtype({"use_amp": True, "amp_dtype": "bfloat16"}, xla) is torch.bfloat16
    # fp16 is never used on TPU even if a stale config asks for it.
    assert trainer.resolve_amp_dtype({"use_amp": True, "amp_dtype": "float16"}, xla) is torch.bfloat16
    assert trainer.resolve_amp_dtype({"use_amp": False}, xla) is None


def test_resolve_amp_dtype_unchanged_for_cpu_and_cuda():
    trainer = load_trainer()
    assert trainer.resolve_amp_dtype({"use_amp": True}, FakeDevice("cpu")) is None
    assert trainer.resolve_amp_dtype({"use_amp": True, "amp_dtype": "fp16"}, FakeDevice("cuda")) is torch.float16


def test_no_gradscaler_for_bf16_and_losses_in_fp32():
    source = TRAINER.read_text()
    assert "scale_gradients = amp_dtype == torch.float16" in source
    assert "logits = to_float32(logits)" in source
    assert "auxiliary_predictions = to_float32(auxiliary_predictions)" in source
    assert "with torch.autocast(device_type=device.type, enabled=False):" in source
    assert "' autocast' if use_amp else ''" in source


def test_reduce_then_clip_then_step_on_xla():
    source = TRAINER.read_text()
    reduce_at = source.index("xm.reduce_gradients(optimizer)")
    clip_at = source.index("torch.nn.utils.clip_grad_norm_(", reduce_at - 2000)
    step_at = source.index("optimizer.step()\n                xm.mark_step()")
    assert reduce_at < clip_at < step_at
    assert "xm.optimizer_step(optimizer, barrier=True)" not in source


def test_rope_cache_is_fp32_even_under_autocast():
    from model.module import RotaryPositionalEmbedding

    rope = RotaryPositionalEmbedding(52)
    q = torch.randn(2, 4, 129, 52)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cos, sin = rope._update_cos_sin_cache(q, 129)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    t = torch.arange(129, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, rope.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    assert torch.equal(cos[0, 0], emb.cos()) and torch.equal(sin[0, 0], emb.sin())


# --------------------------------------------------------------------------
# Environment: no XLA_USE_BF16, LR plan
# --------------------------------------------------------------------------

def _runner_env(tmp_path, extra_env=None):
    runner = load_runner()
    data_root = tmp_path / "data"
    processed = data_root / "processed_datasets"
    processed.mkdir(parents=True)
    (processed / "train_data.pkl").write_bytes(b"train")
    (processed / "val_data.pkl").write_bytes(b"val")
    (data_root / "asset_metadata.csv").write_text("symbol,sector,size_bucket\n")
    (data_root / "data_manifest.json").write_text(json.dumps({"window_contract": {}}))
    predictor = tmp_path / "predictor"
    tokenizer = tmp_path / "tokenizer"
    for model_dir in (predictor, tokenizer):
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"model")
    with patch.dict(os.environ, extra_env or {}, clear=False):
        return runner, runner.build_environment(
            "c1", data_root, predictor, tokenizer, tmp_path / "runtime", "output", ROOT
        )


def test_runner_env_does_not_enable_bf16_storage(tmp_path):
    runner, env = _runner_env(tmp_path, {"XLA_USE_BF16": "1", "XLA_DOWNCAST_BF16": "1"})
    assert "XLA_USE_BF16" not in env
    assert "XLA_DOWNCAST_BF16" not in env
    assert env["KRONOS_USE_AMP"] == "1"
    assert env["KRONOS_AMP_DTYPE"] == "bfloat16"
    assert env["KRONOS_TPU_LAUNCHER"] == "auto"
    assert '"XLA_USE_BF16": "1"' not in TPU_RUNNER.read_text()


def test_runner_lr_plan_back_to_3e5_with_short_warmup(tmp_path):
    runner, env = _runner_env(tmp_path)
    assert env["KRONOS_PREDICTOR_LEARNING_RATE"] == "3e-5"
    assert env["KRONOS_CONDITION_LEARNING_RATE"] == "3e-5"
    assert env["KRONOS_PREDICTOR_WARMUP_START_LR"] == "3e-6"
    assert env["KRONOS_CONDITION_WARMUP_START_LR"] == "3e-6"
    assert env["KRONOS_PREDICTOR_MIN_LR"] == "3e-5"
    assert env["KRONOS_CONDITION_MIN_LR"] == "3e-5"
    assert env["KRONOS_SCHEDULER_WARMUP_RATIO"] == "0.002"
    assert runner.STAGES["c1"]["output"] == "beta_v2_1_c1_tpu_fp32"
    assert runner.DEFAULT_OUTPUT_SUFFIX == ""
    assert runner.DEFAULT_MAX_RUNTIME_SECONDS == "32400"


def test_tpu_entry_refuses_bf16_storage_flags():
    fake_xmp = types.ModuleType("torch_xla.distributed.xla_multiprocessing")
    modules = {
        "torch_xla": types.ModuleType("torch_xla"),
        "torch_xla.distributed": types.ModuleType("torch_xla.distributed"),
        "torch_xla.distributed.xla_multiprocessing": fake_xmp,
    }
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {"XLA_USE_BF16": "1"}):
        with pytest.raises(SystemExit, match="XLA_USE_BF16"):
            runpy.run_path(str(TPU_ENTRY), run_name="not_main")
    assert 'os.environ.setdefault("XLA_USE_BF16"' not in TPU_ENTRY.read_text()


# --------------------------------------------------------------------------
# Launcher selection
# --------------------------------------------------------------------------

def _load_entry_module():
    fake_xmp = types.ModuleType("torch_xla.distributed.xla_multiprocessing")
    modules = {
        "torch_xla": types.ModuleType("torch_xla"),
        "torch_xla.distributed": types.ModuleType("torch_xla.distributed"),
        "torch_xla.distributed.xla_multiprocessing": fake_xmp,
    }
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
        os.environ.pop("XLA_USE_BF16", None)
        return runpy.run_path(str(TPU_ENTRY), run_name="not_main")


def test_launcher_selection_prefers_replicated_threads():
    entry = _load_entry_module()
    select = entry["select_launcher"]
    pjrt = types.SimpleNamespace(_run_thread_per_device=object(), _SpawnFn=object(), spawn_threads=object())
    assert select(1, "auto", pjrt) == "single"
    assert select(8, "auto", pjrt) == "replicated_threads"
    assert select(8, "replicated_threads", pjrt) == "replicated_threads"
    assert select(8, "spawn", pjrt) == "spawn"
    assert select(8, "auto", None) == "spawn"
    with pytest.raises(ValueError, match="spawn_threads is disabled"):
        select(8, "spawn_threads", pjrt)
    with pytest.raises(RuntimeError):
        select(8, "replicated_threads", types.SimpleNamespace())


def test_replicated_initializer_sets_replication_and_checks_world_size():
    entry = _load_entry_module()
    calls = {}
    state = {"replicated": 0}
    fake_xla = types.ModuleType("torch_xla")
    fake_xla.device = lambda: "xla:0"
    fake_xm = types.ModuleType("torch_xla.core.xla_model")
    fake_xm.get_xla_supported_devices = lambda: [f"xla:{i}" for i in range(8)]

    def set_replication(device, devices):
        calls["set_replication"] = (device, list(devices))
        state["replicated"] = len(devices)

    fake_xm.set_replication = set_replication
    fake_xr = types.ModuleType("torch_xla.runtime")
    fake_xr._WORLD_SIZE = 1  # stale cache from an earlier call
    fake_xr.world_size = lambda: fake_xr._WORLD_SIZE or state["replicated"] or 1
    modules = {
        "torch_xla": fake_xla,
        "torch_xla.core": types.ModuleType("torch_xla.core"),
        "torch_xla.core.xla_model": fake_xm,
        "torch_xla.runtime": fake_xr,
    }
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {"KRONOS_TPU_CORES": "8"}):
        entry["initialize_replicated_threads"](0, 1)
    assert calls["set_replication"] == ("xla:0", [f"xla:{i}" for i in range(8)])
    assert fake_xr._WORLD_SIZE is None

    # Without replication the old failure mode (world_size 1) is fatal.
    fake_xm.set_replication = lambda device, devices: None
    state["replicated"] = 0
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {"KRONOS_TPU_CORES": "8"}):
        with pytest.raises(RuntimeError, match="xla_world_size_check_failed"):
            entry["initialize_replicated_threads"](0, 1)


def test_setup_ddp_thread_mode_requires_runtime_world_size(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "kronos_training_utils_fp32", FINETUNE / "utils/training_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = types.ModuleType("torch_xla.runtime")
    runtime.global_ordinal = lambda: 2
    runtime.local_ordinal = lambda: 2
    runtime.addressable_device_count = lambda: 8
    runtime.world_size = lambda: 8
    monkeypatch.setitem(sys.modules, "torch_xla", types.ModuleType("torch_xla"))
    monkeypatch.setitem(sys.modules, "torch_xla.core", types.ModuleType("torch_xla.core"))
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", types.ModuleType("torch_xla.core.xla_model"))
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", runtime)
    monkeypatch.setenv("KRONOS_DEVICE", "xla")
    monkeypatch.setenv("KRONOS_XLA_SINGLE_PROCESS", "0")
    monkeypatch.setenv("KRONOS_XLA_THREAD_PER_DEVICE", "1")
    monkeypatch.setenv("KRONOS_TPU_CORES", "8")
    assert module.setup_ddp() == (2, 8, 2)
    runtime.world_size = lambda: 1  # the old spawn_threads bug
    with pytest.raises(RuntimeError, match="xla_world_size_check_failed"):
        module.setup_ddp()


# --------------------------------------------------------------------------
# Self-check logic with simulated collectives
# --------------------------------------------------------------------------

class SimulatedReplicas:
    """Run one function per rank in threads with a real SUM all-reduce."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.lock = threading.Lock()
        self.slots = {}

    def all_reduce_sum(self, rank, value):
        key = ("sum", threading.get_ident())
        with self.lock:
            self.slots.setdefault("buffer", {})[rank] = value
        self.barrier.wait()
        total = sum(self.slots["buffer"][r] for r in range(self.world_size))
        self.barrier.wait()
        if rank == 0:
            self.slots["buffer"] = {}
        self.barrier.wait()
        return total

    def run(self, fn):
        results, errors = {}, {}

        def worker(rank):
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # noqa: BLE001
                errors[rank] = exc
                self.barrier.abort()

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(self.world_size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        return results, errors


def test_check_collectives_passes_with_real_reduction():
    sim = SimulatedReplicas(4)

    def fn(rank):
        reduce = lambda v: float(sim.all_reduce_sum(rank, torch.tensor([float(v)])).item())
        gather = lambda v: checks.gather_rows_via_all_reduce(
            torch.tensor([float(v)]), rank, 4, "cpu",
            lambda t: sim.all_reduce_sum(rank, t),
        ).reshape(-1).tolist()
        return checks.check_collectives(rank, 4, reduce, gather)

    results, errors = sim.run(fn)
    assert not errors, errors
    assert all("xla_collective_check_passed" in results[r] for r in range(4))
    assert "all_gather=[0, 1, 2, 3]" in results[0]


def test_check_collectives_fails_when_collectives_are_noops():
    # world_size()==1 torch_xla: all_reduce/all_gather return the local value.
    with pytest.raises(checks.SelfCheckError, match="xla_collective_check_failed"):
        checks.check_collectives(0, 8, lambda v: v, lambda v: [v])


def test_gather_rows_via_all_reduce_is_exact():
    sim = SimulatedReplicas(3)
    values = {r: torch.tensor([0.1 * (r + 1), -3.3e-7 * r, 12345.678]) for r in range(3)}

    def fn(rank):
        return checks.gather_rows_via_all_reduce(
            values[rank], rank, 3, "cpu", lambda t: sim.all_reduce_sum(rank, t)
        )

    results, errors = sim.run(fn)
    assert not errors
    for r in range(3):
        for row in range(3):
            assert torch.equal(results[r][row], values[row].float())


def test_check_world_size():
    assert "xla_world_size_check_passed" in checks.check_world_size(8, 8)
    with pytest.raises(checks.SelfCheckError, match="xla_world_size_check_failed"):
        checks.check_world_size(1, 8)


def test_bf16_env_guard():
    checks.assert_no_bf16_storage_env({})
    checks.assert_no_bf16_storage_env({"XLA_USE_BF16": "0"})
    with pytest.raises(checks.SelfCheckError, match="fp32_master_weight_check_failed"):
        checks.assert_no_bf16_storage_env({"XLA_USE_BF16": "1"})
    with pytest.raises(checks.SelfCheckError):
        checks.assert_no_bf16_storage_env({"XLA_DOWNCAST_BF16": "true"})


def test_fp32_storage_check_detects_bf16_rounded_weights():
    torch.manual_seed(0)
    fp32 = torch.randn(4096) * 0.04
    info = checks.check_fp32_storage(fp32, [torch.float32])
    assert info["param_dtype"] == "float32"
    assert info["probe_non_bf16_fraction"] > 0.99
    rounded = fp32.bfloat16().float()
    with pytest.raises(checks.SelfCheckError, match="fp32_master_weight_check_failed"):
        checks.check_fp32_storage(rounded, [torch.float32])
    assert checks.check_fp32_storage(rounded, [torch.float32], allow_bf16_parent=True)
    with pytest.raises(checks.SelfCheckError):
        checks.check_fp32_storage(fp32, [torch.bfloat16])


def test_optimizer_state_dtype_check():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(checks.SelfCheckError):
        checks.check_optimizer_state_dtypes(optimizer)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    assert checks.check_optimizer_state_dtypes(optimizer)["optimizer_state_dtype"] == "float32"
    model.to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4, dtype=torch.bfloat16)).sum().backward()
    optimizer.step()
    with pytest.raises(checks.SelfCheckError, match="optimizer_state_check_failed"):
        checks.check_optimizer_state_dtypes(optimizer)


def _first_step_kwargs(**overrides):
    rows = [[0.1, 0.2, 0.3]] * 8
    kwargs = dict(
        world_size=8,
        gathered_params_after=rows,
        gathered_grads_before_reduce=[[0.01 * r, 0.02, 0.03] for r in range(8)],
        gathered_grads_after_reduce=[[0.5, 0.6, 0.7]] * 8,
        changed_fraction=0.98,
        global_batch_rows=512,
        local_batch_rows=64,
        optimizer_state_info={"optimizer_state_dtype": "float32", "optimizer_state_tensors": 10},
    )
    kwargs.update(overrides)
    return kwargs


def test_first_step_check_passes_for_synchronized_fp32_step():
    result = checks.evaluate_first_step(**_first_step_kwargs())
    line = checks.format_passed(result)
    assert line.startswith("grad_sync_check_passed ")
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["effective_global_batch"] == 512
    assert payload["param_probe_max_abs_diff_across_ranks"] == 0.0


@pytest.mark.parametrize("override, message", [
    ({"gathered_params_after": [[0.1, 0.2, 0.3]] * 7 + [[0.1, 0.2, 0.3000001]]}, "parameters differ"),
    ({"gathered_grads_after_reduce": [[0.5, 0.6, 0.7]] * 7 + [[0.4, 0.6, 0.7]]}, "gradients differ"),
    ({"gathered_grads_before_reduce": [[0.1, 0.2, 0.3]] * 8}, "same samples"),
    ({"changed_fraction": 0.003}, "rounded away"),
    ({"global_batch_rows": 64}, "global batch"),
    ({"gathered_params_after": [[0.1, 0.2, 0.3]]}, "rows, expected 8"),
])
def test_first_step_check_fails_fast(override, message):
    with pytest.raises(checks.SelfCheckError, match="grad_sync_check_failed") as info:
        checks.evaluate_first_step(**_first_step_kwargs(**override))
    assert message in str(info.value)


def test_watchdog_fires_and_cancels():
    fired = threading.Event()
    with checks.Watchdog(0.05, "unit", exit_fn=fired.set):
        fired.wait(1)
    assert fired.is_set()
    quiet = threading.Event()
    with checks.Watchdog(0.5, "unit", exit_fn=quiet.set):
        pass
    assert not quiet.wait(0.8)


def test_first_step_sync_check_end_to_end_with_simulated_xla(monkeypatch):
    """Drive the trainer's begin/finish helpers with 4 simulated replicas."""
    trainer = load_trainer()
    world = 4
    sim = SimulatedReplicas(world)
    local = threading.local()

    class FakeXm:
        REDUCE_SUM = "sum"

        @staticmethod
        def all_reduce(kind, tensor):
            return sim.all_reduce_sum(local.rank, tensor)

        @staticmethod
        def mark_step():
            return None

        @staticmethod
        def reduce_gradients(optimizer):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.copy_(sim.all_reduce_sum(local.rank, parameter.grad.clone()) / world)

    monkeypatch.setattr(trainer, "xm", FakeXm)

    torch.manual_seed(0)
    initial = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Linear(5, 1)).state_dict()
    shards = {r: torch.randn(64, 6, generator=torch.Generator().manual_seed(100 + r)) for r in range(world)}

    def fn(rank, reduce=True):
        local.rank = rank
        model = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Linear(5, 1))
        model.load_state_dict(initial)  # identical replicas (process-global RNG is racy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
        batch = shards[rank]  # distinct shard per rank
        model(batch).pow(2).mean().backward()
        state = trainer.begin_first_step_sync_check(model, batch, "cpu", rank, world)
        if reduce:
            FakeXm.reduce_gradients(optimizer)
        optimizer.step()
        return trainer.finish_first_step_sync_check(state, model, optimizer, "cpu", rank, world)

    results, errors = sim.run(fn)
    assert not errors, errors
    assert results[0]["effective_global_batch"] == 256
    assert results[0]["param_probe_max_abs_diff_across_ranks"] == 0.0
    assert results[0]["local_grad_probe_max_abs_diff_across_ranks"] > 0.0
    assert results[0]["optimizer_state_dtype"] == "float32"

    sim2 = SimulatedReplicas(world)
    sim.__dict__.update(sim2.__dict__)
    results, errors = sim.run(lambda rank: fn(rank, reduce=False))
    assert errors and all(isinstance(e, checks.SelfCheckError) for e in errors.values())
    assert "grad_sync_check_failed" in str(next(iter(errors.values())))


# --------------------------------------------------------------------------
# Resume guard
# --------------------------------------------------------------------------

def test_refuses_resume_from_pre_fix_tpu_checkpoints():
    trainer = load_trainer()
    xla = FakeDevice("xla")
    with pytest.raises(ValueError, match="pre-fix TPU trainer"):
        trainer.refuse_low_precision_checkpoint({"amp_dtype": "disabled"}, xla, "last_state.pt")
    with pytest.raises(ValueError):
        trainer.refuse_low_precision_checkpoint(
            {"master_weight_dtype": "float32", "xla_gradient_all_reduce": False}, xla, "x"
        )
    trainer.refuse_low_precision_checkpoint(
        {"master_weight_dtype": "float32", "xla_gradient_all_reduce": True}, xla, "x"
    )
    trainer.refuse_low_precision_checkpoint({}, FakeDevice("cuda"), "x")


def test_master_weight_dtype_reports_bf16_env(monkeypatch):
    trainer = load_trainer()
    model = torch.nn.Linear(2, 2)
    monkeypatch.delenv("XLA_USE_BF16", raising=False)
    assert trainer.master_weight_dtype(model) == "float32"
    monkeypatch.setenv("XLA_USE_BF16", "1")
    assert trainer.master_weight_dtype(model) == "bfloat16"


def test_resume_state_records_precision_and_sync_markers():
    source = TRAINER.read_text()
    assert "master_weight_dtype=master_weight_dtype(core_model)" in source
    assert "xla_gradient_all_reduce=bool(device.type == 'xla' and world_size > 1)" in source
    assert source.count("refuse_low_precision_checkpoint(") >= 3


def test_no_rank0_only_rendezvous_inside_resume_save():
    source = TRAINER.read_text()
    body = source[source.index("def save_resume_state"):source.index("def save_pretrained_with_retry")]
    assert "distributed_barrier(" not in body


def test_startup_checks_run_on_every_replica_before_training():
    source = TRAINER.read_text()
    assert "run_xla_startup_checks(model, device, rank, world_size)" in source
    main_body = source[source.index("def main(config"):]
    assert main_body.index("model.to(device)") < main_body.index("run_xla_startup_checks(")
    assert "GlobalBatch: " in source


# --------------------------------------------------------------------------
# Probe staging builder
# --------------------------------------------------------------------------

def test_probe_runner_rewrites_only_probe_defaults():
    sys.path.insert(0, str(FINETUNE))
    import build_kaggle_beta_v21_c1_tpu_probe_kernel as probe

    source = TPU_RUNNER.read_text()
    runner = probe.build_probe_runner(source, "UEFZTE9BRA==")
    assert '\nDEFAULT_MAX_RUNTIME_SECONDS = "3600"\n' in runner
    assert '\nDEFAULT_OUTPUT_SUFFIX = "_probe"\n' in runner
    assert '_probe"\n' in runner.split("\nEXPERIMENT_NAME = ", 1)[1].split("\n", 1)[0] + "\n"
    assert 'EMBEDDED_KRONOS_ARCHIVE_B64 = """\nUEFZTE9BRA==\n"""' in runner
    assert '\nDEFAULT_MAX_RUNTIME_SECONDS = "32400"\n' in source
