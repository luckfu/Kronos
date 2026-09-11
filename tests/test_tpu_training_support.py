import ast
import importlib.util
import json
import os
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
TPU_ENTRY = ROOT / "finetune/tpu_train_entry.py"
TPU_RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_tpu.py"
TRAINER = ROOT / "finetune/train_predictor.py"


def test_tpu_entry_bootstraps_project_import_paths_before_worker_imports():
    source = TPU_ENTRY.read_text()

    assert 'PROJECT_ROOT = Path(__file__).resolve().parents[1]' in source
    assert 'FINETUNE_ROOT = PROJECT_ROOT / "finetune"' in source
    assert source.index("sys.path.insert") < source.index("from config import Config")
    assert source.index('os.environ.setdefault("PJRT_DEVICE"') < source.index(
        "import torch_xla"
    )
    assert source.index('os.environ.setdefault("XLA_USE_BF16"') < source.index(
        "import torch_xla"
    )


def test_tpu_entry_spawns_single_worker_and_runs_trainer_config():
    calls = []
    fake_xla = types.ModuleType("torch_xla")
    fake_xla.__path__ = []
    fake_distributed = types.ModuleType("torch_xla.distributed")
    fake_distributed.__path__ = []
    fake_xmp = types.ModuleType("torch_xla.distributed.xla_multiprocessing")

    def spawn(worker, *, nprocs, start_method):
        calls.append(("spawn", nprocs, start_method))
        worker(0)

    fake_xmp.spawn = spawn
    fake_config = types.ModuleType("config")

    class FakeConfig:
        def __init__(self):
            self.marker = "configured"

    fake_config.Config = FakeConfig
    fake_trainer = types.ModuleType("train_predictor")
    fake_trainer.main = lambda config: calls.append(("main", config))
    modules = {
        "torch_xla": fake_xla,
        "torch_xla.distributed": fake_distributed,
        "torch_xla.distributed.xla_multiprocessing": fake_xmp,
        "config": fake_config,
        "train_predictor": fake_trainer,
    }

    with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
        runpy.run_path(str(TPU_ENTRY), run_name="__main__")
        assert os.environ["KRONOS_DEVICE"] == "xla"
        assert os.environ["PJRT_DEVICE"] == "TPU"
        assert os.environ["XLA_USE_BF16"] == "1"

    assert calls == [
        ("spawn", 1, "fork"),
        ("main", {"marker": "configured"}),
    ]


def test_kaggle_runner_uses_extracted_repo_for_child_pythonpath():
    tree = ast.parse(TPU_RUNNER.read_text())
    build_environment = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_environment"
    )
    argument_names = [argument.arg for argument in build_environment.args.args]

    assert "repo_root" in argument_names
    assert 'str(repo_root / "finetune")' in TPU_RUNNER.read_text()


def test_c1_runner_uses_hardcoded_swanlab_api_key():
    source = TPU_RUNNER.read_text()

    assert 'SWANLAB_API_KEY = "fmEPDGk4IItxgqSZKGLi8"' in source
    assert 'swanlab.login(api_key=SWANLAB_API_KEY)' in source
    assert 'UserSecretsClient' not in source


def test_c1_kernel_requests_tpu_machine_shape():
    metadata = json.loads(
        (ROOT / "finetune/kaggle_beta_v21_c1_tpu_smoke_kernel/kernel-metadata.json").read_text()
    )

    assert metadata["machine_shape"] == "TpuV5E8"
    assert "accelerator" not in metadata


def test_xla_launcher_cannot_be_overridden_by_torchrun_env():
    source = TPU_RUNNER.read_text()

    assert 'is_xla = env.get("KRONOS_DEVICE") == "xla"' in source
    assert 'nproc = int(env.get("KRONOS_TORCHRUN_NPROC_PER_NODE", "1"))' in source
    assert 'if is_xla:' in source


def test_xla_loader_skips_disabled_quick_validation_loader():
    source = TRAINER.read_text()

    guard = "if val_loader is not None:"
    wrapper = "val_loader = MpDeviceLoader(val_loader, device)"
    assert guard in source
    assert source.index(guard) < source.index(wrapper)


def load_tpu_runner():
    spec = importlib.util.spec_from_file_location("kronos_tpu_runner", TPU_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c1_runner_environment_passes_config_validation(tmp_path):
    runner = load_tpu_runner()
    data_root = tmp_path / "data"
    processed = data_root / "processed_datasets"
    processed.mkdir(parents=True)
    (processed / "train_data.pkl").write_bytes(b"train")
    (processed / "val_data.pkl").write_bytes(b"val")
    (data_root / "asset_metadata.csv").write_text("symbol,sector,size_bucket\n")
    (data_root / "data_manifest.json").write_text(json.dumps({
        "window_contract": {
            "validation_signal_start": "2026-01-01",
            "validation_signal_end": "2026-12-31",
        }
    }))
    predictor = tmp_path / "predictor"
    tokenizer = tmp_path / "tokenizer"
    for model_dir in (predictor, tokenizer):
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"model")
    (predictor / "config.json").write_text(json.dumps({
        "n_layers": 12,
        "d_model": 832,
        "n_heads": 16,
        "ff_dim": 2048,
        "context_layer": 10,
    }))
    (tokenizer / "config.json").write_text("{}")

    env = runner.build_environment(
        "c1", data_root, predictor, tokenizer, tmp_path / "runtime", "output", ROOT
    )
    with patch.dict(os.environ, env, clear=True):
        config_spec = importlib.util.spec_from_file_location(
            "kronos_tpu_config", ROOT / "finetune/config.py"
        )
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        config = config_module.Config()

    assert config.use_beta_v21_auxiliary
    assert config.beta_v21_auto_calibrate
    assert config.best_selection_metric == "beta_v21_score"
    assert config.validation_full_only
    assert config.n_train_iter == 512
    assert config.n_val_iter == 512
    assert config.beta_v21_consistency_samples == 128
    assert config.batch_size == 64
    assert env["PJRT_DEVICE"] == "TPU"
    assert env["XLA_USE_BF16"] == "1"
    assert env["KRONOS_XLA_SINGLE_PROCESS"] == "1"
    assert env["KRONOS_NUM_WORKERS"] == "0"


def test_xla_resume_checkpoint_uses_xla_serializer_and_rng_state():
    source = TRAINER.read_text()

    assert "capture_rng_state(include_xla=is_xla)" in source
    assert "xm.save(payload, temporary)" in source
    assert "xm.set_rng_state(state['xla'])" in source


def test_xla_export_uses_cpu_state_dict_serializer():
    source = TRAINER.read_text()

    assert "_save_xla_pretrained(model, path, config)" in source
    assert "xm.save(model.state_dict(), state_path)" in source
    assert "save_model_as_safetensor" in source
    assert "map_location='cpu'" in source


def test_xla_setup_supports_pjrt_runtime_topology_api():
    source = (ROOT / "finetune/utils/training_utils.py").read_text()

    assert "import torch_xla.runtime as xr" in source
    assert '"global_ordinal", "get_ordinal", 0' in source
    assert '"world_size", "xrt_world_size", 1' in source
    assert '"local_ordinal", "get_local_ordinal", rank' in source
    assert "KRONOS_XLA_SINGLE_PROCESS" in source


def test_xla_setup_can_force_single_process_topology(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "kronos_training_utils_single_process",
        ROOT / "finetune/utils/training_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    xla_model = types.ModuleType("torch_xla.core.xla_model")
    runtime = types.ModuleType("torch_xla.runtime")
    runtime.global_ordinal = lambda: 3
    runtime.world_size = lambda: 8
    runtime.local_ordinal = lambda: 3
    torch_xla = types.ModuleType("torch_xla")
    torch_xla_core = types.ModuleType("torch_xla.core")
    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", torch_xla_core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", runtime)
    monkeypatch.setenv("KRONOS_DEVICE", "xla")
    monkeypatch.setenv("KRONOS_XLA_SINGLE_PROCESS", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert module.setup_ddp() == (0, 1, 0)


def test_xla_setup_reads_pjrt_runtime_topology(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "kronos_training_utils", ROOT / "finetune/utils/training_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    xla_model = types.ModuleType("torch_xla.core.xla_model")
    runtime = types.ModuleType("torch_xla.runtime")
    runtime.global_ordinal = lambda: 3
    runtime.world_size = lambda: 8
    runtime.local_ordinal = lambda: 3
    torch_xla = types.ModuleType("torch_xla")
    torch_xla_core = types.ModuleType("torch_xla.core")
    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", torch_xla_core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", runtime)
    monkeypatch.setenv("KRONOS_DEVICE", "xla")
    for name in ("ORDINAL", "WORLD_SIZE", "LOCAL_ORDINAL", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)

    assert module.setup_ddp() == (3, 8, 3)


def test_manifest_records_parent_model_architecture(tmp_path):
    runner = load_tpu_runner()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "processed_datasets").mkdir()
    for name in ("train_data.pkl", "val_data.pkl"):
        (data_root / "processed_datasets" / name).write_bytes(name.encode())
    predictor = tmp_path / "predictor"
    tokenizer = tmp_path / "tokenizer"
    predictor.mkdir()
    tokenizer.mkdir()
    architecture = {
        "n_layers": 12,
        "d_model": 832,
        "n_heads": 16,
        "ff_dim": 2048,
        "context_layer": 10,
    }
    (predictor / "config.json").write_text(json.dumps(architecture))
    (predictor / "model.safetensors").write_bytes(b"predictor")
    (tokenizer / "model.safetensors").write_bytes(b"tokenizer")

    output_root = tmp_path / "output"
    runner.write_manifest(output_root, "c1", data_root, predictor, tokenizer)
    manifest = json.loads((output_root / "small_v21_manifest.json").read_text())

    assert manifest["architecture"] == {
        "model": "Kronos-base-compatible",
        **architecture,
        "physical_layer_expansion": False,
    }
    assert manifest["validation"] == {
        "selection": "beta_v21_score",
        "full_symbol_holdout_required": False,
        "profile": "tpu_smoke_512",
        "quick_validation_is_telemetry_only": True,
    }
