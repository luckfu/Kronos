"""KRONOS_EVAL_ONLY: validate the untouched parent once, zero optimizer steps."""

import base64
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
RUNNER = FINETUNE / "kaggle_beta_v21_c1_tpu.py"
EVAL_STAGING = FINETUNE / "kaggle_beta_v21_c1_eval_kernel"
MAIN_STAGING = FINETUNE / "kaggle_beta_v21_c1_tpu_kernel"

for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name, path):
    # Keep staging dirs free of __pycache__ (they are pushed to Kaggle).
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- runner/staging
def test_source_runner_keeps_eval_only_off_and_normal_soft_stop():
    source = RUNNER.read_text()
    assert '\nDEFAULT_EVAL_ONLY = "0"\n' in source
    assert '\nDEFAULT_MAX_RUNTIME_SECONDS = "32400"\n' in source


def test_eval_staging_runner_defaults_on_and_only_differs_in_defaults():
    staged = (EVAL_STAGING / RUNNER.name).read_text()
    source = RUNNER.read_text()
    strip = lambda text: re.sub(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""', "ARCHIVE", text, flags=re.DOTALL
    )
    staged_lines = strip(staged).splitlines()
    source_lines = strip(source).splitlines()
    assert len(staged_lines) == len(source_lines)
    differing = {
        (a, b) for a, b in zip(source_lines, staged_lines) if a != b
    }
    assert differing == {
        ('EXPERIMENT_NAME = "beta_v2_1_c1_tpu_fp32_allreduce"',
         'EXPERIMENT_NAME = "beta_v2_1_c1_tpu_base_eval"'),
        ('DEFAULT_EVAL_ONLY = "0"', 'DEFAULT_EVAL_ONLY = "1"'),
        ('DEFAULT_MAX_RUNTIME_SECONDS = "32400"', 'DEFAULT_MAX_RUNTIME_SECONDS = "2400"'),
    }


def test_eval_staging_metadata_matches_main_except_identity():
    main = json.loads((MAIN_STAGING / "kernel-metadata.json").read_text())
    staged = json.loads((EVAL_STAGING / "kernel-metadata.json").read_text())
    assert staged["id"] == "user281434/kronos-beta-v2-1-c1-base-eval"
    assert staged["title"] == "Kronos Beta V2 1 C1 Base Eval"
    for key in ("enable_tpu", "machine_shape", "enable_internet", "docker_image",
                "dataset_sources", "code_file", "is_private", "enable_gpu"):
        assert staged[key] == main[key], key
    assert staged["enable_tpu"] is True and staged["machine_shape"] == "TpuV5E8"


def test_eval_staging_archive_matches_working_tree():
    staged = (EVAL_STAGING / RUNNER.name).read_text()
    payload = re.search(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n(.*?)\n"""', staged, re.DOTALL
    ).group(1)
    names = []
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                rel = Path(member.name).relative_to("Kronos")
                assert tar.extractfile(member).read() == (ROOT / rel).read_bytes(), rel
                names.append(str(rel))
    assert "finetune/train_predictor.py" in names
    assert "finetune/validation_precision.py" in names


def _fake_data_root(tmp_path):
    data_root = tmp_path / "data"
    processed = data_root / "processed_datasets"
    processed.mkdir(parents=True)
    (processed / "train_data.pkl").write_bytes(b"train")
    (processed / "val_data.pkl").write_bytes(b"val")
    (data_root / "asset_metadata.csv").write_text("symbol,sector,size_bucket\n")
    (data_root / "data_manifest.json").write_text(json.dumps({"window_contract": {}}))
    models = []
    for name in ("predictor", "tokenizer"):
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"model")
        (model_dir / "config.json").write_text("{}")
        models.append(model_dir)
    return data_root, *models


@pytest.mark.parametrize("flag,expected", [(None, "0"), ("1", "1"), ("0", "0")])
def test_runner_environment_eval_only_flag(tmp_path, flag, expected):
    runner = load_module("kronos_c1_runner_eval_test", RUNNER)
    data_root, predictor, tokenizer = _fake_data_root(tmp_path)
    output_root = tmp_path / "runtime/outputs/models/output/checkpoints"
    output_root.mkdir(parents=True)
    (output_root / "last_state.pt").write_bytes(b"x")
    overrides = {} if flag is None else {"KRONOS_EVAL_ONLY": flag}
    with patch.dict(os.environ, overrides, clear=False):
        if flag is None:
            os.environ.pop("KRONOS_EVAL_ONLY", None)
        env = runner.build_environment(
            "c1", data_root, predictor, tokenizer, tmp_path / "runtime", "output", ROOT
        )
    assert env["KRONOS_EVAL_ONLY"] == expected
    # Eval-only must never resume a previous output tree.
    assert env["KRONOS_RESUME_TRAINING"] == ("0" if expected == "1" else "1")


def test_staged_runner_environment_defaults_to_eval_only(tmp_path):
    runner = load_module("kronos_c1_eval_staged_runner", EVAL_STAGING / RUNNER.name)
    data_root, predictor, tokenizer = _fake_data_root(tmp_path)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("KRONOS_EVAL_ONLY", None)
        os.environ.pop("KRONOS_MAX_RUNTIME_SECONDS", None)
        env = runner.build_environment(
            "c1", data_root, predictor, tokenizer, tmp_path / "runtime", "output", ROOT
        )
    assert env["KRONOS_EVAL_ONLY"] == "1"
    assert env["KRONOS_MAX_RUNTIME_SECONDS"] == "2400"


# ---------------------------------------------------------------- trainer path
LOOKBACK, PREDICT, BATCH, N_VAL = 12, 10, 8, 20


class _TinyDataset(torch.utils.data.Dataset):
    validation_period_names = {}

    def __init__(self, n):
        generator = torch.Generator().manual_seed(0)
        self.items = []
        for index in range(n):
            x = torch.randn(LOOKBACK + PREDICT, 6, generator=generator)
            stamp = torch.randint(0, 5, (LOOKBACK + PREDICT, 5), generator=generator).float()
            labels = {
                "return_targets": torch.randn(4, generator=generator).clamp(-3, 3),
                "return_scales": torch.rand(4, generator=generator) + 0.1,
                "barrier_target": torch.tensor(index % 3, dtype=torch.long),
                "barrier_valid": torch.tensor(index % 3 != 1),
                "utility": torch.randn((), generator=generator),
                "date_id": torch.tensor(index // 4, dtype=torch.long),
                "feature_means": torch.rand(6, generator=generator) + 1.0,
                "feature_stds": torch.rand(6, generator=generator) + 0.1,
            }
            self.items.append((
                x, stamp, torch.tensor(index % 4, dtype=torch.long),
                torch.tensor(0, dtype=torch.long), torch.rand(()), labels,
            ))
        self.total_samples = self.n_samples = n

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _tiny_models():
    from model.kronos import Kronos, KronosTokenizer

    torch.manual_seed(0)
    tokenizer = KronosTokenizer(
        d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05,
        group_size=4,
    ).eval()
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=16, n_heads=2, ff_dim=32,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True, num_sectors=4, context_layer=1,
        use_size_percentile=True, use_beta_v21_auxiliary=True,
    )
    return tokenizer, model


def _trainer_config(tmp_path):
    return {
        "batch_size": BATCH,
        "lookback_window": LOOKBACK,
        "predict_window": PREDICT,
        "max_context": 512,
        "clip": 5.0,
        "forecast_horizon_weights": (
            1.364, 1.364, 1.364, 1.136, 1.136, 0.909, 0.909, 0.682, 0.682, 0.455
        ),
        "predictor_loss_mode": "forecast",
        "history_loss_weight": 0.02,
        "best_selection_metric": "beta_v21_score",
        "use_beta_v21_auxiliary": True,
        "beta_v21_auto_calibrate": True,
        "beta_v21_validation_denominators": "",
        "beta_v21_consistency_samples": 4,
        "beta_v21_consistency_sample_count": 1,
        "validation_full_only": True,
        "use_sector_features": True,
        "use_size_features": False,
        "use_size_percentile": True,
        "use_amp": False,
        "pretrained_predictor_path": str(tmp_path / "parent"),
        "epochs": 3,
    }


def test_trainer_eval_only_validates_parent_once_without_optimizer(tmp_path, monkeypatch):
    trainer = load_module("kronos_trainer_eval_only", FINETUNE / "train_predictor.py")
    dataset = _TinyDataset(N_VAL)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False)
    monkeypatch.setattr(
        trainer, "create_dataloaders",
        lambda config, rank, world_size: (loader, None, loader, dataset, dataset),
    )
    calls = []
    real_evaluate = trainer.evaluate_validation

    def counting_evaluate(*args, **kwargs):
        calls.append(kwargs.get("period_names"))
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(trainer, "evaluate_validation", counting_evaluate)

    def forbidden(*args, **kwargs):
        raise AssertionError("eval-only must not build an optimizer")

    monkeypatch.setattr(trainer.torch.optim, "AdamW", forbidden)
    monkeypatch.setenv("KRONOS_EVAL_ONLY", "1")

    tokenizer, model = _tiny_models()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    config = _trainer_config(tmp_path)
    result = trainer.train_model(
        model, tokenizer, torch.device("cpu"), config, str(save_dir), None, 0, 1
    )

    assert len(calls) == 2  # denominator calibration + base-model validation
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert result["optimizer_steps"] == 0
    metrics = result["metrics"]
    assert metrics["samples"] == N_VAL
    # Denominators are calibrated from this same untouched model on the same
    # validation set, so the base model's score is 1.0 by construction.
    assert metrics["beta_v21_score"] == pytest.approx(1.0, abs=1e-12)
    assert result["beta_v21_denominator_source"] == (
        "auto_calibrated_same_run_parent_checkpoint"
    )
    written = json.loads((save_dir / "base_model_validation.json").read_text())
    assert written["metrics"]["objective_loss"] == pytest.approx(metrics["objective_loss"])
    assert written["calibration_metrics"]["weighted_forecast_loss"] == pytest.approx(
        metrics["weighted_forecast_loss"]
    )
    progress = json.loads((save_dir / "progress.json").read_text())
    assert progress["status"] == "eval_only_completed"
    record = json.loads((save_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert record["type"] == "base_model_validation"
    assert not (save_dir / "checkpoints").exists()


def test_trainer_eval_only_with_fixed_denominators_scores_relative_to_them(
    tmp_path, monkeypatch
):
    trainer = load_module("kronos_trainer_eval_only_fixed", FINETUNE / "train_predictor.py")
    dataset = _TinyDataset(N_VAL)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False)
    monkeypatch.setattr(
        trainer, "create_dataloaders",
        lambda config, rank, world_size: (loader, None, loader, dataset, dataset),
    )
    monkeypatch.setenv("KRONOS_EVAL_ONLY", "1")
    tokenizer, model = _tiny_models()
    config = _trainer_config(tmp_path)
    config["beta_v21_validation_denominators"] = "2.0,2.0,1.0,1.0,1.0"
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    result = trainer.train_model(
        model, tokenizer, torch.device("cpu"), config, str(save_dir), None, 0, 1
    )
    m = result["metrics"]
    expected = (
        0.5 * m["weighted_forecast_loss"] / 2.0 + 0.2 * m["return_loss"]
        + 0.2 * m["barrier_loss"] + 0.1 * m["ranking_loss"]
    )
    assert m["beta_v21_score"] == pytest.approx(expected)
    assert result["calibration_metrics"] is None


def test_trainer_default_does_not_enter_eval_only(monkeypatch):
    trainer = load_module("kronos_trainer_default", FINETUNE / "train_predictor.py")
    monkeypatch.delenv("KRONOS_EVAL_ONLY", raising=False)
    assert trainer.eval_only_requested() is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("KRONOS_EVAL_ONLY", value)
        assert trainer.eval_only_requested() is True
