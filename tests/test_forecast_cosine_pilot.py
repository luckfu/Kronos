"""Forecast cosine pilot: snapshot hook + runner recipe (tiny CPU model)."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
RUNNER = FINETUNE / "kaggle_beta_v21_c1_forecast_cosine_pilot.py"
STAGING = FINETUNE / "kaggle_beta_v21_c1_forecast_cosine_pilot_kernel"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_eval_only_mode import _TinyDataset, BATCH, LOOKBACK, PREDICT  # noqa: E402


class _SegmentDataset(_TinyDataset):
    selection_report = {}

    def set_epoch_seed(self, epoch):
        self.epoch = epoch


def load_module(name, path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pilot_env(tmp_path):
    pilot = load_module("pilot_runner", RUNNER)
    data = tmp_path / "data"
    (data / "processed_datasets").mkdir(parents=True)
    (data / "data_manifest.json").write_text("{}")
    out = tmp_path / "models" / pilot.OUTPUT_NAME
    env = pilot.build_environment(data, tmp_path / "p", tmp_path / "t", out, ROOT)
    return pilot, env, out


def test_pilot_recipe_is_forecast_only_uniform_cosine(tmp_path, monkeypatch):
    pilot, env, _ = _pilot_env(tmp_path)
    pilot.verify_recipe(env)
    for key, value in env.items():
        if key.startswith("KRONOS_"):
            monkeypatch.setenv(key, value)
    from config import Config

    config = Config().__dict__
    assert config["scheduler_type"] == "uniform_cosine"
    assert config["use_beta_v21_auxiliary"] is False
    assert config["same_day_ranking_batches"] is False  # shuffled coverage order
    assert config["split_trunk_head_learning_rate"] is False
    assert config["trainable_transformer_layers"] == -1
    assert config["best_selection_metric"] == "forecast"
    assert config["epochs"] == 12 and config["require_full_coverage"] is False
    assert config["predictor_min_learning_rate"] == pytest.approx(1e-6)


def test_staged_metadata_is_new_slug_dual_t4_pinned():
    meta = json.loads((STAGING / "kernel-metadata.json").read_text())
    assert meta["id"] == "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot"
    assert meta["enable_tpu"] is False and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["docker_image"].endswith(
        "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
    )
    assert "luckfu/kronos-beta-v21-c1-seg155-forecast-best" in meta["dataset_sources"]


def test_snapshot_hook_saves_every_n_segments(tmp_path, monkeypatch):
    from model.kronos import Kronos, KronosTokenizer

    trainer = load_module("pilot_trainer", FINETUNE / "train_predictor.py")
    train = _SegmentDataset(16)
    val = _SegmentDataset(8)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH, shuffle=False)
    monkeypatch.setattr(
        trainer, "create_dataloaders",
        lambda config, rank, world_size: (train_loader, None, val_loader, train, val),
    )
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
        use_size_percentile=True, use_beta_v21_auxiliary=False,
    )
    _, env, _ = _pilot_env(tmp_path)
    for key, value in env.items():
        if key.startswith("KRONOS_") and key != "KRONOS_SNAPSHOT_DIR":
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("KRONOS_SNAPSHOT_EVERY_SEGMENTS", "2")
    monkeypatch.setenv("KRONOS_KEEP_EXISTING_BEST", "0")
    monkeypatch.setenv("KRONOS_EPOCHS", "4")
    monkeypatch.setenv("KRONOS_MAX_SEGMENTS_PER_RUN", "4")
    monkeypatch.setenv("KRONOS_USE_AMP", "0")
    from config import Config

    config = Config().__dict__
    config.update({
        "batch_size": BATCH, "lookback_window": LOOKBACK, "predict_window": PREDICT,
        "num_sectors": 4, "context_layer": 1, "use_amp": False,
        "pretrained_predictor_path": str(tmp_path / "parent"),
        "use_swanlab": False, "use_comet": False,
    })
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    trainer.train_model(model, tokenizer, torch.device("cpu"), config, str(save_dir), None, 0, 1)
    snaps = sorted(p.name for p in (save_dir / "snapshots").iterdir())
    assert snaps == ["seg002", "seg004"]
    metric = json.loads((save_dir / "snapshots/seg004/snapshot_metric.json").read_text())
    assert metric["segment"] == 4
    assert (save_dir / "snapshots/seg004/model.safetensors").is_file()
    assert metric["learning_rates"][0] == pytest.approx(1e-6, rel=1e-3)
