"""Beta v2.1 heavy-reg retrain on a time-disjoint val: WSD schedule, dropout
overrides, time-disjoint panel guards, val_contract driver path, selection gate,
staging (tiny CPU models; real-data checks skip when local files are absent)."""

import importlib.util
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
RUNNER = FINETUNE / "kaggle_beta_v21_heavy_reg_time_val.py"
BUILDER = FINETUNE / "build_kaggle_beta_v21_heavy_reg_time_val_kernel.py"
STAGING = FINETUNE / "kaggle_beta_v21_heavy_reg_time_val_kernel"
DOCKER_SHA = "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
PROBE = Path("/workspace/scratch/heavyreg_probe")
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


def _runner():
    return load_module("heavy_reg_runner", RUNNER)


def _tiny_kronos(aux=False, dropout=0.0, num_sectors=4):
    from model.kronos import Kronos

    return Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=16, n_heads=2, ff_dim=32,
        ffn_dropout_p=dropout, attn_dropout_p=0.0, resid_dropout_p=dropout,
        token_dropout_p=0.0, learn_te=True, num_sectors=num_sectors, context_layer=1,
        use_size_percentile=True, use_beta_v21_auxiliary=aux,
    )


def _tiny_tokenizer():
    from model.kronos import KronosTokenizer

    return KronosTokenizer(
        d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05,
        group_size=4,
    ).eval()


def _env(tmp_path, runner):
    data = tmp_path / "data"
    (data / "processed_datasets").mkdir(parents=True, exist_ok=True)
    (data / "data_manifest.json").write_text("{}")
    time_val = {"panel_path": str(tmp_path / "time_val.pkl"), "panel_sha256": "x"}
    out = tmp_path / "models" / runner.OUTPUT_NAME
    return runner.build_environment(data, time_val, tmp_path / "p", tmp_path / "t", out, ROOT), out


# ------------------------------------------------------------------ recipe
def test_recipe_is_heavy_reg_wsd_forecast_only(tmp_path, monkeypatch):
    runner = _runner()
    env, _ = _env(tmp_path, runner)
    runner.verify_recipe(env)
    for key, value in env.items():
        if key.startswith("KRONOS_"):
            monkeypatch.setenv(key, value)
    from config import Config

    config = Config().__dict__
    assert config["scheduler_type"] == "warmup_constant_cosine"
    assert config["scheduler_decay_start_ratio"] == pytest.approx(2 / 3)
    assert config["scheduler_warmup_ratio"] == pytest.approx(1 / 48)
    assert config["predictor_resid_dropout_p"] == 0.25
    assert config["predictor_ffn_dropout_p"] == 0.25
    assert config["predictor_attn_dropout_p"] == 0.1
    assert config["predictor_token_dropout_p"] == 0.1
    assert config["adam_weight_decay"] == pytest.approx(0.1)
    assert config["use_beta_v21_auxiliary"] is False
    assert config["same_day_ranking_batches"] is False  # shuffled coverage order
    assert config["trainable_transformer_layers"] == -1
    assert config["epochs"] == 48 and config["max_segments_per_run"] == 4
    assert config["val_signal_start"] == "2026-07-17" and config["val_signal_end"] == "2026-08-10"
    assert config["train_signal_end"] == "2026-07-02"
    assert config["val_data_paths"] == (str(tmp_path / "time_val.pkl"),)
    with pytest.raises(SystemExit):
        runner.verify_recipe({**env, "KRONOS_ATTN_DROPOUT_P": "0.0"})
    with pytest.raises(SystemExit):
        runner.verify_recipe({**env, "KRONOS_VAL_DATA_PATHS": "/x/temporal_symbol_validation_v1/v.pkl"})


def test_config_rejects_bad_wsd_and_dropout(monkeypatch):
    from config import Config

    monkeypatch.setenv("KRONOS_SCHEDULER", "warmup_constant_cosine")
    monkeypatch.setenv("KRONOS_SCHEDULER_WARMUP_RATIO", "0.1")
    monkeypatch.setenv("KRONOS_SCHEDULER_DECAY_START_RATIO", "0.05")
    with pytest.raises(ValueError):
        Config()
    monkeypatch.setenv("KRONOS_SCHEDULER_DECAY_START_RATIO", "0.5")
    monkeypatch.setenv("KRONOS_ATTN_DROPOUT_P", "1.5")
    with pytest.raises(ValueError):
        Config()
    monkeypatch.delenv("KRONOS_ATTN_DROPOUT_P")
    assert Config().predictor_attn_dropout_p is None  # unset = keep parent


def test_schedule_constants_and_wsd_multiplier_agree():
    runner = _runner()
    trainer = load_module("heavy_trainer_sched", FINETUNE / "train_predictor.py")
    assert runner.STEPS_PER_SEGMENT == 313 and runner.TOTAL_STEPS == 15024
    assert runner.EXPECTED_WARMUP_STEPS == 313  # = segment 1
    assert runner.EXPECTED_DECAY_START_STEP == 32 * 313
    for step in (0, 100, 313, 314, 5000, 10016, 10017, 12520, 15024):
        multiplier = trainer.warmup_constant_cosine_multiplier(
            step, 15024, 313, 10016, 1e-6, 1e-5, 1e-6)
        assert multiplier * 1e-5 == pytest.approx(runner.lr_at_step(step), rel=1e-12)
    assert runner.lr_at_step(0) == pytest.approx(1e-6)
    assert runner.lr_at_segment_end(1) == pytest.approx(1e-5)
    assert runner.lr_at_segment_end(32) == pytest.approx(1e-5)
    assert runner.lr_at_segment_end(40) == pytest.approx(5.5e-6)
    assert runner.lr_at_segment_end(48) == pytest.approx(1e-6)
    # Monotone non-increasing after warmup.
    values = [runner.lr_at_step(s) for s in range(313, 15025, 50)]
    assert all(b <= a + 1e-18 for a, b in zip(values, values[1:]))


def test_resume_guard_new_fields_are_opt_in():
    trainer = load_module("heavy_trainer_guard", FINETUNE / "train_predictor.py")
    base = {"scheduler_type": "uniform_cosine", "scheduler_decay_start_ratio": 0.8,
            "predictor_resid_dropout_p": None, "predictor_attn_dropout_p": None}
    guard = trainer.build_resume_guard(base, 12, 100)
    assert "scheduler_decay_start_ratio" not in guard
    assert not any(key.startswith("predictor_") and "dropout" in key for key in guard)
    heavy = {**base, "scheduler_type": "warmup_constant_cosine",
             "predictor_resid_dropout_p": 0.25, "predictor_attn_dropout_p": 0.1}
    guard = trainer.build_resume_guard(heavy, 48, 473)
    assert guard["scheduler_decay_start_ratio"] == 0.8
    assert guard["predictor_resid_dropout_p"] == 0.25 and guard["predictor_attn_dropout_p"] == 0.1
    with pytest.raises(ValueError):  # old checkpoint without the fields cannot resume it
        trainer.validate_resume_guard(trainer.build_resume_guard(base, 48, 473), guard)
    with pytest.raises(ValueError):
        trainer.validate_resume_guard(guard, {**guard, "predictor_attn_dropout_p": 0.0})


# --------------------------------------------------------------- dropout
def test_dropout_override_applies_on_pretrained_load_and_export(tmp_path):
    from model.kronos import Kronos

    trainer = load_module("heavy_trainer_drop", FINETUNE / "train_predictor.py")
    runner = _runner()
    torch.manual_seed(0)
    parent = _tiny_kronos(aux=True, dropout=0.2)
    parent.save_pretrained(tmp_path / "parent")
    kwargs = json.loads((tmp_path / "parent" / "config.json").read_text())
    assert kwargs["resid_dropout_p"] == 0.2 and kwargs["attn_dropout_p"] == 0.0
    kwargs["use_beta_v21_auxiliary"] = False
    applied = trainer.apply_predictor_dropout_overrides(kwargs, {
        "predictor_resid_dropout_p": 0.25, "predictor_ffn_dropout_p": 0.25,
        "predictor_attn_dropout_p": 0.1, "predictor_token_dropout_p": 0.1,
    })
    assert applied["attn_dropout_p"] == {"parent": 0.0, "override": 0.1}
    model = Kronos.from_pretrained(str(tmp_path / "parent"), **kwargs)
    report = trainer.predictor_dropout_report(model)
    assert report["resid_dropout_p"] == [0.25] and report["ffn_dropout_p"] == [0.25]
    assert report["attn_dropout_p"] == [0.1] and report["token_dropout_p"] == [0.1]
    line = trainer.format_predictor_dropout_line(report)
    assert runner.check_dropout_line(line) is True
    assert runner.check_dropout_line(line.replace("attn=0.1000", "attn=0.0000")) is False
    assert runner.check_dropout_line("unrelated") is None
    parent_state = parent.state_dict()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, parent_state[name]), name  # weights untouched
    # Attention dropout really fires in train mode and is off in eval mode.
    block = model.transformer[0].self_attn
    x = torch.randn(2, 6, 16)
    model.train()
    assert not torch.allclose(block(x), block(x))
    model.eval()
    assert torch.allclose(block(x), block(x))
    # Exports carry the new dropout config.
    exported = trainer.model_export_config(model, {"use_beta_v21_auxiliary": False,
                                                   "num_sectors": 4, "context_layer": 1,
                                                   "use_size_percentile": True})
    assert exported["resid_dropout_p"] == 0.25 and exported["token_dropout_p"] == 0.1


def test_runner_parent_and_dropout_check(tmp_path):
    runner = _runner()
    torch.manual_seed(0)
    _tiny_kronos(aux=True, dropout=0.2).save_pretrained(tmp_path / "aux")
    env = {"KRONOS_NUM_SECTORS": "4", "KRONOS_NUM_SIZE_BUCKETS": "0",
           "KRONOS_CONTEXT_LAYER": "1", "KRONOS_USE_SIZE_PERCENTILE": "1",
           "KRONOS_SIZE_MLP_HIDDEN_DIM": "64", "KRONOS_USE_BETA_V21_AUXILIARY": "0",
           **{name: str(runner.DROPOUT[key]) for key, name in runner.DROPOUT_ENV.items()}}
    report = runner.check_parent_and_dropout(ROOT, tmp_path / "aux", env)
    assert sorted(report["dropped_aux"]) == [
        "barrier_head.bias", "barrier_head.weight", "return_head.bias", "return_head.weight"]
    assert report["module_dropout"]["attn_dropout_p"] == [0.1]
    assert report["loaded_not_equal_to_parent"] == []
    with pytest.raises(SystemExit):
        runner.check_parent_and_dropout(ROOT, tmp_path / "aux",
                                        {**env, "KRONOS_ATTN_DROPOUT_P": "0.0"})
    with pytest.raises(SystemExit):
        runner.check_parent_and_dropout(ROOT, tmp_path / "aux", {**env, "KRONOS_NUM_SECTORS": "7"})


# ------------------------------------------------- trainer: WSD + chunks
def _train(trainer, tmp_path, monkeypatch, save_dir, model, extra_env):
    train = _SegmentDataset(16)
    val = _SegmentDataset(8)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH, shuffle=False)
    monkeypatch.setattr(trainer, "create_dataloaders",
                        lambda config, rank, world_size: (train_loader, None, val_loader, train, val))
    env, _ = _env(tmp_path, _runner())
    for key, value in env.items():
        if key.startswith("KRONOS_") and key != "KRONOS_SNAPSHOT_DIR":
            monkeypatch.setenv(key, value)
    monkeypatch.delenv("KRONOS_SNAPSHOT_DIR", raising=False)
    for key, value in {"KRONOS_SNAPSHOT_EVERY_SEGMENTS": "1", "KRONOS_EPOCHS": "4",
                       "KRONOS_USE_AMP": "0", "KRONOS_SCHEDULER_WARMUP_RATIO": "0.25",
                       "KRONOS_SCHEDULER_DECAY_START_RATIO": "0.5",
                       "KRONOS_TRAIN_SIGNAL_END": "", "KRONOS_VAL_SIGNAL_START": "",
                       "KRONOS_VAL_SIGNAL_END": "", **extra_env}.items():
        monkeypatch.setenv(key, value)
    from config import Config

    config = Config().__dict__
    config.update({"batch_size": BATCH, "lookback_window": LOOKBACK, "predict_window": PREDICT,
                   "num_sectors": 4, "context_layer": 1, "use_amp": False,
                   "pretrained_predictor_path": str(tmp_path / "parent"),
                   "use_swanlab": False, "use_comet": False})
    save_dir.mkdir(exist_ok=True)
    return trainer.train_model(model, _tiny_tokenizer(), torch.device("cpu"), config,
                               str(save_dir), None, 0, 1)


def _lrs(save_dir):
    return {p.name: json.loads((p / "snapshot_metric.json").read_text())["learning_rates"][0]
            for p in sorted((save_dir / "snapshots").iterdir())}


def test_wsd_chunked_resume_matches_one_plan(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    trainer = load_module("heavy_trainer_chunks", FINETUNE / "train_predictor.py")
    torch.manual_seed(0)
    single = tmp_path / "single"
    _train(trainer, tmp_path, monkeypatch, single, _tiny_kronos(), {"KRONOS_MAX_SEGMENTS_PER_RUN": "4"})
    lrs = _lrs(single)
    # warmup = segment 1, hold through segment 2, cosine over 3-4.
    assert lrs["seg001"] == pytest.approx(1e-5) and lrs["seg002"] == pytest.approx(1e-5)
    assert lrs["seg003"] == pytest.approx(5.5e-6) and lrs["seg004"] == pytest.approx(1e-6)

    torch.manual_seed(0)
    chunked = tmp_path / "chunked"
    first = _train(trainer, tmp_path, monkeypatch, chunked, _tiny_kronos(),
                   {"KRONOS_MAX_SEGMENTS_PER_RUN": "2", "KRONOS_RESUME_TRAINING": "0"})
    assert first["completed_segments"] == 2
    torch.manual_seed(0)
    restarted = _tiny_kronos()
    noise = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in restarted.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=noise))
    second = _train(trainer, tmp_path, monkeypatch, chunked, restarted,
                    {"KRONOS_MAX_SEGMENTS_PER_RUN": "2", "KRONOS_RESUME_TRAINING": "1"})
    assert second["status"] == "completed"
    assert _lrs(chunked) == pytest.approx(lrs, rel=1e-9)
    wa = load_file(str(single / "snapshots/seg004/model.safetensors"))
    wb = load_file(str(chunked / "snapshots/seg004/model.safetensors"))
    for name in wa:
        assert torch.allclose(wa[name], wb[name], atol=1e-6), name
    # Changing the decay start cannot silently resume this run.
    with pytest.raises(ValueError, match="scheduler_decay_start_ratio"):
        _train(trainer, tmp_path, monkeypatch, chunked, _tiny_kronos(),
               {"KRONOS_MAX_SEGMENTS_PER_RUN": "2", "KRONOS_RESUME_TRAINING": "1",
                "KRONOS_EPOCHS": "4", "KRONOS_SCHEDULER_DECAY_START_RATIO": "0.75"})


# ------------------------------------------------ time-disjoint panel
def _frame(dates, seed, sector="S1"):
    rng = np.random.default_rng(seed)
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01, "low": close * 0.98, "close": close,
        "volume": rng.uniform(1e5, 2e5, len(dates)), "amount": rng.uniform(1e6, 2e6, len(dates)),
        "size_bucket": 3.0, "size_percentile": 0.4, "sector": sector,
    }, index=pd.DatetimeIndex(dates))


def _synthetic(tmp_path):
    dates = pd.bdate_range("2025-12-01", "2026-09-10")
    source = {f"S{i}": _frame(dates, i) for i in range(6)}
    holdout = {}
    for symbol in ("S1", "S3"):
        frame = source[symbol].loc[source[symbol].index <= "2026-07-17"].copy()
        frame.index.name = "datetime"
        holdout[symbol] = frame
    return source, holdout


def test_time_val_panel_restricts_truncates_and_checks_overlap(tmp_path):
    tv = load_module("tv_synth", FINETUNE / "build_time_disjoint_val_panel.py")
    source, holdout = _synthetic(tmp_path)
    panel, audit = tv.build_panel(source, holdout, expected_symbols=2)
    assert sorted(panel) == ["S1", "S3"] and audit["symbols"] == 2
    for frame in panel.values():
        assert frame.index.name == "datetime"
        assert frame.index.max() <= pd.Timestamp(tv.MAX_BAR_DATE)
        assert list(frame.columns) == tv.COLUMNS
    records = tv.build_records(panel)
    dates = sorted({r["asof_date"] for r in records})
    assert dates[0] == "2026-07-17" and dates[-1] == "2026-08-10"
    assert all(r["target_date"] <= "2026-08-24" for r in records)
    assert all(r["asof_date"] < tv.SEALED_SIGNAL_START for r in records)
    # Panel-vs-holdout disagreement aborts.
    bad = {k: v.copy() for k, v in source.items()}
    bad["S3"].iloc[50, bad["S3"].columns.get_loc("close")] += 1.0
    with pytest.raises(RuntimeError, match="disagrees"):
        tv.build_panel(bad, holdout, expected_symbols=2)
    with pytest.raises(RuntimeError, match="missing"):
        tv.build_panel({"S1": source["S1"]}, holdout, expected_symbols=2)
    # QlibDataset contract: reset_index() must yield a 'datetime' column.
    assert "datetime" in panel["S1"].reset_index().columns


def test_time_val_refuses_sealed_package(tmp_path):
    tv = load_module("tv_guard", FINETUNE / "build_time_disjoint_val_panel.py")
    sealed = {"name": "kronos_beta_v2_time_oos_through_20260903",
              "artifacts": {"panel_sha256": tv.SOURCE_PANEL_SHA256},
              "model_contract": {"lookback": 120, "predict": 10}}
    with pytest.raises(RuntimeError, match="sealed"):
        tv.check_source_manifest(sealed)
    good = {**sealed, "name": tv.SOURCE_MANIFEST_NAME}
    tv.check_source_manifest(good)
    with pytest.raises(RuntimeError, match="pinned"):
        tv.check_source_manifest({**good, "artifacts": {"panel_sha256": "0" * 64}})
    raw = tmp_path / "august_raw.csv"
    raw.write_text("x")
    with pytest.raises(RuntimeError, match="sealed"):
        tv.verify_inputs(raw, raw, raw)
    # Discovery only accepts the 2026-08-26 package subfolder, never the dataset root.
    root = tmp_path / "input" / "a-share-120d-temporal-symbol-holdout"
    root.mkdir(parents=True)
    (root / "evaluation_panel.pkl").write_bytes(b"sealed")
    (root / "evaluation_manifest.json").write_text(json.dumps(sealed))
    with pytest.raises(RuntimeError, match="Expected one"):
        tv.find_source(tmp_path / "input")
    package = root / tv.SOURCE_PACKAGE_DIR / "evaluation"
    package.mkdir(parents=True)
    (package / "evaluation_panel.pkl").write_bytes(b"old")
    (package / "evaluation_manifest.json").write_text(json.dumps(good))
    panel, manifest = tv.find_source(tmp_path / "input")
    assert panel == package / "evaluation_panel.pkl"


def test_runner_never_references_sealed_root_files():
    runner_text = RUNNER.read_text()
    assert "august_raw.csv" not in runner_text
    assert "evaluation_samples.jsonl" not in runner_text
    assert "evaluation_panel.pkl" not in runner_text.replace(
        '"evaluation_panel" in val_path', "")  # only the guard mentions it
    # The panel builder only globs the 2026-08-26 package subfolder.
    module = (FINETUNE / "build_time_disjoint_val_panel.py").read_text()
    code = module.split('"""', 2)[2]  # skip the module docstring
    assert "evaluation_samples.jsonl" not in code
    assert re.findall(r'glob\((.*?)\)', code) == [
        'f"**/{SOURCE_PACKAGE_DIR}/evaluation/evaluation_panel.pkl"',
        '"**/*temporal_symbol_validation_v1/processed_datasets/val_data.pkl"',
    ]
    runner = _runner()
    assert runner.TIME_VAL_SIGNAL_END < runner.SEALED_SIGNAL_START == "2026-08-11"


@pytest.mark.skipif(not (PROBE / "small01/evaluation_panel.pkl").is_file(),
                    reason="local copy of the 2026-08-26 package not present")
def test_real_time_val_panel_matches_runner_contract(tmp_path, monkeypatch):
    tv = load_module("tv_real", FINETUNE / "build_time_disjoint_val_panel.py")
    runner = _runner()
    manifest = tv.build(PROBE / "small01/evaluation_panel.pkl",
                        PROBE / "small01_evaluation_manifest.json",
                        PROBE / "tsv/val_data.pkl", tmp_path)
    assert manifest["windows"] == runner.TIME_VAL_WINDOWS == 8784
    assert manifest["signal_dates"] == runner.TIME_VAL_DATES == 17
    assert manifest["identities_sha256"] == runner.TIME_VAL_IDENTITIES_SHA256
    assert manifest["strict_subset_signal_dates"][0] == "2026-07-31"
    assert len(manifest["strict_subset_signal_dates"]) == 7
    # Training-time QlibDataset('val') sees exactly the same windows.
    monkeypatch.setenv("KRONOS_VAL_DATA_PATHS", str(tmp_path / manifest["panel_file"]))
    monkeypatch.setenv("KRONOS_VAL_SIGNAL_START", runner.TIME_VAL_SIGNAL_START)
    monkeypatch.setenv("KRONOS_VAL_SIGNAL_END", runner.TIME_VAL_SIGNAL_END)
    monkeypatch.setenv("KRONOS_METADATA_PATH", "")
    monkeypatch.setenv("KRONOS_USE_SECTOR_FEATURES", "0")
    monkeypatch.setenv("KRONOS_USE_SIZE_PERCENTILE", "1")
    monkeypatch.setenv("KRONOS_VALIDATION_SAMPLES", "0")
    monkeypatch.setenv("KRONOS_LOOKBACK_WINDOW", "120")
    monkeypatch.setenv("KRONOS_PREDICT_WINDOW", "10")
    from dataset import QlibDataset

    dataset = QlibDataset("val")
    assert dataset.total_samples == 8784
    identities = {(s, int(i)) for s, i in dataset.indices}
    records = tv.build_records(pickle.load(open(tmp_path / manifest["panel_file"], "rb")))
    assert identities == {(r["symbol"], r["start_index"]) for r in records}


# --------------------------------------------- driver val_contract (CPU e2e)
def test_driver_val_contract_end_to_end_cpu(tmp_path, monkeypatch):
    import val_gen_ic_driver as driver
    import build_time_disjoint_val_panel as tv
    from evaluate_beta_v21_val_gen_ic import build_val_records

    torch.manual_seed(0)
    model_dir, tok_dir = tmp_path / "ckpt", tmp_path / "tok"
    _tiny_kronos(num_sectors=86).save_pretrained(model_dir)
    _tiny_tokenizer().save_pretrained(tok_dir)
    dates = pd.bdate_range("2026-01-01", "2026-08-25")
    panel = {}
    for i in range(12):
        frame = _frame(dates, 100 + i, sector=f"SEC{i % 3:02d}")
        frame.index.name = "datetime"
        panel[f"X{i}"] = frame
    val_path = tmp_path / "tv.pkl"
    with val_path.open("wb") as handle:
        pickle.dump(panel, handle)
    sectors = tmp_path / "asset_metadata.csv"
    pd.DataFrame({"sector": [f"SEC{i:02d}" for i in range(86)]}).to_csv(sectors, index=False)
    records = build_val_records(panel, "2026-08-06", "2026-08-10")
    n_dates = len({r["asof_date"] for r in records})
    spec = {
        "kernel": "test", "output_dir": str(tmp_path / "eval"), "tokenizer_dir": str(tok_dir),
        "input_root": str(tmp_path / "nowhere"), "deadline": 1e12, "world": 1,
        "effective_batch": 64,
        "checkpoints": [{"label": "seg000_x", "path": str(model_dir), "segment": 0}],
        "val_contract": {
            "name": "unit_time_val", "val_data": str(val_path),
            "val_sha256": driver.__dict__.get("sha256_file", None) or "",
            "sector_metadata": str(sectors), "signal_start": "2026-08-06",
            "signal_end": "2026-08-10", "expected_samples": len(records),
            "expected_dates": n_dates, "identities_sha256": tv.identities_sha256(records),
            "subsample_dates": "all", "same_contract_as": "unit",
        },
    }
    from evaluate_beta_v21_val_gen_ic import sha256_file

    spec["val_contract"]["val_sha256"] = sha256_file(val_path)
    monkeypatch.setenv("KRONOS_VALGENIC_ALLOW_CPU", "1")
    monkeypatch.delenv("KRONOS_VALGENIC_LOCAL_TEST", raising=False)
    monkeypatch.setattr(driver, "POLL_SECONDS", 0.5)
    comparison = driver.run(spec)
    assert comparison["contract"]["val_dataset"] == "unit_time_val"
    assert comparison["contract"]["subsample_dates"] == sorted({r["asof_date"] for r in records})
    summary = json.loads((tmp_path / "eval/results/seg000_x_val_summary.json").read_text())
    assert summary["samples"] == len(records) == 12 * n_dates
    assert summary["signal_dates"] == n_dates
    # Contract drift aborts before any decode.
    spec["val_contract"]["expected_samples"] += 1
    spec["output_dir"] = str(tmp_path / "eval2")
    with pytest.raises(RuntimeError, match="drift"):
        driver.run(spec)


# ------------------------------------------------------- selection gate
def _summary(label, segment, by_date):
    ics = list(by_date.values())
    return {"label": label, "segment": segment,
            "return10d_rank_ic_daily": float(np.mean(ics)), "return10d_rank_ic_se": 0.01,
            "top_bottom_decile_return10d": 0.01, "weighted_forecast_loss_subsample": 2.3,
            "by_signal_date": [{"asof_date": d, "return10d_rank_ic": v} for d, v in by_date.items()]}


def test_selection_gate_and_paired_stats():
    runner = _runner()
    dates = [f"2026-07-{d:02d}" for d in range(10, 27)]  # 17 synthetic dates
    assert len(dates) == runner.TIME_VAL_DATES
    rng = np.random.default_rng(0)
    base = {d: 0.10 + 0.05 * rng.standard_normal() for d in dates}
    good = {d: v + 0.02 + 0.002 * rng.standard_normal() for d, v in base.items()}
    flat = {d: v + 0.001 * (-1) ** i for i, (d, v) in enumerate(base.items())}
    summaries = {
        runner.SEG0_LABEL: _summary(runner.SEG0_LABEL, 0, base),
        "ref_pilot_seg009": _summary("ref_pilot_seg009", None, {d: v - 0.01 for d, v in base.items()}),
        "heavyreg_seg024": _summary("heavyreg_seg024", 24, good),
        "heavyreg_seg048": _summary("heavyreg_seg048", 48, flat),
    }
    selection = runner.build_selection(summaries)
    gate = selection["gate"]
    assert gate["c_star"] == "heavyreg_seg024"
    assert gate["paired"]["n"] == 17 and gate["paired"]["wins"] == 17
    assert gate["t_pass"] and gate["wins_pass"]
    assert gate["endpoint_label"] == "heavyreg_seg048"
    assert gate["endpoint_delta"] == pytest.approx(0.001 / 17, abs=1e-12)
    assert gate["endpoint_pass"] and gate["complete"] and gate["pass"]
    assert selection["reference_seg9"]["agrees_with_sealed_oos"] is True
    roles = {row["label"]: row["role"] for row in selection["checkpoints"]}
    assert roles == {runner.SEG0_LABEL: "parent", "ref_pilot_seg009": "reference",
                     "heavyreg_seg024": "candidate", "heavyreg_seg048": "candidate"}
    # Without the endpoint the gate is incomplete -> no pass.
    del summaries["heavyreg_seg048"]
    assert runner.build_selection(summaries)["gate"]["pass"] is False
    stats = runner.paired_stats({"a": 1.0, "b": 2.0, "c": 4.0}, {"a": 0.0, "b": 1.0, "c": 1.0})
    assert stats["n"] == 3 and stats["mean_delta"] == pytest.approx(5 / 3)
    sd = math.sqrt(((1 - 5 / 3) ** 2 * 2 + (3 - 5 / 3) ** 2) / 2)
    assert stats["t"] == pytest.approx((5 / 3) / (sd / math.sqrt(3)))


def test_scorer_uses_time_val_contract_and_is_cumulative(tmp_path, monkeypatch):
    runner = _runner()
    out = tmp_path / "out"
    for seg in (4, 8):
        snap = out / "snapshots" / f"seg{seg:03d}"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"w%d" % seg)
        (snap / "config.json").write_text("{}")
        (snap / "snapshot_metric.json").write_text(json.dumps({"segment": seg}))
    specs, logged = [], []
    dates = [f"2026-07-{d:02d}" for d in range(10, 27)]

    def fake_run(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        specs.append(spec)
        results = Path(spec["output_dir"]) / "results"
        results.mkdir(parents=True, exist_ok=True)
        for item in spec["checkpoints"]:
            bump = 0.001 * int(item["segment"] or 0)
            (results / f"{item['label']}_val_summary.json").write_text(json.dumps(
                _summary(item["label"], item["segment"], {d: 0.1 + bump for d in dates})))
        import subprocess

        return subprocess.CompletedProcess(command, 0)

    class _Run:
        def log(self, payload, step=None):
            logged.append((step, payload))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    time_val = {"panel_path": "/tmp/tv.pkl", "panel_sha256": "abc"}
    scorer = runner.Scorer(ROOT, out, tmp_path / "tok", tmp_path, time_val,
                           tmp_path / "meta.csv", runner.SafeRun(_Run()))
    seg0 = {"label": runner.SEG0_LABEL, "path": str(tmp_path), "sha256": "x", "segment": 0}
    scorer.score([seg0] + scorer.snapshot_items())
    contract = specs[-1]["val_contract"]
    assert contract["val_data"] == "/tmp/tv.pkl" and contract["val_sha256"] == "abc"
    assert contract["subsample_dates"] == "all" and contract["expected_samples"] == 8784
    assert contract["signal_start"] == "2026-07-17" and contract["signal_end"] == "2026-08-10"
    assert [c["label"] for c in specs[-1]["checkpoints"]] == [
        runner.SEG0_LABEL, "heavyreg_seg004", "heavyreg_seg008"]
    assert scorer.snapshot_items() == []
    selection = json.loads((out / "heavy_reg_selection.json").read_text())
    assert selection["gate"]["c_star"] == "heavyreg_seg008"
    steps = sorted(step for step, payload in logged if "valgenic/return10d_ic_daily" in payload)
    assert steps == [0, 4 * 313, 8 * 313]


# --------------------------------------------------------------- staging
def test_staged_kernel_metadata_and_runner():
    runner = _runner()
    meta = json.loads((STAGING / "kernel-metadata.json").read_text())
    assert meta["id"] == runner.KERNEL_ID == "wynstonliu/kronos-beta-v21-heavy-reg-time-val"
    assert meta["is_private"] is True and meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["docker_image"].endswith(DOCKER_SHA)
    assert meta["dataset_sources"] == list(runner.DATASET_SOURCES)
    assert meta["code_file"] == RUNNER.name
    strip = lambda text: re.sub(r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""', "ARCHIVE",
                                text, flags=re.DOTALL)
    staged_text = (STAGING / RUNNER.name).read_text()
    assert 'EMBEDDED_KRONOS_ARCHIVE_B64 = """\nPLACEHOLDER\n"""' not in staged_text
    assert strip(staged_text) == strip(RUNNER.read_text())


def test_staged_archive_matches_current_sources():
    import base64
    import io
    import tarfile

    builder = load_module("heavy_builder", BUILDER)
    assert "finetune/build_time_disjoint_val_panel.py" in builder.FILES
    assert "finetune/val_gen_ic_driver.py" in builder.FILES
    staged_text = (STAGING / RUNNER.name).read_text()
    payload = re.search(r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n(.*?)\n"""', staged_text,
                        flags=re.DOTALL).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as archive:
        for relative in builder.FILES:
            member = archive.extractfile(f"Kronos/{relative}")
            assert member.read() == (ROOT / relative).read_bytes(), relative
