"""Forecast cosine pilot: parent selection, aux-key loading, chunked resume,
snapshot hook, between-chunk scoring bookkeeping (tiny CPU models)."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
RUNNER = FINETUNE / "kaggle_beta_v21_c1_forecast_cosine_pilot.py"
BUILDER = FINETUNE / "build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel.py"
STAGING_SEG155 = FINETUNE / "kaggle_beta_v21_c1_forecast_cosine_pilot_kernel"
STAGING_BEST475 = FINETUNE / "kaggle_beta_v21_c1_forecast_cosine_pilot_best475_kernel"
DOCKER_SHA = "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_eval_only_mode import _TinyDataset, BATCH, LOOKBACK, PREDICT  # noqa: E402


class _SegmentDataset(_TinyDataset):
    selection_report = {}

    def set_epoch_seed(self, epoch):
        self.epoch = epoch


def load_module(name, path, parent=None):
    sys.dont_write_bytecode = True
    if parent is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    source = Path(path).read_text()
    source = re.sub(r'^PILOT_PARENT = "[a-z0-9]+"$', f'PILOT_PARENT = "{parent}"',
                    source, count=1, flags=re.MULTILINE)
    module = type(sys)(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _pilot_env(tmp_path, parent=None):
    pilot = load_module(f"pilot_runner_{parent}", RUNNER, parent)
    data = tmp_path / "data"
    (data / "processed_datasets").mkdir(parents=True, exist_ok=True)
    (data / "data_manifest.json").write_text("{}")
    out = tmp_path / "models" / pilot.OUTPUT_NAME
    env = pilot.build_environment(data, tmp_path / "p", tmp_path / "t", out, ROOT)
    return pilot, env, out


def _tiny_kronos(aux):
    from model.kronos import Kronos

    return Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=16, n_heads=2, ff_dim=32,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True, num_sectors=4, context_layer=1,
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


# ------------------------------------------------------------- recipe / parents
@pytest.mark.parametrize("parent", ["best475", "seg155"])
def test_pilot_recipe_is_forecast_only_uniform_cosine(tmp_path, monkeypatch, parent):
    pilot, env, _ = _pilot_env(tmp_path, parent)
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
    assert config["max_segments_per_run"] == 3  # chunked: score between chunks
    assert config["predictor_min_learning_rate"] == pytest.approx(1e-6)
    assert config["keep_existing_best"] is (parent == "seg155")


def test_parent_profiles_have_distinct_slugs_outputs_and_swanlab():
    best = load_module("p_best", RUNNER, "best475")
    seg = load_module("p_seg", RUNNER, "seg155")
    assert best.PILOT_PARENT == "best475"  # source default
    assert best.KERNEL_ID == "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475"
    assert seg.KERNEL_ID == "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot"
    assert best.OUTPUT_NAME != seg.OUTPUT_NAME
    assert best.SWANLAB_RUN_ID == best.OUTPUT_NAME
    assert best.SWANLAB_URL.endswith("/runs/beta_v2_1_c1_forecast_cosine_pilot_best475")
    assert best.PROFILE["runtime_dir"] != seg.PROFILE["runtime_dir"]
    # Best@475 = val-gen-ic kernel's beta_v21_release_best475 (same repo + SHA).
    valgenic = (FINETUNE / "kaggle_beta_v21_c1_val_gen_ic.py").read_text()
    assert best.MODEL_REPO in valgenic and best.EXPECTED_PARENT_MODEL_SHA256 in valgenic
    assert best.PARENT_LABEL == "beta_v21_release_best475"
    assert best.SEG0_LABEL == "seg000_beta_v21_release_best475"


def test_staged_metadata_both_parents_dual_t4_pinned():
    best = json.loads((STAGING_BEST475 / "kernel-metadata.json").read_text())
    seg = json.loads((STAGING_SEG155 / "kernel-metadata.json").read_text())
    assert best["id"] == "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475"
    assert seg["id"] == "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot"
    for meta in (best, seg):
        assert meta["enable_tpu"] is False and meta["machine_shape"] == "NvidiaTeslaT4"
        assert meta["docker_image"].endswith(DOCKER_SHA)
        assert "luckfu/a-share-120d-temporal-symbol-holdout" in meta["dataset_sources"]
    assert "luckfu/kronos-beta-v21-c1-seg155-forecast-best" in seg["dataset_sources"]
    assert "luckfu/kronos-beta-v21-c1-seg155-forecast-best" not in best["dataset_sources"]


def test_staged_runners_only_differ_in_parent_pin_and_archive():
    strip = lambda text: re.sub(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""', "ARCHIVE", text, flags=re.DOTALL
    )
    source = strip(RUNNER.read_text()).splitlines()
    for staging, parent in ((STAGING_BEST475, "best475"), (STAGING_SEG155, "seg155")):
        staged_text = (staging / RUNNER.name).read_text()
        assert 'EMBEDDED_KRONOS_ARCHIVE_B64 = """\nPLACEHOLDER\n"""' not in staged_text
        staged = strip(staged_text).splitlines()
        assert len(staged) == len(source)
        diff = {(a, b) for a, b in zip(source, staged) if a != b}
        expected = set() if parent == "best475" else {
            ('PILOT_PARENT = "best475"', 'PILOT_PARENT = "seg155"')
        }
        assert diff == expected


def test_builder_stage_source_pins_parent():
    builder = load_module("pilot_builder", BUILDER)
    staged = builder.stage_source(RUNNER.read_text(), "seg155", "QUJD")
    assert '\nPILOT_PARENT = "seg155"\n' in staged
    assert 'EMBEDDED_KRONOS_ARCHIVE_B64 = """\nQUJD\n"""' in staged
    assert builder.build_metadata("best475")["id"].endswith("-best475")


# ---------------------------------------------------- aux-head robust loading
def test_classify_parent_keys_tolerates_aux_only():
    pilot = load_module("p_cls", RUNNER, "best475")
    parent = {"a.weight": [2, 2], "return_head.weight": [4, 8], "barrier_head.bias": [3]}
    model = {"a.weight": [2, 2]}
    diff = pilot.classify_parent_keys(parent, model)
    assert diff["dropped_aux"] == ["barrier_head.bias", "return_head.weight"]
    assert not diff["bad_missing"] and not diff["bad_unexpected"]
    reverse = pilot.classify_parent_keys(model, parent)
    assert reverse["missing_aux"] == ["barrier_head.bias", "return_head.weight"]
    bad = pilot.classify_parent_keys({"a.weight": [2, 3], "x": [1]}, {"a.weight": [2, 2]})
    assert bad["shape_mismatch"] == ["a.weight"] and bad["bad_unexpected"] == ["x"]


def test_aux_parent_loads_into_forecast_only_model_and_back(tmp_path):
    from model.kronos import Kronos

    torch.manual_seed(0)
    aux_parent = _tiny_kronos(aux=True)
    aux_parent.save_pretrained(tmp_path / "aux")
    config = json.loads((tmp_path / "aux" / "config.json").read_text())
    assert config["use_beta_v21_auxiliary"] is True
    # Train path: config + override aux off (as train_predictor main()).
    kwargs = {**config, "use_beta_v21_auxiliary": False}
    plain = Kronos.from_pretrained(str(tmp_path / "aux"), **kwargs)
    assert plain.return_head is None and plain.barrier_head is None
    parent_state = aux_parent.state_dict()
    for name, tensor in plain.state_dict().items():
        assert torch.equal(tensor, parent_state[name]), name
    # Reverse: forecast-only parent into an aux model keeps fresh heads.
    plain.save_pretrained(tmp_path / "plain")
    back = Kronos.from_pretrained(str(tmp_path / "plain"),
                                  **{**config, "use_beta_v21_auxiliary": True})
    assert back.return_head is not None
    # Eval-worker path: no kwargs, config.json drives construction.
    reloaded = Kronos.from_pretrained(str(tmp_path / "plain"), local_files_only=True)
    assert reloaded.return_head is None


def test_check_parent_state_dict_drops_aux_and_rejects_mismatch(tmp_path):
    pilot = load_module("p_chk", RUNNER, "best475")
    torch.manual_seed(0)
    _tiny_kronos(aux=True).save_pretrained(tmp_path / "aux")
    env = {"KRONOS_NUM_SECTORS": "4", "KRONOS_NUM_SIZE_BUCKETS": "0",
           "KRONOS_CONTEXT_LAYER": "1", "KRONOS_USE_SIZE_PERCENTILE": "1",
           "KRONOS_SIZE_MLP_HIDDEN_DIM": "64", "KRONOS_USE_BETA_V21_AUXILIARY": "0"}
    report = pilot.check_parent_state_dict(ROOT, tmp_path / "aux", env)
    assert sorted(report["dropped_aux"]) == [
        "barrier_head.bias", "barrier_head.weight", "return_head.bias", "return_head.weight"]
    assert report["parent_config_aux"] is True and report["model_aux"] is False
    with pytest.raises(SystemExit):
        pilot.check_parent_state_dict(ROOT, tmp_path / "aux",
                                      {**env, "KRONOS_NUM_SECTORS": "7"})


# -------------------------------------------------- trainer: snapshots + chunks
def _train(trainer, tmp_path, monkeypatch, save_dir, model, extra_env):
    train = _SegmentDataset(16)
    val = _SegmentDataset(8)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH, shuffle=False)
    monkeypatch.setattr(
        trainer, "create_dataloaders",
        lambda config, rank, world_size: (train_loader, None, val_loader, train, val),
    )
    _, env, _ = _pilot_env(tmp_path, "best475")
    for key, value in env.items():
        if key.startswith("KRONOS_") and key != "KRONOS_SNAPSHOT_DIR":
            monkeypatch.setenv(key, value)
    monkeypatch.delenv("KRONOS_SNAPSHOT_DIR", raising=False)
    monkeypatch.setenv("KRONOS_SNAPSHOT_EVERY_SEGMENTS", "2")
    monkeypatch.setenv("KRONOS_KEEP_EXISTING_BEST", "0")
    monkeypatch.setenv("KRONOS_EPOCHS", "4")
    monkeypatch.setenv("KRONOS_USE_AMP", "0")
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)
    from config import Config

    config = Config().__dict__
    config.update({
        "batch_size": BATCH, "lookback_window": LOOKBACK, "predict_window": PREDICT,
        "num_sectors": 4, "context_layer": 1, "use_amp": False,
        "pretrained_predictor_path": str(tmp_path / "parent"),
        "use_swanlab": False, "use_comet": False,
    })
    save_dir.mkdir(exist_ok=True)
    return trainer.train_model(model, _tiny_tokenizer(), torch.device("cpu"), config,
                               str(save_dir), None, 0, 1)


def test_snapshot_hook_saves_every_n_segments(tmp_path, monkeypatch):
    trainer = load_module("pilot_trainer", FINETUNE / "train_predictor.py")
    torch.manual_seed(0)
    save_dir = tmp_path / "out"
    _train(trainer, tmp_path, monkeypatch, save_dir, _tiny_kronos(False),
           {"KRONOS_MAX_SEGMENTS_PER_RUN": "4"})
    snaps = sorted(p.name for p in (save_dir / "snapshots").iterdir())
    assert snaps == ["seg002", "seg004"]
    metric = json.loads((save_dir / "snapshots/seg004/snapshot_metric.json").read_text())
    assert metric["segment"] == 4
    assert (save_dir / "snapshots/seg004/model.safetensors").is_file()
    assert metric["learning_rates"][0] == pytest.approx(1e-6, rel=1e-3)


def test_chunked_resume_matches_one_cosine_plan(tmp_path, monkeypatch):
    """2+2 segments with last_state resume == one 4-segment cosine plan."""
    from safetensors.torch import load_file

    trainer = load_module("pilot_trainer_chunks", FINETUNE / "train_predictor.py")
    torch.manual_seed(0)
    single = tmp_path / "single"
    _train(trainer, tmp_path, monkeypatch, single, _tiny_kronos(False),
           {"KRONOS_MAX_SEGMENTS_PER_RUN": "4"})

    torch.manual_seed(0)
    chunked = tmp_path / "chunked"
    first = _train(trainer, tmp_path, monkeypatch, chunked, _tiny_kronos(False),
                   {"KRONOS_MAX_SEGMENTS_PER_RUN": "2", "KRONOS_RESUME_TRAINING": "0"})
    assert first["stop_reason"] == "segment_limit" and first["completed_segments"] == 2
    assert [p.name for p in (chunked / "snapshots").iterdir()] == ["seg002"]
    # A restarted process builds the predictor from the parent again; perturb it so
    # the test proves last_state.pt (not the constructor) supplies the weights.
    # (Same global-seed position as the other runs: _TinyDataset uses torch.rand.)
    torch.manual_seed(0)
    restarted = _tiny_kronos(False)
    noise = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in restarted.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=noise))
    second = _train(trainer, tmp_path, monkeypatch, chunked, restarted,
                    {"KRONOS_MAX_SEGMENTS_PER_RUN": "2", "KRONOS_RESUME_TRAINING": "1"})
    assert second["status"] == "completed"
    assert sorted(p.name for p in (chunked / "snapshots").iterdir()) == ["seg002", "seg004"]
    for seg in ("seg002", "seg004"):
        a = json.loads((single / "snapshots" / seg / "snapshot_metric.json").read_text())
        b = json.loads((chunked / "snapshots" / seg / "snapshot_metric.json").read_text())
        assert b["learning_rates"] == pytest.approx(a["learning_rates"], rel=1e-9)
    wa = load_file(str(single / "snapshots/seg004/model.safetensors"))
    wb = load_file(str(chunked / "snapshots/seg004/model.safetensors"))
    for name in wa:
        assert torch.allclose(wa[name], wb[name], atol=1e-6), name


# ----------------------------------------------------- scoring bookkeeping
def _summary(label, segment, ic, wfl, full=None):
    return {"label": label, "segment": segment, "return10d_rank_ic_daily": ic,
            "return10d_rank_ic_se": 0.025, "weighted_forecast_loss_subsample": wfl,
            "full_val_weighted_forecast_loss": full}


def test_build_selection_picks_ic_not_wfl():
    pilot = load_module("p_sel", RUNNER, "best475")
    selection = pilot.build_selection([
        _summary(pilot.SEG0_LABEL, 0, 0.3145, 2.3205),
        _summary("pilot_seg003", 3, 0.3200, 2.3300, full=2.3300),
        _summary("pilot_seg006", 6, 0.3100, 2.3000, full=2.3100),
    ])
    assert selection["best_by_ic"] == "pilot_seg003"
    assert selection["best_is_seg0"] is False
    rows = {row["label"]: row for row in selection["candidates"]}
    assert rows["pilot_seg003"]["wfl_full_val_above_red_line"] is True
    assert rows["pilot_seg006"]["wfl_full_val_above_red_line"] is False
    assert rows["pilot_seg003"]["delta_ic_vs_seg0"] == pytest.approx(0.0055)
    assert selection["wfl_red_line_full_val"] == pytest.approx(2.32736787)


def test_scorer_is_cumulative_and_logs_each_checkpoint(tmp_path, monkeypatch):
    pilot = load_module("p_scorer", RUNNER, "best475")
    logged = []

    class _Run:
        def log(self, payload, step=None):
            logged.append((step, payload))

    out = tmp_path / "out"
    snap = out / "snapshots" / "seg003"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"w3")
    (snap / "config.json").write_text("{}")
    (snap / "snapshot_metric.json").write_text(json.dumps(
        {"segment": 3, "weighted_forecast_loss": 2.33, "learning_rates": [8.68e-6]}))
    specs = []

    def fake_run(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        specs.append([c["label"] for c in spec["checkpoints"]])
        results = Path(spec["output_dir"]) / "results"
        results.mkdir(parents=True, exist_ok=True)
        for item in spec["checkpoints"]:
            ic = 0.31 + 0.001 * int(item["segment"])
            (results / f"{item['label']}_val_summary.json").write_text(json.dumps(
                _summary(item["label"], item["segment"], ic, 2.32,
                         item.get("full_val_wfl"))))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    scorer = pilot.PilotScorer(ROOT, out, tmp_path / "tok", tmp_path, pilot.SafeRun(_Run()))
    scorer.total_steps = 313
    seg0 = {"label": pilot.SEG0_LABEL, "path": str(tmp_path), "sha256": "x",
            "segment": 0, "full_val_wfl": None, "note": "parent"}
    scorer.score([seg0] + scorer.new_snapshots())
    assert specs[-1] == [pilot.SEG0_LABEL, "pilot_seg003"]
    snap6 = out / "snapshots" / "seg006"
    snap6.mkdir()
    (snap6 / "model.safetensors").write_bytes(b"w6")
    (snap6 / "config.json").write_text("{}")
    new = scorer.new_snapshots()
    assert [item["label"] for item in new] == ["pilot_seg006"]
    scorer.score(new)
    assert specs[-1] == [pilot.SEG0_LABEL, "pilot_seg003", "pilot_seg006"]
    selection = json.loads((out / "pilot_selection.json").read_text())
    assert selection["best_by_ic"] == "pilot_seg006"
    steps = [step for step, payload in logged if "valgenic/return10d_ic_daily" in payload]
    assert steps == [0, 3 * 313, 6 * 313]
