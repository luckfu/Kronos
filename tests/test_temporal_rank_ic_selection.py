"""Temporal Rank IC checkpoint selection. No symbol-holdout fallback."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from temporal_rank_ic_selection import (
    DEFAULT_INTERVAL_SEGMENTS,
    MIN_NAMES_PER_DATE,
    N_PATHS,
    PATH_SEED,
    PROTOCOL_ID,
    SELECTION_METRIC,
    TemporalRankICSelectionError,
    assert_frozen_manifest_present,
    build_evaluation_rows,
    build_rank_ic_best_metric,
    checkpoint_should_update,
    daily_d10_rank_ic_mean,
    isolated_path_seed,
    load_temporal_selection_manifest,
    manifest_sha256,
    median_path_d10_return,
    rank_ic_is_improvement,
    reconcile_resumed_rank_ic_threshold,
    require_pinned_panel_sha,
    resolve_panel_dataset_slug,
    resolve_training_selection_inputs,
    sha256_file,
    should_run_temporal_rank_ic,
    temporal_rank_ic_due,
    validate_manifest_calendar,
)
from train_predictor import (
    selection_metric_higher_is_better,
    should_run_large_validation,
)


DATES = ["2026-07-06", "2026-07-07", "2026-07-08"]


def _manifest_body(dates, panel_sha, **overrides):
    body = {
        "protocol_id": PROTOCOL_ID,
        "split_kind": "temporal",
        "horizon": "D10",
        "n_paths": N_PATHS,
        "path_aggregation": "median",
        "temperature": 0.65,
        "top_p": 0.8,
        "top_k": 0,
        "path_seed": 20260906,
        "lookback": 120,
        "predict": 10,
        "train_time_upper_bound": "2026-07-03",
        "signal_dates": list(dates),
        "sha256": manifest_sha256(dates),
        "panel_sha256": panel_sha,
        "panel_filename": "temporal_rank_ic_panel.pkl",
    }
    body.update(overrides)
    return body


def _write_manifest(directory, dates=DATES, panel_name="temporal_rank_ic_panel.pkl", **overrides):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    panel_path = directory / panel_name
    panel_path.write_bytes(b"temporal-panel-bytes")
    body = _manifest_body(dates, sha256_file(panel_path), **overrides)
    path = directory / "selection_window.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path, panel_path, body


def test_sparse_schedule_matches_large_validation():
    assert DEFAULT_INTERVAL_SEGMENTS == 10
    for segment in range(1, 521):
        assert should_run_temporal_rank_ic(segment, 520, 10) == (
            should_run_large_validation(segment, 520, 10)
        )


def test_first_and_last_of_a_run_are_scored():
    assert temporal_rank_ic_due(1, 520, 10)
    assert temporal_rank_ic_due(10, 520, 10)
    assert temporal_rank_ic_due(520, 520, 10)
    assert temporal_rank_ic_due(11, 520, 10) is False
    assert temporal_rank_ic_due(41, 520, 10, first_of_run=True)
    assert temporal_rank_ic_due(7, 520, 10, force_final=True)
    with pytest.raises(ValueError):
        should_run_temporal_rank_ic(1, 10, 0)


def test_higher_rank_ic_wins_and_near_ties_keep_the_earlier_score():
    assert rank_ic_is_improvement(0.2, None)
    assert rank_ic_is_improvement(0.2, float("-inf"))
    assert rank_ic_is_improvement(0.20 + 1e-6, 0.20)
    assert rank_ic_is_improvement(0.20 + 1e-7, 0.20) is False
    assert rank_ic_is_improvement(0.19, 0.20) is False
    assert rank_ic_is_improvement(float("nan"), 0.1) is False
    assert selection_metric_higher_is_better(SELECTION_METRIC)


def test_val_loss_does_not_select_when_rank_ic_mode_is_on():
    assert checkpoint_should_update(
        protocol_enabled=True,
        rank_ic_due=False,
        rank_ic_mean=None,
        incumbent_rank_ic=0.1,
        val_loss_improved=True,
    ) is False
    assert checkpoint_should_update(
        protocol_enabled=True,
        rank_ic_due=True,
        rank_ic_mean=0.2,
        incumbent_rank_ic=0.1,
        val_loss_improved=False,
    ) is True
    assert checkpoint_should_update(
        protocol_enabled=True,
        rank_ic_due=True,
        rank_ic_mean=0.1000000001,
        incumbent_rank_ic=0.1,
        val_loss_improved=True,
    ) is False
    assert checkpoint_should_update(
        protocol_enabled=False,
        rank_ic_due=False,
        rank_ic_mean=None,
        incumbent_rank_ic=None,
        val_loss_improved=True,
    ) is True
    assert checkpoint_should_update(
        protocol_enabled=False,
        rank_ic_due=True,
        rank_ic_mean=0.9,
        incumbent_rank_ic=0.1,
        val_loss_improved=False,
    ) is False


def test_daily_mean_is_the_arithmetic_mean_of_same_day_spearman():
    rows = []
    for day, actuals in (
        ("2026-07-06", [1.0, 2.0, 3.0]),
        ("2026-07-07", [3.0, 2.0, 1.0]),
    ):
        for predicted, actual in zip([1.0, 2.0, 3.0], actuals):
            rows.append({
                "asof_date": day, "predicted": predicted, "actual": actual,
            })
    summary = daily_d10_rank_ic_mean(rows, min_names=3)
    assert summary["n_dates"] == 2
    assert summary["daily_rank_ic_mean"] == pytest.approx(0.0)
    assert summary["per_date"]["2026-07-06"] == pytest.approx(1.0)
    assert summary["per_date"]["2026-07-07"] == pytest.approx(-1.0)
    predicted = pd.Series([1.0, 1.0, 2.0])
    actual = pd.Series([1.0, 3.0, 2.0])
    # Same definition as finetune/small_validation_rank_ic: Pearson of average ranks.
    pandas_score = float(predicted.rank().corr(actual.rank()))
    tied = daily_d10_rank_ic_mean(
        [
            {"asof_date": "2026-07-06", "predicted": pred, "actual": act}
            for pred, act in zip(predicted, actual)
        ],
        min_names=3,
    )
    assert tied["per_date"]["2026-07-06"] == pytest.approx(pandas_score)


def test_short_cross_section_fails_closed():
    rows = [
        {"asof_date": "2026-07-06", "predicted": 1.0, "actual": 1.0},
        {"asof_date": "2026-07-06", "predicted": 2.0, "actual": 2.0},
    ]
    with pytest.raises(TemporalRankICSelectionError, match="requires"):
        daily_d10_rank_ic_mean(rows, min_names=MIN_NAMES_PER_DATE)


def test_path_median_matches_numpy_even_count():
    closes = [float(value) for value in range(1, 17)]
    last = 10.0
    expected = float(np.median(np.asarray(closes) / last - 1.0))
    assert median_path_d10_return(closes, last) == pytest.approx(expected)
    with pytest.raises(TemporalRankICSelectionError):
        median_path_d10_return(closes[:-1], last)


def test_best_metric_document_carries_protocol_fields():
    document = build_rank_ic_best_metric(
        daily_rank_ic_mean=0.123456789,
        segment=10,
        step=6400,
        n_dates=12,
        manifest_sha256="abc",
        objective_loss=2.2,
        forecast_loss=2.1,
        history_loss=1.5,
        full_sequence_loss=2.4,
    )
    assert document["selection_protocol"] == PROTOCOL_ID
    assert document["selection_metric"] == SELECTION_METRIC
    assert document["daily_rank_ic_mean"] == pytest.approx(0.123456789)
    assert document["selection_loss"] == document["daily_rank_ic_mean"]
    assert document["segment"] == 10
    assert document["step"] == 6400
    assert document["n_dates"] == 12
    assert document["n_paths"] == 16
    assert document["path_agg"] == "median"
    assert document["manifest_sha256"] == "abc"
    assert document["val_loss_selects_best"] is False


def test_resume_ignores_a_val_loss_best_and_keeps_an_exported_rank_ic():
    foreign = {"selection_metric": "forecast", "selection_loss": 2.26, "objective_loss": 2.26}
    assert reconcile_resumed_rank_ic_threshold("forecast", 2.26, foreign) == float("-inf")
    saved = {
        "selection_protocol": PROTOCOL_ID,
        "selection_metric": SELECTION_METRIC,
        "daily_rank_ic_mean": 0.22,
        "selection_loss": 0.22,
    }
    assert reconcile_resumed_rank_ic_threshold("", 2.26, saved) == pytest.approx(0.22)
    assert reconcile_resumed_rank_ic_threshold(
        PROTOCOL_ID, 0.10, saved,
    ) == pytest.approx(0.22)
    with pytest.raises(TemporalRankICSelectionError, match="better temporal Rank IC"):
        reconcile_resumed_rank_ic_threshold(PROTOCOL_ID, 0.30, saved)


def test_manifest_roundtrip_and_fail_closed(tmp_path):
    path, panel, _body = _write_manifest(tmp_path)
    loaded = load_temporal_selection_manifest(
        path, panel_path=panel, val_data_paths=[tmp_path / "other_val.pkl"],
    )
    assert loaded["signal_dates"] == DATES
    assert loaded["sha256"] == manifest_sha256(DATES)

    missing = tmp_path / "missing.json"
    with pytest.raises(TemporalRankICSelectionError, match="missing"):
        load_temporal_selection_manifest(missing, panel_path=panel)

    tampered = json.loads(path.read_text())
    tampered["signal_dates"] = ["2026-07-06", "2026-07-08", "2026-07-09"]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(TemporalRankICSelectionError, match="SHA"):
        load_temporal_selection_manifest(path, panel_path=panel)

    forbidden, forbidden_panel, _body = _write_manifest(
        tmp_path / "forbidden", dates=["2026-09-18", "2026-09-21"],
        train_time_upper_bound="2026-07-03",
    )
    with pytest.raises(TemporalRankICSelectionError, match="sealed_8d"):
        load_temporal_selection_manifest(forbidden, panel_path=forbidden_panel)

    late_train, late_panel, _body = _write_manifest(
        tmp_path / "late", train_time_upper_bound="2026-07-06",
    )
    with pytest.raises(TemporalRankICSelectionError, match="strictly earlier"):
        load_temporal_selection_manifest(late_train, panel_path=late_panel)

    holdout_manifest, holdout_panel, _body = _write_manifest(
        tmp_path / "holdout", source="a-share-120d-temporal-symbol-holdout",
    )
    with pytest.raises(TemporalRankICSelectionError, match="symbol-holdout"):
        load_temporal_selection_manifest(holdout_manifest, panel_path=holdout_panel)

    named = tmp_path / "val_data.pkl"
    named.write_bytes(b"temporal-panel-bytes")
    plain, _panel, body = _write_manifest(tmp_path / "plain")
    body["panel_sha256"] = sha256_file(named)
    plain.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(TemporalRankICSelectionError, match="val_data.pkl"):
        load_temporal_selection_manifest(plain, panel_path=named)

    same = tmp_path / "same"
    same.mkdir()
    manifest_path, panel_path, _body = _write_manifest(same)
    with pytest.raises(TemporalRankICSelectionError, match="val_data"):
        load_temporal_selection_manifest(
            manifest_path, panel_path=panel_path, val_data_paths=[panel_path],
        )


PINNED_PANEL_SHA256 = (
    "8539246be495db60ac89f23c4a79f19abae730af6d2a23a3847d477abf376784"
)


def test_frozen_repo_calendar_passes_with_pinned_panel_sha():
    from ashare_sse_calendar_2026 import (
        SELECTION_HOLIDAY_WEEKDAYS,
        is_sse_session,
        selection_window_sessions,
        sessions_between,
    )

    path = assert_frozen_manifest_present(ROOT)
    document = json.loads(path.read_text(encoding="utf-8"))
    calendar = validate_manifest_calendar(document)
    expected = selection_window_sessions()
    assert calendar["signal_dates"] == expected
    assert document["signal_dates"] == expected
    assert len(expected) == 60
    assert document["n_dates"] == 60
    assert calendar["train_time_upper_bound"] == "2024-12-31"
    assert calendar["sha256"] == manifest_sha256(expected)
    assert document["sha256"] == calendar["sha256"]
    assert document["panel_filename"] == "temporal_rank_ic_panel.pkl"
    assert document["panel_sha256"] == PINNED_PANEL_SHA256
    assert require_pinned_panel_sha(document) == PINNED_PANEL_SHA256
    assert expected[0] == "2026-04-01"
    assert expected[-1] == "2026-06-30"
    business = [
        stamp.date().isoformat()
        for stamp in pd.bdate_range("2026-04-01", "2026-06-30")
    ]
    assert set(business) - set(expected) == set(SELECTION_HOLIDAY_WEEKDAYS)
    assert set(expected) - set(business) == set()
    for day in (
        "2026-04-01", "2026-04-30", "2026-05-11", "2026-06-22", "2026-06-26",
    ):
        assert day in expected
    assert "2026-07-02" not in expected
    assert is_sse_session("2026-07-02")
    assert sessions_between("2026-07-01", "2026-07-31")[9] == "2026-07-14"
    for day in ("2026-08-03", "2026-09-18", "2026-09-25", "2026-10-09", "2026-10-23"):
        assert day not in expected
    assert is_sse_session("2026-09-18")
    assert not is_sse_session("2026-09-25")
    protocol = (ROOT / "docs/VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md").read_text(
        encoding="utf-8",
    )
    assert calendar["sha256"] in protocol
    for day in expected:
        assert day in protocol
    launch = (ROOT / "finetune/small_clean_bigbatch_relay/launch.py").read_text(
        encoding="utf-8",
    )
    push = (ROOT / "finetune/small_clean_bigbatch_relay/push_chunk.py").read_text(
        encoding="utf-8",
    )
    assert "KRONOS_CHECKPOINT_SELECTION_PROTOCOL" in launch
    assert "temporal_rank_ic_v1" in launch
    assert "KRONOS_TEMPORAL_RANK_IC_INTERVAL_SEGMENTS" in launch
    assert "require_pinned_panel_sha" in push
    assert "KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET" in push
    assert "[DATASET, panel_dataset]" in push
    training = "smmt315/a-share-120d-temporal-symbol-holdout"
    with pytest.raises(TemporalRankICSelectionError, match="KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET"):
        resolve_panel_dataset_slug("", training_dataset=training)
    with pytest.raises(TemporalRankICSelectionError, match="symbol-holdout"):
        resolve_panel_dataset_slug(training, training_dataset=training)
    with pytest.raises(TemporalRankICSelectionError, match="val_data.pkl"):
        resolve_panel_dataset_slug("owner/val_data.pkl", training_dataset=training)
    assert resolve_panel_dataset_slug(
        "owner/kronos-temporal-rank-ic-panel-v1", training_dataset=training,
    ) == "owner/kronos-temporal-rank-ic-panel-v1"


def test_empty_panel_sha_fails_push_dry_run(tmp_path, monkeypatch):
    """Empty panel_sha256 still fail-closes push, including --dry-run."""
    import runpy

    import temporal_rank_ic_selection as selection

    frozen = json.loads(
        (ROOT / "finetune/manifests/temporal_rank_ic_v1/selection_window.json").read_text(
            encoding="utf-8",
        )
    )
    assert frozen["panel_sha256"] == PINNED_PANEL_SHA256
    frozen["panel_sha256"] = ""
    empty_path = tmp_path / "selection_window.json"
    empty_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(selection, "default_manifest_path", lambda _repo_root: empty_path)

    ns = runpy.run_path(str(ROOT / "finetune/small_clean_bigbatch_relay/push_chunk.py"))

    def forbid_subprocess(*_args, **_kwargs):
        raise AssertionError("empty panel SHA must fail before any subprocess")

    monkeypatch.setattr(ns["subprocess"], "run", forbid_subprocess)
    monkeypatch.setattr(ns["subprocess"], "check_output", forbid_subprocess)
    monkeypatch.setenv(
        "KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET",
        "wynstonliu/kronos-temporal-rank-ic-panel-v1",
    )
    monkeypatch.setattr(sys, "argv", ["push_chunk.py", "1", "--dry-run"])
    with pytest.raises(TemporalRankICSelectionError, match="panel_sha256"):
        ns["main"]()


def test_resolve_inputs_does_not_pick_symbol_holdout_val(tmp_path):
    repo = tmp_path / "repo"
    manifest_dir = repo / "finetune/manifests/temporal_rank_ic_v1"
    manifest_dir.mkdir(parents=True)
    input_root = tmp_path / "input"
    panel_dir = input_root / "temporal_window"
    panel_dir.mkdir(parents=True)
    panel = panel_dir / "temporal_rank_ic_panel.pkl"
    panel.write_bytes(b"panel")
    body = _manifest_body(DATES, sha256_file(panel))
    (manifest_dir / "selection_window.json").write_text(json.dumps(body), encoding="utf-8")
    holdout = input_root / "a-share-120d-temporal-symbol-holdout"
    holdout.mkdir(parents=True)
    (holdout / "val_data.pkl").write_bytes(b"holdout-val")
    manifest_path, panel_path, loaded = resolve_training_selection_inputs(
        repo, input_root, val_data_paths=[holdout / "val_data.pkl"],
    )
    assert manifest_path.name == "selection_window.json"
    assert panel_path == panel
    assert loaded["split_kind"] == "temporal"
    assert "symbol-holdout" not in panel_path.as_posix()

    holdout_copy = holdout / "temporal_rank_ic_panel.pkl"
    holdout_copy.write_bytes(b"do-not-use")
    again_manifest, again_panel, _loaded = resolve_training_selection_inputs(
        repo, input_root, val_data_paths=[holdout / "val_data.pkl"],
    )
    assert again_manifest == manifest_path
    assert again_panel == panel

    only = tmp_path / "only_holdout"
    only_repo = only / "repo"
    only_manifest_dir = only_repo / "finetune/manifests/temporal_rank_ic_v1"
    only_manifest_dir.mkdir(parents=True)
    only_input = only / "input" / "a-share-120d-temporal-symbol-holdout"
    only_input.mkdir(parents=True)
    planted = only_input / "temporal_rank_ic_panel.pkl"
    planted.write_bytes(b"holdout-panel")
    planted_body = _manifest_body(DATES, sha256_file(planted))
    (only_manifest_dir / "selection_window.json").write_text(
        json.dumps(planted_body), encoding="utf-8",
    )
    with pytest.raises(TemporalRankICSelectionError, match="symbol-holdout"):
        resolve_training_selection_inputs(only_repo, only / "input")


def test_window_builder_uses_the_manifest_calendar_only():
    signal = "2026-07-06"
    signal_ts = pd.Timestamp(signal)
    before = pd.bdate_range(end=signal_ts, periods=120)
    after = pd.bdate_range(start=signal_ts + pd.offsets.BDay(1), periods=15)
    index = before.append(after)
    assert signal_ts in index
    location = int(index.get_loc(signal_ts))
    assert location >= 119 and location + 10 < len(index)
    panel = {}
    for symbol, scale in (("AAA", 1.0), ("BBB", 2.0)):
        close = np.linspace(10.0, 20.0, len(index)) * scale
        frame = pd.DataFrame({
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.ones(len(index)), "amount": np.ones(len(index)),
            "sector_id": np.full(len(index), 3),
            "size_percentile": np.full(len(index), 0.4),
        }, index=index)
        panel[symbol] = frame
    manifest = {
        "signal_dates": [signal],
        "lookback": 120,
        "predict": 10,
    }
    rows = build_evaluation_rows(manifest, panel, min_names=2)
    assert [row["symbol"] for row in rows] == ["AAA", "BBB"]
    assert {row["asof_date"] for row in rows} == {signal}
    for row in rows:
        assert row["x"].shape == (120, 6)
        assert row["stamp"].shape[0] == 130
    aaa = panel["AAA"]["close"].to_numpy()
    expected = float(aaa[location + 10] / aaa[location] - 1.0)
    assert rows[0]["actual"] == pytest.approx(expected)


def test_config_flag_defaults_off_and_bigbatch_interval_stays_ten(monkeypatch):
    from config import Config

    monkeypatch.delenv("KRONOS_CHECKPOINT_SELECTION_PROTOCOL", raising=False)
    monkeypatch.setenv("KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS", "1000000")
    config = Config()
    assert config.checkpoint_selection_protocol == ""
    assert config.temporal_rank_ic_interval_segments == 10

    monkeypatch.setenv("KRONOS_CHECKPOINT_SELECTION_PROTOCOL", "val_loss")
    assert Config().checkpoint_selection_protocol == ""

    monkeypatch.setenv("KRONOS_CHECKPOINT_SELECTION_PROTOCOL", "temporal_rank_ic_v1")
    monkeypatch.setenv("KRONOS_LOOKBACK_WINDOW", "120")
    monkeypatch.setenv("KRONOS_PREDICT_WINDOW", "10")
    with pytest.raises(ValueError, match="KRONOS_TEMPORAL_RANK_IC_MANIFEST"):
        Config()
    monkeypatch.setenv("KRONOS_TEMPORAL_RANK_IC_MANIFEST", "selection_window.json")
    monkeypatch.setenv("KRONOS_TEMPORAL_RANK_IC_PANEL", "temporal_rank_ic_panel.pkl")
    enabled = Config()
    assert enabled.checkpoint_selection_protocol == "temporal_rank_ic_v1"
    assert enabled.temporal_rank_ic_interval_segments == 10
    assert enabled.best_selection_metric == "objective"

    monkeypatch.setenv("KRONOS_CHECKPOINT_SELECTION_PROTOCOL", "symbol_holdout")
    with pytest.raises(ValueError, match="temporal_rank_ic_v1"):
        Config()


def test_present_manifest_calendar_is_checked_without_a_panel(tmp_path):
    repo = tmp_path / "repo"
    manifest_dir = repo / "finetune/manifests/temporal_rank_ic_v1"
    manifest_dir.mkdir(parents=True)
    body = _manifest_body(DATES, "panel-sha-not-opened")
    path = manifest_dir / "selection_window.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    assert assert_frozen_manifest_present(repo) == path

    body["signal_dates"] = ["2026-09-18"]
    body["sha256"] = manifest_sha256(body["signal_dates"])
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(TemporalRankICSelectionError, match="sealed_8d"):
        assert_frozen_manifest_present(repo)


def test_path_seed_does_not_replace_the_training_rng():
    import random
    import torch

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected_torch = torch.rand(3).clone()
    expected_numpy = np.random.rand(3).copy()
    expected_python = random.random()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with isolated_path_seed():
        assert torch.initial_seed() == PATH_SEED
        torch.rand(8)
        np.random.rand(8)
        random.random()
    assert torch.equal(torch.rand(3), expected_torch)
    assert np.allclose(np.random.rand(3), expected_numpy)
    assert random.random() == expected_python


def test_unpickle_guard_rejects_non_dicts(tmp_path):
    path = tmp_path / "panel.pkl"
    with path.open("wb") as handle:
        pickle.dump(["not", "a", "panel"], handle)
    from temporal_rank_ic_selection import load_price_panel
    with pytest.raises(TemporalRankICSelectionError, match="dict"):
        load_price_panel(path)
