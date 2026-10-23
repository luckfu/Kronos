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
    assert document["n_paths"] == 1
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
    assert not str(document.get("rows_sha256") or "").strip()
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


def _long_panel():
    signal_dates = ["2026-07-06", "2026-07-07"]
    index = pd.date_range("2025-06-01", "2026-08-01", freq="D")
    panel = {}
    for symbol, scale in (("AAA", 1.0), ("BBB", 2.0)):
        close = np.linspace(10.0, 40.0, len(index)) * scale
        panel[symbol] = pd.DataFrame({
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(index), 5.0),
            "amount": np.full(len(index), 7.0),
            "sector_id": np.full(len(index), 4),
            "size_percentile": np.full(len(index), 0.25),
        }, index=index)
    late = pd.date_range("2026-07-20", "2026-08-01", freq="D")
    late_close = np.linspace(1.0, 2.0, len(late))
    panel["CCC"] = pd.DataFrame({
        "open": late_close,
        "high": late_close,
        "low": late_close,
        "close": late_close,
        "volume": np.ones(len(late)),
        "amount": np.ones(len(late)),
    }, index=late)
    return signal_dates, index, panel


def test_evaluation_rows_use_a_per_symbol_date_index():
    signal_dates, index, panel = _long_panel()
    manifest = {"signal_dates": signal_dates, "lookback": 120, "predict": 10}
    rows = build_evaluation_rows(manifest, panel, min_names=2, progress=False)
    assert [row["symbol"] for row in rows] == ["AAA", "BBB", "AAA", "BBB"]
    assert [row["asof_date"] for row in rows] == [
        "2026-07-06", "2026-07-06", "2026-07-07", "2026-07-07",
    ]
    location = int(index.get_loc(pd.Timestamp("2026-07-06")))
    assert location > 200
    close = panel["AAA"]["close"].to_numpy()
    expected = float(close[location + 10] / close[location] - 1.0)
    assert rows[0]["actual"] == pytest.approx(expected)
    assert rows[0]["x"].shape == (120, 6)
    signal_stamp = rows[0]["stamp"][119]
    assert signal_stamp.tolist() == pytest.approx([0, 0, 0, 6, 7])
    bare = dict(panel)
    bare["BBB"] = bare["BBB"].drop(columns=["sector_id"])
    with pytest.raises(TemporalRankICSelectionError, match="sector_id"):
        build_evaluation_rows(manifest, bare, min_names=2, progress=False)


def test_row_build_progress_is_flushed(monkeypatch):
    signal_dates, _index, panel = _long_panel()
    manifest = {"signal_dates": signal_dates[:1], "lookback": 120, "predict": 10}
    calls = []

    def capture(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("builtins.print", capture)
    build_evaluation_rows(manifest, panel, min_names=2, progress=True)
    assert calls
    assert all(call.get("flush") is True for call in calls)


def _wrote_rows(tmp_path, rows, *, panel_sha, calendar_sha):
    from temporal_rank_ic_selection import write_evaluation_rows

    path = tmp_path / "temporal_rank_ic_rows.pkl"
    digest = write_evaluation_rows(
        rows, path, panel_sha256=panel_sha, calendar_sha256=calendar_sha,
    )
    return path, digest


def test_precomputed_rows_load_and_fail_closed(tmp_path, monkeypatch):
    from temporal_rank_ic_selection import (
        load_precomputed_rows,
        materialize_evaluation_rows,
        prepare_rank_ic_runtime,
        write_evaluation_rows,
    )

    signal_dates, _index, panel = _long_panel()
    manifest_body = {
        "signal_dates": signal_dates,
        "lookback": 120,
        "predict": 10,
        "panel_path": str(tmp_path / "temporal_rank_ic_panel.pkl"),
    }
    rows = build_evaluation_rows(manifest_body, panel, min_names=2, progress=False)
    panel_sha = "a" * 64
    calendar_sha = manifest_sha256(signal_dates)
    manifest_body["panel_sha256"] = panel_sha
    manifest_body["sha256"] = calendar_sha
    path, digest = _wrote_rows(
        tmp_path, rows, panel_sha=panel_sha, calendar_sha=calendar_sha,
    )
    loaded = load_precomputed_rows(
        path, manifest_body, expected_sha=digest, min_names=2,
    )
    assert len(loaded) == len(rows)
    assert loaded[0]["symbol"] == "AAA"

    def forbid_rebuild(*_args, **_kwargs):
        raise AssertionError("row build must not run when an artifact is expected")

    monkeypatch.setattr(
        "temporal_rank_ic_selection.build_evaluation_rows", forbid_rebuild,
    )
    monkeypatch.setattr(
        "temporal_rank_ic_selection.load_price_panel", forbid_rebuild,
    )
    manifest_body["rows_sha256"] = "b" * 64
    with pytest.raises(TemporalRankICSelectionError, match="rows SHA"):
        materialize_evaluation_rows(manifest_body, min_names=2, progress=False)
    manifest_body["rows_sha256"] = digest
    path.unlink()
    with pytest.raises(TemporalRankICSelectionError, match="missing"):
        materialize_evaluation_rows(manifest_body, min_names=2, progress=False)

    path, digest = _wrote_rows(
        tmp_path, rows, panel_sha="c" * 64, calendar_sha=calendar_sha,
    )
    manifest_body["rows_sha256"] = digest
    with pytest.raises(TemporalRankICSelectionError, match="panel SHA"):
        load_precomputed_rows(path, manifest_body, expected_sha=digest, min_names=2)

    matching, matching_sha = _wrote_rows(
        tmp_path, rows, panel_sha=panel_sha, calendar_sha=calendar_sha,
    )
    del manifest_body["rows_sha256"]
    monkeypatch.delenv("KRONOS_TEMPORAL_RANK_IC_ROWS", raising=False)
    loaded_rows, source = materialize_evaluation_rows(
        manifest_body, min_names=2, progress=False,
    )
    assert source == "precomputed"
    assert len(loaded_rows) == len(rows)
    assert matching == tmp_path / "temporal_rank_ic_rows.pkl"
    assert matching_sha

    monkeypatch.setenv(
        "KRONOS_TEMPORAL_RANK_IC_ROWS",
        str(tmp_path / "missing-temporal_rank_ic_rows.pkl"),
    )
    with pytest.raises(TemporalRankICSelectionError, match="missing"):
        materialize_evaluation_rows(manifest_body, min_names=2, progress=False)
    monkeypatch.setenv(
        "KRONOS_TEMPORAL_RANK_IC_ROWS",
        str(tmp_path / "symbol-holdout" / "temporal_rank_ic_rows.pkl"),
    )
    with pytest.raises(TemporalRankICSelectionError, match="symbol-holdout"):
        materialize_evaluation_rows(manifest_body, min_names=2, progress=False)

    # Full prepare path: cheap SHA check plus load, with no panel unpickle.
    monkeypatch.delenv("KRONOS_TEMPORAL_RANK_IC_ROWS", raising=False)
    panel_file = tmp_path / "temporal_rank_ic_panel.pkl"
    panel_file.write_bytes(b"panel-bytes-not-unpickled")
    from temporal_rank_ic_selection import sha256_file as file_sha
    body = _manifest_body(signal_dates, file_sha(panel_file))
    expanded = []
    for row in rows:
        for index in range(15):
            clone = dict(row)
            clone["symbol"] = f"{row['symbol']}{index:02d}"
            expanded.append(clone)
    body["rows_sha256"] = write_evaluation_rows(
        expanded,
        tmp_path / "temporal_rank_ic_rows.pkl",
        panel_sha256=body["panel_sha256"],
        calendar_sha256=body["sha256"],
    )
    manifest_path = tmp_path / "selection_window.json"
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    config = {
        "checkpoint_selection_protocol": PROTOCOL_ID,
        "temporal_rank_ic_manifest_path": str(manifest_path),
        "temporal_rank_ic_panel_path": str(panel_file),
    }
    runtime = prepare_rank_ic_runtime(config, load_panel=True)
    assert runtime["rows_source"] == "precomputed"
    assert runtime["panel"] is None
    assert len(runtime["rows"]) == len(expanded)
    checked = prepare_rank_ic_runtime(config, load_panel=False)
    assert checked["rows"] is None
    assert checked["manifest"]["panel_sha256"] == body["panel_sha256"]


def test_materialize_builds_when_no_rows_artifact_is_expected(tmp_path, monkeypatch):
    from temporal_rank_ic_selection import materialize_evaluation_rows

    signal_dates, _index, panel = _long_panel()
    panel.pop("CCC")
    panel_path = tmp_path / "temporal_rank_ic_panel.pkl"
    with panel_path.open("wb") as handle:
        pickle.dump(panel, handle)
    manifest = {
        "signal_dates": signal_dates,
        "lookback": 120,
        "predict": 10,
        "panel_path": str(panel_path),
        "panel_sha256": sha256_file(panel_path),
        "sha256": manifest_sha256(signal_dates),
    }
    monkeypatch.delenv("KRONOS_TEMPORAL_RANK_IC_ROWS", raising=False)
    rows, source = materialize_evaluation_rows(
        manifest, min_names=2, progress=False,
    )
    assert source == "built"
    assert len(rows) == 4


def test_prepare_failure_names_the_underlying_error_and_stays_closed(tmp_path):
    from temporal_rank_ic_selection import coordinate_rank_ic_prepare

    def cheap():
        raise TemporalRankICSelectionError("panel SHA does not match the manifest")

    def materialize(_checked):
        raise AssertionError("rows must not be built after a failed manifest check")

    with pytest.raises(RuntimeError, match="panel SHA does not match") as caught:
        coordinate_rank_ic_prepare(
            rank=0,
            world_size=1,
            directory=tmp_path,
            run_id="solo",
            cheap_check=cheap,
            materialize=materialize,
            exchange_errors=lambda _error: [],
            timeout=5,
        )
    message = str(caught.value)
    assert "TemporalRankICSelectionError" in message
    assert "val loss" in message
    assert "symbol holdout" in message


def test_rank1_timeout_does_not_enter_the_collective(tmp_path):
    from temporal_rank_ic_selection import coordinate_rank_ic_prepare

    calls = []

    def exchange(local_error):
        calls.append(local_error)
        return [local_error] if local_error else []

    with pytest.raises(RuntimeError, match="all_reduce") as caught:
        coordinate_rank_ic_prepare(
            rank=1,
            world_size=2,
            directory=tmp_path,
            run_id="timeout-run",
            cheap_check=lambda: {"manifest": {"signal_dates": ["2026-07-06"]}},
            materialize=lambda _checked: (_ for _ in ()).throw(
                AssertionError("rank 1 built rows")
            ),
            exchange_errors=exchange,
            timeout=0.2,
            poll_seconds=0.01,
            heartbeat_seconds=30,
        )
    assert calls == []
    assert "val loss" in str(caught.value)
    assert "NCCL" in str(caught.value)


def test_other_ranks_wait_outside_the_collective_during_row_build(tmp_path):
    import threading

    from temporal_rank_ic_selection import coordinate_rank_ic_prepare

    order = []
    lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()
    failures = []
    builds = []

    def cheap(rank):
        def _cheap():
            with lock:
                order.append((rank, "cheap"))
            return {"manifest": {"signal_dates": ["2026-07-06"]}, "rows": None}
        return _cheap

    def materialize(_checked):
        builds.append("build")
        started.set()
        with lock:
            order.append((0, "materialize-start"))
        assert release.wait(timeout=5)
        with lock:
            order.append((0, "materialize-end"))
        return {
            "manifest": {"signal_dates": ["2026-07-06"]},
            "rows": [{"symbol": "AAA"}],
            "rows_source": "built",
        }

    def exchange(rank):
        def _exchange(local_error):
            with lock:
                order.append((rank, "exchange"))
                assert (0, "materialize-end") in order
            return [local_error] if local_error else []
        return _exchange

    def run(rank):
        try:
            coordinate_rank_ic_prepare(
                rank=rank,
                world_size=2,
                directory=tmp_path,
                run_id="ddp-run",
                cheap_check=cheap(rank),
                materialize=materialize if rank == 0 else (
                    lambda _checked: (_ for _ in ()).throw(
                        AssertionError("rank 1 built rows")
                    )
                ),
                exchange_errors=exchange(rank),
                timeout=5,
                poll_seconds=0.01,
                heartbeat_seconds=30,
            )
        except Exception as exc:
            failures.append((rank, exc))

    workers = [threading.Thread(target=run, args=(rank,)) for rank in (1, 0)]
    for worker in workers:
        worker.start()
    assert started.wait(timeout=5)
    with lock:
        assert not any(item[1] == "exchange" for item in order)
    release.set()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert failures == []
    assert builds == ["build"]
    with lock:
        assert (0, "materialize-end") in order
        assert (1, "exchange") in order
        assert order.index((0, "materialize-end")) < order.index((1, "exchange"))


def test_rank0_build_error_reaches_every_rank(tmp_path):
    import threading

    from temporal_rank_ic_selection import coordinate_rank_ic_prepare

    barrier = threading.Barrier(2)
    collected = {}
    failures = []
    builds = []

    def materialize(_checked):
        builds.append("build")
        raise TemporalRankICSelectionError("missing columns ['sector_id']")

    def exchange(rank):
        def _exchange(local_error):
            collected[rank] = local_error
            barrier.wait(timeout=5)
            return [
                f"rank{index}: {text}"
                for index, text in sorted(collected.items())
                if text
            ]
        return _exchange

    def run(rank):
        try:
            coordinate_rank_ic_prepare(
                rank=rank,
                world_size=2,
                directory=tmp_path,
                run_id="err-run",
                cheap_check=lambda: {
                    "manifest": {"signal_dates": ["2026-07-06"]},
                    "rows": None,
                },
                materialize=materialize,
                exchange_errors=exchange(rank),
                timeout=5,
                poll_seconds=0.01,
                heartbeat_seconds=30,
            )
        except Exception as exc:
            failures.append((rank, exc))

    workers = [threading.Thread(target=run, args=(rank,)) for rank in (0, 1)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert builds == ["build"]
    assert sorted(rank for rank, _exc in failures) == [0, 1]
    for _rank, exc in failures:
        assert isinstance(exc, RuntimeError)
        text = str(exc)
        assert "sector_id" in text
        assert "TemporalRankICSelectionError" in text
        assert "val loss" in text
        assert "symbol holdout" in text


def test_launch_prepare_keeps_rank1_out_of_all_reduce(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    import train_predictor as trainer

    events = []
    lock = threading.Lock()
    failures = []

    def fake_prepare(config, *, load_panel, manifest=None):
        del config, manifest
        with lock:
            events.append(("prepare", bool(load_panel)))
        if load_panel:
            time_mod = __import__("time")
            time_mod.sleep(0.2)
            with lock:
                assert "all_reduce" not in events
            return {
                "manifest": {
                    "signal_dates": ["2026-07-06"],
                    "sha256": "abc",
                    "train_time_upper_bound": "2024-12-31",
                },
                "panel": None,
                "rows": [{"symbol": "AAA"}],
                "rows_source": "built",
            }
        return {
            "manifest": {
                "signal_dates": ["2026-07-06"],
                "sha256": "abc",
                "train_time_upper_bound": "2024-12-31",
            },
            "panel": None,
            "rows": None,
            "rows_source": "",
        }

    def forbid_all_reduce(*_args, **_kwargs):
        with lock:
            events.append("all_reduce")
        raise AssertionError("all_reduce during Temporal Rank IC prepare")

    def set_timeout(timeout, group=None):
        del group
        with lock:
            events.append(("timeout", float(timeout.total_seconds())))

    def exchange(local_error, device, rank, world_size):
        del device, world_size
        with lock:
            events.append(("exchange", rank))
            assert ("prepare", True) in events
        return [local_error] if local_error else []

    monkeypatch.setattr(trainer, "prepare_rank_ic_runtime", fake_prepare)
    monkeypatch.setattr(trainer, "_exchange_prepare_errors", exchange)
    monkeypatch.setattr(trainer.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer.dist, "all_reduce", forbid_all_reduce)
    monkeypatch.setattr(trainer.dist, "set_timeout", set_timeout, raising=False)
    device = SimpleNamespace(type="cpu")

    def run(rank):
        try:
            runtime = trainer.prepare_temporal_rank_ic_for_launch(
                {"checkpoint_selection_protocol": PROTOCOL_ID},
                device,
                rank,
                2,
                str(tmp_path),
            )
            if rank == 0:
                assert len(runtime["rows"]) == 1
        except Exception as exc:
            failures.append((rank, exc))

    workers = [threading.Thread(target=run, args=(rank,)) for rank in (1, 0)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert not worker.is_alive()
    assert failures == []
    with lock:
        assert events.count(("prepare", True)) == 1
        assert events.count(("prepare", False)) == 2
        prepare_at = events.index(("prepare", True))
        exchange_at = min(
            index for index, item in enumerate(events)
            if isinstance(item, tuple) and item and item[0] == "exchange"
        )
        assert prepare_at < exchange_at
        assert "all_reduce" not in events
        timeouts = [item[1] for item in events if isinstance(item, tuple) and item[0] == "timeout"]
    assert 2700 in timeouts
    assert 600 in timeouts


def test_rank1_score_wait_does_not_enter_broadcast(tmp_path):
    from temporal_rank_ic_selection import coordinate_rank_ic_score

    calls = []

    def score():
        calls.append("score")
        raise AssertionError("rank 1 must not score")

    with pytest.raises(RuntimeError, match="broadcast") as caught:
        coordinate_rank_ic_score(
            rank=1,
            world_size=2,
            directory=tmp_path,
            run_id="score-timeout",
            segment=1,
            score_fn=score,
            timeout=0.2,
            poll_seconds=0.01,
            heartbeat_seconds=30,
        )
    assert calls == []
    assert "val loss" in str(caught.value)
    assert "NCCL" in str(caught.value) or "broadcast" in str(caught.value)


def test_other_ranks_wait_outside_broadcast_during_score(tmp_path):
    import threading

    from temporal_rank_ic_selection import coordinate_rank_ic_score

    order = []
    lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()
    failures = []
    results = {}

    def score():
        with lock:
            order.append((0, "score-start"))
        started.set()
        assert release.wait(timeout=5)
        with lock:
            order.append((0, "score-end"))
        return {
            "daily_rank_ic_mean": 0.042,
            "n_dates": 60,
            "n_paths": 1,
            "path_agg": "median",
            "manifest_sha256": "abc",
            "samples": 100,
            "selection_protocol": PROTOCOL_ID,
        }

    def run(rank):
        try:
            summary = coordinate_rank_ic_score(
                rank=rank,
                world_size=2,
                directory=tmp_path,
                run_id="score-ddp",
                segment=1,
                score_fn=score if rank == 0 else (
                    lambda: (_ for _ in ()).throw(
                        AssertionError("rank 1 scored")
                    )
                ),
                timeout=5,
                poll_seconds=0.01,
                heartbeat_seconds=30,
            )
            results[rank] = summary["daily_rank_ic_mean"]
            with lock:
                order.append((rank, "returned"))
        except Exception as exc:
            failures.append((rank, exc))

    workers = [threading.Thread(target=run, args=(rank,)) for rank in (1, 0)]
    for worker in workers:
        worker.start()
    assert started.wait(timeout=5)
    with lock:
        assert (0, "score-start") in order
        assert not any(item[1] == "returned" for item in order)
    release.set()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert failures == []
    assert results == {0: 0.042, 1: 0.042}
    with lock:
        assert order.index((0, "score-end")) < order.index((1, "returned"))


def test_rank0_score_error_reaches_every_rank(tmp_path):
    import threading

    from temporal_rank_ic_selection import (
        TemporalRankICSelectionError,
        coordinate_rank_ic_score,
    )

    failures = []
    scores = []

    def score():
        scores.append("score")
        raise TemporalRankICSelectionError("non-finite Rank IC on 2026-07-06")

    def run(rank):
        try:
            coordinate_rank_ic_score(
                rank=rank,
                world_size=2,
                directory=tmp_path,
                run_id="score-err",
                segment=1,
                score_fn=score if rank == 0 else (
                    lambda: (_ for _ in ()).throw(
                        AssertionError("rank 1 scored")
                    )
                ),
                timeout=5,
                poll_seconds=0.01,
                heartbeat_seconds=30,
            )
        except Exception as exc:
            failures.append((rank, exc))

    workers = [threading.Thread(target=run, args=(rank,)) for rank in (0, 1)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert scores == ["score"]
    assert sorted(rank for rank, _exc in failures) == [0, 1]
    for _rank, exc in failures:
        assert isinstance(exc, RuntimeError)
        text = str(exc)
        assert "non-finite Rank IC" in text
        assert "val-loss" in text or "val loss" in text


def test_train_score_keeps_rank1_out_of_broadcast(tmp_path, monkeypatch):
    """Mimic Seg1: due-flag broadcast is fine; score float broadcast is not."""
    import threading
    from types import SimpleNamespace

    import train_predictor as trainer

    events = []
    lock = threading.Lock()
    failures = []
    measured = {}

    def fake_evaluate(model, tokenizer, runtime, device):
        del model, tokenizer, runtime, device
        with lock:
            events.append("evaluate")
            assert "broadcast_float" not in events
        time_mod = __import__("time")
        time_mod.sleep(0.25)
        with lock:
            assert "broadcast_float" not in events
        return {
            "daily_rank_ic_mean": 0.11,
            "n_dates": 60,
            "n_paths": 1,
            "path_agg": "median",
            "manifest_sha256": "deadbeef",
            "samples": 12,
            "selection_protocol": PROTOCOL_ID,
        }

    def fake_broadcast_flag(flag, device, rank):
        del device
        with lock:
            events.append(("broadcast_flag", rank, bool(flag)))
        return True

    def forbid_broadcast_float(value, device, rank):
        del value, device, rank
        with lock:
            events.append("broadcast_float")
        raise AssertionError("broadcast_rank0_float during Rank IC score")

    monkeypatch.setattr(trainer, "evaluate_temporal_d10_rank_ic", fake_evaluate)
    monkeypatch.setattr(trainer, "broadcast_rank0_flag", fake_broadcast_flag)
    monkeypatch.setattr(trainer, "broadcast_rank0_float", forbid_broadcast_float)
    monkeypatch.setattr(
        trainer, "rank_ic_prepare_run_id", lambda: "unit-score-run"
    )
    monkeypatch.setattr(trainer, "score_timeout_seconds", lambda: 5.0)

    # Build a minimal closure environment similar to train_model's helper.
    def run(rank):
        try:
            use_temporal_rank_ic = True
            rank_ic_evaluated = False
            next_segment = 1
            rank_ic_interval = 10
            world_size = 2
            save_dir = str(tmp_path)
            device = SimpleNamespace(type="cpu")
            rank_ic_runtime = {
                "rows": [{"symbol": "AAA"}] if rank == 0 else None,
                "manifest": {"signal_dates": ["2026-07-06"]},
            }
            core_model = object()
            tokenizer = object()

            def score_temporal_rank_ic_segment():
                nonlocal rank_ic_evaluated
                if not use_temporal_rank_ic or rank_ic_evaluated:
                    return
                due = rank == 0
                if not trainer.broadcast_rank0_flag(due, device, rank):
                    return

                def _run_rank0_score():
                    return trainer.evaluate_temporal_d10_rank_ic(
                        core_model, tokenizer, rank_ic_runtime, device,
                    )

                summary = trainer.coordinate_rank_ic_score(
                    rank=rank,
                    world_size=world_size,
                    directory=save_dir,
                    run_id=f"{trainer.rank_ic_prepare_run_id()}-seg{next_segment}",
                    segment=next_segment,
                    score_fn=_run_rank0_score if rank == 0 else (
                        lambda: (_ for _ in ()).throw(
                            AssertionError("non-zero rank must not score")
                        )
                    ),
                    timeout=trainer.score_timeout_seconds(),
                    poll_seconds=0.01,
                    heartbeat_seconds=30,
                )
                measured[rank] = float(summary["daily_rank_ic_mean"])
                rank_ic_evaluated = True

            score_temporal_rank_ic_segment()
        except Exception as exc:
            failures.append((rank, exc))

    workers = [threading.Thread(target=run, args=(rank,)) for rank in (1, 0)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert not worker.is_alive()
    assert failures == []
    assert measured == {0: 0.11, 1: 0.11}
    with lock:
        assert events.count("evaluate") == 1
        assert "broadcast_float" not in events
        assert any(
            item[0] == "broadcast_flag" for item in events if isinstance(item, tuple)
        )
