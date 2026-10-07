"""Validation-only C2 decode + per-path estimators kernel.

Covers: path estimators, per-arm scoring / pre-registered rule 1, paired stats,
rank-blend grid, style-shrink == the documented analysis transform, decode with paths
== decode_records (same RNG), C2 window conditioning == the C2 OOS WindowStore
(commit e4b92bb), builder / staged metadata / embedded archive (old C2 model code),
sealed-package guard, and a CPU end-to-end run of the staged runner (tiny models,
synthetic val panel, three phases, C2 phase on the e4b92bb model code).
"""

import base64
import importlib.util
import io
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
RUNNER = FINETUNE / "kaggle_beta_v21_c1_val_c2_path_estimators.py"
BUILDER = FINETUNE / "build_kaggle_beta_v21_c1_val_c2_path_estimators_kernel.py"
STAGING = FINETUNE / "kaggle_beta_v21_c1_val_c2_path_estimators_kernel"
STAGED_RUNNER = STAGING / RUNNER.name
DOCKER_SHA = "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
C2_COMMIT = "e4b92bb32aa47d676ebcba70b5c8427bbc404c03"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import val_c2_path_estimators as est  # noqa: E402


def load(name, path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis():
    return load("why_c2_analysis_t", FINETUNE / "analysis" / "beta_v21_c1_why_c2_better.py")


# ------------------------------------------------------------------- estimators
def test_estimators_n5_known_values():
    paths = np.array([[0.01, -0.02, 0.50, 0.03, 0.00],     # one explosive path
                      [0.1, 0.1, 0.1, 0.1, 0.1]])
    out = est.estimator_scores(paths)
    assert np.allclose(out["mean"], [0.104, 0.1])
    assert np.allclose(out["median"], [0.01, 0.1])
    assert np.allclose(out["trimmed_mean"], [(0.0 + 0.01 + 0.03) / 3, 0.1])
    assert np.allclose(out["mean_ex_max"], [(-0.02 + 0.0 + 0.01 + 0.03) / 4, 0.1])


def test_estimators_n16_drop_single_extremes():
    rng = np.random.default_rng(0)
    paths = rng.normal(size=(7, 16))
    out = est.estimator_scores(paths)
    s = np.sort(paths, axis=1)
    assert np.allclose(out["trimmed_mean"], s[:, 1:15].mean(1))
    assert np.allclose(out["mean_ex_max"], s[:, :15].mean(1))
    assert np.allclose(out["median"], np.median(paths, 1))
    with pytest.raises(ValueError):
        est.estimator_scores(np.zeros(5))


def test_paired_stats_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    a = {f"d{i}": float(x) for i, x in enumerate(rng.normal(0.3, 0.05, 24))}
    b = {f"d{i}": float(x) for i, x in enumerate(rng.normal(0.29, 0.05, 24))}
    out = est.paired_stats(a, b)
    keys = sorted(a)
    t, _ = stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys])
    assert out["n"] == 24 and math.isclose(out["t"], float(t), rel_tol=1e-9)
    assert out["wins"] == sum(a[k] > b[k] for k in keys)


def _synthetic_arm(n_dates=6, n_per=60, n_paths=5, seed=0, explosive=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        signal = rng.normal(size=n_per)
        ret = 0.05 * signal + rng.normal(scale=0.05, size=n_per)
        paths = 0.02 * signal[:, None] + rng.normal(scale=0.03, size=(n_per, n_paths))
        if explosive:
            hit = rng.random(n_per) < 0.15
            paths[hit, 0] += explosive  # one spiking path, unrelated to returns
        for i in range(n_per):
            row = {"symbol": f"s{i:03d}", "asof_date": f"2025-0{d + 1}-01",
                   "return_10d": ret[i], "utility": ret[i] + rng.normal(scale=0.01),
                   "predicted_return_d10": float(paths[i].mean())}
            for p in range(n_paths):
                row[f"{est.PATH_PREFIX}{p:02d}"] = float(paths[i, p])
            rows.append(row)
    return pd.DataFrame(rows)


def test_score_arm_metrics_and_rule1_logic():
    frame = _synthetic_arm(explosive=0.5)
    result = est.score_arm(frame)
    assert set(result["estimators"]) == set(est.ESTIMATORS)
    assert result["sample_count"] == 5 and result["samples"] == len(frame)
    mean = result["estimators"]["mean"]
    assert mean["signal_dates"] == 6 and len(mean["by_signal_date"]) == 6
    for key in ("return10d_rank_ic_daily", "return10d_rank_ic_pooled", "return10d_rank_icir",
                "top_bottom_decile_return10d", "utility_rank_ic_daily", "top_decile_within_ic"):
        assert mean[key] is not None and np.isfinite(mean[key])
    # Robust estimators beat mean when one path spikes at random.
    vm = result["versus_mean"]
    assert vm["median"]["daily_ic"]["mean_diff"] > 0
    assert set(vm) == {"median", "trimmed_mean", "mean_ex_max"}
    # Rule 1 decision is exactly the conjunction of its three parts.
    for v in vm.values():
        assert v["rule1_adopt"] == (v["rule1_gain_ok"] and v["rule1_t_ok"]
                                    and v["rule1_top_decile_fixed"])
    # A corrupted saved mean must be caught.
    bad = frame.copy()
    bad.loc[0, "predicted_return_d10"] += 1e-3
    with pytest.raises(RuntimeError):
        est.score_arm(bad)


def test_rule1_thresholds():
    base = {"top_decile_within_ic": 0.05}
    good = {"top_decile_within_ic": 0.06}
    ok = est.estimator_decision({"mean_diff": 0.011, "t": 2.1}, good, base)
    assert ok["rule1_adopt"]
    assert not est.estimator_decision({"mean_diff": 0.009, "t": 5}, good, base)["rule1_adopt"]
    assert not est.estimator_decision({"mean_diff": 0.02, "t": 1.9}, good, base)["rule1_adopt"]
    assert not est.estimator_decision({"mean_diff": 0.02, "t": 3}, base, base)["rule1_adopt"]
    assert not est.estimator_decision({"mean_diff": 0.02, "t": 3},
                                      {"top_decile_within_ic": -0.01},
                                      {"top_decile_within_ic": -0.05})["rule1_adopt"]


def test_day_metrics_decile_convention_matches_val_gen_ic():
    evaluator = load("valgenic_t", FINETUNE / "evaluate_beta_v21_val_gen_ic.py")
    frame = _synthetic_arm(n_dates=3)
    ref = evaluator.score_val_checkpoint(frame)
    ours = est.summarize_score(frame, "predicted_return_d10")
    for key in ("return10d_rank_ic_daily", "return10d_rank_ic_pooled", "return10d_rank_icir",
                "utility_rank_ic_daily", "top_bottom_decile_return10d",
                "top_bottom_quintile_return10d"):
        assert math.isclose(ours[key], ref[key], rel_tol=1e-12, abs_tol=1e-12), key


def test_blend_grid_endpoints_and_choice():
    c2 = est.add_estimator_columns(_synthetic_arm(seed=3))
    ours = est.add_estimator_columns(_synthetic_arm(seed=3))
    rng = np.random.default_rng(5)
    ours["est_mean"] = ours["est_mean"] + rng.normal(scale=0.05, size=len(ours))
    merged = est.merged_pair(c2, ours)
    grid = est.blend_grid(merged)
    frame = grid.pop("_frame")
    assert grid["weights_on_c2"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    by = grid["by_weight"]
    ic_ours = est.summarize_score(merged, "s_ours")["return10d_rank_ic_daily"]
    ic_c2 = est.summarize_score(merged, "s_c2")["return10d_rank_ic_daily"]
    assert math.isclose(by["0.00"]["return10d_rank_ic_daily"], ic_ours, abs_tol=1e-12)
    assert math.isclose(by["1.00"]["return10d_rank_ic_daily"], ic_c2, abs_tol=1e-12)
    best = max(by, key=lambda k: by[k]["return10d_rank_ic_daily"])
    assert grid["chosen_weight_on_c2"] == float(best)
    assert "blend_w0.50" in frame
    with pytest.raises(RuntimeError):
        est.merged_pair(c2, ours.iloc[:-1])


def test_style_shrink_matches_documented_transform(analysis):
    rng = np.random.default_rng(11)
    rows = []
    sectors = ["C39计算机", "C38电气", "C26化学", "K70房地产", "J66货币", "B06煤炭"]
    for d in range(4):
        for i in range(80):
            rows.append({"symbol": f"s{i}", "asof_date": f"2025-0{d + 1}-01",
                         "sector": sectors[i % len(sectors)] if i < 70 else f"Z{i}",
                         "size_pct": rng.random(), **{c: rng.normal() for c in est.STYLE_COLUMNS},
                         "mom120": rng.normal(), "est_mean": rng.normal(),
                         "return_10d": rng.normal(), "utility": 0.0})
    v = pd.DataFrame(rows)
    v.loc[3, "vol20"] = np.nan  # fillna(median) path
    features = v[["symbol", "asof_date", "sector", "size_pct", "mom120", *est.STYLE_COLUMNS]]
    report = est.style_shrink_report(analysis, v, features, "est_mean")
    ref_in = v.rename(columns={"est_mean": "s_b475", "return_10d": "ret"}).assign(disp_b475=0.0)
    ref = analysis.val_postproc(ref_in, models=("b475",))["b475"]
    assert math.isclose(report["style_shrink"]["return10d_rank_ic_daily"],
                        ref["fitted_plus_half_resid"]["ic"], abs_tol=1e-12)
    assert math.isclose(report["raw"]["return10d_rank_ic_daily"], ref["raw"]["ic"], abs_tol=1e-12)
    assert report["lambda"] == 0.5


# ------------------------------------------------------------- decode / window store
SECTORS = [f"s{i:02d}" for i in range(86)]  # production: 86 labels, unknown -> 86


def _val_panel(n_symbols=24, n_days=160, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2025-01-02", periods=n_days, name="datetime")
    panel = {}
    for s in range(n_symbols):
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        panel[f"{600000 + s}"] = pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.003, n_days)), "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": rng.uniform(1e5, 2e5, n_days), "amount": rng.uniform(1e6, 2e6, n_days),
            "size_bucket": s % 5, "size_percentile": (s + 0.5) / n_symbols,
            "sector": SECTORS[s % 3] if s != 5 else np.nan,
        }, index=index)
    return panel


def _tiny_tokenizer():
    from model.kronos import KronosTokenizer

    return KronosTokenizer(
        d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
    ).eval()


def _tiny_predictor(seed, aux=False):
    import torch
    from model.kronos import Kronos

    torch.manual_seed(seed)
    return Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=16, n_heads=2, ff_dim=32,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0,
        learn_te=True, num_sectors=len(SECTORS), context_layer=1,
        use_size_percentile=True, use_beta_v21_auxiliary=aux,
    ).eval()


def test_decode_with_paths_reproduces_decode_records():
    torch = pytest.importorskip("torch")
    from evaluate_beta_v21_generative_return_oos import decode_records
    from evaluate_beta_v21_val_gen_ic import ValWindowStore, build_val_records
    from model.kronos import auto_regressive_inference

    panel = _val_panel(n_symbols=7)
    records = [r for r in build_val_records(panel) if r["asof_date"] == "2025-07-01"]
    assert len(records) == 7
    store = ValWindowStore(panel, SECTORS)
    tok, model = _tiny_tokenizer(), _tiny_predictor(1)
    device = torch.device("cpu")
    arm = est.ARMS["prod_t065_p80_n5"]
    ref = decode_records(arm, model, tok, records, store, device, effective_batch=10,
                         use_amp=False, label="x")
    ours = est.decode_with_paths(arm, model, tok, records, store, device, 10, False, "x",
                                 auto_regressive_inference)
    assert list(ours["symbol"]) == list(ref["symbol"])
    for column in ref.columns:
        assert ref[column].equals(ours[column]) or np.allclose(
            ref[column].astype(float), ours[column].astype(float), equal_nan=True), column
    paths = est.path_matrix(ours)
    assert paths.shape == (7, 5)
    assert np.allclose(paths.mean(1), ours["predicted_return_d10"], atol=1e-15)
    assert list(ours["start_index"]) == [r["start_index"] for r in records]
    n16 = est.decode_with_paths(est.ARMS["prod_t065_p80_n16"], model, tok, records, store,
                                device, 256, False, "x", auto_regressive_inference)
    assert est.path_matrix(n16).shape == (7, 16)


def test_c2_conditioning_matches_c2_oos_window_store(tmp_path):
    """ValWindowStore (used here for C2) == WindowStore of the C2 OOS kernel (e4b92bb)."""
    source = subprocess.check_output(
        ["git", "show", f"{C2_COMMIT}:finetune/evaluate_v1_beta_checkpoints.py"], cwd=ROOT)
    old_path = tmp_path / "evaluate_v1_beta_checkpoints_e4b92bb.py"
    old_path.write_bytes(source)
    old = load("ev1_e4b92bb", old_path)
    from evaluate_beta_v21_val_gen_ic import ValWindowStore, build_val_records

    panel = _val_panel(n_symbols=9)
    labels = SECTORS
    new_store = ValWindowStore(panel, labels)
    old_store = old.WindowStore.__new__(old.WindowStore)  # skip vocabulary equality check
    old_store.panel = panel
    old_store.sector_map = {value: index for index, value in enumerate(labels)}
    records = build_val_records(panel)[:40]
    for record in records:
        a = new_store.prepare(record)
        b = old_store.prepare({**record, "direction": 0, "size_decile": 0, "return_10d": 0.0})
        for key in ("x", "stamp", "mean", "std"):
            assert np.array_equal(a[key], b[key]), key
        assert a["sector_id"] == b["sector_id"]
        assert a["size_percentile"] == b["size_percentile"]
    import torch

    batch_new = __import__("evaluate_beta_v21_generative_return_oos").stack_batch(
        [new_store.prepare(r) for r in records[:4]], torch.device("cpu"))
    batch_old = old.stack_batch([old_store.prepare(
        {**r, "direction": 0, "size_decile": 0, "return_10d": 0.0}) for r in records[:4]],
        torch.device("cpu"))
    for key in ("x", "stamp", "sector", "percentile"):
        assert batch_new[key].dtype == batch_old[key].dtype, key
        assert torch.equal(batch_new[key], batch_old[key]), key


# ----------------------------------------------------------------- builder / staging
def test_builder_metadata_and_owner_options():
    builder = load("valc2_builder_t", BUILDER)
    meta = builder.build_metadata("wynstonliu", "wynstonliu")
    staged = json.loads((STAGING / "kernel-metadata.json").read_text())
    assert staged == meta
    assert staged["id"] == "wynstonliu/kronos-val-c2-and-path-estimators"
    assert staged["dataset_sources"] == ["wynstonliu/kronos-val-c2-path-inputs"]
    assert not any("temporal-symbol-holdout" in s for s in staged["dataset_sources"])
    assert staged["is_private"] is True and staged["enable_internet"] is True
    assert staged["enable_gpu"] is True and staged["machine_shape"] == "NvidiaTeslaT4"
    assert staged["docker_image"].endswith(DOCKER_SHA)
    other = builder.build_metadata("someone", "other")
    assert other["id"] == "someone/kronos-val-c2-and-path-estimators"
    assert other["dataset_sources"] == ["other/kronos-val-c2-path-inputs"]


def test_staged_runner_matches_tracked_and_constants():
    tracked, staged = RUNNER.read_text(), STAGED_RUNNER.read_text()
    blob = re.compile(r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""', re.S)
    assert blob.search(tracked).group(0) == blob.search(staged).group(0)
    assert tracked.replace('KERNEL_ID = "luckfu/', 'KERNEL_ID = "wynstonliu/') == staged
    runner = load("valc2_runner_t", STAGED_RUNNER)
    assert runner.KERNEL_ID == "wynstonliu/kronos-val-c2-and-path-estimators"
    assert [p["key"] for p in runner.PHASES] == ["c2_n5", "b475_n5", "b475_n16"]
    assert [p["required"] for p in runner.PHASES] == [True, True, False]
    c2 = runner.MODELS[runner.C2_LABEL]
    assert c2["sha256"].startswith("4ee469d4") and c2["effective_batch"] == 64
    assert c2["code"] == "c2_code_e4b92bb"
    assert c2["load_kwargs"] == {"num_sectors": 86, "num_size_buckets": 0, "context_layer": 6,
                                 "use_size_percentile": True, "size_mlp_hidden_dim": 64}
    b475 = runner.MODELS[runner.B475_LABEL]
    assert b475["sha256"].startswith("e1bd5584") and b475["effective_batch"] == 256
    assert runner.EXPECTED_VAL_SHA256.startswith("4cce31bc")
    assert runner.EXPECTED_VAL_SHA256.endswith("19bf7")
    assert len(runner.EXPECTED_SUBSAMPLE_DATES) == 24
    assert runner.EXPECTED_SUBSAMPLE_WINDOWS == 12256
    assert runner.HARD_LIMIT_SECONDS <= 11 * 3600
    assert abs(np.mean(list(runner.B475_VALGENIC_IC_BY_DATE.values())) - 0.31447) < 1e-4
    arms = est.ARMS
    for arm in arms.values():
        assert (arm["temperature"], arm["top_p"], arm["seed"]) == (0.65, 0.8, 20260906)
    assert arms["prod_t065_p80_n5"]["sample_count"] == 5
    assert arms["prod_t065_p80_n16"]["sample_count"] == 16


def test_embedded_archive_has_current_and_e4b92bb_model_code():
    runner = load("valc2_runner_t2", STAGED_RUNNER)
    raw = base64.b64decode(runner.EMBEDDED_KRONOS_ARCHIVE_B64.strip())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        names = set(archive.getnames())
        read = lambda n: archive.extractfile(n).read()  # noqa: E731
        assert read("Kronos/finetune/val_c2_path_estimators.py") == (
            FINETUNE / "val_c2_path_estimators.py").read_bytes()
        assert read("Kronos/model/kronos.py") == (ROOT / "model/kronos.py").read_bytes()
        old = subprocess.check_output(["git", "show", f"{C2_COMMIT}:model/kronos.py"], cwd=ROOT)
        assert read("Kronos/c2_code_e4b92bb/model/kronos.py") == old
        assert read("Kronos/c2_code_e4b92bb/SOURCE_COMMIT").decode() == C2_COMMIT
    assert "Kronos/c2_code_e4b92bb/model/__init__.py" in names
    assert "Kronos/finetune/analysis/beta_v21_c1_why_c2_better.py" in names


def test_sealed_package_guard(tmp_path, monkeypatch):
    runner = load("valc2_runner_t3", STAGED_RUNNER)
    monkeypatch.setattr(runner, "INPUT_ROOT", tmp_path)
    runner.assert_no_sealed_package()
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "evaluation_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError):
        runner.assert_no_sealed_package()


# ------------------------------------------------------------------- CPU end-to-end
def test_end_to_end_cpu_three_phases(tmp_path):
    pytest.importorskip("torch")
    snapshot = tmp_path / "modelscope_snapshot"
    _tiny_tokenizer().save_pretrained(str(snapshot / "tokenizer"))
    _tiny_predictor(1, aux=True).save_pretrained(str(snapshot))  # Best@475 stand-in
    (snapshot / "sector_vocabulary.json").write_text(json.dumps({"sector_labels": SECTORS}))
    ds = tmp_path / "input" / "datasets" / "wynstonliu" / "kronos-val-c2-path-inputs"
    c2 = ds / "c2_small_best_seg179"
    _tiny_predictor(2).save_pretrained(str(c2))
    (c2 / "best_metric.json").write_text(json.dumps({"segment": 179,
                                                     "forecast_loss": 2.2944018841}))
    val = ds / "temporal_symbol_validation_v1"
    (val / "processed_datasets").mkdir(parents=True)
    with (val / "processed_datasets" / "val_data.pkl").open("wb") as handle:
        pickle.dump(_val_panel(), handle)
    (val / "sector_labels.json").write_text(json.dumps({"sector_labels": SECTORS}))
    working = tmp_path / "working"
    env = os.environ.copy()
    env.update({
        "KRONOS_LOCAL_TEST": "1", "KRONOS_ALLOW_CPU_WORKER": "1", "KRONOS_POLL_SECONDS": "1",
        "KRONOS_WORKING_ROOT": str(working), "KRONOS_INPUT_ROOT": str(tmp_path / "input"),
        "KRONOS_MODEL_SNAPSHOT": str(snapshot), "KRONOS_DISABLE_SWANLAB": "1",
        "KRONOS_VAL_DATES": "3", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
    })
    proc = subprocess.run([sys.executable, str(STAGED_RUNNER)], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-6000:]
    out = working / "beta_v2_1_c1_val_c2_path_estimators"
    results = out / "results"
    records = json.loads((out / "val_subsample_records.json").read_text())
    dates = sorted({r["asof_date"] for r in records})
    assert len(dates) == 3
    loaded = [json.loads(line) for line in proc.stdout.splitlines()
              if line.startswith('{"phase": "worker_model_loaded"')]
    by_phase = {}
    for event in loaded:
        by_phase.setdefault(event["phase_key"], set()).add(event["model_code"])
    assert all("c2_code_e4b92bb" in p for p in by_phase["c2_n5"])
    assert not any("c2_code_e4b92bb" in p for p in by_phase["b475_n5"] | by_phase["b475_n16"])
    for label, arm, n in (("c2_small_best_seg179", "prod_t065_p80_n5", 5),
                          ("beta_v21_release_best475", "prod_t065_p80_n5", 5),
                          ("beta_v21_release_best475", "prod_t065_p80_n16", 16)):
        for date in dates:
            shard = pd.read_csv(out / "shards" / f"{label}__{arm}__{date}.csv.gz")
            assert {"symbol", "asof_date", "start_index", "return_10d"} <= set(shard.columns)
            assert len(est.path_columns(shard)) == n
        preds = pd.read_csv(results / f"{label}__{arm}_val_predictions.csv.gz")
        assert len(preds) == len(records)
        assert {f"est_{e}" for e in est.ESTIMATORS} <= set(preds.columns)
        summary = json.loads((results / f"{label}__{arm}_summary.json").read_text())
        assert set(summary["estimators"]) == set(est.ESTIMATORS)
        assert summary["sample_count"] == n
    comparison = json.loads((results / "comparison.json").read_text())
    assert comparison["final"] is True and comparison["sealed_oos_read"] is False
    assert comparison["phase_order"] == ["c2_n5", "b475_n5", "b475_n16"]
    assert all(v["status"] == "scored" for v in comparison["phase_status"].values())
    assert set(comparison["arms"]) == {"c2_n5", "b475_n5", "b475_n16"}
    assert "rule1_path_estimator_best475_n5" in comparison["decisions"]
    assert comparison["decisions"]["rule2_blend"]["chosen_weight_on_c2"] in (
        0.0, 0.25, 0.5, 0.75, 1.0)
    assert set(comparison["decisions"]["rule3_style_shrink"]) >= {"c2_n5", "b475_n5"}
    assert "blend_c2_n5__b475_n5" in comparison["extras"]
    assert "sanity_best475_n5_mean" in comparison["extras"]
    assert comparison["c2_minus_best475_n5_mean"]["n"] == 3
    events = [json.loads(line) for line in (out / "run.log").read_text().splitlines()
              if line.startswith("{")]
    scored = [e["phase_key"] for e in events if e.get("phase") == "arm_scored"]
    assert scored == ["c2_n5", "b475_n5", "b475_n16"]
