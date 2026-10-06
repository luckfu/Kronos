"""Sealed-OOS gen-return final eval kernel for cosine-pilot Seg9 + Seg0 (Best@475).

Covers: builder variant / staged metadata, Seg9-first task order, dynamic
claims vs legacy round-robin, paired stats, and a CPU end-to-end run of the
staged runner (tiny models, synthetic sealed package, two workers) checking that
each checkpoint is scored and saved as soon as its shards land.
"""

import base64
import hashlib
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
RUNNER = FINETUNE / "kaggle_beta_v21_c1_gen_return_oos_pilot.py"
BUILDER = FINETUNE / "build_kaggle_beta_v21_c1_gen_return_oos_kernel.py"
STAGING = FINETUNE / "kaggle_beta_v21_c1_gen_return_oos_pilot_seg9_kernel"
STAGED_RUNNER = STAGING / RUNNER.name
DOCKER_SHA = "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
SEG9_SHA = "f9d3da03f8e5b55824bff28e31e00cc039ee021126d71891bda7b5daeb76e3c8"
LAUNCH_OWNER = "wynstonliu"
BEST475_SHA = "e1bd55842996b7690a21c34c4d74e1128702bca9c16164788b741e3b5d052f97"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load(name, path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load("gen_return_oos_pilot_runner", STAGED_RUNNER)


@pytest.fixture(scope="module")
def evaluator():
    return load("evaluate_beta_v21_generative_return_oos_t", FINETUNE / "evaluate_beta_v21_generative_return_oos.py")


def test_builder_variant_and_staged_metadata():
    builder = load("gen_return_builder_t", BUILDER)
    variant = builder.VARIANTS["pilot_seg9"]
    assert variant["kernel_id"] == "luckfu/kronos-beta-v21-c1-gen-return-oos-pilot-seg9"
    assert variant["arm_filter"] == ("prod_t065_p80_n5",)
    assert variant["runner"] == RUNNER
    # Legacy variants keep their runner and Seg155 dataset.
    for name in ("both", "prod"):
        meta = builder.build_metadata(builder.VARIANTS[name])
        assert meta["code_file"] == "kaggle_beta_v21_c1_gen_return_oos.py"
        assert "luckfu/kronos-beta-v21-c1-seg155-forecast-best" in meta["dataset_sources"]
    # Default owner keeps the legacy ids unchanged.
    assert builder.resolve_variant("pilot_seg9") == variant
    for name in ("both", "prod"):
        assert builder.resolve_variant(name) == builder.VARIANTS[name]
    staged = json.loads((STAGING / "kernel-metadata.json").read_text())
    # Staging is built for the launching account (--owner wynstonliu); the input
    # datasets stay under luckfu (public holdout + group-shared Seg9 weights).
    assert staged == builder.build_metadata(
        builder.resolve_variant("pilot_seg9", owner=LAUNCH_OWNER))
    assert staged["id"] == f"{LAUNCH_OWNER}/kronos-beta-v21-c1-gen-return-oos-pilot-seg9"
    assert staged["code_file"] == RUNNER.name
    assert staged["is_private"] is True
    assert staged["enable_gpu"] is True and staged["enable_tpu"] is False
    assert staged["machine_shape"] == "NvidiaTeslaT4"
    assert staged["docker_image"].endswith(DOCKER_SHA)
    assert staged["dataset_sources"] == [
        "luckfu/a-share-120d-temporal-symbol-holdout",
        "luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9",
    ]


def test_builder_owner_options_reown_kernel_and_datasets():
    builder = load("gen_return_builder_owner_t", BUILDER)
    v = builder.resolve_variant("pilot_seg9", owner="someone", dataset_owner="other")
    assert v["kernel_id"] == "someone/kronos-beta-v21-c1-gen-return-oos-pilot-seg9"
    assert v["dataset_sources"] == (
        "other/a-share-120d-temporal-symbol-holdout",
        "other/kronos-beta-v21-c1-cosine-pilot-best475-seg9",
    )
    # resolve_variant must not mutate the module-level table.
    assert builder.VARIANTS["pilot_seg9"]["kernel_id"].startswith("luckfu/")
    meta = builder.build_metadata(v)
    assert meta["id"] == v["kernel_id"]
    assert meta["dataset_sources"] == list(v["dataset_sources"])


def test_staged_runner_matches_tracked_runner_and_constants(runner):
    tracked = RUNNER.read_text()
    staged = STAGED_RUNNER.read_text()
    blob = re.compile(r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""', re.S)
    assert blob.search(tracked).group(0) == blob.search(staged).group(0)
    assert runner.KERNEL_ID == f"{LAUNCH_OWNER}/kronos-beta-v21-c1-gen-return-oos-pilot-seg9"
    assert runner.OUTPUT_NAME == "beta_v2_1_c1_gen_return_oos_pilot_seg9"
    assert runner.ARM_FILTER == ("prod_t065_p80_n5",)
    assert [c["label"] for c in runner.CHECKPOINTS] == [
        "pilot_seg009", "seg000_beta_v21_release_best475"]
    assert runner.CHECKPOINTS[0]["sha256"] == SEG9_SHA
    assert runner.CHECKPOINTS[1]["sha256"] == BEST475_SHA
    assert runner.EXPECTED_EVAL_NAME == "kronos_beta_v2_time_oos_through_20260903"
    assert runner.EXPECTED_SIGNAL_DATES == 18 and runner.EXPECTED_SAMPLES == 92751
    assert runner.HARD_LIMIT_SECONDS < 12 * 3600
    assert abs(np.mean(list(runner.SEG155_OOS_PROD_BY_DATE.values())) - 0.11702) < 1e-4
    assert abs(np.mean(list(runner.C2_OOS_PROD_BY_DATE.values())) - 0.17687) < 1e-4


def test_embedded_archive_carries_multi_checkpoint_worker(runner):
    raw = base64.b64decode(runner.EMBEDDED_KRONOS_ARCHIVE_B64.strip())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        names = archive.getnames()
        source = archive.extractfile(
            "Kronos/finetune/evaluate_beta_v21_generative_return_oos.py").read().decode()
    assert "Kronos/model/kronos.py" in names
    assert source == (FINETUNE / "evaluate_beta_v21_generative_return_oos.py").read_text()
    assert "dynamic_claims" in source and "def shard_name" in source


def test_prod_arm_is_locked(evaluator):
    arm = {a["name"]: a for a in evaluator.ARMS}["prod_t065_p80_n5"]
    assert (arm["temperature"], arm["top_p"], arm["sample_count"], arm["seed"]) == (
        0.65, 0.8, 5, 20260906)


def test_tasks_are_checkpoint_major_seg9_first(runner):
    arms = [{"name": "prod_t065_p80_n5"}]
    dates = [f"2026-08-{d:02d}" for d in (11, 12, 13)]
    tasks = runner.build_tasks(runner.CHECKPOINTS, arms, dates)
    assert [t["checkpoint"] for t in tasks] == ["pilot_seg009"] * 3 + [
        "seg000_beta_v21_release_best475"] * 3
    assert [t["date"] for t in tasks[:3]] == dates


def test_dynamic_claims_take_all_seg9_before_seg0(tmp_path, runner, evaluator):
    dates = [f"d{i:02d}" for i in range(18)]
    tasks = runner.build_tasks(runner.CHECKPOINTS, [{"name": "prod_t065_p80_n5"}], dates)
    plan = {"tasks": tasks, "dynamic_claims": True, "claims": str(tmp_path / "claims")}
    workers = [evaluator.iter_worker_tasks(plan, r, 2) for r in range(2)]
    taken, live = [], [0, 1]
    turn = 0
    while live:  # uneven interleaving: rank 0 takes two per turn
        rank = live[turn % len(live)]
        for _ in range(2 if rank == 0 else 1):
            try:
                taken.append((rank, next(workers[rank])))
            except StopIteration:
                live.remove(rank)
                break
        turn += 1
    names = [evaluator.shard_name(t) for _, t in taken]
    assert len(names) == len(set(names)) == 36
    labels = [t["checkpoint"] for _, t in taken]
    assert labels == ["pilot_seg009"] * 18 + ["seg000_beta_v21_release_best475"] * 18
    assert evaluator.shard_name(tasks[0]) == "pilot_seg009__prod_t065_p80_n5_d00.csv.gz"


def test_legacy_round_robin_and_shard_names_unchanged(evaluator):
    tasks = [{"arm": "prod_t065_p80_n5", "date": f"d{i}"} for i in range(5)]
    plan = {"tasks": tasks}
    assert [t["date"] for t in evaluator.iter_worker_tasks(plan, 0, 2)] == ["d0", "d2", "d4"]
    assert [t["date"] for t in evaluator.iter_worker_tasks(plan, 1, 2)] == ["d1", "d3"]
    assert evaluator.shard_name(tasks[0]) == "prod_t065_p80_n5_d0.csv.gz"


def test_paired_stats_matches_scipy(runner):
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = {f"d{i}": float(x) for i, x in enumerate(rng.normal(0.2, 0.05, 18))}
    b = {f"d{i}": float(x) for i, x in enumerate(rng.normal(0.18, 0.05, 18))}
    out = runner.paired_stats(a, b)
    keys = sorted(a)
    t, _ = stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys])
    assert out["n"] == 18
    assert math.isclose(out["t"], float(t), rel_tol=1e-9)
    assert out["wins"] == sum(a[k] > b[k] for k in keys)


# ---------------------------------------------------------------------------
# CPU end-to-end: staged runner, tiny models, synthetic sealed package.

SECTORS = ["s0", "s1"]
N_DAYS = 160
LOOKBACK, PREDICT = 120, 10


def _tiny_models(tmp_path):
    from model.kronos import Kronos, KronosTokenizer

    tok = KronosTokenizer(
        d_in=6, d_model=16, n_heads=2, ff_dim=32, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
    ).eval()

    def predictor(seed, aux):
        import torch

        torch.manual_seed(seed)
        return Kronos(
            s1_bits=4, s2_bits=4, n_layers=2, d_model=16, n_heads=2, ff_dim=32,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            token_dropout_p=0.0, learn_te=True, num_sectors=len(SECTORS) + 1,
            context_layer=1, use_size_percentile=True, use_beta_v21_auxiliary=aux,
        ).eval()

    snapshot = tmp_path / "modelscope_snapshot"
    tok.save_pretrained(str(snapshot / "tokenizer"))
    predictor(1, True).save_pretrained(str(snapshot))  # stands in for Best@475 (aux heads)
    seg9 = (tmp_path / "input" / "datasets" / "luckfu"
            / "kronos-beta-v21-c1-cosine-pilot-best475-seg9" / "checkpoints" / "best_model")
    predictor(2, False).save_pretrained(str(seg9))
    return snapshot, seg9


def _sealed_package(root: Path, n_symbols=6, n_dates=3):
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2026-01-01", periods=N_DAYS)
    panel, records = {}, []
    for s in range(n_symbols):
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, N_DAYS)))
        frame = pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.003, N_DAYS)),
            "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": rng.uniform(1e5, 2e5, N_DAYS), "amount": rng.uniform(1e6, 2e6, N_DAYS),
            "sector": SECTORS[s % 2], "size_percentile": (s + 0.5) / n_symbols,
        }, index=index)
        symbol = f"{600000 + s}"
        panel[symbol] = frame
        for d in range(n_dates):
            start = d
            asof = index[start + LOOKBACK - 1]
            target = index[start + LOOKBACK - 1 + PREDICT]
            records.append({
                "set": "incremental_future_all", "symbol": symbol, "start_index": start,
                "asof_date": str(asof.date()), "target_date": str(target.date()),
                "direction": None,
                "return_10d": float(close[start + LOOKBACK - 1 + PREDICT]
                                    / close[start + LOOKBACK - 1] - 1),
            })
    root.mkdir(parents=True, exist_ok=True)
    with (root / "panel.pkl").open("wb") as handle:
        pickle.dump(panel, handle)
    with (root / "samples.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    dates = sorted({r["asof_date"] for r in records})
    manifest = {
        "name": "kronos_beta_v2_time_oos_through_20260903",
        "purpose": "evaluation_only_never_train_or_tune",
        "temporal_isolation": {"targets_strictly_after_training_target_end": True,
                               "incremental_signal_start": dates[0],
                               "incremental_signal_end": dates[-1]},
        "artifacts": {"panel_file": "panel.pkl", "samples_file": "samples.jsonl",
                      "panel_sha256": sha(root / "panel.pkl"),
                      "samples_sha256": sha(root / "samples.jsonl")},
        "model_contract": {"sector_labels": SECTORS},
    }
    (root / "evaluation_manifest.json").write_text(json.dumps(manifest))
    return records, dates


def test_end_to_end_cpu_scores_each_checkpoint_seg9_first(tmp_path):
    pytest.importorskip("torch")
    snapshot, seg9 = _tiny_models(tmp_path)
    package = tmp_path / "input" / "datasets" / "luckfu" / "a-share-120d-temporal-symbol-holdout" / "pkg"
    records, dates = _sealed_package(package)
    working = tmp_path / "working"
    env = os.environ.copy()
    env.update({
        "KRONOS_LOCAL_TEST": "1",
        "KRONOS_ALLOW_CPU_WORKER": "1",
        "KRONOS_POLL_SECONDS": "1",
        "KRONOS_WORKING_ROOT": str(working),
        "KRONOS_INPUT_ROOT": str(tmp_path / "input"),
        "KRONOS_MODEL_SNAPSHOT": str(snapshot),
        "KRONOS_DISABLE_SWANLAB": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    proc = subprocess.run([sys.executable, str(STAGED_RUNNER)], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    out = working / "beta_v2_1_c1_gen_return_oos_pilot_seg9"
    results = out / "results"
    labels = ["pilot_seg009", "seg000_beta_v21_release_best475"]
    for label in labels:
        stem = f"{label}_prod_t065_p80_n5"
        summary = json.loads((results / f"{stem}_summary.json").read_text())
        assert summary["samples"] == len(records)
        assert summary["signal_dates"] == len(dates)
        assert summary["decode"]["sample_count"] == 5
        assert summary["decode"]["temperature"] == 0.65
        assert len(summary["by_signal_date"]) == len(dates)
        assert set(summary["top_bottom_decile_by_date"]) == set(dates)
        preds = pd.read_csv(results / f"{stem}_predictions.csv.gz")
        assert len(preds) == len(records) and set(preds["model"]) == {label}
        assert len(pd.read_csv(results / f"{stem}_by_date.csv")) == len(dates)
        for date in dates:  # per-date predictions persisted immediately
            assert (out / "shards" / f"{label}__prod_t065_p80_n5_{date}.csv.gz").is_file()
    comparison = json.loads((results / "comparison.json").read_text())
    assert comparison["final"] is True and comparison["incomplete"] == []
    assert comparison["checkpoint_order"] == labels
    assert [c["label"] for c in comparison["checkpoints"]] == labels
    pair = comparison["primary_pair_seg9_minus_seg0"]["prod_t065_p80_n5"]
    assert pair["n"] == len(dates) and pair["a"] == "pilot_seg009"
    progress = json.loads((results / "progress.json").read_text())
    assert progress["shards_done"] == progress["shards_total"] == 2 * len(dates)
    log = (out / "run.log").read_text().splitlines()
    events = [json.loads(line) for line in log if line.startswith("{")]
    scored = [e["checkpoint"] for e in events if e.get("phase") == "checkpoint_scored"]
    assert scored == labels
    assert sum(e.get("phase") == "shard_scored" for e in events) == 2 * len(dates)
    first_seg0_shard = min(i for i, e in enumerate(events) if e.get("phase") == "shard_scored"
                           and e["checkpoint"] == labels[1])
    seg9_scored = next(i for i, e in enumerate(events) if e.get("phase") == "checkpoint_scored")
    seg9_shards = [i for i, e in enumerate(events) if e.get("phase") == "shard_scored"
                   and e["checkpoint"] == labels[0]]
    assert max(seg9_shards) < seg9_scored
    assert len(seg9_shards) == len(dates) and min(seg9_shards) < first_seg0_shard
    # Claims: every Seg9 task was claimed before any Seg0 task.
    claims = {p.name: json.loads(p.read_text())["time"] for p in (out / "claims").iterdir()}
    seg9_times = [t for n, t in claims.items() if n.startswith("pilot_seg009__")]
    seg0_times = [t for n, t in claims.items() if n.startswith("seg000_")]
    assert len(seg9_times) == len(seg0_times) == len(dates)
    assert max(seg9_times) <= min(seg0_times)
