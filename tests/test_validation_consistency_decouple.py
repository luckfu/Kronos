"""Structural fix: scalar validation finishes before AR consistency on TPU.

v7 probe SIGKILL'd during Seg1 large validation with consistency_samples=128
because auto_regressive_inference ran inside the teacher-forcing batch loop
with no xm.mark_step per decode step, after fp32 master + AdamW were already
allocated. These tests lock in the decoupled second-pass design and the
per-decode mark_step on the AR path.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
FINETUNE = ROOT / "finetune"
TRAINER = FINETUNE / "train_predictor.py"
KRONOS = ROOT / "model" / "kronos.py"

for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_train_predictor():
    # Prefer the same loader used by the bf16 smoke suite when available.
    try:
        from tests.test_real_kronos_bf16_autocast import load_train_predictor
        return load_train_predictor()
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "kronos_train_predictor_consistency", TRAINER
        )
        module = importlib.util.module_from_spec(spec)
        # Ensure finetune/ is first for config/dataset imports.
        sys.path.insert(0, str(FINETUNE))
        spec.loader.exec_module(module)
        return module


def test_source_consistency_is_second_pass_after_scalar_score():
    source = TRAINER.read_text()
    evaluate = source[
        source.index("def evaluate_validation("):
        source.index("def reset_cuda_peak_memory(")
    ]
    assert "run_return_path_consistency(" in evaluate
    assert "auto_regressive_inference" not in evaluate
    assert evaluate.index("beta_v21_validation_score") < evaluate.index(
        "run_return_path_consistency("
    )
    assert "[VAL] Scalar validation complete" in evaluate
    assert "def run_return_path_consistency(" in source
    assert "def prepare_model_for_validation(" in source
    assert source.count("prepare_model_for_validation(model, device, optimizer)") == 2
    assert "[VAL] Consistency pass starting" in source
    assert "[VAL] Consistency pass finished" in source


def test_ar_inference_calls_xla_mark_step_each_decode(monkeypatch):
    import model.kronos as kronos_mod

    recorded = []
    monkeypatch.setattr(
        kronos_mod, "_xla_mark_step", lambda device: recorded.append(device.type)
    )

    class TinyTok(torch.nn.Module):
        def encode(self, x, half=False):
            batch, seq, _ = x.shape
            # two token streams
            return (
                torch.zeros(batch, seq, dtype=torch.long, device=x.device),
                torch.zeros(batch, seq, dtype=torch.long, device=x.device),
            )

        def decode(self, tokens, half=False):
            pre, post = tokens
            batch, seq = pre.shape
            return torch.zeros(batch, seq, 6, device=pre.device)

    class TinyModel(torch.nn.Module):
        def decode_s1(self, s1, s2, stamp, sector_id=None, size_bucket=None,
                      size_percentile=None):
            batch, seq = s1.shape
            logits = torch.zeros(batch, seq, 4, device=s1.device)
            context = torch.zeros(batch, seq, 8, device=s1.device)
            return logits, context

        def decode_s2(self, context, sample_pre):
            batch, seq, _ = context.shape
            return torch.zeros(batch, seq, 4, device=context.device)

    pred_len = 10
    batch = 2
    lookback = 16
    x = torch.randn(batch, lookback, 6)
    x_stamp = torch.zeros(batch, lookback, 5)
    y_stamp = torch.zeros(batch, pred_len, 5)
    out = kronos_mod.auto_regressive_inference(
        TinyTok(), TinyModel(), x, x_stamp, y_stamp,
        max_context=32, pred_len=pred_len, sample_count=1, sample_logits=False,
    )
    assert out.shape[0] == batch
    assert recorded == ["cpu"] * pred_len


def test_xla_mark_step_helper_noops_without_xla_and_fires_on_xla(monkeypatch):
    import model.kronos as kronos_mod
    import types
    import sys as _sys

    # Non-xla: no-op
    kronos_mod._xla_mark_step(torch.device("cpu"))

    calls = []
    fake_xm = types.ModuleType("torch_xla.core.xla_model")
    fake_xm.mark_step = lambda: calls.append("step")
    fake_core = types.ModuleType("torch_xla.core")
    fake_core.xla_model = fake_xm
    fake_torch_xla = types.ModuleType("torch_xla")
    fake_torch_xla.core = fake_core
    monkeypatch.setitem(_sys.modules, "torch_xla", fake_torch_xla)
    monkeypatch.setitem(_sys.modules, "torch_xla.core", fake_core)
    monkeypatch.setitem(_sys.modules, "torch_xla.core.xla_model", fake_xm)

    kronos_mod._xla_mark_step(SimpleNamespace(type="xla"))
    assert calls == ["step"]


def test_prepare_model_for_validation_clears_grads_and_marks_xla(monkeypatch):
    tp = _load_train_predictor()
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    (model(x) ** 2).sum().backward()
    assert any(p.grad is not None for p in model.parameters())

    marks = []
    monkeypatch.setattr(tp, "xm", SimpleNamespace(mark_step=lambda: marks.append(1)))
    tp.prepare_model_for_validation(
        model, SimpleNamespace(type="xla"), optimizer=opt
    )
    assert all(p.grad is None for p in model.parameters())
    assert marks == [1]


def test_consistency_decoupled_order_and_calibration_skip(monkeypatch, tmp_path):
    """beta_v21_score is computed before any AR call; samples=0 skips AR."""
    from tests.test_real_kronos_bf16_autocast import (
        SyntheticBetaV21Dataset, build_models, c1_config, load_train_predictor,
        assert_finite_metrics, VALIDATION_KEYS,
    )

    tp = load_train_predictor()
    config = c1_config(monkeypatch, tmp_path)
    config["beta_v21_validation_denominators"] = "1,1,1,1,1"
    tokenizer, model = build_models()
    loader = DataLoader(
        SyntheticBetaV21Dataset(6, seed=7, data_type="val"), batch_size=4
    )

    order = []
    real_score = tp.beta_v21_validation_score
    real_ar = tp.auto_regressive_inference

    def tracking_score(*args, **kwargs):
        order.append("score")
        return real_score(*args, **kwargs)

    def tracking_ar(*args, **kwargs):
        order.append("ar")
        return real_ar(*args, **kwargs)

    monkeypatch.setattr(tp, "beta_v21_validation_score", tracking_score)
    monkeypatch.setattr(tp, "auto_regressive_inference", tracking_ar)

    metrics = tp.evaluate_validation(
        model, tokenizer, loader, torch.device("cpu"), config, torch.bfloat16,
        period_names={}, rank=0,
    )
    assert_finite_metrics(metrics, VALIDATION_KEYS)
    assert "score" in order and "ar" in order
    assert order.index("score") < order.index("ar")
    assert metrics["return_path_consistency"]["samples"] == 4

    # Calibration path: consistency_samples=0 must not invoke AR.
    order.clear()
    calibration = dict(config)
    calibration["beta_v21_consistency_samples"] = 0
    metrics = tp.evaluate_validation(
        model, tokenizer, loader, torch.device("cpu"), calibration,
        torch.bfloat16, period_names={}, rank=0,
    )
    assert_finite_metrics(metrics, VALIDATION_KEYS)
    assert order == ["score"]
    assert metrics["return_path_consistency"]["samples"] == 0


def test_consistency_ar_batch_cap_defaults_for_xla():
    tp = _load_train_predictor()
    assert tp.consistency_ar_batch_cap({}, SimpleNamespace(type="xla")) == 8
    assert tp.consistency_ar_batch_cap({}, SimpleNamespace(type="cuda")) == 8
    assert tp.consistency_ar_batch_cap({}, SimpleNamespace(type="cpu")) == 0
    assert tp.consistency_ar_batch_cap(
        {"beta_v21_consistency_ar_batch": 4}, SimpleNamespace(type="xla")
    ) == 4


def test_source_ar_loop_contains_xla_mark_step():
    source = KRONOS.read_text()
    assert "def _xla_mark_step(device):" in source
    tree = ast.parse(source)
    ar = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "auto_regressive_inference"
    )
    calls = [
        node for node in ast.walk(ar)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_xla_mark_step"
    ]
    assert len(calls) == 1
