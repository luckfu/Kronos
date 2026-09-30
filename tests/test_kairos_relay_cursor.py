"""Exercise the actual runner AST without importing CUDA or training models."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module(ROOT / "modernbert_finance/audit_kairos_relay_control_flow.py")
builder = load_module(ROOT / "finetune/build_kairos_lr_probe.py")


@pytest.mark.parametrize("chunk", range(1, 5))
@pytest.mark.parametrize("sizes", ([5, 5, 3], [4, 4, 4], [1, 1, 1], [2, 2, 2, 2, 3], [13], [3, 8]))
def test_every_boundary_and_relay(chunk, sizes):
    path = ROOT / f"finetune/kaggle_kairos_r2_chunk{chunk}/kaggle_kairos_r2_chunk{chunk}.py"
    uninterrupted = audit.run_r2_cursor(path, sizes, 100)
    assert uninterrupted["matches_contiguous_expected"]
    assert uninterrupted["cursor"] == [len(sizes), 0]
    joined = []
    group, offset = 0, 0
    while group < len(sizes):
        result = audit.run_r2_cursor(path, sizes, 1, offset, group)
        assert result["matches_contiguous_expected"]
        assert result["duplicate_visits"] == 0
        joined.extend(result["observed"])
        group, offset = result["cursor"]
    assert joined == uninterrupted["observed"]


@pytest.mark.parametrize("arm", builder.ARMS)
def test_generated_probe(arm, tmp_path):
    destination = builder.build(arm, tmp_path / arm)
    module = load_module(destination / "lr_probe.py")
    assert module.LR_PROBE
    assert module.LEARNING_RATE_OVERRIDE == builder.ARMS[arm]
    assert module.MAX_SEGMENTS_THIS_RUN == 4
    assert module.GPU_BUDGET_SECONDS == 5100
    assert module.SWANLAB_API_KEY_FALLBACK == ""


def test_early_stop_and_coverage_guards(tmp_path):
    module = load_module(builder.build("1e5", tmp_path) / "lr_probe.py")
    assert module.probe_stop_reason(0.9, 0.67, 0, 0.67)[0] == "validation_degradation"
    assert module.probe_stop_reason(0.7, 0.67, 1, 0.67)[0] == "validation_patience"
    assert module.probe_stop_reason(float("nan"), 0.67, 0, 0.67)[0] == "nonfinite_validation"
    assert module.probe_stop_reason(0.60, 0.67, 1, 0.67) == ("", 0.60, 0)
    rows = [{"symbol": "x", "start_index": i, "asof_date": "2024-01-01"} for i in range(3)]
    expected = [module.row_identity(row) for row in rows]
    seen = set()
    assert len(module.verify_segment_coverage(rows, expected, 0, seen)) == 64
    with pytest.raises(RuntimeError, match="duplicate"):
        module.verify_segment_coverage(rows, expected, 0, seen)
    with pytest.raises(RuntimeError, match="prefix"):
        module.verify_segment_coverage(rows[::-1], expected, 0, set())


def test_offline_dashboard_does_not_access_online_url(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    module = load_module(builder.build("1e5", tmp_path) / "lr_probe.py")

    class OfflineRun:
        @property
        def url(self):
            raise ValueError("offline URL unavailable")

    run = OfflineRun()
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "swanlab", SimpleNamespace(init=lambda **kwargs: run))
    assert module.start_swanlab()[1] is run


def test_initial_validation_is_recorded_before_gate(tmp_path):
    source = (builder.build("3e5", tmp_path) / "lr_probe.py").read_text()
    assert source.index("log('probe_initial_validation'") < source.index(
        "if not np.isfinite(probe_initial_score)"
    )
    assert source.index("'initial_validation.json'") < source.index(
        "if not np.isfinite(probe_initial_score)"
    )
    assert "tolerance=0.002" in source


def test_diagnostic_exits_before_training():
    import ast

    tree = ast.parse(builder.SOURCE.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    preflight = next(node for node in main.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "LR_PROBE"
                     and any(isinstance(child, ast.If) and ast.unparse(child.test) == "DIAGNOSTIC_ONLY"
                             for child in node.body))
    diagnostic = next(node for node in preflight.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "DIAGNOSTIC_ONLY")
    assert isinstance(diagnostic.body[-1], ast.Return)
    assert main.body.index(preflight) < next(i for i, node in enumerate(main.body) if isinstance(node, ast.While))
    calls = [ast.unparse(node.func) for node in ast.walk(diagnostic) if isinstance(node, ast.Call)]
    assert "full_validation" in calls
    assert not any("optimizer" in name or "backward" in name for name in calls)


def test_observation_window_and_fail_closed(tmp_path):
    module = load_module(builder.build("1e4", tmp_path) / "lr_probe.py")
    module.EARLY_STOP_PATIENCE = 4
    for completed in range(1, 8):
        reason, _, _ = module.trial_stop_reason(1.4, 0.67, 10, completed)
        assert reason == ""
    assert module.trial_stop_reason(1.4, 0.67, 10, 8)[0] == "validation_patience_after_observation"
    assert module.trial_stop_reason(float("nan"), 0.67, 0, 1)[0] == "nonfinite_validation"
    assert module.trial_stop_reason(0.60, 0.67, 10, 8) == ("", 0.60, 0)


def test_source_artifact_isolation(tmp_path):
    module = load_module(builder.build("1e4", tmp_path / "module") / "lr_probe.py")
    for slug in ("chunk1", "trial"):
        path = tmp_path / "input" / slug / "last_checkpoint.pt"
        path.parent.mkdir(parents=True)
        path.touch()
    assert module.source_files("**/last_checkpoint.pt", "chunk1", tmp_path / "input") == [
        tmp_path / "input/chunk1/last_checkpoint.pt"
    ]
    assert module.source_files("**/last_checkpoint.pt", "missing", tmp_path / "input") == []


def test_rope_numeric_guard_and_canonical_order(tmp_path):
    import torch
    from types import SimpleNamespace

    module = load_module(builder.build("1e4", tmp_path) / "lr_probe.py")
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.rotary_emb = torch.nn.Module()
    model.backbone.config = SimpleNamespace(
        hidden_size=768, num_attention_heads=12,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
        },
    )
    rotary = model.backbone.rotary_emb
    for kind, params in model.backbone.config.rope_parameters.items():
        value = 1.0 / (params["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        rotary.register_buffer(f"{kind}_inv_freq", value, persistent=False)
        rotary.register_buffer(f"{kind}_original_inv_freq", value.clone(), persistent=False)
    module.canonicalize_buffers(model)
    fingerprint = module.rope_fingerprint(model, torch)
    names = [item[0] for item in fingerprint["ordered_buffers"]]
    assert names == sorted(names)
    assert not model.state_dict()
    with torch.no_grad():
        rotary.full_attention_inv_freq.copy_(rotary.sliding_attention_inv_freq)
    with pytest.raises(AssertionError):
        module.rope_fingerprint(model, torch)


def test_bounded_trial_configuration(tmp_path, monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(ROOT / "finetune"))
    trial_builder = load_module(ROOT / "finetune/build_kairos_rope_fixed_trial.py")
    directory = trial_builder.build_trial(tmp_path)
    module = load_module(directory / "lr_probe.py")
    assert module.REPAIRED_TRIAL and module.LR_PROBE and not module.DIAGNOSTIC_ONLY
    assert module.LEARNING_RATE_OVERRIDE == 1e-4
    assert module.GPU_BUDGET_SECONDS == 10800
    assert module.MAX_SEGMENTS_THIS_RUN == 12
    assert module.MIN_OBSERVATION_SEGMENTS == 8
    assert module.EARLY_STOP_PATIENCE == 4
    assert module.SOURCE_KERNEL_SLUG != module.REASSESS_KERNEL_SLUG
    metadata = json.loads((directory / "kernel-metadata.json").read_text())
    assert metadata["id"] == "wynstonliu/kairos-rope-fixed-bounded-trial"
    assert len(metadata["kernel_sources"]) == 2
    source = (directory / "lr_probe.py").read_text()
    assert source.index("if not repeat_ok:") < source.index("scaler.scale(loss).backward()")
