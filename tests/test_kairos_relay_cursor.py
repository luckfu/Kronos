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
