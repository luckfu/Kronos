"""CPU smoke tests: no data, no checkpoints, no GPU, no network.

Purpose: a fresh machine can verify the repo is wired correctly before it has
the 4.3 GB dataset or any model weights.

    pytest tests/test_smoke_cpu.py -q

On the macOS conda base (torch 2.13 + OpenMP duplication) run with:

    KMP_DUPLICATE_LIB_OK=TRUE DYLD_LIBRARY_PATH=/opt/miniconda3/lib \
        python3 -m pytest tests/test_smoke_cpu.py -q
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
for path in (ROOT, ROOT / "finetune"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

torch = pytest.importorskip("torch")


def _tiny_kronos(module, **kwargs):
    return module.Kronos(
        s1_bits=4,
        s2_bits=4,
        n_layers=1,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=True,
        num_sectors=8,
        context_layer=0,
        use_size_percentile=True,
        **kwargs,
    )


def test_tiny_model_forward_cpu():
    """One forward pass on synthetic tokens; shapes and the cond path are live."""
    from model import kronos as head_kronos

    torch.manual_seed(0)
    model = _tiny_kronos(head_kronos).eval()

    ids_s1 = torch.randint(0, 16, (2, 12))
    ids_s2 = torch.randint(0, 16, (2, 12))
    sector = torch.tensor([0, 3])
    percentile = torch.tensor([0.2, 0.9])

    with torch.no_grad():
        s1_logits, s2_logits = model(
            ids_s1,
            ids_s2,
            use_teacher_forcing=True,
            s1_targets=ids_s1,
            sector_id=sector,
            size_percentile=percentile,
        )

    assert s1_logits.shape == (2, 12, 16)
    assert s2_logits.shape == (2, 12, 16)
    assert torch.isfinite(s1_logits).all()
    assert torch.isfinite(s2_logits).all()

    # Sector/size embeddings start at zero (bare-add contract). Fill them and
    # check the injection path actually moves the logits.
    with torch.no_grad():
        model.sector_emb.weight.normal_()
        model.size_mlp[0].weight.normal_()
        model.size_mlp[0].bias.normal_()
        base = model(
            ids_s1, ids_s2, use_teacher_forcing=True, s1_targets=ids_s1,
            sector_id=torch.zeros(2, dtype=torch.long), size_percentile=percentile,
        )[0]
        other = model(
            ids_s1, ids_s2, use_teacher_forcing=True, s1_targets=ids_s1,
            sector_id=torch.full((2,), 3, dtype=torch.long), size_percentile=percentile,
        )[0]

    assert not torch.allclose(base, other), "sector conditioning had no effect"


def test_condition_scale_snapshot(tmp_path):
    """The archived 2x2 code accepts use_condition_scale; HEAD does not.

    The snapshot is a stand-in package: HEAD supplies __init__.py/module.py
    (byte-identical to the payload), the snapshot supplies kronos.py.
    """
    snapshot = ROOT / "finetune" / "cond_identity" / "payload_snapshot" / "model" / "kronos.py"
    if not snapshot.is_file():
        pytest.skip("cond_identity snapshot not present")

    package = tmp_path / "model"
    package.mkdir()
    shutil.copy(ROOT / "model" / "__init__.py", package / "__init__.py")
    shutil.copy(ROOT / "model" / "module.py", package / "module.py")
    shutil.copy(snapshot, package / "kronos.py")

    code = """
import torch
from model.kronos import Kronos

def build(**kwargs):
    return Kronos(
        s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True, num_sectors=8, context_layer=0,
        use_size_percentile=True, **kwargs,
    )

off = build()
assert off.condition_scale is None, "default must stay bare-add (no gamma)"

on = build(use_condition_scale=True, condition_scale_init=25.0)
assert on.condition_scale is not None
assert abs(float(on.condition_scale) - 25.0) < 1e-6
print("snapshot-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "snapshot-ok" in result.stdout


def test_cond_identity_harness_imports_without_running():
    """Launch script must stay importable: constants + resolve_c2, no side effects."""
    import importlib.util

    path = ROOT / "finetune" / "cond_identity" / "launch_cond_identity_2x2.py"
    if not path.is_file():
        pytest.skip("cond identity harness not present")

    spec = importlib.util.spec_from_file_location("cond_identity_20261027", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DATASET == "smmt315/a-share-120d-temporal-symbol-holdout"
    assert callable(module.main)
    assert callable(module.resolve_c2)
