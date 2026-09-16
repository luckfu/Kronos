import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / 'finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py'
KERNEL = ROOT / 'finetune/kaggle_stage3_freeze_probe_b/stage3_freeze_probe_b.py'
META = ROOT / 'finetune/kaggle_stage3_freeze_probe_b/kernel-metadata.json'


def test_smoke_fetches_pinned_commit_not_master_tip():
    source = SMOKE.read_text()
    assert "['git', 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT]" in source
    assert "'--branch', 'master'" not in source
    assert 'STAGE3_TRAINABLE_MASK' in source
    assert "'--trainable-mask', TRAINABLE_MASK" in source
    assert 'tests/test_stage3_trainable_mask.py' in source
    assert 'freeze_audit.json' in source


def test_freeze_probe_b_kernel_is_dep_layer_five_segment_weighted_ce():
    source = KERNEL.read_text()
    required = (
        "STAGE3_TARGET_SEGMENTS='5'",
        "STAGE3_CHUNK='freeze_probe_b'",
        "STAGE3_TRAINABLE_MASK='dependency_layer'",
        "STAGE3_LAMBDA_PATH='0'",
        "STAGE3_CE_RANK='1'",
        "STAGE3_LAMBDA_RANK='0'",
        "STAGE3_HISTORY_LOSS_WEIGHT='0.02'",
        "STAGE3_MILESTONE_SEGMENTS='1,3,5'",
        "STAGE3_BASELINE_BEFORE_RESUME='1'",
        "os.environ.pop('STAGE3_RESUME_KERNEL', None)",
    )
    for declaration in required:
        assert declaration in source, declaration
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-stage3-freeze-probe-b'
    assert metadata['code_file'] == 'stage3_freeze_probe_b.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == ['smmt315/kronos-small-0-1-stage2-cosine-refinement-c2']
