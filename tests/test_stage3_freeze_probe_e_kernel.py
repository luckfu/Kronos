import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_freeze_probe_e/stage3_freeze_probe_e.py'
META = ROOT / 'finetune/kaggle_stage3_freeze_probe_e/kernel-metadata.json'


def test_freeze_probe_e_kernel_is_forecast_head_five_segment_weighted_ce():
    source = KERNEL.read_text()
    required = (
        "STAGE3_TARGET_SEGMENTS='5'",
        "STAGE3_CHUNK='freeze_probe_e'",
        "STAGE3_TRAINABLE_MASK='forecast_head'",
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
    assert metadata['id'] == 'luckfu/kronos-small-0-1-stage3-freeze-probe-e'
    assert metadata['code_file'] == 'stage3_freeze_probe_e.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == ['smmt315/kronos-small-0-1-stage2-cosine-refinement-c2']
