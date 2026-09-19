import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_c2' / 'stage3_c4_best_1e5_ce_only_c2.py'
META = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_c2' / 'kernel-metadata.json'


def test_c2_resumes_1e5_ce_only_for_eleven_hours():
    source = KERNEL.read_text()
    for token in (
        "STAGE3_RESUME_KERNEL='user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only'",
        "STAGE3_TARGET_SEGMENTS='80'",
        "STAGE3_CHUNK='c2'",
        "STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_c4_best_1e5_ce_only_v1'",
        "STAGE3_LR='1e-5'",
        "STAGE3_SEED='20260919'",
        "STAGE3_LAMBDA_PATH='0'",
        "STAGE3_MAX_RUNTIME_SECONDS='39600'",
        "STAGE3_HARD_TIMEOUT_SECONDS='42000'",
        "STAGE3_BASELINE_BEFORE_RESUME='0'",
        "STAGE3_MILESTONE_SEGMENTS='20,30,40,50,60,70,80'",
        "'completed_segments': 15",
        "'next_epoch': 15",
        "'step': 4695",
        'next_segment=16',
        'last_segment=80',
        'resume_step=4695',
        'new_segments=65',
        "os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)",
        "os.environ.pop('STAGE3_CE_RANK', None)",
        "STAGE3_SOURCE_COMMIT=COMMIT",
        "COMMIT = 'f73c3e04612f4904c58cb18bb23800a6ed0f33fb'",
    ):
        assert token in source, token
    assert "STAGE3_LR='2e-6'" not in source
    assert "STAGE3_SEED='20260918'" not in source
    assert "STAGE3_CE_RANK='1'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only-c2'
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
        'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only',
    ]
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
