import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_vol_prod_smoke' / 'stage3_vol_prod_smoke.py'
META = ROOT / 'finetune/kaggle_stage3_vol_prod_smoke' / 'kernel-metadata.json'


def test_vol_prod_smoke_matches_locked_production_decode():
    source = KERNEL.read_text()
    for token in (
        "COMMIT = '2da36f03513a975022183e1371947efd950013d8'",
        "STAGE3_SOURCE_COMMIT=COMMIT",
        "STAGE3_TARGET_SEGMENTS='8'",
        "STAGE3_CHUNK='vol_prod_smoke'",
        "STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_vol_prod_decode_from_c2_v1'",
        "STAGE3_SEED='20260921'",
        "STAGE3_LAMBDA_PATH='0'",
        "STAGE3_LAMBDA_VOL='0.15'",
        "STAGE3_VOL_TEMPERATURE='0.65'",
        "STAGE3_VOL_TOP_P='0.8'",
        "STAGE3_VOL_SAMPLES='5'",
        "STAGE3_BASELINE_BEFORE_RESUME='1'",
        "STAGE3_MAX_RUNTIME_SECONDS='36000'",
        "STAGE3_HARD_TIMEOUT_SECONDS='39600'",
        "os.environ.pop('STAGE3_RESUME_KERNEL', None)",
        "os.environ.pop('STAGE3_CE_RANK', None)",
        'resume_c3=False',
        'vol_temperature=0.65',
        'vol_samples=5',
    ):
        assert token in source, token
    assert "STAGE3_RESUME_KERNEL='" not in source
    assert 'vol_cal_from_c2_best_v1' not in source
    assert 'joint_path_alignment_from_c2_best' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-s3-vol-prod-smoke'
    assert len(metadata['id'].split('/')[1]) <= 50
    assert metadata['code_file'] == 'stage3_vol_prod_smoke.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
    ]
    assert 'kronos-small-0-1-stage3-joint-path-c3' not in metadata['kernel_sources']
    import re
    assert re.search(r"COMMIT = '([0-9a-f]{40})'", source)
