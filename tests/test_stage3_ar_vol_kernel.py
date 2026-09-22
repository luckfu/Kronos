import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_ar_vol_smoke' / 'stage3_ar_vol_smoke.py'
META = ROOT / 'finetune/kaggle_stage3_ar_vol_smoke' / 'kernel-metadata.json'


def test_ar_vol_smoke_is_one_fresh_segment_at_the_production_decode():
    source = KERNEL.read_text()
    for token in (
        "STAGE3_SOURCE_COMMIT=COMMIT",
        "STAGE3_TARGET_SEGMENTS='1'",
        "STAGE3_CHUNK='ar_vol_smoke'",
        "STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_ar_vol_from_c2_v1'",
        "STAGE3_SEED='20260922'",
        "STAGE3_LAMBDA_PATH='0'",
        "STAGE3_LAMBDA_VOL='0.15'",
        "STAGE3_VOL_TEMPERATURE='0.65'",
        "STAGE3_VOL_TOP_P='0.8'",
        "STAGE3_VOL_SAMPLES='5'",
        "STAGE3_AR_VOL='1'",
        "STAGE3_BATCH='8'",
        "STAGE3_BASELINE_BEFORE_RESUME='1'",
        "os.environ.pop('STAGE3_RESUME_KERNEL', None)",
        "os.environ.pop('STAGE3_CE_RANK', None)",
        'resume_c3=False',
        'teacher_forced_vol=False',
        'ar_ratio_closer_to_1_and_token_ce_increase_lt_0.03',
    ):
        assert token in source, token
    assert "STAGE3_RESUME_KERNEL='" not in source
    assert 'vol_cal_from_c2_best_v1' not in source
    assert 'vol_prod_decode_from_c2_v1' not in source
    assert 'joint_path_alignment_from_c2_best' not in source
    commit = re.search(r"COMMIT = '([0-9a-f]{40})'", source)
    assert commit
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-s3-ar-vol-smoke'
    assert len(metadata['id'].split('/')[1]) <= 50
    assert metadata['code_file'] == 'stage3_ar_vol_smoke.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
    ]


def test_trainer_and_smoke_wire_autoregressive_vol_without_teacher_forcing():
    trainer = (ROOT / 'finetune/train_stage3_path_alignment.py').read_text()
    smoke = (ROOT / 'finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py').read_text()
    model = (ROOT / 'finetune/stage3_training_model.py').read_text()
    objective = (ROOT / 'finetune/stage3_ar_vol.py').read_text()
    for token in (
        "p.add_argument('--ar-vol', action='store_true')",
        'model.no_sync()',
        'autoregressive vol requires lambda_vol>0, lambda_path=0, and ce_rank disabled',
        'Autoregressive vol requires its own SwanLab run id',
        'Refusing to reuse the autoregressive vol dashboard',
        'scalar_keys = scalar_keys + AR_METRIC_KEYS',
    ):
        assert token in trainer, token
    for token in (
        "AR_VOL = os.environ.get('STAGE3_AR_VOL', '0') == '1'",
        "BATCH = int(os.environ.get('STAGE3_BATCH', '32'))",
        "'--batch', str(BATCH)",
        "train_args.append('--ar-vol')",
        'finetune.stage3_ar_probe',
        'checkpoints/last_model',
        'tests/test_stage3_ar_vol.py',
    ):
        assert token in smoke, token
    assert "return 'stage3_ar_vol_calibration_v1'" in model
    assert 'Score-function grads are already in .grad' in model
    assert 'return_generated_tokens=True' in objective
    assert 'top_k=0' in objective
    assert 'adv_flat.detach()' in objective
    assert 'within_window_advantage' in objective
