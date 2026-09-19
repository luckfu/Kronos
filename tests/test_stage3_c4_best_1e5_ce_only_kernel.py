import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / 'finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py'
KERNEL = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only' / 'stage3_c4_best_1e5_ce_only.py'
META = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only' / 'kernel-metadata.json'


def test_smoke_lr_seed_and_parent_are_env_configurable():
    source = SMOKE.read_text()
    for token in (
        "LR = float(os.environ.get('STAGE3_LR', '2e-6'))",
        "SEED = int(os.environ.get('STAGE3_SEED', '20260915'))",
        "STAGE3_PARENT_BEST_GLOB",
        "STAGE3_PARENT_BEST_SHA",
        "STAGE3_TOKENIZER_REPO",
        "KRONOS_COVERAGE_SEED=str(SEED)",
        "'--lr', str(LR)",
        "'--seed', str(SEED)",
        'PARENT_BEST_GLOB',
    ):
        assert token in source, token
    assert "'--lr', '2e-6'" not in source
    assert "KRONOS_COVERAGE_SEED='20260915'" not in source


def test_c4_best_1e5_ce_only_is_stage3_rehearsal_from_wc_c4_best():
    source = KERNEL.read_text()
    for token in (
        "STAGE3_TARGET_SEGMENTS='15'",
        "STAGE3_CHUNK='c4_best_1e5_ce_only'",
        "STAGE3_LR='1e-5'",
        "STAGE3_SEED='20260919'",
        "STAGE3_LAMBDA_PATH='0'",
        "STAGE3_MILESTONE_SEGMENTS='5,10'",
        "STAGE3_PARENT_BEST_SHA='9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075'",
        "small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors",
        "STAGE3_TOKENIZER_REPO='NeoQuasar/Kronos-Tokenizer-base'",
        "finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py",
    ):
        assert token in source, token
    assert "STAGE3_LR='2e-6'" not in source
    assert "STAGE3_SEED='20260915'" not in source
    assert "STAGE3_SEED='20260918'" not in source
    assert 'wc_2e5' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
    ]
