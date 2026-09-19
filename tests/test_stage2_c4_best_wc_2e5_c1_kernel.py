import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c4_best_wc_2e5_c1_kernel'
    / 'kaggle_small_0_1_stage2_c4_best_wc_2e5_c1.py'
)
META = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c4_best_wc_2e5_c1_kernel'
    / 'kernel-metadata.json'
)


def test_c4_best_wc_2e5_c1_is_stage2_warmup_constant_100_segments():
    source = KERNEL.read_text()
    for token in (
        'OUTPUT_NAME = "small_0.1_stage2_c4_best_wc_2e5"',
        'SWANLAB_RUN_ID = "small_0.1_stage2_c4_best_wc_2e5"',
        'COVERAGE_SEED = "20260919"',
        'COVERAGE_EPOCH_OFFSET = 0',
        'KRONOS_COVERAGE_EPOCH_OFFSET": str(COVERAGE_EPOCH_OFFSET)',
        'MAX_SEGMENTS_PER_RUN = 100',
        'TARGET_SEGMENTS = 100',
        'FIXED_LR = "2e-5"',
        'KRONOS_SCHEDULER": "warmup_constant"',
        'KRONOS_SCHEDULER_WARMUP_RATIO": "0"',
        'KRONOS_EPOCHS": str(TARGET_SEGMENTS)',
        'KRONOS_REQUIRE_FULL_COVERAGE": "0"',
        'KRONOS_PREDICTOR_LEARNING_RATE": FIXED_LR',
        'KRONOS_PREDICTOR_WARMUP_START_LR": FIXED_LR',
        'KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1"',
        'small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors',
        'EXPECTED_C4_BEST_SHA = "9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075"',
        'EXPECTED_C4_BEST_SEGMENT = 21',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'c4_best_weights_fresh_adamw',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        'finetune/kaggle_kronos_small_v21.py',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert 'KRONOS_SCHEDULER_TRANSITION_STATE' not in source
    assert 'SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5"' not in source
    assert 'FIXED_LR = "1e-5"' not in source
    assert 'COVERAGE_SEED = "20260918"' not in source
    assert 'COVERAGE_EPOCH_OFFSET = 507' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage2-c4-best-wc-2e5-c1'
    assert metadata['code_file'] == 'kaggle_small_0_1_stage2_c4_best_wc_2e5_c1.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
    ]
