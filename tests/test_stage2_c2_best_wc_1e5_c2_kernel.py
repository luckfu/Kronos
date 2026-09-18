import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_c2_kernel'
    / 'kaggle_small_0_1_stage2_c2_best_wc_1e5_c2.py'
)
META = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_c2_kernel'
    / 'kernel-metadata.json'
)


def test_c2_best_wc_1e5_c2_continues_c1_last_for_200_segments():
    source = KERNEL.read_text()
    for token in (
        'OUTPUT_NAME = "small_0.1_stage2_c2_best_wc_1e5_c2"',
        'COVERAGE_SEED = "20260918"',
        'COVERAGE_EPOCH_OFFSET = 100',
        'EXPECTED_PARENT_SEGMENT = 100',
        'EXPECTED_PARENT_BEST_SEGMENT = 95',
        'TARGET_SEGMENTS = 200',
        'MAX_SEGMENTS_PER_RUN = 200',
        'FIXED_LR = "1e-5"',
        'EXPECTED_C1_LAST_SHA = "c00fd8096f35496a395721959dd22b2832c1913bc7ce7c51c268577a930729b3"',
        'KRONOS_SCHEDULER": "warmup_constant"',
        'KRONOS_SCHEDULER_WARMUP_RATIO": "0"',
        'KRONOS_EPOCHS": str(TARGET_SEGMENTS)',
        'KRONOS_REQUIRE_FULL_COVERAGE": "0"',
        'KRONOS_COVERAGE_EPOCH_OFFSET": str(COVERAGE_EPOCH_OFFSET)',
        'KRONOS_SCHEDULER_TRANSITION_STATE": str(source_state)',
        'KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1"',
        'c1_last_state_preserve_adamw_new_200_segment_stage',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'finetune/kaggle_kronos_small_v21.py',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert 'KRONOS_SMALL_V21_CONTINUATION_ROOT' not in source
    assert 'FIXED_LR = "3.2e-6"' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-stage2-c2-best-wc-1e5-c2'
    assert metadata['code_file'] == 'kaggle_small_0_1_stage2_c2_best_wc_1e5_c2.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'luckfu/kronos-small-0-1-stage2-c2-best-wc-1e5-c1',
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
    ]
