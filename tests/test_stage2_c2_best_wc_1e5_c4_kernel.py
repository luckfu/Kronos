import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_c4_kernel'
    / 'kaggle_small_0_1_stage2_c2_best_wc_1e5_c4.py'
)
META = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_c4_kernel'
    / 'kernel-metadata.json'
)


def test_c2_best_wc_1e5_c4_finishes_last_27_slices_from_c3_last():
    source = KERNEL.read_text()
    for token in (
        'OUTPUT_NAME = "small_0.1_stage2_c2_best_wc_1e5_c4"',
        'SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5"',
        'COVERAGE_SEED = "20260918"',
        'COVERAGE_EPOCH_OFFSET = 507',
        'EXPECTED_PARENT_SEGMENT = 207',
        'EXPECTED_PARENT_BEST_SEGMENT = 176',
        'TARGET_SEGMENTS = 27',
        'MAX_SEGMENTS_PER_RUN = 27',
        'FIXED_LR = "1e-5"',
        'EXPECTED_C3_LAST_SHA = "1bd30e3c66aefd1364967bb906ac77652bcf9bfad89044f385995c3e931f9a49"',
        'small_0.1_stage2_c2_best_wc_1e5_c3/checkpoints/last_state.pt',
        'c3_last_state_preserve_adamw_new_27_segment_stage',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        'KRONOS_SCHEDULER_TRANSITION_STATE": str(source_state)',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert 'SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5_c4"' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4'
    assert metadata['code_file'] == 'kaggle_small_0_1_stage2_c2_best_wc_1e5_c4.py'
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c3',
    ]
