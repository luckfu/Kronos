import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_c3_kernel'
    / 'kaggle_small_0_1_stage2_c2_best_wc_1e5_c3.py'
)
META = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_c3_kernel'
    / 'kernel-metadata.json'
)


def test_c2_best_wc_1e5_c3_finishes_coverage_from_c2_last():
    source = KERNEL.read_text()
    for token in (
        'OUTPUT_NAME = "small_0.1_stage2_c2_best_wc_1e5_c3"',
        'SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5"',
        'KRONOS_SWANLAB_SEGMENT_OFFSET": str(COVERAGE_EPOCH_OFFSET)',
        'COVERAGE_SEED = "20260918"',
        'COVERAGE_EPOCH_OFFSET = 300',
        'EXPECTED_PARENT_SEGMENT = 200',
        'EXPECTED_PARENT_BEST_SEGMENT = 189',
        'TARGET_SEGMENTS = 234',
        'MAX_SEGMENTS_PER_RUN = 234',
        'FIXED_LR = "1e-5"',
        'EXPECTED_C2_LAST_SHA = "7f0dba2304d26c7dd466d463b8e4ee1597370d237bd68106e73258980f448877"',
        'small_0.1_stage2_c2_best_wc_1e5_c2/checkpoints/last_state.pt',
        'KRONOS_SCHEDULER": "warmup_constant"',
        'KRONOS_SCHEDULER_TRANSITION_STATE": str(source_state)',
        'KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1"',
        'c2_last_state_preserve_adamw_new_234_segment_stage',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        'def resolve_tokenizer():',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'finetune/kaggle_kronos_small_v21.py',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert 'KRONOS_SMALL_V21_CONTINUATION_ROOT' not in source
    assert 'SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5_c3"' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c3'
    assert metadata['code_file'] == 'kaggle_small_0_1_stage2_c2_best_wc_1e5_c3.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == [
        'luckfu/kronos-small-0-1-stage2-c2-best-wc-1e5-c2',
    ]
