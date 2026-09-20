import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_cosine_c1_kernel'
    / 'kaggle_small_0_1_stage2_c2_best_wc_1e5_cosine_c1.py'
)
META = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_1e5_cosine_c1_kernel'
    / 'kernel-metadata.json'
)


def test_wc_1e5_cosine_c1_is_new_stage_from_c4_last():
    source = KERNEL.read_text()
    for token in (
        'OUTPUT_NAME = "small_0.1_stage2_c2_best_wc_1e5_cosine"',
        'SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5_cosine"',
        'COVERAGE_SEED = "20260920"',
        'EXPECTED_PARENT_SEGMENT = 27',
        'EXPECTED_PARENT_BEST_SEGMENT = 21',
        'REFINEMENT_SEGMENTS = 267',
        'MAX_SEGMENTS_PER_RUN = 170',
        'MAX_RUNTIME_SECONDS = 36000',
        'EXPECTED_OPTIMIZER_STEPS = 83571',
        'EXPECTED_C4_LAST_SHA = "c285a652419841c05e3b2c7b696cb4e1d86d92bc6cfae0ec00e30f076d407f7b"',
        'small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/last_state.pt',
        'new_stage_preserve_model_and_adamw_replace_scheduler',
        'KRONOS_SCHEDULER": "uniform_cosine"',
        'KRONOS_SCHEDULER_WARMUP_RATIO": "0"',
        'KRONOS_PREDICTOR_MIN_LR": "1e-6"',
        'KRONOS_CONDITION_MIN_LR": "1e-6"',
        'KRONOS_SCHEDULER_TRANSITION_STATE": str(source_state)',
        'KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1"',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert 'SWANLAB_RUN_ID = "small_0.1_stage2_c2_best_wc_1e5"' not in source
    assert 'COVERAGE_SEED = "20260918"' not in source
    assert 'KRONOS_COVERAGE_EPOCH_OFFSET' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-cosine-c1'
    assert metadata['code_file'] == 'kaggle_small_0_1_stage2_c2_best_wc_1e5_cosine_c1.py'
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
    ]
