import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_3p2e6_c1_kernel'
    / 'kaggle_small_0_1_stage2_c2_best_wc_3p2e6_c1.py'
)
META = (
    ROOT
    / 'finetune/kaggle_small_0_1_stage2_c2_best_wc_3p2e6_c1_kernel'
    / 'kernel-metadata.json'
)


def test_c2_best_wc_3p2e6_c1_is_stage2_warmup_constant_from_c2_best():
    source = KERNEL.read_text()
    for token in (
        'OUTPUT_NAME = "small_0.1_stage2_c2_best_wc_3p2e6"',
        'COVERAGE_SEED = "20260917"',
        'MAX_SEGMENTS_PER_RUN = 30',
        'FIXED_LR = "3.2e-6"',
        'KRONOS_SCHEDULER": "warmup_constant"',
        'KRONOS_SCHEDULER_WARMUP_RATIO": "0"',
        'KRONOS_PREDICTOR_LEARNING_RATE": FIXED_LR',
        'KRONOS_CONDITION_LEARNING_RATE": FIXED_LR',
        'KRONOS_PREDICTOR_WARMUP_START_LR": FIXED_LR',
        'KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION": "1"',
        'KRONOS_COVERAGE_PASSES": "1"',
        'checkpoints/best_model/model.safetensors',
        'EXPECTED_C2_BEST_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'c2_best_weights_fresh_adamw',
        'finetune/kaggle_kronos_small_v21.py',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert '"--branch", "master"' not in source
    assert 'COVERAGE_SEED = "20260910"' not in source
    assert 'COVERAGE_SEED = "20260912"' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-stage2-c2-best-wc-3p2e6-c1'
    assert metadata['code_file'] == 'kaggle_small_0_1_stage2_c2_best_wc_3p2e6_c1.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
    ]
