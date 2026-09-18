import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_stage2_c2_best_wc_1e5_best_oos'
    / 'stage2_c2_best_wc_1e5_best_oos.py'
)
META = ROOT / 'finetune/kaggle_stage2_c2_best_wc_1e5_best_oos' / 'kernel-metadata.json'


def test_wc_1e5_best_oos_evaluates_campaign_best_without_training():
    source = KERNEL.read_text()
    for token in (
        'LABEL = "wc_1e5_best_seg189"',
        'EXPECTED_WC_BEST_SEGMENT = 189',
        'EXPECTED_WC_BEST_FORECAST = 2.279949903488159',
        'EXPECTED_WC_BEST_SHA = "8fb3d3d7a76fc2bb0f0277d8682b19c912a3c5b657c91416a3d5e3b5e7c3e268"',
        'C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.18440434213929857',
        'small_0.1_stage2_c2_best_wc_1e5_c2/checkpoints/best_model/model.safetensors',
        '"c2_best_segment_179"',
        '"training_performed": False',
        '"c2-alpha-oos-evaluation"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'evaluate_predictions(',
        '20260906',
        'EXPECTED_C2_BEST_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert 'KRONOS_SMALL_V21_STAGE' not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-stage2-c2-best-wc-1e5-best-oos'
    assert metadata['code_file'] == 'stage2_c2_best_wc_1e5_best_oos.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaP100'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'luckfu/kronos-small-0-1-stage2-c2-best-wc-1e5-c2',
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
        'smmt315/kronos-small-0-1-c2-alpha-oos-evaluation',
    ]
