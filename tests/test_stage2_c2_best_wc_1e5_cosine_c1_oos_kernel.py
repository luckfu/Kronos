import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT
    / 'finetune/kaggle_stage2_c2_best_wc_1e5_cosine_c1_oos'
    / 'stage2_c2_best_wc_1e5_cosine_c1_oos.py'
)
META = ROOT / 'finetune/kaggle_stage2_c2_best_wc_1e5_cosine_c1_oos' / 'kernel-metadata.json'


def test_wc_1e5_cosine_c1_oos_evaluates_only_this_round_best():
    source = KERNEL.read_text()
    for token in (
        'LABEL = "wc_1e5_cosine_c1_best_seg148"',
        'EXPECTED_LOCAL_SEGMENT = 148',
        'EXPECTED_FORECAST = 2.2693750858306885',
        'EXPECTED_SHA = "0201e44c93094ff4a277ff9f7cddfb9e534f7d44e578cb74c903612d4e33f055"',
        'EXPECTED_PARENT_COMPLETED = 170',
        'small_0.1_stage2_c2_best_wc_1e5_cosine/checkpoints/best_model/model.safetensors',
        '"training_performed": False',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'already_measured_not_rerun',
        'WC_1E5_C4_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.17918741695116097',
    ):
        assert token in source, token
    assert 'c4_last' not in source
    assert 'last_model/model.safetensors' not in source
    assert "'--branch', 'master'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-cos-c1-oos'
    assert metadata['code_file'] == 'stage2_c2_best_wc_1e5_cosine_c1_oos.py'
    assert metadata['machine_shape'] == 'NvidiaTeslaP100'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-cosine-c1',
    ]
