import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_oos' / 'stage3_c4_best_1e5_ce_only_oos.py'
META = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_oos' / 'kernel-metadata.json'


def test_stage3_1e5_oos_evaluates_only_this_round_best():
    source = KERNEL.read_text()
    for token in (
        'LABEL = "stage3_c4_best_1e5_ce_only_seg15"',
        'EXPECTED_SEGMENT = 15',
        'EXPECTED_STEP = 4695',
        'EXPECTED_TOKEN_CE = 1.919825055859587',
        'EXPECTED_SHA = "20e3bb6fa401d4a739054bdf88e3ed38331a45fea322b7287a6e16e8c8c88113"',
        'stage3_joint_path_smoke/checkpoints/best_model/model.safetensors',
        '"training_performed": False',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'already_measured_not_rerun',
        'WC_1E5_C4_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.17918741695116097',
        'P0A_2E6_LAST_D10_POOLED_RANK_IC = 0.0642869478967645',
    ):
        assert token in source, token
    assert 'seg05' not in source
    assert 'seg10' not in source
    assert 'c4_last' not in source
    assert "'--branch', 'master'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only-oos'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only',
    ]
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['machine_shape'] == 'NvidiaTeslaP100'
