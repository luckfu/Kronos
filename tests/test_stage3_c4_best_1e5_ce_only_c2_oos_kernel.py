import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_c2_oos' / 'stage3_c4_best_1e5_ce_only_c2_oos.py'
META = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_c2_oos' / 'kernel-metadata.json'


def test_stage3_1e5_c2_oos_evaluates_only_this_round_best():
    source = KERNEL.read_text()
    for token in (
        'LABEL = "stage3_c4_best_1e5_ce_only_seg80"',
        'EXPECTED_SEGMENT = 80',
        'EXPECTED_STEP = 25040',
        'EXPECTED_TOKEN_CE = 1.8161181209835073',
        'EXPECTED_SHA = "a448e0f760f308c133dd363bbd886403052f046dcdbb49b73a8d4d828a93308f"',
        'stage3_joint_path_smoke/checkpoints/best_model/model.safetensors',
        '"training_performed": False',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'already_measured_not_rerun',
        'STAGE3_1E5_SEG15_D10_POOLED_RANK_IC = 0.11323220411413162',
        'WC_1E5_C4_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.17918741695116097',
    ):
        assert token in source, token
    assert 'milestone' not in source
    assert 'EXPECTED_SEGMENT = 15' not in source
    assert "'--branch', 'master'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only-c2-oos'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only-c2',
    ]
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['machine_shape'] == 'NvidiaTeslaP100'
