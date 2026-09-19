import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage2_c2_best_wc_1e5_c4_best_oos' / 'stage2_c2_best_wc_1e5_c4_best_oos.py'
META = ROOT / 'finetune/kaggle_stage2_c2_best_wc_1e5_c4_best_oos' / 'kernel-metadata.json'


def test_wc_1e5_c4_best_oos_evaluates_only_this_round_best():
    source = KERNEL.read_text()
    for token in (
        'LABEL = "wc_1e5_c4_best_g528"',
        'EXPECTED_LOCAL_SEGMENT = 21',
        'EXPECTED_GLOBAL_SEGMENT = 528',
        'EXPECTED_FORECAST = 2.2728769779205322',
        'EXPECTED_SHA = "9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075"',
        'small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors',
        '"training_performed": False',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'already_measured_not_rerun',
    ):
        assert token in source, token
    assert 'c4_last' not in source
    assert 'c3_best' not in source
    assert "'--branch', 'master'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4-best-oos'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
    ]
