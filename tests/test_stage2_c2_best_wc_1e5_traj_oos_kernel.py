import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage2_c2_best_wc_1e5_traj_oos' / 'stage2_c2_best_wc_1e5_traj_oos.py'
META = ROOT / 'finetune/kaggle_stage2_c2_best_wc_1e5_traj_oos' / 'kernel-metadata.json'


def test_wc_1e5_traj_oos_covers_saved_late_checkpoints():
    source = KERNEL.read_text()
    for token in (
        '("wc_1e5_c4_best_g528"',
        '("wc_1e5_c4_last_g534"',
        '("wc_1e5_c3_best_g476"',
        '("wc_1e5_c3_last_g507"',
        '("wc_1e5_c2_last_g300"',
        '9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075',
        'c285a652419841c05e3b2c7b696cb4e1d86d92bc6cfae0ec00e30f076d407f7b',
        '7f0dba2304d26c7dd466d463b8e4ee1597370d237bd68106e73258980f448877',
        'ad417595d73431f6d5708b0f76e03f14b822f16fe4ea6991be3cfdcb6e598c68',
        '1bd30e3c66aefd1364967bb906ac77652bcf9bfad89044f385995c3e931f9a49',
        '"training_performed": False',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'already_measured_not_rerun',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-traj-oos'
    assert metadata['kernel_sources'] == [
        'luckfu/kronos-small-0-1-stage2-c2-best-wc-1e5-c2',
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c3',
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
    ]
