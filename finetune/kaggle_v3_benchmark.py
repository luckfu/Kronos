"""Kaggle Kernel entrypoint for the three-segment V3 GPU benchmark."""

import os
import subprocess
from pathlib import Path


repo = Path('/kaggle/working/Kronos')
if not (repo / '.git').is_dir():
    subprocess.run(
        ['git', 'clone', 'https://github.com/luckfu/Kronos.git', str(repo)],
        check=True,
    )

env = os.environ.copy()
env.setdefault('KRONOS_KAGGLE_DATASET_INPUT', '/kaggle/input/kronos-train-set-a')
env['KRONOS_KAGGLE_ROOT'] = '/kaggle/working/kronos_a_share_v3_benchmark'
env['KRONOS_PREDICTOR_SAVE_FOLDER'] = 'a_share_size_full_market_v3_kaggle_benchmark'
env['KRONOS_EPOCHS'] = '3'
env['KRONOS_COVERAGE_PASSES'] = '1'
env['KRONOS_REQUIRE_FULL_COVERAGE'] = '0'
env['KRONOS_EARLY_STOPPING_PATIENCE'] = '0'
env['KRONOS_RESUME_TRAINING'] = '0'

subprocess.run(['bash', 'finetune/kaggle_v3_bootstrap.sh'], cwd=repo, env=env, check=True)
subprocess.run(['bash', 'finetune/kaggle_v3_train.sh'], cwd=repo, env=env, check=True)
