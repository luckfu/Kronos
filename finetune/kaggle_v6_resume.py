"""Kaggle entrypoint for the final V6 one-pass continuation (121 through 284)."""
import os
import subprocess
from pathlib import Path


repo = Path('/kaggle/working/Kronos')
if not (repo / '.git').exists():
    subprocess.run(
        ['git', 'clone', 'https://github.com/luckfu/Kronos.git', str(repo)],
        check=True,
    )

env = os.environ.copy()
env['KRONOS_KAGGLE_ROOT'] = '/kaggle/working/kronos_a_share_v6_forecast'
env['KRONOS_PREDICTOR_SAVE_FOLDER'] = 'a_share_v6_forecast_only_context120_2pass'
env['KRONOS_BATCH_SIZE'] = '32'
# The first task ends at 120. This continuation ends exactly at the end of the
# first coverage pass, Segment 284, while retaining the stored global schedule.
env['KRONOS_MAX_SEGMENTS_PER_RUN'] = '164'

subprocess.run(
    ['bash', 'finetune/kaggle_v6_bootstrap.sh'],
    cwd=repo, env=env, check=True,
)
subprocess.run(
    ['bash', 'finetune/kaggle_v6_train.sh'],
    cwd=repo, env=env, check=True,
)
