"""Kaggle entrypoint for corrected-context V4 recent/replay A/B training."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path


REPO = Path('/kaggle/working/Kronos')
RUNTIME = Path('/kaggle/working/kronos_a_share_v4_ab')


def locate_v3_base():
    candidates = list(Path('/kaggle/input').rglob('base_model/v3_last/model.safetensors'))
    if len(candidates) == 1:
        return candidates[0].parent
    if len(candidates) > 1:
        raise RuntimeError(f'Multiple V3 base models found: {candidates}')
    bundles = list(Path('/kaggle/input').rglob('kronos_a_share_2026_incremental.tar.gz'))
    if len(bundles) != 1:
        raise RuntimeError(f'Expected one V3 base bundle; found {bundles}')
    target = RUNTIME / 'input_base'
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundles[0], 'r:gz') as archive:
        members = [
            member for member in archive.getmembers()
            if member.name.startswith('base_model/v3_last/')
        ]
        archive.extractall(target, members=members, filter='data')
    return target / 'base_model/v3_last'


if not (REPO / '.git').is_dir():
    subprocess.run(
        ['git', 'clone', 'https://github.com/luckfu/Kronos.git', str(REPO)],
        check=True,
    )

base_model = locate_v3_base()
env = os.environ.copy()
env['KRONOS_KAGGLE_ROOT'] = str(RUNTIME)
env['KRONOS_KAGGLE_DATASET_INPUT'] = '/kaggle/input/kronos-train-set-a'
env['KRONOS_PREDICTOR_PATH'] = str(base_model)

subprocess.run(['bash', 'finetune/kaggle_v3_bootstrap.sh'], cwd=REPO, env=env, check=True)
for variant in ('recent_only', 'replay20'):
    subprocess.run(
        ['bash', 'finetune/kaggle_v4_ab_train.sh', variant],
        cwd=REPO,
        env=env,
        check=True,
    )
subprocess.run(
    ['bash', 'finetune/kaggle_v4_ab_evaluate.sh'],
    cwd=REPO,
    env=env,
    check=True,
)

shutil.rmtree(RUNTIME / 'data', ignore_errors=True)
shutil.rmtree(RUNTIME / 'input_base', ignore_errors=True)
shutil.rmtree(REPO, ignore_errors=True)
