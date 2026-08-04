"""Kaggle entrypoint for one resumable A-share V3 training chunk."""

import os
import shutil
import subprocess
from pathlib import Path


REPO = Path('/kaggle/working/Kronos')
RUNTIME = Path('/kaggle/working/kronos_a_share_v3_long')
OUTPUT_NAME = 'a_share_size_full_market_v3_kaggle_bs32'


def find_previous_run() -> Path | None:
    candidates = []
    for checkpoint in Path('/kaggle/input').rglob('last_state.pt'):
        if OUTPUT_NAME not in checkpoint.parts:
            continue
        candidates.append(checkpoint.parent.parent)
    if len(candidates) > 1:
        paths = '\n'.join(str(path) for path in candidates)
        raise RuntimeError(f'Multiple resume checkpoints found:\n{paths}')
    return candidates[0] if candidates else None


if not (REPO / '.git').is_dir():
    subprocess.run(
        ['git', 'clone', 'https://github.com/luckfu/Kronos.git', str(REPO)],
        check=True,
    )

env = os.environ.copy()
env.setdefault('KRONOS_KAGGLE_DATASET_INPUT', '/kaggle/input/kronos-train-set-a')
env['KRONOS_KAGGLE_ROOT'] = str(RUNTIME)
env['KRONOS_PREDICTOR_SAVE_FOLDER'] = OUTPUT_NAME
env['KRONOS_EPOCHS'] = '50'
env['KRONOS_COVERAGE_PASSES'] = '2'
env['KRONOS_REQUIRE_FULL_COVERAGE'] = '1'
env['KRONOS_EARLY_STOPPING_PATIENCE'] = '5'
env['KRONOS_MAX_SEGMENTS_PER_RUN'] = os.getenv(
    'KRONOS_MAX_SEGMENTS_PER_RUN', '180'
)

subprocess.run(['bash', 'finetune/kaggle_v3_bootstrap.sh'], cwd=REPO, env=env, check=True)

previous_run = find_previous_run()
target_run = RUNTIME / 'outputs' / 'models' / OUTPUT_NAME
if previous_run is not None:
    print(f'[kaggle-v3] Restoring previous run from {previous_run}', flush=True)
    shutil.copytree(previous_run, target_run, dirs_exist_ok=True)
    env['KRONOS_RESUME_TRAINING'] = '1'
else:
    print('[kaggle-v3] No prior checkpoint found; starting segment 1.', flush=True)
    env['KRONOS_RESUME_TRAINING'] = '0'

subprocess.run(['bash', 'finetune/kaggle_v3_train.sh'], cwd=REPO, env=env, check=True)

# The base model is reproducibly downloaded by bootstrap. Removing this copy
# keeps each Kernel output focused on the resume checkpoint and best model.
shutil.rmtree(RUNTIME / 'models', ignore_errors=True)
shutil.rmtree(REPO, ignore_errors=True)
