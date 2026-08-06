"""Train the corrected-context V4 B model for two full coverage passes."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


REPO = Path('/kaggle/working/Kronos')
RUNTIME = Path('/kaggle/working/kronos_a_share_v4_b_2pass')
OUTPUT_NAME = 'a_share_v4_corrected_2026_replay20_2pass'


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


def export_last_model(output_root):
    import torch

    sys.path.insert(0, str(REPO))
    from model import Kronos

    checkpoint_dir = output_root / 'checkpoints'
    best_dir = checkpoint_dir / 'best_model'
    last_dir = checkpoint_dir / 'last_model'
    state = torch.load(
        checkpoint_dir / 'last_state.pt',
        map_location='cpu',
        weights_only=False,
    )
    with open(best_dir / 'config.json') as handle:
        config = json.load(handle)
    model = Kronos.from_pretrained(best_dir)
    model.load_state_dict(state['model'], strict=True)
    model.save_pretrained(last_dir, config=config)
    shutil.copy2(best_dir / 'README.md', last_dir / 'README.md')
    print(f'[v4-b-2pass] Last model exported to {last_dir}', flush=True)


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
env['KRONOS_V4_OUTPUT_NAME'] = OUTPUT_NAME
env['KRONOS_V4_COVERAGE_PASSES'] = '2'

subprocess.run(['bash', 'finetune/kaggle_v3_bootstrap.sh'], cwd=REPO, env=env, check=True)
subprocess.run(
    ['bash', 'finetune/kaggle_v4_ab_train.sh', 'replay20'],
    cwd=REPO,
    env=env,
    check=True,
)
export_last_model(RUNTIME / 'outputs' / 'models' / OUTPUT_NAME)

shutil.rmtree(RUNTIME / 'data', ignore_errors=True)
shutil.rmtree(RUNTIME / 'input_base', ignore_errors=True)
shutil.rmtree(REPO, ignore_errors=True)
