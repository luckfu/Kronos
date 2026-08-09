"""Kaggle entrypoint for V6 forecast-only training from V5 Last."""
import json
import os
import shutil
import subprocess
from pathlib import Path


repo = Path('/kaggle/working/Kronos')
if not (repo / '.git').exists():
    subprocess.run(
        ['git', 'clone', 'https://github.com/luckfu/Kronos.git', str(repo)],
        check=True,
    )

env = os.environ.copy()
runtime = Path('/kaggle/working/kronos_a_share_v6_forecast')
output_name = 'a_share_v6_forecast_only_context120_2pass'
output_root = runtime / 'outputs' / 'models' / output_name
env.setdefault('KRONOS_KAGGLE_ROOT', str(runtime))
env.setdefault('KRONOS_PREDICTOR_SAVE_FOLDER', output_name)
env.setdefault('KRONOS_BATCH_SIZE', '32')
env.setdefault('KRONOS_MAX_SEGMENTS_PER_RUN', '120')

subprocess.run(
    ['bash', 'finetune/kaggle_v6_bootstrap.sh'],
    cwd=repo, env=env, check=True,
)
result = subprocess.run(
    ['bash', 'finetune/kaggle_v6_train.sh'],
    cwd=repo, env=env,
)

oom_marker = output_root / 'oom.json'
if result.returncode and oom_marker.exists() and env['KRONOS_BATCH_SIZE'] == '32':
    print('[kaggle-v6] CUDA OOM at batch 32; restarting from V5 Last with batch 24.')
    print(oom_marker.read_text())
    shutil.rmtree(output_root)
    env['KRONOS_BATCH_SIZE'] = '24'
    env['KRONOS_RESUME_TRAINING'] = '0'
    subprocess.run(
        ['bash', 'finetune/kaggle_v6_train.sh'],
        cwd=repo, env=env, check=True,
    )
elif result.returncode:
    raise subprocess.CalledProcessError(
        result.returncode, ['bash', 'finetune/kaggle_v6_train.sh']
    )

run_manifest = {
    'base': 'V5 Last weights only',
    'loss_mode': 'forecast',
    'history_loss_weight': 0,
    'batch_size': int(env['KRONOS_BATCH_SIZE']),
    'coverage_passes': 2,
    'max_segments_per_run': int(env['KRONOS_MAX_SEGMENTS_PER_RUN']),
}
(runtime / 'v6_run_manifest.json').write_text(json.dumps(run_manifest, indent=2))
