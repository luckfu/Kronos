"""Clone this repository inside a Colab VM and optionally start training."""

import argparse
import os
import subprocess
from pathlib import Path


def run(command, cwd=None, env=None):
    print('[colab]', ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-url', default='https://github.com/luckfu/Kronos.git')
    parser.add_argument('--ref', default='master')
    parser.add_argument('--repo-dir', default='/content/Kronos')
    parser.add_argument('--runtime-root')
    parser.add_argument('--base-model', choices=('production', 'original'), default='production')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args()

    drive_root = Path('/content/drive/MyDrive')
    runtime_root = args.runtime_root or str(
        drive_root / 'kronos_a_share' if drive_root.exists()
        else Path('/content/kronos_runtime')
    )

    repo_dir = Path(args.repo_dir)
    if (repo_dir / '.git').exists():
        run(['git', 'fetch', '--depth=1', 'origin', args.ref], cwd=repo_dir)
        run(['git', 'merge', '--ff-only', 'FETCH_HEAD'], cwd=repo_dir)
    else:
        if repo_dir.exists():
            raise SystemExit(f'{repo_dir} exists but is not a Git checkout')
        run(['git', 'clone', '--depth=1', '--branch', args.ref, args.repo_url, str(repo_dir)])

    env = os.environ.copy()
    env['KRONOS_COLAB_ROOT'] = runtime_root
    env['KRONOS_COLAB_BASE_MODEL'] = args.base_model
    if args.prepare_only:
        env['KRONOS_PREPARE_ONLY'] = '1'
    run(['bash', 'finetune/colab_bootstrap.sh'], cwd=repo_dir, env=env)
    if args.prepare_only:
        return

    output_name = env.get(
        'KRONOS_PREDICTOR_SAVE_FOLDER',
        'a_share_size_full_coverage_colab_v1',
    )
    state_path = Path(runtime_root) / 'outputs' / 'models' / output_name / 'checkpoints' / 'last_state.pt'
    if state_path.exists():
        env['KRONOS_RESUME_TRAINING'] = '1'
    if args.foreground:
        run(['bash', 'finetune/colab_train.sh'], cwd=repo_dir, env=env)
        return

    runtime_path = Path(runtime_root)
    runtime_path.mkdir(parents=True, exist_ok=True)
    pid_path = runtime_path / 'training.pid'
    log_path = runtime_path / 'colab-training.log'
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text().strip())
            os.kill(existing_pid, 0)
        except (OSError, ValueError):
            pass
        else:
            raise SystemExit(f'Training is already running with PID {existing_pid}')

    with log_path.open('ab', buffering=0) as log_handle:
        process = subprocess.Popen(
            ['bash', 'finetune/colab_train.sh'],
            cwd=repo_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid))
    print(f'[colab] Training started with PID {process.pid}')
    print(f'[colab] Log: {log_path}')
    print(f'[colab] Checkpoints: {Path(runtime_root) / "outputs" / "models" / output_name}')


if __name__ == '__main__':
    main()
