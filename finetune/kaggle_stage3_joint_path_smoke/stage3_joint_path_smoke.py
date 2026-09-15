"""One fresh Stage3 segment after a disposable dual-T4 correctness gate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

SOURCE_COMMIT = os.environ.get('STAGE3_SOURCE_COMMIT', 'a854641420068dbbc6daf2dcfa329c8ecbaa425a')
RUN_ID = os.environ.get('STAGE3_SWANLAB_RUN_ID', 'small_0.1_stage3_joint_path_alignment_from_c2_best_v2')
PARENT = 'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2'
OUTPUT = Path('/kaggle/working/stage3_joint_path_smoke')
RESUME_KERNEL = os.environ.get('STAGE3_RESUME_KERNEL', '')
TARGET_SEGMENTS = int(os.environ.get('STAGE3_TARGET_SEGMENTS', '1'))
CHUNK = os.environ.get('STAGE3_CHUNK', 'c1')
HARD_TIMEOUT_SECONDS = int(os.environ.get('STAGE3_HARD_TIMEOUT_SECONDS', '18000'))
MAX_RUNTIME_SECONDS = int(os.environ.get('STAGE3_MAX_RUNTIME_SECONDS', '10800'))
BASELINE_BEFORE_RESUME = os.environ.get('STAGE3_BASELINE_BEFORE_RESUME', '0') == '1'
LAMBDA_PATH = float(os.environ.get('STAGE3_LAMBDA_PATH', '0.05'))
MILESTONE_SEGMENTS = os.environ.get('STAGE3_MILESTONE_SEGMENTS', '')
HASHES = {
    'best': '4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a',
    'tokenizer': '59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee',
    'train': '034c5315547e35e38e6f3a3279ecc3ce01b625f1139229e6371d3f6c4e927a90',
    'val': '4cce31bc3e70eab83d5b7ea05f19fce04aa57a87f3acf00b882ddfbac4219bf7',
    'metadata': '697cadd672d53b8fa0a990f6c5b7fba2f88cb26aad88c3b91cfcc3865d0a1e3c',
    'manifest': '32cfcbf606dcee81c9416f9ad399ee7303b6790cf9025bf5888638f552d66598',
}
_log = None
_lock = threading.Lock()
_phase = 'started'


def emit(line):
    with _lock:
        print(line, end='' if line.endswith('\n') else '\n', flush=True)
        if _log:
            _log.write(line if line.endswith('\n') else line + '\n'); _log.flush()


def phase(name, **fields):
    global _phase
    _phase = name
    emit(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': name, **fields}))


def run(command, cwd=None, env=None):
    proc = subprocess.Popen(command, cwd=cwd, env=env or os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        emit(line)
    if proc.wait(): raise subprocess.CalledProcessError(proc.returncode, command)


def resolve_inputs(root):
    manifests = [p for p in root.rglob('data_manifest.json')
                 if (p.parent / 'processed_datasets/train_data.pkl').is_file()
                 and (p.parent / 'processed_datasets/val_data.pkl').is_file()]
    best = list(root.glob('**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors'))
    tokenizer = list(root.glob('**/models/Kronos-Tokenizer-base/model.safetensors'))
    for name, matches in [('public dataset', manifests), ('C2 best', best), ('tokenizer', tokenizer)]:
        if len(matches) != 1: raise RuntimeError(f'Expected unique {name}; got {matches}')
    data_root = manifests[0].parent
    return {'manifest': manifests[0], 'train': data_root / 'processed_datasets/train_data.pkl',
            'val': data_root / 'processed_datasets/val_data.pkl',
            'metadata': data_root / 'asset_metadata.csv', 'best': best[0], 'tokenizer': tokenizer[0]}


def verify_hashes(inputs):
    for name, path in inputs.items():
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != HASHES[name]: raise RuntimeError(f'{name} SHA256 mismatch: {actual}')
        phase('input_verified', input_name=name, path=str(path), sha256=actual)


def main():
    global _log
    os.environ['PYTHONUNBUFFERED'] = '1'
    phase('started', run_id=RUN_ID, segments=TARGET_SEGMENTS, batch_per_gpu=32, global_batch=64,
          lr=2e-6, amp=False, lambda_path=LAMBDA_PATH, milestone_segments=MILESTONE_SEGMENTS,
          hard_timeout_seconds=HARD_TIMEOUT_SECONDS,
          max_runtime_seconds=MAX_RUNTIME_SECONDS, chunk=CHUNK)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _log = (OUTPUT / 'run.log').open('a', buffering=1)
    phase('log_ready', output=str(OUTPUT))
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(20):
            emit(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(),
                             'phase': 'heartbeat', 'active_phase': _phase}))
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        if len(SOURCE_COMMIT) != 40: raise RuntimeError('Unpinned source commit')
        phase('clone_source', commit=SOURCE_COMMIT, max_runtime_seconds=MAX_RUNTIME_SECONDS,
              hard_timeout_seconds=HARD_TIMEOUT_SECONDS, baseline_before_resume=BASELINE_BEFORE_RESUME)
        for attempt in range(3):
            repo = Path(tempfile.mkdtemp(prefix='kronos-stage3-source-')) / 'repo'
            try:
                run(['git', 'clone', '--depth', '5', '--branch', 'master',
                     'https://github.com/luckfu/Kronos.git', str(repo)],
                    env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
                run(['git', 'checkout', '--detach', SOURCE_COMMIT], cwd=repo)
                break
            except subprocess.CalledProcessError:
                if attempt == 2: raise
                phase('clone_retry', attempt=attempt + 1); time.sleep(3 * (attempt + 1))
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        if actual != SOURCE_COMMIT: raise RuntimeError('Source commit mismatch')
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(repo), 'OMP_NUM_THREADS': '2'}
        if RESUME_KERNEL:
            phase('verify_continuation', parent_kernel=RESUME_KERNEL)
            matches = [p for p in Path('/kaggle/input').rglob('last_state.pt')
                       if (p.parent.parent / 'gpu_probe.json').is_file()
                       and (p.parent.parent / 'experiment_manifest.json').is_file()]
            if len(matches) != 1: raise RuntimeError(f'Expected unique Stage3 state: {matches}')
            source = matches[0].parent.parent
            required = ['run.log', 'metrics.jsonl', 'progress.json', 'summary.json',
                        'experiment_manifest.json', 'gpu_probe.json', 'checkpoints/last_state.pt',
                        'checkpoints/best_model/model.safetensors', 'checkpoints/best_model/config.json',
                        'checkpoints/best_model/best_metric.json', 'checkpoints/last_model/model.safetensors',
                        'checkpoints/last_model/config.json']
            for name in required:
                if not (source / name).is_file() or not (source / name).stat().st_size:
                    raise RuntimeError(f'Incomplete continuation output: {name}')
            with matches[0].open('rb') as f:
                state_sha = hashlib.file_digest(f, 'sha256').hexdigest()
            expected_sha = os.environ.get('STAGE3_RESUME_STATE_SHA256', '').strip()
            if expected_sha and state_sha != expected_sha:
                raise RuntimeError(f'Continuation state hash mismatch: {state_sha}')
            expected_progress = json.loads(os.environ['STAGE3_EXPECTED_PROGRESS'])
            progress = json.loads((source / 'progress.json').read_text())
            if progress != expected_progress:
                raise RuntimeError(f'Unexpected source progress: {progress}')
            next_segment = int(progress['next_epoch']) + 1
            global_step = int(progress['step'])
            # Preserve history without replacing the already-open wrapper log.
            shutil.copytree(source, OUTPUT, dirs_exist_ok=True, ignore=shutil.ignore_patterns('run.log'))
            phase('continuation_log_begin')
            with (source / 'run.log').open() as f:
                for line in f: emit(line)
            phase('continuation_log_end', source=str(source), next_segment=next_segment,
                  global_step=global_step, state_sha256=state_sha, expected_sha256=expected_sha or None,
                  historical_metric_lines=sum(1 for _ in (source / 'metrics.jsonl').open()))
        phase('install_t4_torch', version='2.6.0+cu124')
        run([sys.executable, '-u', '-m', 'pip', 'install', '--disable-pip-version-check',
             '--progress-bar', 'off', 'torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124'], env=env)
        phase('install_dependencies')
        run([sys.executable, '-u', '-m', 'pip', 'install', '--disable-pip-version-check', '--progress-bar', 'off',
             'numpy==2.2.6', 'pandas==2.2.3', 'huggingface_hub==0.33.1', 'safetensors==0.6.2',
             'einops==0.8.1', 'tqdm==4.67.1', 'swanlab==0.9.8', 'pytest'], env=env)
        # Import only after installing; no stale pre-install torch in this process.
        import torch
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if len(devices) != 2 or any('T4' not in d for d in devices): raise RuntimeError(devices)
        if not torch.__version__.startswith('2.6.0') or 'sm_75' not in torch.cuda.get_arch_list():
            raise RuntimeError('Unverified T4 PyTorch build')
        phase('device_ready', devices=devices, torch=torch.__version__, cuda=torch.version.cuda)
        # User explicitly authorized this existing key; never print it.
        key = os.environ.get('SWANLAB_API_KEY')
        if not key:
            try:
                from kaggle_secrets import UserSecretsClient
                key = UserSecretsClient().get_secret('SWANLAB_API_KEY')
            except Exception:
                # Reuse the authorized key in the existing runner, without
                # executing that runner or duplicating the key in this file.
                import ast
                old = repo / 'finetune/kaggle_small_0_1_stage3_path_alignment_smoke_kernel/kaggle_small_0_1_stage3_path_alignment_smoke.py'
                tree = ast.parse(old.read_text())
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SWANLAB_KEY' for t in node.targets):
                        key = ast.literal_eval(node.value.args[1])
        if not key: raise RuntimeError('No SwanLab credentials configured')
        env.update(SWANLAB_API_KEY=key, SWANLAB_RUN_ID=RUN_ID, SWANLAB_EXPERIMENT_NAME=RUN_ID)
        phase('dashboard_init', run_id=RUN_ID, credentials_configured=True)
        import swanlab
        swanlab.login(api_key=key)
        dashboard = swanlab.init(id=RUN_ID, resume='allow', project='finance', workspace='roc_fu',
                                 experiment_name=RUN_ID, config={'source_commit': SOURCE_COMMIT, 'target_segments': TARGET_SEGMENTS}, mode='cloud')
        if not dashboard.url: raise RuntimeError('SwanLab has no URL')
        dashboard_url = dashboard.url
        phase('dashboard_ready', SWANLAB_RUN_URL=dashboard_url)
        dashboard.finish()
        phase('verify_inputs')
        inputs = resolve_inputs(Path('/kaggle/input')); verify_hashes(inputs)
        env.update(KRONOS_DATASET_PATH=str(inputs['train'].parent), KRONOS_METADATA_PATH=str(inputs['metadata']),
                   KRONOS_LOOKBACK_WINDOW='120', KRONOS_PREDICT_WINDOW='10', KRONOS_USE_SIZE_PERCENTILE='1',
                   KRONOS_NUM_SIZE_BUCKETS='0', KRONOS_VALIDATION_SAMPLES='0',
                   KRONOS_VAL_SIGNAL_START='2025-07-01', KRONOS_VAL_SIGNAL_END='2026-07-02',
                   KRONOS_COVERAGE_SEED='20260915', KRONOS_TRAIN_SAMPLES_PER_SEGMENT='20000')
        manifest = {'source_commit': SOURCE_COMMIT, 'parent_kernel': PARENT, 'parent_checkpoint': 'best_model',
                    'initialization': 'model_weights_only', 'optimizer': 'fresh_AdamW', 'scheduler': 'fixed',
                    'lr': 2e-6, 'amp': False, 'seed': 20260915, 'segments': TARGET_SEGMENTS, 'batch_per_gpu': 32,
                    'global_batch': 64, 'train_samples': 20000, 'validation_samples': 123836,
                    'lookback': 120, 'horizon': 10, 'loss': f'CE+{LAMBDA_PATH:g}*EMA_normalized_six_feature_Huber',
                    'lambda_path': LAMBDA_PATH, 'milestone_segments': MILESTONE_SEGMENTS,
                    'huber_delta': 0.02, 'top_k': 16, 'candidates': 16, 'ema_decay': 0.99,
                    'dependency_causal': True, 'devices': devices, 'torch': torch.__version__,
                    'run_id': RUN_ID, 'swanlab_url': dashboard_url, 'sha256': HASHES,
                    'inputs': {k: str(v) for k, v in inputs.items()}, 'oos_used': False}
        if RESUME_KERNEL:
            previous = json.loads((OUTPUT / 'experiment_manifest.json').read_text())
            (OUTPUT / 'parent_experiment_manifest.json').write_text(json.dumps(previous, indent=2))
            manifest.update(continuation_kernel=RESUME_KERNEL, continuation_state_sha256=state_sha,
                            optimizer='resumed_AdamW', initialization='exact_stage3_resume',
                            next_segment=next_segment, initial_global_step=global_step, chunk=CHUNK,
                            baseline_before_resume=BASELINE_BEFORE_RESUME,
                            max_runtime_seconds=MAX_RUNTIME_SECONDS)
        (OUTPUT / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2))
        phase('cpu_preflight')
        run([sys.executable, '-u', '-m', 'pytest', 'tests/test_stage3_conditional_joint.py',
             'tests/test_stage3_training_framework.py', '-q'], cwd=repo,
            env={**env, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'CUDA_VISIBLE_DEVICES': ''})
        torchrun = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
        common = ['--model-dir', str(inputs['best'].parent), '--tokenizer-dir', str(inputs['tokenizer'].parent)]
        if not RESUME_KERNEL:
            phase('gpu_probe')
            run(torchrun + ['finetune.stage3_gpu_probe'] + common + ['--batch', '32', '--output', str(OUTPUT / 'gpu_probe.json')], cwd=repo, env=env)
        probe = json.loads((OUTPUT / 'gpu_probe.json').read_text())
        if probe['status'] != 'passed': raise RuntimeError('GPU gate did not pass')
        resume_step = global_step if RESUME_KERNEL else 0
        phase('training', segments=TARGET_SEGMENTS, optimizer='resume' if RESUME_KERNEL else 'fresh',
              global_step=resume_step, baseline_before_resume=BASELINE_BEFORE_RESUME,
              max_runtime_seconds=MAX_RUNTIME_SECONDS)
        resume_args = []
        if RESUME_KERNEL:
            resume_args = ['--resume-state', str(OUTPUT / 'checkpoints/last_state.pt'), '--chunk', CHUNK]
        if BASELINE_BEFORE_RESUME:
            resume_args.append('--baseline-before-resume')
        if MILESTONE_SEGMENTS:
            resume_args += ['--milestone-segments', MILESTONE_SEGMENTS]
        run(torchrun + ['finetune.train_stage3_path_alignment'] + common +
            ['--output-dir', str(OUTPUT), '--segments', str(TARGET_SEGMENTS), '--batch', '32', '--lr', '2e-6',
             '--seed', '20260915', '--log-interval', '10', '--lambda-path', str(LAMBDA_PATH),
             '--max-runtime-seconds', str(MAX_RUNTIME_SECONDS)] + resume_args, cwd=repo, env=env)
        phase('verify_output')
        progress = json.loads((OUTPUT / 'progress.json').read_text())
        summary = json.loads((OUTPUT / 'summary.json').read_text())
        completed = int(progress['completed_segments'])
        assert progress['status'] == 'completed' and progress['next_epoch'] == completed
        assert completed == len(summary['segments']) and completed <= TARGET_SEGMENTS
        if RESUME_KERNEL:
            assert completed > int(expected_progress['completed_segments'])
        else:
            assert completed == TARGET_SEGMENTS
        assert all(row['validation']['samples'] == 123836 for row in summary['segments'])
        for name in ['run.log', 'metrics.jsonl', 'experiment_manifest.json', 'checkpoints/last_state.pt',
                     'checkpoints/best_model/model.safetensors', 'checkpoints/best_model/best_metric.json',
                     'checkpoints/last_model/model.safetensors']:
            assert (OUTPUT / name).is_file(), name
        phase('completed', completed_segments=TARGET_SEGMENTS, validation_samples=123836, output=str(OUTPUT))
    except Exception as exc:
        phase('failed', error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        stopped.set()
        if _log: _log.close()


if __name__ == '__main__':
    main()
