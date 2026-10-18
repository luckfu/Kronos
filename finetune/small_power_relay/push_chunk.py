"""Submit the sampled Small Clean continuation under the authenticated owner."""

import argparse
import base64
import hashlib
import io
import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OWNER = 'user281434'
DATASET = 'smmt315/a-share-120d-temporal-symbol-holdout'
PARENT = 'luckfu/kronos-small-clean-wc-dual-t4-c03'


def slug(chunk):
    return f'kronos-small-clean-wc-dual-t4-c{chunk:02d}'


def source_bundle():
    legacy = runpy.run_path(str(ROOT / 'finetune/small_clean_relay/push_chunk.py'))
    stream = io.BytesIO(legacy['source_bundle']())
    with zipfile.ZipFile(stream, 'a', zipfile.ZIP_DEFLATED) as archive:
        for path in (ROOT / 'finetune/power_law_sampler.py', HERE / 'launch.py'):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def available_gpu_seconds(quota):
    return (quota.total_time_allowed - quota.time_used - quota.time_reserved).total_seconds()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('chunk', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stage-plan', type=Path)
    args = parser.parse_args()
    if args.chunk < 4:
        parser.error('Second round starts at C04')
    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    stage = json.loads(args.stage_plan.read_text()) if args.stage_plan else None
    if stage is None and args.chunk > 4:
        previous_path = HERE / 'build' / slug(args.chunk - 1) / 'build_manifest.json'
        if previous_path.is_file():
            stage = json.loads(previous_path.read_text()).get('stage')
    first_chunk = int(stage['first_chunk']) if stage else 4
    if args.chunk < first_chunk:
        raise RuntimeError('Chunk precedes the frozen stage plan')
    if args.chunk > first_chunk:
        previous = json.loads((HERE / 'build' / slug(args.chunk - 1) / 'build_manifest.json').read_text())
        if previous.get('stage') != stage:
            raise RuntimeError('Sampling stage plan changed within a round')
        if digest != previous['source_sha256']:
            raise RuntimeError('Source changed within the second round')
        commit = previous['source_commit']
    parent = (stage['parent_kernel'] if stage else PARENT) if args.chunk == first_chunk else f'{OWNER}/{slug(args.chunk - 1)}'
    config = {'chunk': args.chunk, 'parent_kernel': parent,
              'source_sha256': digest, 'source_commit': commit, 'working_tree_bundle': True}
    if stage:
        config['stage'] = stage
    legacy = runpy.run_path(str(ROOT / 'finetune/small_clean_relay/push_chunk.py'))
    code = f'''import base64, hashlib, io, json, os, runpy, sys, time, zipfile
from pathlib import Path
print(json.dumps({{"phase": "started", "time": time.time(), "chunk": {args.chunk}}}), flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("SWANLAB_API_KEY", {legacy['local_credential']()!r})
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/small_power_source")
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.log = open("/kaggle/working/entry.log", "a", buffering=1)
    def write(self, text):
        self.log.write(text)
        return self.stream.write(text)
    def flush(self):
        self.log.flush()
        self.stream.flush()
sys.stdout = Tee(sys.stdout)
namespace = runpy.run_path(str(repo / "finetune/small_power_relay/launch.py"))
namespace["main"].__globals__["CHUNK_CONFIG"] = {config!r}
namespace["main"]()
'''
    compile(code, 'train.py', 'exec')
    build = HERE / 'build' / slug(args.chunk)
    build.mkdir(parents=True, exist_ok=True)
    (build / 'train.py').write_text(code)
    metadata = {
        'id': f'{OWNER}/{slug(args.chunk)}',
        'title': f'Kronos Small Clean WC Dual T4 C{args.chunk:02d}',
        'code_file': 'train.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [DATASET], 'kernel_sources': [parent],
    }
    (build / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (build / 'build_manifest.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'id': metadata['id'], **config, 'hard_timeout_seconds': 43200,
                      'budget_seconds': 40500, 'sample_draws': 4000000}, indent=2))
    if args.dry_run:
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if api.config_values.get('username') != OWNER:
        raise RuntimeError('Authenticated Kaggle owner mismatch')
    with api.build_kaggle_client() as client:
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics().gpu_quota
    available_seconds = available_gpu_seconds(quota)
    if available_seconds < 43200:
        raise RuntimeError(f'12-hour invocation needs more GPU quota; available seconds={available_seconds:.0f}')
    print(json.dumps({'gpu_quota_available_seconds': available_seconds}), flush=True)
    cli = [sys.executable, '-m', 'kaggle']
    status = subprocess.run(cli + ['kernels', 'status', metadata['id']], capture_output=True, text=True)
    subprocess.run(cli + ['kernels', 'files', metadata['id'], '--format', 'json'], capture_output=True, text=True)
    if status.returncode == 0:
        raise RuntimeError('Target exists; inspect terminal state before any replacement')
    error = (status.stdout + status.stderr).lower()
    if not any(token in error for token in ('403', '404', 'not found', 'permission', 'cannot access kernel')):
        raise RuntimeError(f'Unverified target status: {error}')
    page = 1
    while True:
        owned = api.kernels_list(mine=True, page=page, page_size=200) or []
        if any(item.ref == metadata['id'] for item in owned):
            raise RuntimeError('Target exists but its status is inaccessible')
        if len(owned) < 200:
            break
        page += 1
    parent_status = subprocess.check_output(cli + ['kernels', 'status', parent], text=True)
    if 'COMPLETE' not in parent_status:
        raise RuntimeError('Parent has not completed')
    listing = subprocess.check_output(cli + ['kernels', 'files', parent, '--page-size', '200', '--format', 'json'], text=True)
    files = json.loads(listing)
    names = {Path(item['name']).name for item in files}
    required = {'last_state.pt', 'model.safetensors', 'best_metric.json', 'summary.json',
                'progress.json', 'experiment_manifest.json', 'metrics.jsonl', 'run.log'}
    if not required <= names:
        raise RuntimeError(f'Incomplete parent Output: {required - names}')
    subprocess.run(cli + ['kernels', 'push', '-p', str(build), '--timeout', '43200',
                          '--accelerator', 'NvidiaTeslaT4'], check=True)
    subprocess.run(cli + ['kernels', 'status', metadata['id']], check=True)


if __name__ == '__main__':
    main()
