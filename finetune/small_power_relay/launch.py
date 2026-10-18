"""Second Small Clean round, preserving C03 optimizer and global dashboard axes."""

import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


CHUNK_CONFIG = {}
REPO = Path(__file__).resolve().parents[2]
LEGACY = runpy.run_path(str(REPO / 'finetune/small_clean_relay/launch.py'))
EXPERIMENT = LEGACY['EXPERIMENT']
OUTPUT_NAME = EXPERIMENT + '_power_round2'
RUN_ID = LEGACY['RUN_ID']
START_SEGMENT = 520
PARENT_GLOBAL_STEP = 162724
ROUND_SEGMENTS = 200
PLAN_SHA256 = 'fa746204cda4f373d9b5992f30bfa1c98eb112c604ca471fdcc700d09508d71c'
TOTAL_SEGMENTS = START_SEGMENT + ROUND_SEGMENTS
MAX_SEGMENTS = 0
BUDGET_SECONDS = 40500
phase = LEGACY['phase']
command = LEGACY['command']
sha256 = LEGACY['sha256']
find_one = LEGACY['find_one']
ROUND_NUMBER = 2
SAMPLING_SEED = 20261013
FIRST_CHUNK = 4
STAGE_PARENT = 'luckfu/kronos-small-clean-wc-dual-t4-c03'


def configure_stage(stage):
    global START_SEGMENT, PARENT_GLOBAL_STEP, ROUND_SEGMENTS, PLAN_SHA256
    global TOTAL_SEGMENTS, ROUND_NUMBER, SAMPLING_SEED, FIRST_CHUNK, STAGE_PARENT, OUTPUT_NAME
    START_SEGMENT = int(stage['start_segment'])
    PARENT_GLOBAL_STEP = int(stage['parent_global_step'])
    ROUND_SEGMENTS = int(stage['round_segments'])
    PLAN_SHA256 = stage['sampling_plan_sha256']
    TOTAL_SEGMENTS = START_SEGMENT + ROUND_SEGMENTS
    ROUND_NUMBER = int(stage['round'])
    SAMPLING_SEED = int(stage['seed'])
    FIRST_CHUNK = int(stage['first_chunk'])
    STAGE_PARENT = stage['parent_kernel']
    OUTPUT_NAME = EXPERIMENT + f'_power_round{ROUND_NUMBER}'


def recipe():
    return {
        'experiment': EXPERIMENT, 'round': ROUND_NUMBER, 'run_id': RUN_ID,
        'parent_kernel': STAGE_PARENT,
        'data_manifest_sha256': LEGACY['DATA_SHA'],
        'source_sha256': CHUNK_CONFIG['source_sha256'],
        'source_commit': CHUNK_CONFIG['source_commit'],
        'sampler': 'capped_power_mixture_v1', 'seed': SAMPLING_SEED,
        'weighted_ratio': 0.6, 'reference': 0.05, 'exponent': 1.0, 'weight_cap': 4.0,
        'amplitude': 'max_h1_h10_abs_close_return_from_last_input_close',
        'replacement': True, 'start_segment': START_SEGMENT,
        'sampling_plan_sha256': PLAN_SHA256,
        'parent_global_step': PARENT_GLOBAL_STEP, 'round_segments': ROUND_SEGMENTS,
        'total_segments': TOTAL_SEGMENTS, 'samples_per_segment': 20000,
        'global_batch_size': 64, 'scheduler': 'warmup_constant', 'learning_rate': 1e-5,
        'warmup_steps': 1627, 'validation_samples': 123836,
    }


def validate_output(root, torch, expected_recipe, parent=False):
    missing = [name for name in LEGACY['REQUIRED'] if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f'Incomplete parent Output: {missing}')
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    progress = json.loads((root / 'progress.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    summary = summary.get('final_result', summary)
    state = torch.load(root / 'checkpoints/last_state.pt', map_location='cpu', weights_only=False)
    boundary = int(state['next_epoch'])
    expected_steps = PARENT_GLOBAL_STEP + (boundary - START_SEGMENT) * 313
    if (state.get('resume_step') != 0 or not START_SEGMENT <= boundary <= TOTAL_SEGMENTS
            or int(state['batch_idx_global']) != expected_steps
            or state['scheduler']['last_epoch'] != expected_steps
            or state['scheduler_type'] != 'warmup_constant'
            or int(state['scheduler_warmup_steps']) != 1627
            or not state['optimizer']['state'] or 'amp_scaler' not in state or not state.get('rng_state')
            or int(progress.get('completed_segments', progress.get('current_segment', -1))) != boundary
            or int(summary.get('completed_segments', -1)) != boundary
            or progress['status'] not in {'stopped', 'completed'}
            or summary['status'] not in {'stopped', 'completed'}):
        raise RuntimeError('Parent global segment/step/optimizer contract mismatch')
    if parent:
        if 'stage' in CHUNK_CONFIG:
            if (boundary != START_SEGMENT or state['effective_epochs'] != START_SEGMENT
                    or manifest['recipe'] != CHUNK_CONFIG['stage']['parent_recipe']
                    or sha256(root / 'checkpoints/last_state.pt') != CHUNK_CONFIG['stage']['parent_state_sha256']):
                raise RuntimeError('Not the verified completed parent round')
        elif (boundary != START_SEGMENT or state['effective_epochs'] != START_SEGMENT
                or manifest['recipe']['source_sha256'] != 'd505207d6d12ac0d5a7443f7e06ac63c46ce8fe312acf00ad41d7fad82d01b57'
                or manifest['recipe']['data_manifest_sha256'] != LEGACY['DATA_SHA']):
            raise RuntimeError('Not the verified C03 first-round checkpoint')
    elif manifest['recipe'] != expected_recipe or state['effective_epochs'] != TOTAL_SEGMENTS:
        raise RuntimeError('Second-round recipe changed across chunks')
    phase('output_verified', next_epoch=boundary, next_segment=boundary + 1,
          global_step=expected_steps, state_sha256=sha256(root / 'checkpoints/last_state.pt'))
    return boundary


def training_environment(runtime, data_root, parent, tokenizer, continuation, boundary):
    env = {key: value for key, value in os.environ.items() if not key.startswith('KRONOS_')}
    env.update({
        'PYTHONUNBUFFERED': '1', 'KRONOS_SMALL_V21_STAGE': 'main',
        'KRONOS_SMALL_V21_RUNTIME': str(runtime), 'KRONOS_SMALL_V21_DATA_ROOT': str(data_root),
        'KRONOS_SMALL_V21_PARENT_MODEL': str(parent / 'checkpoints/last_model'),
        'KRONOS_SMALL_V21_TOKENIZER': str(tokenizer), 'KRONOS_SMALL_V21_OUTPUT_NAME': OUTPUT_NAME,
        'KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION': '0' if continuation else '1',
        'KRONOS_EPOCHS': str(TOTAL_SEGMENTS), 'KRONOS_REQUIRE_FULL_COVERAGE': '0',
        'KRONOS_COVERAGE_PASSES': '1', 'KRONOS_COVERAGE_SEED': '20261011',
        'KRONOS_SAMPLING_SEED': str(SAMPLING_SEED),
        'KRONOS_SAMPLING_PLAN_SHA256': PLAN_SHA256,
        'KRONOS_TRAIN_SAMPLING': 'capped_power_mixture_v1',
        'KRONOS_SAMPLING_START_SEGMENT': str(START_SEGMENT),
        'KRONOS_SAMPLING_PARENT_GLOBAL_STEP': str(PARENT_GLOBAL_STEP),
        'KRONOS_SAMPLING_SEGMENTS': str(ROUND_SEGMENTS),
        'KRONOS_SAMPLING_WEIGHTED_RATIO': '0.6', 'KRONOS_SAMPLING_REFERENCE': '0.05',
        'KRONOS_SAMPLING_EXPONENT': '1.0', 'KRONOS_SAMPLING_WEIGHT_CAP': '4.0',
        'KRONOS_SAMPLING_WARMUP_STEPS': '1627',
        'KRONOS_SWANLAB_REPLAY_AFTER_SEGMENT': str(boundary),
        'KRONOS_MAX_SEGMENTS_PER_RUN': str(MAX_SEGMENTS),
        'KRONOS_MAX_RUNTIME_SECONDS': str(BUDGET_SECONDS),
        'KRONOS_TORCHRUN_NPROC_PER_NODE': '2', 'KRONOS_BATCH_SIZE': '32', 'KRONOS_NUM_WORKERS': '2',
        'KRONOS_SCHEDULER': 'warmup_constant', 'KRONOS_SCHEDULER_WARMUP_RATIO': '0.01',
        'KRONOS_PREDICTOR_LEARNING_RATE': '1e-5', 'KRONOS_CONDITION_LEARNING_RATE': '1e-5',
        'KRONOS_USE_AMP': '1', 'KRONOS_AMP_DTYPE': 'float16',
        'KRONOS_REQUIRE_SWANLAB': '1', 'SWANLAB_PROJECT': 'finance',
        'SWANLAB_WORKSPACE': 'roc_fu', 'SWANLAB_EXPERIMENT_NAME': EXPERIMENT, 'SWANLAB_RUN_ID': RUN_ID,
    })
    if continuation:
        env['KRONOS_SMALL_V21_CONTINUATION_ROOT'] = str(parent)
    else:
        env['KRONOS_SAMPLING_TRANSITION_STATE'] = str(parent / 'checkpoints/last_state.pt')
    return env


def main():
    if 'stage' in CHUNK_CONFIG:
        configure_stage(CHUNK_CONFIG['stage'])
    started = time.monotonic()
    phase('started', owner='user281434', chunk=CHUNK_CONFIG['chunk'], recipe=recipe())
    command([sys.executable, '-u', '-m', 'pip', 'install',
             'torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124'])
    command([sys.executable, '-u', '-m', 'pip', 'install',
             'numpy==2.1.3', 'pandas==2.2.3', 'einops==0.8.1',
             'huggingface_hub==0.33.1', 'safetensors==0.6.2', 'tqdm==4.67.1', 'swanlab==0.9.8',
             'packaging', 'requests', 'regex', 'protobuf', 'pyyaml', 'hf-xet'])
    import torch
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(devices) != 2 or any('T4' not in device for device in devices):
        raise RuntimeError(f'Expected dual T4; found {devices}')
    phase('device_ready', devices=devices, torch=torch.__version__, cuda=torch.version.cuda)
    import swanlab
    token = os.environ.get('SWANLAB_API_KEY', '').strip()
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('SWANLAB_API_KEY')
    swanlab.login(api_key=token)
    os.environ['SWANLAB_API_KEY'] = token
    for key in ('SWANLAB_PROJECT', 'SWANLAB_WORKSPACE', 'SWANLAB_EXPERIMENT_NAME'):
        os.environ.pop(key, None)
    tracker = swanlab.init(id=RUN_ID, resume='must', project='finance', workspace='roc_fu',
                           experiment_name=EXPERIMENT, config=recipe())
    phase('dashboard_ready', run_id=RUN_ID, url=getattr(tracker, 'url', None))
    swanlab.finish()
    input_root = Path('/kaggle/input')
    runtime = Path('/kaggle/working/small_power_runtime')
    data_root = LEGACY['prepare_data'](input_root, runtime / 'data')
    tokenizer = find_one(input_root, '**/Kronos-Tokenizer-base/model.safetensors').parent
    if sha256(tokenizer / 'model.safetensors') != LEGACY['TOKENIZER_SHA']:
        raise RuntimeError('Tokenizer mismatch')
    parent = find_one(input_root, '**/last_state.pt').parent.parent
    continuation = CHUNK_CONFIG['chunk'] > FIRST_CHUNK
    boundary = validate_output(parent, torch, recipe(), parent=not continuation)
    if boundary >= TOTAL_SEGMENTS:
        raise RuntimeError('Second round has already completed')
    output = runtime / 'outputs/models' / OUTPUT_NAME
    if not continuation:
        output.mkdir(parents=True)
        for name in ('run.log', 'metrics.jsonl'):
            shutil.copy2(parent / name, output / name)
        (output / f'round{ROUND_NUMBER - 1}_parent.json').write_text(json.dumps({
            'kernel': recipe()['parent_kernel'], 'state_sha256': sha256(parent / 'checkpoints/last_state.pt'),
            'best': json.loads((parent / 'checkpoints/best_model/best_metric.json').read_text()),
            'next_epoch': boundary, 'global_step': PARENT_GLOBAL_STEP,
            'parent_recipe': json.loads((parent / 'experiment_manifest.json').read_text())['recipe'],
        }, indent=2))
    env = training_environment(runtime, data_root, parent, tokenizer, continuation, boundary)
    env['KRONOS_MAX_RUNTIME_SECONDS'] = str(max(60, BUDGET_SECONDS - int(time.monotonic() - started)))
    phase('training_started', recipe=recipe(), next_segment=boundary + 1,
          next_global_step=PARENT_GLOBAL_STEP + (boundary - START_SEGMENT) * 313 + 1)
    command([sys.executable, '-u', 'finetune/kaggle_kronos_small_v21.py'], cwd=REPO, env=env)
    (output / 'experiment_manifest.json').write_text(json.dumps({
        'recipe': recipe(), 'chunk': CHUNK_CONFIG['chunk'], 'parent_kernel': CHUNK_CONFIG['parent_kernel'],
    }, indent=2))
    entry_log = Path('/kaggle/working/entry.log')
    if entry_log.is_file():
        with (output / 'run.log').open('a') as handle:
            handle.write(entry_log.read_text())
    validate_output(output, torch, recipe())
    shutil.copytree(tokenizer, runtime / 'models/Kronos-Tokenizer-base', dirs_exist_ok=True)
    phase('finished', output=str(output), elapsed_seconds=time.monotonic() - started)


if __name__ == '__main__':
    main()
