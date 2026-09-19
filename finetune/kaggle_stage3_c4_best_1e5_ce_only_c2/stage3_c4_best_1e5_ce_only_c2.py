"""Resume Stage3 1e-5 CE-only from seg15 for ~11h on the same dashboard."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import urllib.request
from datetime import datetime, timezone

COMMIT = 'f73c3e04612f4904c58cb18bb23800a6ed0f33fb'


def main():
    expected_progress = {
        'completed_segments': 15,
        'next_epoch': 15,
        'step': 4695,
        'status': 'completed',
    }
    os.environ.update(
        PYTHONUNBUFFERED='1',
        STAGE3_SOURCE_COMMIT=COMMIT,
        STAGE3_RESUME_KERNEL='user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only',
        STAGE3_EXPECTED_PROGRESS=json.dumps(expected_progress, separators=(',', ':')),
        STAGE3_TARGET_SEGMENTS='80',
        STAGE3_CHUNK='c2',
        STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_c4_best_1e5_ce_only_v1',
        STAGE3_LR='1e-5',
        STAGE3_SEED='20260919',
        STAGE3_PARENT_KERNEL='user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
        STAGE3_PARENT_BEST_GLOB='**/small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors',
        STAGE3_PARENT_BEST_SHA='9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075',
        STAGE3_TOKENIZER_REPO='NeoQuasar/Kronos-Tokenizer-base',
        STAGE3_LAMBDA_PATH='0',
        STAGE3_MILESTONE_SEGMENTS='20,30,40,50,60,70,80',
        STAGE3_BASELINE_BEFORE_RESUME='0',
        STAGE3_MAX_RUNTIME_SECONDS='39600',
        STAGE3_HARD_TIMEOUT_SECONDS='42000',
    )
    os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)
    os.environ.pop('STAGE3_CE_RANK', None)
    output = Path('/kaggle/working/stage3_joint_path_smoke')
    output.mkdir(parents=True, exist_ok=True)

    def log(phase, **fields):
        line = json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields})
        print(line, flush=True)
        with (output / 'run.log').open('a', buffering=1) as handle:
            handle.write(line + '\n')

    log('started', source_commit=COMMIT, next_segment=16, last_segment=80, resume_step=4695,
        new_segments=65, lambda_path=0, lr=1e-5, coverage_seed=20260919,
        parent='c4_best_g528', run_id='small_0.1_stage3_c4_best_1e5_ce_only_v1',
        hard_timeout_seconds=42000, max_runtime_seconds=39600, baseline_before_resume=False)
    url = f'https://raw.githubusercontent.com/luckfu/Kronos/{COMMIT}/finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py'
    for attempt in range(3):
        log('fetch_pinned_entry', attempt=attempt + 1)
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                code = response.read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    path = output / 'pinned_entry.py'
    path.write_bytes(code)
    log('pinned_entry_ready', sha256=hashlib.sha256(code).hexdigest())
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
