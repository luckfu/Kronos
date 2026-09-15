"""Fresh Stage3 CE-only causal control from C2 best, dual T4."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import urllib.request
from datetime import datetime, timezone

COMMIT = 'bf83ef0bcafe4d8fc6ddebf0a17a92ec929ee5cf'


def main():
    os.environ.update(
        PYTHONUNBUFFERED='1',
        STAGE3_SOURCE_COMMIT=COMMIT,
        STAGE3_TARGET_SEGMENTS='15',
        STAGE3_CHUNK='ce_only_control',
        STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_ce_only_control_from_c2_best_v1',
        STAGE3_LAMBDA_PATH='0',
        STAGE3_MILESTONE_SEGMENTS='5,10',
        STAGE3_BASELINE_BEFORE_RESUME='1',
        STAGE3_MAX_RUNTIME_SECONDS='12600',
        STAGE3_HARD_TIMEOUT_SECONDS='15000',
    )
    os.environ.pop('STAGE3_RESUME_KERNEL', None)
    os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)
    os.environ.pop('STAGE3_EXPECTED_PROGRESS', None)
    output = Path('/kaggle/working/stage3_joint_path_smoke')
    output.mkdir(parents=True, exist_ok=True)
    def log(phase, **fields):
        line = json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields})
        print(line, flush=True)
        with (output / 'run.log').open('a', buffering=1) as f: f.write(line + '\n')
    log('started', source_commit=COMMIT, segments=15, lambda_path=0,
        milestone_segments='5,10', run_id='small_0.1_stage3_ce_only_control_from_c2_best_v1',
        hard_timeout_seconds=15000, max_runtime_seconds=12600, baseline_before_resume=True)
    url = f'https://raw.githubusercontent.com/luckfu/Kronos/{COMMIT}/finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py'
    for attempt in range(3):
        log('fetch_pinned_entry', attempt=attempt + 1)
        try:
            with urllib.request.urlopen(url, timeout=20) as response: code = response.read()
            break
        except Exception:
            if attempt == 2: raise
            time.sleep(2)
    path = output / 'pinned_entry.py'
    path.write_bytes(code)
    log('pinned_entry_ready', sha256=hashlib.sha256(code).hexdigest())
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
