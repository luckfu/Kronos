"""Resume Stage3 segment 7..15 within remaining dual-T4 GPU quota."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import urllib.request
from datetime import datetime, timezone

COMMIT = 'PINNED_STAGE3_C3_SOURCE_COMMIT_PLACEHOLDER0'


def main():
    expected_progress = {
        'completed_segments': 6,
        'next_epoch': 6,
        'step': 1878,
        'status': 'completed',
    }
    os.environ.update(
        PYTHONUNBUFFERED='1',
        STAGE3_SOURCE_COMMIT=COMMIT,
        STAGE3_RESUME_KERNEL='smmt315/kronos-small-0-1-stage3-joint-path-c2',
        STAGE3_TARGET_SEGMENTS='15',
        STAGE3_CHUNK='c3',
        STAGE3_EXPECTED_PROGRESS=json.dumps(expected_progress, separators=(',', ':')),
        STAGE3_MAX_RUNTIME_SECONDS='7200',
        STAGE3_HARD_TIMEOUT_SECONDS='9000',
        STAGE3_BASELINE_BEFORE_RESUME='0',
    )
    os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)
    output = Path('/kaggle/working/stage3_joint_path_smoke')
    output.mkdir(parents=True, exist_ok=True)
    def log(phase, **fields):
        line = json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields})
        print(line, flush=True)
        with (output / 'run.log').open('a', buffering=1) as f: f.write(line + '\n')
    log('started', source_commit=COMMIT, next_segment=7, last_segment=15, resume_step=1878,
        new_segments=9, hard_timeout_seconds=9000, max_runtime_seconds=7200,
        baseline_before_resume=False)
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
