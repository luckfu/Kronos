"""Resume Stage3 segment 2..6 with the same optimizer, data order and dashboard."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import urllib.request
from datetime import datetime, timezone

COMMIT = '3595dc2381fd6c920cac9bad7514ebba46e490f0'


def main():
    os.environ.update(PYTHONUNBUFFERED='1', STAGE3_SOURCE_COMMIT=COMMIT,
                     STAGE3_RESUME_KERNEL='smmt315/kronos-small-0-1-stage3-joint-path-smoke',
                     STAGE3_RESUME_STATE_SHA256='cf4a62e4084e6d380cd39165cee2bd6a5c48535264a01ab1960d01e6d8fe53f8',
                     STAGE3_TARGET_SEGMENTS='6', STAGE3_CHUNK='c2')
    output = Path('/kaggle/working/stage3_joint_path_smoke')
    output.mkdir(parents=True, exist_ok=True)
    def log(phase, **fields):
        line = json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields})
        print(line, flush=True)
        with (output / 'run.log').open('a', buffering=1) as f: f.write(line + '\n')
    log('started', source_commit=COMMIT, next_segment=2, last_segment=6, resume_step=313,
        hard_timeout_seconds=18000)
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
