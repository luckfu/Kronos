"""Stage3 freeze probe E: train forecast_head only from C2 best, 5 segments."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import urllib.request
from datetime import datetime, timezone

COMMIT = '0000000000000000000000000000000000000000'


def main():
    os.environ.update(
        PYTHONUNBUFFERED='1',
        STAGE3_SOURCE_COMMIT=COMMIT,
        STAGE3_TARGET_SEGMENTS='5',
        STAGE3_CHUNK='freeze_probe_e',
        STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_freeze_probe_e_forecast_head_from_c2_best_v1',
        STAGE3_LAMBDA_PATH='0',
        STAGE3_CE_RANK='1',
        STAGE3_LAMBDA_RANK='0',
        STAGE3_HISTORY_LOSS_WEIGHT='0.02',
        STAGE3_FORECAST_HORIZON_WEIGHTS='1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455',
        STAGE3_TRAINABLE_MASK='forecast_head',
        STAGE3_MILESTONE_SEGMENTS='1,3,5',
        STAGE3_BASELINE_BEFORE_RESUME='1',
        STAGE3_MAX_RUNTIME_SECONDS='7200',
        STAGE3_HARD_TIMEOUT_SECONDS='9000',
    )
    os.environ.pop('STAGE3_RESUME_KERNEL', None)
    os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)
    os.environ.pop('STAGE3_EXPECTED_PROGRESS', None)
    output = Path('/kaggle/working/stage3_joint_path_smoke')
    output.mkdir(parents=True, exist_ok=True)

    def log(phase, **fields):
        line = json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields})
        print(line, flush=True)
        with (output / 'run.log').open('a', buffering=1) as handle:
            handle.write(line + '\n')

    log('started', source_commit=COMMIT, segments=5, lambda_path=0, ce_rank=True,
        lambda_rank=0, history_weight=0.02, trainable_mask='forecast_head',
        milestone_segments='1,3,5',
        run_id='small_0.1_stage3_freeze_probe_e_forecast_head_from_c2_best_v1',
        hard_timeout_seconds=9000, max_runtime_seconds=7200, baseline_before_resume=True)
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
