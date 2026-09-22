"""One autoregressive vol-calibration segment from C2 best.

Production decode T=0.65 / top_p=0.8 / N=5. Score function on the five
sampled paths. Not a C3 resume and not either teacher-forced vol smoke.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import urllib.request
from datetime import datetime, timezone

COMMIT = 'c13bb2699718de32716af3defeaab121058293fc'


def main():
    os.environ.update(
        PYTHONUNBUFFERED='1',
        STAGE3_SOURCE_COMMIT=COMMIT,
        STAGE3_TARGET_SEGMENTS='1',
        STAGE3_CHUNK='ar_vol_smoke',
        STAGE3_SWANLAB_RUN_ID='small_0.1_stage3_ar_vol_from_c2_v1',
        STAGE3_SEED='20260922',
        STAGE3_LAMBDA_PATH='0',
        STAGE3_LAMBDA_VOL='0.15',
        STAGE3_VOL_TEMPERATURE='0.65',
        STAGE3_VOL_TOP_P='0.8',
        STAGE3_VOL_SAMPLES='5',
        STAGE3_AR_VOL='1',
        STAGE3_BATCH='8',
        STAGE3_BASELINE_BEFORE_RESUME='1',
        STAGE3_MAX_RUNTIME_SECONDS='39600',
        STAGE3_HARD_TIMEOUT_SECONDS='43200',
    )
    os.environ.pop('STAGE3_RESUME_KERNEL', None)
    os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)
    os.environ.pop('STAGE3_EXPECTED_PROGRESS', None)
    os.environ.pop('STAGE3_CE_RANK', None)
    output = Path('/kaggle/working/stage3_joint_path_smoke')
    output.mkdir(parents=True, exist_ok=True)
    def log(phase, **fields):
        line = json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields})
        print(line, flush=True)
        with (output / 'run.log').open('a', buffering=1) as f: f.write(line + '\n')
    log('started', source_commit=COMMIT, segments=1, batch_per_gpu=8, lambda_path=0, lambda_vol=0.15,
        ar_vol=True, vol_temperature=0.65, vol_top_p=0.8, vol_samples=5, seed=20260922,
        run_id='small_0.1_stage3_ar_vol_from_c2_v1',
        hard_timeout_seconds=43200, max_runtime_seconds=39600, baseline_before_resume=True,
        resume_c3=False, teacher_forced_vol=False,
        go_rule='ar_ratio_closer_to_1_and_token_ce_increase_lt_0.03')
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
