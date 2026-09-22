"""After the 1-segment AR run, rescore the same four causal-val dates.

Uses last weights, the gap-probe autoregressive path, and the locked go rule.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from finetune.stage3_ar_vol import decide


OUTPUT = Path(os.environ.get('STAGE3_OUTPUT', '/kaggle/working/stage3_joint_path_smoke'))


def _emit(payload):
    print(json.dumps(payload), flush=True)


def _prepare_windows(repo):
    sys.path.insert(0, str(repo))
    os.environ['KRONOS_STAGE3_VOL_LOSS'] = '1'
    os.environ['KRONOS_STAGE3_RANK_LOSS'] = '0'
    from finetune.dataset import QlibDataset
    from finetune.vol_gap_probe import _extract_windows, day_to_date, last_signal_days

    dataset = QlibDataset('val')
    days = last_signal_days(dataset.signal_date_ids, 4)
    windows = _extract_windows(dataset, days)
    del dataset
    import gc
    gc.collect()
    path = OUTPUT / 'ar_probe_windows.npz'
    import numpy as np
    np.savez_compressed(path, **windows)
    dates = [day_to_date(day) for day in days]
    _emit({'phase': 'ar_probe_windows', 'dates': dates, 'samples': int(len(windows['x']))})
    return path, dates


def _worker(rank):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    import numpy as np
    import torch
    from finetune.vol_gap_probe import score_shard
    from model import Kronos, KronosTokenizer

    device = torch.device('cuda:0')
    payload = np.load(os.environ['KRONOS_AR_PROBE_WINDOWS'])
    index = np.arange(len(payload['x']))
    shard_index = index[index % 2 == rank]
    windows = {key: payload[key][shard_index] for key in payload.files}
    tokenizer = KronosTokenizer.from_pretrained(os.environ['KRONOS_AR_PROBE_TOKENIZER']).to(device).eval()
    predictor = Kronos.from_pretrained(os.environ['KRONOS_AR_PROBE_MODEL']).to(device).eval()
    _emit({'phase': 'ar_probe_worker', 'rank': rank, 'samples': int(len(shard_index))})
    rows = score_shard(predictor, tokenizer, windows, device)
    out = OUTPUT / f'ar_probe_shard_{rank}.json'
    out.write_text(json.dumps(rows))
    _emit({'phase': 'ar_probe_worker_done', 'rank': rank, 'rows': len(rows)})


def _decision(rows, dates):
    from finetune.vol_gap_probe import summarize
    measured = summarize(rows)
    baseline = json.loads((OUTPUT / 'c2_causal_baseline.json').read_text())
    summary = json.loads((OUTPUT / 'summary.json').read_text())
    if not summary.get('segments'):
        raise RuntimeError('AR probe found no completed segment')
    verdict = decide(
        measured['ar_t065_n5']['calibration_ratio'],
        baseline['validation']['token_loss'],
        summary['segments'][-1]['validation']['token_loss'],
    )
    verdict.update(
        checkpoint='last_model',
        dates=dates,
        samples=len(rows),
        probe=measured,
    )
    (OUTPUT / 'ar_probe_summary.json').write_text(json.dumps(verdict, indent=2))
    _emit({'phase': 'ar_probe_decision', **{key: value for key, value in verdict.items() if key != 'probe'}})
    return verdict


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get('KRONOS_AR_PROBE_RANK'):
        _worker(int(os.environ['KRONOS_AR_PROBE_RANK']))
        return
    repo = Path(os.environ.get('KRONOS_AR_PROBE_REPO', Path(__file__).resolve().parents[1]))
    windows, dates = _prepare_windows(repo)
    procs = []
    for rank in (0, 1):
        env = os.environ.copy()
        env.update(
            KRONOS_AR_PROBE_RANK=str(rank),
            KRONOS_AR_PROBE_WINDOWS=str(windows),
            CUDA_VISIBLE_DEVICES=str(rank),
            PYTHONPATH=str(repo),
        )
        procs.append(subprocess.Popen([sys.executable, '-u', '-m', 'finetune.stage3_ar_probe'], env=env))
    codes = [proc.wait() for proc in procs]
    if any(codes):
        raise RuntimeError(f'AR probe worker failed: {codes}')
    rows = []
    for rank in (0, 1):
        rows.extend(json.loads((OUTPUT / f'ar_probe_shard_{rank}.json').read_text()))
    _decision(rows, dates)


if __name__ == '__main__':
    main()
