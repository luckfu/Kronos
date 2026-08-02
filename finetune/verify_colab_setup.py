"""Validate a Colab A-share fine-tuning workspace before a long job."""

import argparse
import json
import pickle
from pathlib import Path

import torch


def inspect_split(path, window):
    with path.open('rb') as handle:
        panel = pickle.load(handle)
    rows = sum(len(frame) for frame in panel.values())
    windows = sum(max(len(frame) - window + 1, 0) for frame in panel.values())
    return {'symbols': len(panel), 'rows': rows, 'windows': windows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--min-train-windows', type=int, default=1_000_000)
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_model = Path(args.base_model)
    train_path = data_dir / 'train_data.pkl'
    val_path = data_dir / 'val_data.pkl'
    missing = [str(path) for path in (train_path, val_path) if not path.exists()]
    if missing:
        raise SystemExit(f'Missing prepared dataset files: {missing}')
    for name in ('config.json', 'model.safetensors'):
        if not (base_model / name).exists():
            raise SystemExit(f'Missing base model file: {base_model / name}')

    report = {
        'torch': torch.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'train': inspect_split(train_path, 101),
        'val': inspect_split(val_path, 101),
        'base_model': str(base_model),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Add --allow-cpu only for a deliberate CPU smoke test.')
    if report['train']['windows'] < args.min_train_windows:
        raise SystemExit(
            f"Training dataset is too small: {report['train']['windows']} windows "
            f"< {args.min_train_windows}"
        )


if __name__ == '__main__':
    main()
