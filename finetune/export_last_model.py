"""Export the final model weights from a completed/resumed training state."""
import argparse
import json
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    checkpoint = output / 'checkpoints'
    state_path = checkpoint / 'last_state.pt'
    best = checkpoint / 'best_model'
    last = checkpoint / 'last_model'
    if not state_path.is_file() or not (best / 'config.json').is_file():
        raise SystemExit(f'Incomplete training output: {output}')
    from model import Kronos
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    config = json.loads((best / 'config.json').read_text())
    model = Kronos.from_pretrained(best)
    model.load_state_dict(state['model'], strict=True)
    model.save_pretrained(last, config=config)
    readme = best / 'README.md'
    if readme.is_file():
        shutil.copy2(readme, last / 'README.md')
    print(f'Exported final last_model to {last}')


if __name__ == '__main__':
    main()
