"""Compare baseline and size-conditioned checkpoints on the validation set."""

import argparse
import gc
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset import QlibDataset
from model import Kronos, KronosTokenizer


def evaluate(path, dataset, loader, tokenizer, device, config):
    dataset.set_epoch_seed(0)
    torch.manual_seed(config.seed)
    model = Kronos.from_pretrained(
        path,
        num_sectors=config.num_sectors,
        num_size_buckets=config.num_size_buckets,
        context_layer=config.context_layer,
        use_size_percentile=config.use_size_percentile,
        size_mlp_hidden_dim=config.size_mlp_hidden_dim,
    ).to(device).eval()
    by_bucket = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            x, stamp, size = batch[0].to(device), batch[1].to(device), batch[3].to(device)
            percentile = (
                batch[4].to(device) if len(batch) > 4 and config.use_size_percentile else None
            )
            s1, s2 = tokenizer.encode(x, half=True)
            s1_logits, s2_logits = model(
                s1[:, :-1], s2[:, :-1], stamp[:, :-1], size_bucket=size,
                size_percentile=percentile,
            )
            s1_loss = F.cross_entropy(
                s1_logits.transpose(1, 2), s1[:, 1:], reduction='none'
            ).mean(1)
            s2_loss = F.cross_entropy(
                s2_logits.transpose(1, 2), s2[:, 1:], reduction='none'
            ).mean(1)
            losses = ((s1_loss + s2_loss) / 2).cpu().tolist()
            for bucket, loss in zip(size.cpu().tolist(), losses):
                by_bucket[int(bucket)].append(float(loss))
    result = {
        'overall': sum(sum(values) for values in by_bucket.values()) / sum(len(values) for values in by_bucket.values()),
        'by_bucket': {str(key): sum(values) / len(values) for key, values in sorted(by_bucket.items())},
    }
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return result


def main():
    config = Config()
    parser = argparse.ArgumentParser(description='Evaluate A-share size conditioning')
    parser.add_argument('--base', default=config.pretrained_predictor_path)
    parser.add_argument('--model', default='./outputs/models/a_share_size_kronos_base/checkpoints/best_model')
    parser.add_argument('--output', default='./outputs/models/a_share_size_kronos_base/size_eval.json')
    args = parser.parse_args()

    device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    dataset = QlibDataset('val')
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path).to(device).eval()
    baseline = evaluate(args.base, dataset, loader, tokenizer, device, config)
    finetuned = evaluate(args.model, dataset, loader, tokenizer, device, config)

    result = {'baseline': baseline, 'finetuned': finetuned}
    result['improvement_pct'] = (baseline['overall'] - finetuned['overall']) / baseline['overall'] * 100
    result['bucket_improvement_pct'] = {
        bucket: (baseline['by_bucket'][bucket] - value) / baseline['by_bucket'][bucket] * 100
        for bucket, value in finetuned['by_bucket'].items()
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
