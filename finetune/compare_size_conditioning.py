"""Compare two fixed-seed size-conditioning evaluation artifacts."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Compare size-conditioning evaluations')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.reference) as handle:
        reference = json.load(handle)
    with open(args.candidate) as handle:
        candidate = json.load(handle)

    reference_loss = float(reference['finetuned']['overall'])
    candidate_loss = float(candidate['finetuned']['overall'])
    delta = candidate_loss - reference_loss
    buckets = sorted(
        set(reference['finetuned']['by_bucket'])
        & set(candidate['finetuned']['by_bucket']),
        key=int,
    )
    result = {
        'reference_loss': reference_loss,
        'candidate_loss': candidate_loss,
        'candidate_minus_reference': delta,
        'candidate_relative_change_pct': delta / reference_loss * 100.0,
        'winner': 'candidate' if delta < 0 else 'reference',
        'by_bucket_candidate_minus_reference': {
            bucket: (
                float(candidate['finetuned']['by_bucket'][bucket])
                - float(reference['finetuned']['by_bucket'][bucket])
            )
            for bucket in buckets
        },
    }
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
