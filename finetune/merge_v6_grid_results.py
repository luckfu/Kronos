#!/usr/bin/env python3
"""Merge split Kaggle V6 grid outputs into one validated result table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    (sample_count, temperature, top_p)
    for sample_count in (32, 50)
    for temperature in (0.65, 0.70, 0.75)
    for top_p in (0.7, 0.8, 0.9)
}

METRICS = (
    "observations",
    "stocks",
    "periods",
    "mean_rank_ic",
    "rank_ic_positive_rate",
    "direction_accuracy",
    "return_mae",
    "mean_top_bottom_actual_return_spread",
    "predicted_down_rate",
)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = []
    for source_key, item in payload.items():
        config = item["configuration"]
        sample_count = int(config["sample_count"])
        top_p = float(config["top_p"])
        for result in item["results"].values():
            temperature = float(result["temperature"])
            summary = result["summary"]
            row = {
                "sample_count": sample_count,
                "temperature": temperature,
                "top_p": top_p,
                **{metric: summary[metric] for metric in METRICS},
                "source_file": str(path),
                "source_key": source_key,
            }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[tuple[int, float, float], dict] = {}
    for path in args.inputs:
        for row in load_rows(path):
            key = (row["sample_count"], row["temperature"], row["top_p"])
            if key in merged:
                comparable = {name: row[name] for name in METRICS}
                previous = {name: merged[key][name] for name in METRICS}
                if comparable != previous:
                    raise ValueError(f"Conflicting results for {key}: {path}")
                continue
            merged[key] = row

    unexpected = sorted(set(merged) - EXPECTED)
    if unexpected:
        raise ValueError(f"Unexpected parameter combinations: {unexpected}")

    rows = [merged[key] for key in sorted(merged)]
    missing = [
        {"sample_count": key[0], "temperature": key[1], "top_p": key[2]}
        for key in sorted(EXPECTED - set(merged))
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "merged_grid_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    (args.output_dir / "missing_combinations.json").write_text(
        json.dumps(missing, ensure_ascii=False, indent=2)
    )
    columns = ("sample_count", "temperature", "top_p", *METRICS, "source_file", "source_key")
    with (args.output_dir / "merged_grid_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"merged={len(rows)}/{len(EXPECTED)}")
    print(f"missing={len(missing)}")
    for item in missing:
        print(
            "pending "
            f"sample_count={item['sample_count']} "
            f"temperature={item['temperature']:.2f} top_p={item['top_p']:.1f}"
        )


if __name__ == "__main__":
    main()
