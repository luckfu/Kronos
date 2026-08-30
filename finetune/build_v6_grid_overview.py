#!/usr/bin/env python3
"""Build a stage-aware overview of V6 coarse, fine, and exploratory grids."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from merge_v6_grid_results import METRICS, load_rows


COARSE_EXPECTED = {
    (16, 0.50, 0.8),
    (16, 0.70, 0.6),
    (16, 0.70, 0.8),
    (16, 0.70, 1.0),
    (16, 0.90, 0.8),
}
FINE_EXPECTED = {
    (sample_count, temperature, top_p)
    for sample_count in (32, 50)
    for temperature in (0.65, 0.70, 0.75)
    for top_p in (0.7, 0.8, 0.9)
}


def key(row: dict) -> tuple[int, float, float]:
    return row["sample_count"], row["temperature"], row["top_p"]


def validated_stage(paths: list[Path], stage: str, expected: set[tuple]) -> list[dict]:
    merged = {}
    for path in paths:
        for row in load_rows(path):
            row_key = key(row)
            if row_key not in expected:
                raise ValueError(f"Unexpected {stage} combination {row_key} in {path}")
            if row_key in merged:
                previous = {metric: merged[row_key][metric] for metric in METRICS}
                current = {metric: row[metric] for metric in METRICS}
                if previous != current:
                    raise ValueError(f"Conflicting {stage} results for {row_key}")
                continue
            row["stage"] = stage
            row["status"] = "complete"
            merged[row_key] = row
    return [merged[item] for item in sorted(merged)]


def load_fine_extensions(paths: list[Path]) -> list[dict]:
    """Load explicitly requested fine-grid extensions outside the 18-point core."""
    merged = {}
    for path in paths:
        for row in load_rows(path):
            row_key = key(row)
            if row_key in FINE_EXPECTED:
                continue
            if row_key in merged:
                previous = {metric: merged[row_key][metric] for metric in METRICS}
                current = {metric: row[metric] for metric in METRICS}
                if previous != current:
                    raise ValueError(f"Conflicting fine extension results for {row_key}")
                continue
            row["stage"] = "fine_extension"
            row["status"] = "complete"
            merged[row_key] = row
    return [merged[item] for item in sorted(merged)]


def load_exploratory(path: Path | None) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    rows = []
    for item in payload.get("results", []):
        if item.get("source") != "partial_fine":
            continue
        row = {
            "stage": "exploratory_partial",
            "status": item.get("status", "complete_from_log"),
            "sample_count": int(item["sample_count"]),
            "temperature": float(item["temperature"]),
            "top_p": float(item["top_p"]),
            **{metric: item.get(metric) for metric in METRICS},
            "source_file": str(path),
            "source_key": item.get("notes", ""),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", nargs="+", type=Path, required=True)
    parser.add_argument("--fine", nargs="+", type=Path, required=True)
    parser.add_argument("--fine-extension", nargs="*", type=Path, default=[])
    parser.add_argument("--exploratory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    coarse = validated_stage(args.coarse, "coarse", COARSE_EXPECTED)
    fine = validated_stage(args.fine, "fine", FINE_EXPECTED)
    fine_extensions = load_fine_extensions(args.fine_extension)
    exploratory = load_exploratory(args.exploratory)
    rows = sorted(
        [*coarse, *fine, *fine_extensions, *exploratory],
        key=lambda row: (row["stage"], row["sample_count"], row["temperature"], row["top_p"]),
    )

    missing = {
        "coarse": [list(item) for item in sorted(COARSE_EXPECTED - {key(row) for row in coarse})],
        "fine": [list(item) for item in sorted(FINE_EXPECTED - {key(row) for row in fine})],
    }
    manifest = {
        "coarse": {"complete": len(coarse), "expected": len(COARSE_EXPECTED)},
        "fine": {"complete": len(fine), "expected": len(FINE_EXPECTED)},
        "fine_extensions": {"complete": len(fine_extensions)},
        "exploratory_partial": {"complete": len(exploratory)},
        "missing": missing,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_grid_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    (args.output_dir / "grid_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    columns = (
        "stage", "status", "sample_count", "temperature", "top_p",
        *METRICS, "source_file", "source_key",
    )
    with (args.output_dir / "all_grid_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
