"""Validate every local OOS window and the training sector vocabulary before GPU use."""

import json
from pathlib import Path

import pandas as pd

from finetune.small_clean_oos.evaluate import (
    METADATA_SHA, WindowStore, audit_training_dates, checked, load_oos, phase,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/a_share_full_market_v1_beta_temporal_symbol_validation_no_suspension_v1"


def main():
    metadata = DATA / "asset_metadata.csv"
    checked(metadata, METADATA_SHA)
    labels = set()
    for part in pd.read_csv(metadata, usecols=["sector"], chunksize=500000):
        labels.update(part.sector.dropna().astype(str))
    labels = sorted(labels)
    if len(labels) != 86:
        raise RuntimeError("Training sector vocabulary drift")
    panel, records, manifest = load_oos(DATA / "evaluation_oos_full_through_20261009")
    store = WindowStore(panel, labels)
    for index, record in enumerate(records, 1):
        store.prepare(record)
        if index % 20000 == 0:
            phase("local_window_audit", windows=index, total=len(records))
    result = dict(status="passed", windows=len(records), dates=34,
                  training_sector_labels=labels, artifacts=manifest["artifacts"],
                  raw_labels_verified=True, window=130,
                  training_selection_max_row_dates=audit_training_dates(DATA))
    path = ROOT / "finetune/reports/small_clean_full_oos_preflight.json"
    path.write_text(json.dumps(result, indent=2))
    phase("local_preflight_passed", windows=len(records), report=str(path))


if __name__ == "__main__":
    main()
