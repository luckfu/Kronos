"""Build the temporal Rank IC panel and pin its SHA on the frozen calendar.

The price panel is not stored in git. This script reads the clean Kaggle
dataset ``smmt315/a-share-120d-temporal-symbol-holdout`` (data_manifest SHA
``b85ccc45…``), writes ``temporal_rank_ic_panel.pkl`` and
``temporal_rank_ic_rows.pkl`` beside it, and fills ``rows_sha256``.
``panel_sha256`` is rewritten only when the panel file digest differs from
the value already in the manifest. It does not change ``signal_dates`` or
the calendar SHA, and it does not push a Kaggle kernel.

Score time must not open ``val_data.pkl`` or join ``asset_metadata.csv``.
Those files are inputs here only. Holdout symbols' post-cutoff bars live on
the validation pickle; train symbols' later bars live on the training pickle.
The script concatenates them per symbol, bakes ``sector_id`` and
``size_percentile`` onto each frame, and writes a new pickle.

``sector_id`` uses the same sorted-string vocabulary as
``finetune/asset_metadata.py`` (unknown id = ``KRONOS_NUM_SECTORS``, default
86). Prefer columns already on the frame. Look up asset metadata only when a
frame is missing ``sector_id`` or ``size_percentile``.

Example, from the repo root, after the clean dataset is on disk::

    python finetune/build_temporal_rank_ic_panel.py \\
        --dataset-root /path/to/a-share-120d-temporal-symbol-holdout \\
        --output /path/to/temporal_rank_ic_panel.pkl

Then upload both pickles together as their own Kaggle dataset (the slug must
not be the symbol-holdout training set). Commit ``rows_sha256``. Leave
``panel_sha256`` unchanged when the panel file digest still matches the
manifest (the pinned panel is ``8539246b…``). This script does not push a
Kaggle kernel and will not invent a digest. ``push_chunk`` still refuses to
run while ``panel_sha256`` is empty.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ashare_sse_calendar_2026 import is_sse_session
from temporal_rank_ic_selection import (
    LOOKBACK,
    MIN_NAMES_PER_DATE,
    PANEL_FILENAME,
    PREDICT,
    ROWS_FILENAME,
    TemporalRankICSelectionError,
    build_evaluation_rows,
    manifest_sha256,
    path_mentions_symbol_holdout,
    sha256_file,
    validate_manifest_calendar,
    write_evaluation_rows,
)


CLEAN_DATA_MANIFEST_SHA256 = (
    "b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd"
)
CLEAN_TRAIN_CUTOFF = "2024-12-31"
DEFAULT_MANIFEST = HERE / "manifests" / "temporal_rank_ic_v1" / "selection_window.json"
NUM_SECTORS = 86
PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
PANEL_COLUMNS = PRICE_COLUMNS + ("sector_id", "size_percentile")
PICKLE_PROTOCOL = 4


def find_clean_manifest(dataset_root, *, expected_sha, expected_cutoff):
    """Return the one clean ``data_manifest.json`` whose SHA and cutoff match."""
    root = Path(dataset_root)
    matches = [
        path for path in root.glob("**/data_manifest.json")
        if path.is_file() and sha256_file(path) == expected_sha
    ]
    if len(matches) != 1:
        raise TemporalRankICSelectionError(
            "Expected one clean data_manifest.json with SHA "
            f"{expected_sha}; found {[str(path) for path in matches[:8]]}. "
            "Refusing to build the temporal panel from a different dataset."
        )
    manifest_path = matches[0]
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TemporalRankICSelectionError(
            f"Clean data_manifest.json is not JSON: {manifest_path}"
        ) from exc
    cutoff = str((document.get("split") or {}).get("train_cutoff") or "")
    if cutoff != expected_cutoff:
        raise TemporalRankICSelectionError(
            f"clean data_manifest train_cutoff is {cutoff or '<missing>'}, "
            f"not {expected_cutoff}. The frozen calendar is locked to the "
            "clean split cutoff. Refusing to build the panel."
        )
    return manifest_path, document


def assert_output_path(path) -> Path:
    destination = Path(path).expanduser()
    if destination.name != PANEL_FILENAME or destination.name == "val_data.pkl":
        raise TemporalRankICSelectionError(
            f"Refusing to write {destination}. The selection panel file name "
            f"must be {PANEL_FILENAME}, not val_data.pkl."
        )
    if path_mentions_symbol_holdout(destination):
        raise TemporalRankICSelectionError(
            f"Refusing to write the temporal panel under a symbol-holdout path "
            f"({destination})."
        )
    return destination


def assert_signal_dates_are_sessions(signal_dates):
    bad = [day for day in signal_dates if not is_sse_session(day)]
    if bad:
        raise TemporalRankICSelectionError(
            "signal_dates include non-sessions under the SSE 2026 notice: "
            f"{bad[:8]}. Refusing to build a panel on invented dates."
        )


def _metadata_datetime(metadata: pd.DataFrame) -> pd.Series:
    if "datetime" in metadata.columns:
        column = "datetime"
    elif "date" in metadata.columns:
        column = "date"
    else:
        raise TemporalRankICSelectionError(
            "asset_metadata.csv needs a datetime or date column"
        )
    stamps = pd.to_datetime(metadata[column], errors="coerce")
    if stamps.isna().any():
        raise TemporalRankICSelectionError(
            "asset_metadata.csv has rows with an unparseable datetime"
        )
    return stamps.dt.tz_localize(None).dt.normalize()


def build_sector_map(metadata: pd.DataFrame, num_sectors=NUM_SECTORS) -> dict:
    """Same vocabulary as ``AssetMetadata``: sorted sector strings, unknown = N."""
    if metadata is None:
        raise TemporalRankICSelectionError(
            "asset_metadata.csv is required to map sector names to sector_id"
        )
    if "sector" not in metadata.columns:
        raise TemporalRankICSelectionError(
            "asset_metadata.csv has no sector column; cannot bake sector_id"
        )
    values = metadata["sector"].dropna().astype(str).unique()
    mapping = {value: index for index, value in enumerate(sorted(values))}
    if len(mapping) > int(num_sectors):
        raise TemporalRankICSelectionError(
            f"Found {len(mapping)} sectors but num_sectors={num_sectors}"
        )
    return mapping


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.copy()
    indexed.index = pd.DatetimeIndex(pd.to_datetime(indexed.index)).tz_localize(None)
    indexed.index = indexed.index.normalize()
    if not indexed.index.is_monotonic_increasing:
        indexed = indexed.sort_index()
    if indexed.index.has_duplicates:
        raise TemporalRankICSelectionError("Temporal panel index has duplicate dates")
    indexed.index.name = "datetime"
    return indexed


def _combine_symbol(train_frame, val_frame) -> pd.DataFrame | None:
    parts = []
    for frame in (train_frame, val_frame):
        if frame is not None and len(frame):
            parts.append(_normalize_index(frame))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    combined = pd.concat(parts)
    if combined.index.has_duplicates:
        raise TemporalRankICSelectionError(
            "Train and validation panels overlap on the same symbol and date"
        )
    combined = combined.sort_index()
    combined.index.name = "datetime"
    return combined


def _crop(frame: pd.DataFrame, first_signal: str, last_signal: str) -> pd.DataFrame | None:
    start = pd.Timestamp(first_signal)
    end = pd.Timestamp(last_signal)
    positions = [
        index for index, stamp in enumerate(frame.index)
        if start <= stamp <= end
    ]
    if not positions:
        return None
    lo = max(0, positions[0] - (LOOKBACK - 1))
    hi = min(len(frame) - 1, positions[-1] + PREDICT)
    return frame.iloc[lo:hi + 1]


def _point_in_time(index: pd.DatetimeIndex, rows: pd.DataFrame, column: str) -> pd.Series:
    meta = rows.sort_values("datetime")[["datetime", column]]
    left = pd.DataFrame({"datetime": pd.DatetimeIndex(index)})
    merged = pd.merge_asof(left, meta, on="datetime", direction="backward")
    return merged[column]


def _attach_conditions(frame, symbol, sector_map, metadata_rows, num_sectors):
    """Bake sector_id and size_percentile. Do not leave a score-time join."""
    working = frame.copy()
    rows = None if metadata_rows is None else metadata_rows.get(str(symbol))
    if "sector_id" in working.columns:
        sector_id = pd.to_numeric(working["sector_id"], errors="coerce")
    elif "sector" in working.columns:
        if sector_map is None:
            raise TemporalRankICSelectionError(
                f"{symbol} has sector names but no asset_metadata vocabulary"
            )
        sector_id = working["sector"].map(
            lambda value: sector_map.get(str(value), num_sectors) if pd.notna(value) else num_sectors
        )
        sector_id = pd.to_numeric(sector_id, errors="coerce")
    elif rows is not None and "sector" in rows.columns:
        if sector_map is None:
            raise TemporalRankICSelectionError(
                f"{symbol} needs a sector vocabulary to bake sector_id"
            )
        names = _point_in_time(working.index, rows, "sector")
        sector_id = names.map(
            lambda value: sector_map.get(str(value), num_sectors) if pd.notna(value) else num_sectors
        )
        sector_id = pd.to_numeric(sector_id, errors="coerce")
    else:
        raise TemporalRankICSelectionError(
            f"{symbol} needs sector_id or sector on the panel, or point-in-time "
            "sector rows in asset_metadata.csv. Score time will not join metadata."
        )
    if "size_percentile" in working.columns:
        size = pd.to_numeric(working["size_percentile"], errors="coerce")
    elif rows is not None and "size_percentile" in rows.columns:
        size = pd.to_numeric(
            _point_in_time(working.index, rows, "size_percentile"), errors="coerce",
        )
    else:
        raise TemporalRankICSelectionError(
            f"{symbol} needs size_percentile on the panel or in asset_metadata.csv. "
            "Score time will not join metadata."
        )
    missing = [name for name in PRICE_COLUMNS if name not in working.columns]
    if missing:
        raise TemporalRankICSelectionError(f"{symbol} is missing columns {missing}")
    baked = working.loc[:, list(PRICE_COLUMNS)].copy()
    baked["sector_id"] = sector_id.fillna(num_sectors).astype("int64")
    baked["size_percentile"] = size.astype("float64")
    for name in PRICE_COLUMNS:
        baked[name] = pd.to_numeric(baked[name], errors="coerce").astype("float64")
    baked.index = working.index
    baked.index.name = "datetime"
    return baked.loc[:, list(PANEL_COLUMNS)]


def _metadata_by_symbol(metadata: pd.DataFrame | None) -> dict | None:
    if metadata is None:
        return None
    table = metadata.copy()
    if "symbol" not in table.columns:
        raise TemporalRankICSelectionError("asset_metadata.csv needs a symbol column")
    table["symbol"] = table["symbol"].astype(str)
    table["datetime"] = _metadata_datetime(table)
    grouped = {}
    for symbol, rows in table.groupby("symbol", sort=False):
        grouped[str(symbol)] = rows.sort_values("datetime").reset_index(drop=True)
    return grouped


def assemble_panel(
    train_panel,
    val_panel,
    signal_dates,
    *,
    metadata=None,
    num_sectors=NUM_SECTORS,
    min_names=MIN_NAMES_PER_DATE,
) -> dict:
    """Crop the clean panels to the frozen dates and bake condition columns."""
    if not signal_dates:
        raise TemporalRankICSelectionError("signal_dates is empty")
    first_signal, last_signal = signal_dates[0], signal_dates[-1]
    needs_sector_map = False
    needs_metadata_rows = False
    symbols = sorted(set(train_panel) | set(val_panel), key=str)
    staged = []
    for symbol in symbols:
        combined = _combine_symbol(
            train_panel.get(symbol), val_panel.get(symbol),
        )
        if combined is None:
            continue
        cropped = _crop(combined, first_signal, last_signal)
        if cropped is None or cropped.empty:
            continue
        if "sector_id" not in cropped.columns:
            needs_sector_map = True
        if "sector_id" not in cropped.columns or "size_percentile" not in cropped.columns:
            needs_metadata_rows = True
        staged.append((str(symbol), cropped))
    sector_map = build_sector_map(metadata, num_sectors) if needs_sector_map else None
    by_symbol = _metadata_by_symbol(metadata) if needs_metadata_rows else None
    panel = {}
    for symbol, cropped in staged:
        panel[symbol] = _attach_conditions(
            cropped, symbol, sector_map, by_symbol, int(num_sectors),
        )
    if not panel:
        raise TemporalRankICSelectionError(
            "Temporal panel is empty after cropping to the frozen signal dates"
        )
    manifest = {
        "signal_dates": list(signal_dates),
        "lookback": LOOKBACK,
        "predict": PREDICT,
    }
    rows = build_evaluation_rows(
        manifest, panel, min_names=int(min_names), progress=True,
    )
    return dict(sorted(panel.items())), rows


def write_panel(panel: dict, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = {symbol: panel[symbol] for symbol in sorted(panel)}
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=PICKLE_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return sha256_file(destination)


def pin_selection_digests(manifest_path, *, panel_sha256, rows_sha256) -> dict:
    """Pin panel and rows SHAs without touching the frozen date list.

    ``panel_sha256`` keeps the manifest's current string when that string
    already equals the panel file digest. A different digest means the panel
    bytes changed, and the field is updated to the new digest.
    """
    manifest_path = Path(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    calendar = validate_manifest_calendar(document)
    panel_sha256 = str(panel_sha256)
    rows_sha256 = str(rows_sha256)
    if len(panel_sha256) != 64 or len(rows_sha256) != 64:
        raise TemporalRankICSelectionError(
            "panel and rows digests must be SHA256 hex strings"
        )
    existing_panel = str(document.get("panel_sha256") or "")
    stored_panel = existing_panel if existing_panel == panel_sha256 else panel_sha256
    updated = dict(document)
    updated["panel_sha256"] = stored_panel
    updated["panel_filename"] = PANEL_FILENAME
    updated["rows_sha256"] = rows_sha256
    updated["rows_filename"] = ROWS_FILENAME
    again = validate_manifest_calendar(updated)
    if again["signal_dates"] != calendar["signal_dates"] or again["sha256"] != calendar["sha256"]:
        raise TemporalRankICSelectionError(
            "Refusing to rewrite the frozen signal_dates while pinning selection artifacts"
        )
    if manifest_sha256(again["signal_dates"]) != calendar["sha256"]:
        raise TemporalRankICSelectionError(
            "Calendar SHA drifted while pinning selection artifacts"
        )
    if updated["panel_sha256"] != stored_panel:
        raise TemporalRankICSelectionError(
            "panel_sha256 changed while pinning rows"
        )
    text = json.dumps(updated, ensure_ascii=False, indent=2) + "\n"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, manifest_path)
    return {"panel_sha256": stored_panel, "rows_sha256": rows_sha256}


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict) or not panel:
        raise TemporalRankICSelectionError(f"{path} is not a symbol -> frame panel")
    return panel


def build_from_dataset(
    dataset_root,
    manifest_path,
    output_path,
    *,
    expected_sha=CLEAN_DATA_MANIFEST_SHA256,
    expected_cutoff=CLEAN_TRAIN_CUTOFF,
    min_names=MIN_NAMES_PER_DATE,
) -> dict:
    manifest_path = Path(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    calendar = validate_manifest_calendar(document)
    if calendar["train_time_upper_bound"] != expected_cutoff:
        raise TemporalRankICSelectionError(
            "selection train_time_upper_bound "
            f"{calendar['train_time_upper_bound']} does not match the clean "
            f"split cutoff {expected_cutoff}"
        )
    assert_signal_dates_are_sessions(calendar["signal_dates"])
    data_manifest_path, _data_manifest = find_clean_manifest(
        dataset_root, expected_sha=expected_sha, expected_cutoff=expected_cutoff,
    )
    destination = assert_output_path(output_path)
    root = data_manifest_path.parent
    train_path = root / "processed_datasets" / "train_data.pkl"
    val_path = root / "processed_datasets" / "val_data.pkl"
    metadata_path = root / "asset_metadata.csv"
    for path in (train_path, val_path, metadata_path):
        if not path.is_file():
            raise TemporalRankICSelectionError(f"Clean dataset is missing {path}")
    if destination.resolve() in {train_path.resolve(), val_path.resolve()}:
        raise TemporalRankICSelectionError(
            "Refusing to overwrite train_data.pkl or val_data.pkl"
        )
    metadata = pd.read_csv(metadata_path)
    train_panel = _load_pickle(train_path)
    val_panel = _load_pickle(val_path)
    panel, rows = assemble_panel(
        train_panel,
        val_panel,
        calendar["signal_dates"],
        metadata=metadata,
        min_names=min_names,
    )
    digest = write_panel(panel, destination)
    rows_path = destination.with_name(ROWS_FILENAME)
    rows_digest = write_evaluation_rows(
        rows,
        rows_path,
        panel_sha256=digest,
        calendar_sha256=calendar["sha256"],
    )
    pinned = pin_selection_digests(
        manifest_path, panel_sha256=digest, rows_sha256=rows_digest,
    )
    if pinned["panel_sha256"] != digest or pinned["rows_sha256"] != rows_digest:
        raise TemporalRankICSelectionError(
            "Pinned selection digests do not match the written files "
            f"(panel {pinned['panel_sha256']} vs {digest}, "
            f"rows {pinned['rows_sha256']} vs {rows_digest})"
        )
    return {
        "panel": str(destination),
        "panel_sha256": digest,
        "rows": str(rows_path),
        "rows_sha256": rows_digest,
        "n_symbols": len(panel),
        "n_rows": len(rows),
        "n_dates": len(calendar["signal_dates"]),
        "calendar_sha256": calendar["sha256"],
        "train_time_upper_bound": calendar["train_time_upper_bound"],
        "data_manifest": str(data_manifest_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--min-names", type=int, default=MIN_NAMES_PER_DATE)
    args = parser.parse_args(argv)
    summary = build_from_dataset(
        args.dataset_root,
        args.manifest,
        args.output,
        min_names=args.min_names,
    )
    print(json.dumps(summary, indent=2))
    print(
        "Wrote temporal_rank_ic_panel.pkl and temporal_rank_ic_rows.pkl. "
        "panel_sha256 was updated only if the panel bytes changed. "
        "Commit rows_sha256 with the manifest and upload both pickles in the "
        "panel dataset. This script does not push a Kaggle kernel.",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    main()
