"""Temporal daily D10 Rank IC checkpoint selection (protocol temporal_rank_ic_v1).

Mid-training ``best_model`` uses the mean same-day Spearman IC of a single-path
D10 return on a frozen temporal signal-date manifest (N_PATHS=1 for wall-clock). Validation loss
stays telemetry. Symbol-holdout ``val_data.pkl`` is never a fallback: a missing
manifest, a bad SHA, or a holdout path raises ``TemporalRankICSelectionError``.

Frozen rules: ``docs/VALIDATION_RANK_IC_SELECTION_PROTOCOL_CN.md``.
Calendar: ``finetune/manifests/temporal_rank_ic_v1/selection_window.json``.
Panel bytes: ``finetune/build_temporal_rank_ic_panel.py`` (not stored in git).
Optional precomputed windows live beside that panel as
``temporal_rank_ic_rows.pkl``. Dual-GPU startup hashes the manifest on every
rank, then only rank 0 materializes rows. Other ranks wait on a status file
and do not enter NCCL until that build has finished. Mid-train scoring is the
same shape: only rank 0 runs the temporal forward; peers poll a status
file and must not enter ``broadcast`` until the score finishes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
from datetime import date
from pathlib import Path


PROTOCOL_ID = "temporal_rank_ic_v1"
SELECTION_METRIC = "temporal_daily_d10_rank_ic"
DEFAULT_INTERVAL_SEGMENTS = 10
N_PATHS = 1
PATH_AGG = "median"
TEMPERATURE = 0.65
TOP_P = 0.8
TOP_K = 0
PATH_SEED = 20260906
HORIZON = "D10"
LOOKBACK = 120
PREDICT = 10
WINDOW = LOOKBACK + PREDICT
TIE_EPSILON = 1e-6
MIN_NAMES_PER_DATE = 30
INFERENCE_BATCH = 4
FEATURES = ("open", "high", "low", "close", "volume", "amount")
MANIFEST_RELATIVE = Path(
    "finetune/manifests/temporal_rank_ic_v1/selection_window.json"
)
README_RELATIVE = Path("finetune/manifests/temporal_rank_ic_v1/README.md")
PANEL_FILENAME = "temporal_rank_ic_panel.pkl"
ROWS_FILENAME = "temporal_rank_ic_rows.pkl"
ROWS_FORMAT = "temporal_rank_ic_rows_v1"
ROWS_SHA_KEY = "rows_sha256"
PANEL_DATASET_ENV = "KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET"
ROWS_PATH_ENV = "KRONOS_TEMPORAL_RANK_IC_ROWS"
PREPARE_TIMEOUT_SECONDS = 45 * 60
# Status-file wait while rank 0 scores ~300k single-path windows on one T4.
# With N_PATHS=1 expect ~1h-class Seg1; keep headroom. Override with
# KRONOS_TEMPORAL_RANK_IC_SCORE_TIMEOUT_SECONDS.
SCORE_TIMEOUT_SECONDS = 4 * 60 * 60
_PANEL_SHA = re.compile(r"^[0-9a-f]{64}$")

# Protocol §2.3. Inclusive signal-date ranges. These windows are acceptance
# only and must not enter checkpoint selection.
FORBIDDEN_SIGNAL_WINDOWS = (
    ("package_34d", "2026-08-03", "2026-09-17"),
    ("sealed_8d", "2026-09-18", "2026-09-30"),
    ("section_5_2", "2026-10-09", "2026-10-23"),
)
HOLDOUT_MARKERS = (
    "symbol-holdout",
    "symbol_holdout",
    "temporal-symbol-holdout",
    "temporal_symbol_holdout",
)
HOLDOUT_SOURCE_KEYS = (
    "source",
    "dataset",
    "source_dataset",
    "panel_source",
    "data_path",
    "panel_path",
)
_BROADCAST_EMPTY = -1.0e30


class TemporalRankICSelectionError(RuntimeError):
    """The temporal Rank IC selector cannot run. Do not fall back."""


def temporal_rank_ic_enabled(config) -> bool:
    return str((config or {}).get("checkpoint_selection_protocol") or "") == PROTOCOL_ID


def default_manifest_path(repo_root) -> Path:
    return Path(repo_root) / MANIFEST_RELATIVE


def missing_manifest_message(path) -> str:
    return (
        "Frozen temporal Rank IC manifest is missing at "
        f"{path}. Checkpoint selection requires this date list and its SHA "
        f"({README_RELATIVE.as_posix()}). "
        "Symbol-holdout validation and val loss cannot select best_model "
        "while temporal_rank_ic_v1 is on."
    )


def assert_frozen_manifest_present(repo_root) -> Path:
    """Fail closed before a Kaggle push when the date list is missing or illegal.

    The price panel is not in the repo, so this check stops at the calendar,
    its SHA, and the forbidden windows. Panel bytes are checked at launch.
    """
    path = default_manifest_path(repo_root)
    if not path.is_file():
        raise TemporalRankICSelectionError(missing_manifest_message(path))
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TemporalRankICSelectionError(
            f"Temporal Rank IC manifest is not JSON: {path}"
        ) from exc
    validate_manifest_calendar(document)
    return path


def canonical_date_payload(dates) -> bytes:
    """UTF-8 body whose SHA is stored on the manifest. Dates must already be sorted."""
    rows = [str(item) for item in dates]
    return ("\n".join(rows) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(dates) -> str:
    return sha256_bytes(canonical_date_payload(dates))


def should_run_temporal_rank_ic(segment, total_segments, interval):
    """Same milestones as ``should_run_large_validation`` (first, every N, last)."""
    segment = int(segment)
    total_segments = int(total_segments)
    interval = int(interval)
    if interval <= 0:
        raise ValueError("Rank IC selection interval must be positive")
    return segment == 1 or segment % interval == 0 or segment == total_segments


def temporal_rank_ic_due(
    segment, total_segments, interval, *, first_of_run=False, force_final=False,
):
    """Sparse schedule plus the first and last segment of this invocation.

    Wall-clock and chunk stops set ``force_final`` so the last completed
    segment is scored even when it is not a multiple of N.
    """
    return bool(
        should_run_temporal_rank_ic(segment, total_segments, interval)
        or first_of_run
        or force_final
    )


def rank_ic_is_improvement(candidate, incumbent, epsilon=TIE_EPSILON):
    """Higher daily Rank IC wins. Gaps below ``epsilon`` keep the earlier score."""
    if candidate is None:
        return False
    try:
        candidate = float(candidate)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(candidate):
        return False
    if incumbent is None:
        return True
    try:
        incumbent = float(incumbent)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(incumbent):
        return True
    return (candidate - incumbent) >= float(epsilon)


def checkpoint_should_update(
    *,
    protocol_enabled,
    rank_ic_due,
    rank_ic_mean,
    incumbent_rank_ic,
    val_loss_improved,
):
    """Val loss selects best only when the Rank IC protocol is off."""
    if not protocol_enabled:
        return bool(val_loss_improved)
    if not rank_ic_due:
        return False
    return rank_ic_is_improvement(rank_ic_mean, incumbent_rank_ic)


def rank_ic_threshold_from_saved_metric(metric):
    """Saved temporal IC, or -inf when the file was chosen by another metric."""
    metric = metric or {}
    if str(metric.get("selection_protocol") or "") != PROTOCOL_ID:
        return float("-inf")
    if str(metric.get("selection_metric") or "") != SELECTION_METRIC:
        return float("-inf")
    value = metric.get("daily_rank_ic_mean", metric.get("selection_loss"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if not math.isfinite(value):
        return float("-inf")
    return value


def reconcile_resumed_rank_ic_threshold(state_protocol, state_value, metric):
    """Resume threshold for temporal Rank IC.

    A val-loss ``best_metric.json`` does not become the IC bar. When both the
    state and the export use this protocol, an export that is ahead of
    ``last_state.pt`` wins (the best file is committed first). A state that
    claims a strictly better IC than the export is a broken checkpoint.
    """
    metric_value = rank_ic_threshold_from_saved_metric(metric)
    if str((metric or {}).get("selection_protocol") or "") != PROTOCOL_ID:
        return float("-inf")
    state_is_protocol = str(state_protocol or "") == PROTOCOL_ID
    if (
        not state_is_protocol
        or state_value is None
        or not math.isfinite(float(state_value))
    ):
        return metric_value
    state_value = float(state_value)
    if state_value > metric_value and not math.isclose(
        state_value, metric_value, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise TemporalRankICSelectionError(
            "last_state.pt claims a better temporal Rank IC than best_model: "
            f"{state_value} > {metric_value}"
        )
    if metric_value > state_value and not math.isclose(
        metric_value, state_value, rel_tol=0.0, abs_tol=1e-12,
    ):
        return metric_value
    return state_value


def build_rank_ic_best_metric(
    *,
    daily_rank_ic_mean,
    segment,
    step,
    n_dates,
    manifest_sha256,
    objective_loss,
    forecast_loss,
    history_loss,
    full_sequence_loss,
):
    """Fields required by protocol §4. ``selection_loss`` stores the IC for resume."""
    mean = float(daily_rank_ic_mean)
    return {
        "selection_protocol": PROTOCOL_ID,
        "selection_metric": SELECTION_METRIC,
        "daily_rank_ic_mean": mean,
        # Resume reads selection_loss. Here that number is the Rank IC, not a loss.
        "selection_loss": mean,
        "segment": int(segment),
        "step": int(step),
        "n_dates": int(n_dates),
        "n_paths": N_PATHS,
        "path_agg": PATH_AGG,
        "manifest_sha256": str(manifest_sha256),
        "val_loss_selects_best": False,
        "objective_loss": float(objective_loss),
        "forecast_loss": float(forecast_loss),
        "history_loss": float(history_loss),
        "full_sequence_loss": float(full_sequence_loss),
    }


def _average_ranks(values):
    pairs = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(pairs)
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][1] == pairs[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[pairs[offset][0]] = average
        start = end
    return ranks


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise TemporalRankICSelectionError(
            "Spearman Rank IC needs at least two aligned names"
        )
    rank_left = _average_ranks(left)
    rank_right = _average_ranks(right)
    count = len(rank_left)
    mean_left = sum(rank_left) / count
    mean_right = sum(rank_right) / count
    numerator = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(rank_left, rank_right)
    )
    left_scale = math.sqrt(sum((value - mean_left) ** 2 for value in rank_left))
    right_scale = math.sqrt(sum((value - mean_right) ** 2 for value in rank_right))
    if left_scale == 0.0 or right_scale == 0.0:
        raise TemporalRankICSelectionError(
            "Constant cross-section; daily Rank IC is undefined"
        )
    return numerator / (left_scale * right_scale)


def daily_d10_rank_ic_mean(rows, *, min_names=MIN_NAMES_PER_DATE):
    """Arithmetic mean of per-signal-date Spearman IC. Higher is better."""
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["asof_date"]), []).append(row)
    if not grouped:
        raise TemporalRankICSelectionError("Temporal Rank IC received no rows")
    scores = []
    per_date = {}
    for day in sorted(grouped):
        group = grouped[day]
        if len(group) < int(min_names):
            raise TemporalRankICSelectionError(
                f"{day} has {len(group)} names; temporal Rank IC requires "
                f"{int(min_names)} names on every frozen signal date"
            )
        score = spearman(
            [float(item["predicted"]) for item in group],
            [float(item["actual"]) for item in group],
        )
        if not math.isfinite(score):
            raise TemporalRankICSelectionError(f"Non-finite Rank IC on {day}")
        scores.append(float(score))
        per_date[day] = float(score)
    return {
        "daily_rank_ic_mean": sum(scores) / len(scores),
        "n_dates": len(scores),
        "per_date": per_date,
    }


def median_path_d10_return(path_closes, last_close):
    """D10 return of the path-wise median close. Even N averages the two middle values."""
    closes = [float(value) for value in path_closes]
    if len(closes) != N_PATHS:
        raise TemporalRankICSelectionError(
            f"Expected {N_PATHS} D10 path closes; got {len(closes)}"
        )
    last = float(last_close)
    if not math.isfinite(last) or last == 0.0:
        raise TemporalRankICSelectionError("Invalid last close for D10 return")
    returns = sorted((close / last) - 1.0 for close in closes)
    if any(not math.isfinite(value) for value in returns):
        raise TemporalRankICSelectionError("Non-finite path D10 return")
    mid = len(returns) // 2
    if len(returns) % 2 == 1:
        return float(returns[mid])
    return float(0.5 * (returns[mid - 1] + returns[mid]))


def _parse_iso_date(value, field):
    text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise TemporalRankICSelectionError(
            f"{field} must be YYYY-MM-DD; got {value!r}"
        ) from exc
    if parsed.isoformat() != text:
        raise TemporalRankICSelectionError(
            f"{field} must be canonical YYYY-MM-DD; got {value!r}"
        )
    return text


def _contains_holdout_marker(value) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in HOLDOUT_MARKERS)


def _forbidden_window(signal_date):
    for name, start, end in FORBIDDEN_SIGNAL_WINDOWS:
        if start <= signal_date <= end:
            return name
    return None


def _paths_identical(left, right) -> bool:
    left_path = Path(left).resolve()
    right_path = Path(right).resolve()
    if left_path == right_path:
        return True
    try:
        return left_path.samefile(right_path)
    except OSError:
        return False


def resolve_panel_path(manifest, *, explicit_path="", search_roots=()):
    """Panel path from the environment, or exactly one ``panel_filename`` hit.

    ``val_data.pkl`` is rejected by name so a search cannot land on the
    symbol-holdout validation pickle.
    """
    if str(explicit_path or "").strip():
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise TemporalRankICSelectionError(
                "Temporal Rank IC panel is missing at "
                f"{path}. Symbol-holdout val_data.pkl cannot supply this score."
            )
        return path
    filename = str((manifest or {}).get("panel_filename") or "").strip()
    if not filename or "/" in filename or "\\" in filename or filename == "val_data.pkl":
        raise TemporalRankICSelectionError(
            "Set KRONOS_TEMPORAL_RANK_IC_PANEL to the frozen temporal panel, "
            "or set panel_filename to a base name other than val_data.pkl. "
            "Symbol-holdout validation is not the selection window."
        )
    found = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        found.extend(
            candidate for candidate in root_path.glob(f"**/{filename}")
            if candidate.is_file()
        )
    matches = [
        candidate for candidate in found
        if candidate.name != "val_data.pkl" and not path_mentions_symbol_holdout(candidate)
    ]
    if len(matches) != 1:
        ignored = len(found) - len(matches)
        raise TemporalRankICSelectionError(
            f"Expected one temporal panel named {filename} outside "
            f"symbol-holdout inputs; found {[str(path) for path in matches[:8]]}. "
            f"Ignored {ignored} holdout or val_data.pkl hit(s). "
            "Mount the built panel dataset (KRONOS_TEMPORAL_RANK_IC_PANEL_DATASET) "
            "or set KRONOS_TEMPORAL_RANK_IC_PANEL. Refusing val_data.pkl."
        )
    return matches[0]


def validate_manifest_calendar(document):
    """Lock protocol fields and the date list. Does not open the price panel."""
    if not isinstance(document, dict):
        raise TemporalRankICSelectionError(
            "Temporal Rank IC manifest must be a JSON object"
        )
    if document.get("protocol_id") != PROTOCOL_ID:
        raise TemporalRankICSelectionError(
            f"selection protocol_id must be {PROTOCOL_ID}; "
            f"got {document.get('protocol_id')!r}"
        )
    if document.get("split_kind") != "temporal":
        raise TemporalRankICSelectionError(
            "selection split_kind must be 'temporal'. "
            "Symbol holdout cannot select best_model."
        )
    if document.get("symbol_holdout") or document.get("allow_symbol_holdout"):
        raise TemporalRankICSelectionError(
            "Manifest opts into symbol holdout. temporal_rank_ic_v1 rejects that."
        )
    for key in HOLDOUT_SOURCE_KEYS:
        if key in document and _contains_holdout_marker(document[key]):
            raise TemporalRankICSelectionError(
                f"Manifest field {key} points at a symbol-holdout source. "
                "Temporal selection needs its own date list and panel."
            )
    if document.get("horizon") != HORIZON:
        raise TemporalRankICSelectionError(
            f"selection horizon must be {HORIZON}"
        )
    if int(document.get("n_paths", -1)) != N_PATHS:
        raise TemporalRankICSelectionError(f"n_paths must be {N_PATHS}")
    if document.get("path_aggregation") != PATH_AGG:
        raise TemporalRankICSelectionError(
            f"path_aggregation must be {PATH_AGG}"
        )
    if int(document.get("path_seed", -1)) != PATH_SEED:
        raise TemporalRankICSelectionError(f"path_seed must be {PATH_SEED}")
    if int(document.get("lookback", -1)) != LOOKBACK or int(document.get("predict", -1)) != PREDICT:
        raise TemporalRankICSelectionError(
            f"window must be lookback {LOOKBACK} and predict {PREDICT}"
        )
    if int(document.get("top_k", -1)) != TOP_K:
        raise TemporalRankICSelectionError(f"top_k must be {TOP_K}")
    if not math.isclose(float(document.get("temperature", -1)), TEMPERATURE, rel_tol=0.0, abs_tol=1e-12):
        raise TemporalRankICSelectionError(f"temperature must be {TEMPERATURE}")
    if not math.isclose(float(document.get("top_p", -1)), TOP_P, rel_tol=0.0, abs_tol=1e-12):
        raise TemporalRankICSelectionError(f"top_p must be {TOP_P}")
    filename = str(document.get("panel_filename") or "")
    if filename != PANEL_FILENAME:
        raise TemporalRankICSelectionError(
            f"panel_filename must be {PANEL_FILENAME}. "
            "val_data.pkl is not the temporal selection panel."
        )
    raw_dates = document.get("signal_dates")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise TemporalRankICSelectionError(
            "signal_dates must be a non-empty list of YYYY-MM-DD values"
        )
    signal_dates = [_parse_iso_date(item, "signal_dates") for item in raw_dates]
    if signal_dates != sorted(set(signal_dates)):
        raise TemporalRankICSelectionError(
            "signal_dates must be unique and sorted ascending. "
            "The manifest SHA is the sorted date list; do not reorder at runtime."
        )
    expected_sha = str(document.get("sha256") or "")
    actual_sha = manifest_sha256(signal_dates)
    if expected_sha != actual_sha:
        raise TemporalRankICSelectionError(
            "Temporal manifest SHA does not match signal_dates. "
            f"expected {expected_sha or '<missing>'}, computed {actual_sha}. "
            "Refusing to run with a rewritten calendar."
        )
    for signal_date in signal_dates:
        window = _forbidden_window(signal_date)
        if window:
            raise TemporalRankICSelectionError(
                f"{signal_date} falls in forbidden selection window {window}. "
                "Sealed, §5.2, and the 34-day package are acceptance only."
            )
    train_bound = _parse_iso_date(
        document.get("train_time_upper_bound"), "train_time_upper_bound",
    )
    if not train_bound < signal_dates[0]:
        raise TemporalRankICSelectionError(
            "train_time_upper_bound must be strictly earlier than the first "
            f"signal date ({train_bound} vs {signal_dates[0]})"
        )
    return {
        "signal_dates": signal_dates,
        "sha256": actual_sha,
        "train_time_upper_bound": train_bound,
    }


def require_pinned_panel_sha(document) -> str:
    """Reject a push or launch until the builder has written a real panel SHA.

    The calendar check deliberately allows an empty ``panel_sha256`` so the
    date list can be frozen before the multi-GB panel exists. Push and launch
    call this and stop. Do not fill the field with a placeholder digest.
    """
    if not isinstance(document, dict):
        raise TemporalRankICSelectionError(
            "Temporal Rank IC manifest must be a JSON object"
        )
    filename = str(document.get("panel_filename") or "")
    if filename != PANEL_FILENAME:
        raise TemporalRankICSelectionError(
            f"panel_filename must be {PANEL_FILENAME}. "
            "val_data.pkl is not the temporal selection panel."
        )
    panel_sha = str(document.get("panel_sha256") or "")
    if not _PANEL_SHA.fullmatch(panel_sha):
        raise TemporalRankICSelectionError(
            "panel_sha256 is not a SHA256. Build temporal_rank_ic_panel.pkl with "
            "finetune/build_temporal_rank_ic_panel.py so it can write the digest "
            "into selection_window.json. Do not invent a digest, and do not point "
            "this field at val_data.pkl."
        )
    return panel_sha


def resolve_panel_dataset_slug(value, *, training_dataset) -> str:
    """Kaggle ``owner/slug`` for the built panel. Never the training holdout set."""
    slug = str(value or "").strip()
    training = str(training_dataset or "").strip()
    if not slug or slug.count("/") != 1 or slug.startswith("/") or slug.endswith("/"):
        raise TemporalRankICSelectionError(
            f"Set {PANEL_DATASET_ENV} to the owner/slug of the uploaded "
            f"{PANEL_FILENAME} dataset before push_chunk. "
            "The symbol-holdout training dataset cannot supply this panel."
        )
    owner, name = slug.split("/", 1)
    if not owner or not name or name == "val_data.pkl":
        raise TemporalRankICSelectionError(
            f"{PANEL_DATASET_ENV}={slug!r} is not a panel dataset slug."
        )
    if slug == training or _contains_holdout_marker(slug):
        raise TemporalRankICSelectionError(
            f"{PANEL_DATASET_ENV}={slug!r} points at a symbol-holdout dataset. "
            "Upload temporal_rank_ic_panel.pkl as its own dataset and mount that."
        )
    return slug


def load_temporal_selection_manifest(
    path, *, panel_path, val_data_paths=(), train_signal_end="",
):
    """Load and lock the temporal date list. Raises if anything is off-protocol."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise TemporalRankICSelectionError(missing_manifest_message(manifest_path))
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TemporalRankICSelectionError(
            f"Temporal Rank IC manifest is not JSON: {manifest_path}"
        ) from exc
    calendar = validate_manifest_calendar(document)
    signal_dates = calendar["signal_dates"]
    if str(train_signal_end or "").strip():
        train_end = _parse_iso_date(train_signal_end, "train_signal_end")
        if not train_end < signal_dates[0]:
            raise TemporalRankICSelectionError(
                "Training signal end must be strictly earlier than the temporal "
                f"selection window ({train_end} vs {signal_dates[0]})"
            )
    panel = Path(panel_path)
    if not panel.is_file():
        raise TemporalRankICSelectionError(
            f"Temporal Rank IC panel is missing at {panel}. "
            "Symbol-holdout val_data.pkl cannot supply this score."
        )
    if panel.name == "val_data.pkl" or path_mentions_symbol_holdout(panel):
        raise TemporalRankICSelectionError(
            f"Panel path {panel} is a symbol-holdout validation file. "
            "temporal_rank_ic_v1 requires a separate temporal panel."
        )
    for candidate in val_data_paths or ():
        if candidate and _paths_identical(panel, candidate):
            raise TemporalRankICSelectionError(
                "Temporal Rank IC panel resolves to a training val_data path "
                f"({candidate}). Symbol holdout cannot select best_model."
            )
    panel_sha = str(document.get("panel_sha256") or "")
    actual_panel_sha = sha256_file(panel)
    if panel_sha != actual_panel_sha:
        raise TemporalRankICSelectionError(
            "Temporal panel SHA does not match the manifest. "
            f"expected {panel_sha or '<missing>'}, computed {actual_panel_sha}."
        )
    loaded = dict(document)
    loaded["signal_dates"] = signal_dates
    loaded["train_time_upper_bound"] = calendar["train_time_upper_bound"]
    loaded["sha256"] = calendar["sha256"]
    loaded["panel_sha256"] = actual_panel_sha
    loaded["manifest_path"] = str(manifest_path)
    loaded["panel_path"] = str(panel)
    return loaded


def path_mentions_symbol_holdout(path) -> bool:
    return _contains_holdout_marker(Path(path).as_posix())


def resolve_training_selection_inputs(
    repo_root,
    input_root,
    *,
    explicit_manifest="",
    explicit_panel="",
    val_data_paths=(),
    train_signal_end="",
):
    """Resolve the manifest and panel used by the bigbatch launcher."""
    manifest_path = (
        Path(explicit_manifest).expanduser()
        if str(explicit_manifest or "").strip()
        else default_manifest_path(repo_root)
    )
    if not manifest_path.is_file():
        raise TemporalRankICSelectionError(missing_manifest_message(manifest_path))
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_pinned_panel_sha(document)
    panel_path = resolve_panel_path(
        document,
        explicit_path=explicit_panel or os.environ.get("KRONOS_TEMPORAL_RANK_IC_PANEL", ""),
        search_roots=[input_root] if input_root else [],
    )
    manifest = load_temporal_selection_manifest(
        manifest_path,
        panel_path=panel_path,
        val_data_paths=val_data_paths,
        train_signal_end=train_signal_end,
    )
    return manifest_path, panel_path, manifest


def _frame_dates(frame):
    import pandas as pd

    index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    if index.has_duplicates:
        raise TemporalRankICSelectionError("Temporal panel index has duplicate dates")
    return index


def _positions_by_date(dates):
    """Map ``YYYY-MM-DD`` to a row position once per symbol.

    The old builder scanned ``stamp.date().isoformat()`` for every signal date.
    On the full panel that is a Python walk over every bar, sixty times.
    """
    labels = dates.strftime("%Y-%m-%d").tolist()
    positions = {}
    for position, label in enumerate(labels):
        if label in positions:
            raise TemporalRankICSelectionError(
                "Temporal panel index has duplicate dates"
            )
        positions[label] = position
    return positions


def build_evaluation_rows(
    manifest, panel, *, min_names=MIN_NAMES_PER_DATE, progress=True,
):
    """One normalized window per (signal date, symbol) on the frozen calendar."""
    import numpy as np

    if not isinstance(panel, dict) or not panel:
        raise TemporalRankICSelectionError(
            "Temporal panel must be a non-empty dict of symbol -> frame"
        )
    lookback = int(manifest["lookback"])
    predict = int(manifest["predict"])
    prepared = []
    for symbol in sorted(panel):
        frame = panel[symbol]
        dates = _frame_dates(frame)
        prepared.append({
            "symbol": str(symbol),
            "frame": frame,
            "dates": dates,
            "positions": _positions_by_date(dates),
            "n_bars": len(frame),
            "checked": False,
        })
    signal_dates = list(manifest["signal_dates"])
    rows = []
    n_dates = len(signal_dates)
    if progress:
        print(
            "Temporal Rank IC building rows: "
            f"dates={n_dates} symbols={len(prepared)}",
            flush=True,
        )
    for date_number, signal_text in enumerate(signal_dates, start=1):
        signal_key = str(signal_text)
        day_rows = []
        for item in prepared:
            index = item["positions"].get(signal_key)
            if index is None:
                continue
            if index < lookback - 1 or index + predict >= item["n_bars"]:
                continue
            start = index - (lookback - 1)
            stop = index + predict + 1
            frame = item["frame"]
            window = frame.iloc[start:stop]
            if len(window) != lookback + predict:
                continue
            if not item["checked"]:
                missing = [name for name in FEATURES if name not in frame.columns]
                if missing:
                    raise TemporalRankICSelectionError(
                        f"{item['symbol']} {signal_text} is missing columns {missing}"
                    )
                if (
                    "sector_id" not in frame.columns
                    or "size_percentile" not in frame.columns
                ):
                    raise TemporalRankICSelectionError(
                        f"{item['symbol']} needs sector_id and size_percentile "
                        "columns on the temporal panel. Asset-metadata from the "
                        "symbol-holdout set is not joined in."
                    )
                item["checked"] = True
            values = window.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                continue
            close = values[:, 3]
            actual = float(close[lookback - 1 + predict] / close[lookback - 1] - 1.0)
            size = float(frame["size_percentile"].iloc[index])
            sector = frame["sector_id"].iloc[index]
            if not math.isfinite(actual) or not math.isfinite(size):
                continue
            history = values[:lookback]
            mean = history.mean(axis=0)
            std = history.std(axis=0)
            normalized = np.clip(
                (history - mean) / (std + 1e-5), -5.0, 5.0,
            ).astype(np.float32)
            stamp_index = item["dates"][start:stop]
            stamp = np.column_stack([
                stamp_index.minute,
                stamp_index.hour,
                stamp_index.weekday,
                stamp_index.day,
                stamp_index.month,
            ]).astype(np.float32)
            day_rows.append({
                "symbol": item["symbol"],
                "asof_date": signal_key,
                "actual": actual,
                "x": normalized,
                "stamp": stamp,
                "mean": mean.astype(np.float64),
                "std": std.astype(np.float64),
                "last_close": float(close[lookback - 1]),
                "sector_id": int(sector),
                "size_percentile": float(max(0.0, min(1.0, size))),
            })
        if progress:
            print(
                "Temporal Rank IC rows: "
                f"{date_number}/{n_dates} {signal_key} "
                f"names={len(day_rows)} total={len(rows) + len(day_rows)}",
                flush=True,
            )
        if len(day_rows) < int(min_names):
            raise TemporalRankICSelectionError(
                f"{signal_text} produced {len(day_rows)} finite windows; "
                f"temporal Rank IC requires {int(min_names)} names on every date"
            )
        rows.extend(day_rows)
    rows.sort(key=lambda row: (row["asof_date"], row["symbol"]))
    if progress:
        print(
            f"Temporal Rank IC rows ready: {len(rows)} dates={n_dates}",
            flush=True,
        )
    return rows


def load_price_panel(path):
    import pickle

    with open(path, "rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict) or not panel:
        raise TemporalRankICSelectionError(
            "Temporal panel must be a non-empty dict of symbol -> frame"
        )
    return panel


def pinned_rows_sha(document) -> str:
    """Return a pinned rows digest, or ``""`` when rows are not pinned.

    A present but malformed digest fails closed. An absent field means the
    runtime may build rows from the panel.
    """
    if not isinstance(document, dict) or ROWS_SHA_KEY not in document:
        return ""
    value = str(document.get(ROWS_SHA_KEY) or "").strip()
    if not value:
        return ""
    if not _PANEL_SHA.fullmatch(value):
        raise TemporalRankICSelectionError(
            "rows_sha256 is not a SHA256. Build temporal_rank_ic_rows.pkl with "
            "finetune/build_temporal_rank_ic_panel.py so it can write the digest. "
            "Do not invent one, and do not fall back to val loss."
        )
    return value


def _reject_rows_path(path: Path) -> None:
    if path.name == "val_data.pkl" or path_mentions_symbol_holdout(path):
        raise TemporalRankICSelectionError(
            f"Refusing temporal Rank IC rows at {path}. "
            "val_data.pkl and symbol-holdout paths cannot supply this score."
        )


def resolve_rows_artifact(manifest):
    """Return ``(path, expected_sha, required)``.

    ``required`` means a missing or mismatched file must fail closed. The
    runtime does not rebuild rows in that case. An unpinned sibling named
    ``temporal_rank_ic_rows.pkl`` is also required once it is present: the
    operator put it there to be loaded.
    """
    expected = pinned_rows_sha(manifest)
    explicit = os.environ.get(ROWS_PATH_ENV, "").strip()
    sibling = Path(manifest["panel_path"]).with_name(ROWS_FILENAME)
    if explicit:
        path = Path(explicit).expanduser()
        _reject_rows_path(path)
        return path, expected, True
    if expected:
        _reject_rows_path(sibling)
        return sibling, expected, True
    if sibling.is_file():
        _reject_rows_path(sibling)
        return sibling, "", True
    return None, "", False


def write_evaluation_rows(rows, destination, *, panel_sha256, calendar_sha256) -> str:
    """Atomically write the rows artifact and return its file SHA256."""
    import pickle

    destination = Path(destination)
    if destination.name != ROWS_FILENAME:
        raise TemporalRankICSelectionError(
            f"Refusing to write {destination}. The rows artifact must be named "
            f"{ROWS_FILENAME}."
        )
    _reject_rows_path(destination)
    if not _PANEL_SHA.fullmatch(str(panel_sha256)) or not _PANEL_SHA.fullmatch(
        str(calendar_sha256)
    ):
        raise TemporalRankICSelectionError(
            "Rows artifact requires a SHA256 panel digest and calendar digest"
        )
    payload = {
        "format": ROWS_FORMAT,
        "panel_sha256": str(panel_sha256),
        "calendar_sha256": str(calendar_sha256),
        "n_rows": len(rows),
        "rows": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return sha256_file(destination)


def load_precomputed_rows(path, manifest, *, expected_sha="", min_names=MIN_NAMES_PER_DATE):
    """Load pinned rows. Missing or mismatched bytes raise; they are not rebuilt."""
    import pickle

    path = Path(path)
    _reject_rows_path(path)
    if not path.is_file():
        pinned = f" rows_sha256 is pinned ({expected_sha})." if expected_sha else ""
        raise TemporalRankICSelectionError(
            f"Temporal Rank IC rows are missing at {path}.{pinned} "
            "Refusing to rebuild evaluation rows or select best_model by val loss."
        )
    actual = sha256_file(path)
    if expected_sha and actual != expected_sha:
        raise TemporalRankICSelectionError(
            "Temporal Rank IC rows SHA does not match the manifest. "
            f"expected {expected_sha}, computed {actual}. "
            "Refusing to rebuild evaluation rows from a mismatched artifact."
        )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != ROWS_FORMAT:
        raise TemporalRankICSelectionError(
            f"{path} is not a {ROWS_FORMAT} rows artifact. "
            "Refusing to score it or rebuild from the panel."
        )
    if str(payload.get("panel_sha256") or "") != str(manifest.get("panel_sha256") or ""):
        raise TemporalRankICSelectionError(
            "Precomputed rows were built for a different panel SHA "
            f"(rows {payload.get('panel_sha256')}, "
            f"manifest {manifest.get('panel_sha256')}). "
            "Refusing to score them."
        )
    if str(payload.get("calendar_sha256") or "") != str(manifest.get("sha256") or ""):
        raise TemporalRankICSelectionError(
            "Precomputed rows were built for a different calendar SHA "
            f"(rows {payload.get('calendar_sha256')}, "
            f"manifest {manifest.get('sha256')}). "
            "Refusing to score them."
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TemporalRankICSelectionError(
            "Precomputed temporal rows are empty. "
            "Refusing to select best_model by val loss."
        )
    if int(payload.get("n_rows", -1)) != len(rows):
        raise TemporalRankICSelectionError(
            "Precomputed rows n_rows does not match the payload. "
            "Refusing to score a truncated artifact."
        )
    allowed = {str(day) for day in manifest["signal_dates"]}
    counts = {}
    for row in rows:
        if not isinstance(row, dict) or "asof_date" not in row or "symbol" not in row:
            raise TemporalRankICSelectionError(
                "Precomputed temporal rows have an invalid record"
            )
        day = str(row["asof_date"])
        if day not in allowed:
            raise TemporalRankICSelectionError(
                f"Precomputed rows contain {day}, which is outside the frozen "
                "calendar. Refusing to score them."
            )
        counts[day] = counts.get(day, 0) + 1
    for day in manifest["signal_dates"]:
        found = counts.get(str(day), 0)
        if found < int(min_names):
            raise TemporalRankICSelectionError(
                f"{day} has {found} precomputed rows; temporal Rank IC requires "
                f"{int(min_names)} names on every date. "
                "Refusing to drop the date or fall back to val loss."
            )
    return rows


def materialize_evaluation_rows(
    manifest, *, progress=True, min_names=MIN_NAMES_PER_DATE,
):
    """Load a pinned rows artifact, or build windows from the panel.

    Returns ``(rows, source)`` where source is ``precomputed`` or ``built``.
    """
    path, expected, required = resolve_rows_artifact(manifest)
    if required:
        rows = load_precomputed_rows(
            path, manifest, expected_sha=expected, min_names=min_names,
        )
        print(
            f"Temporal Rank IC loaded precomputed rows: {len(rows)} from {path}",
            flush=True,
        )
        return rows, "precomputed"
    panel = load_price_panel(manifest["panel_path"])
    try:
        rows = build_evaluation_rows(
            manifest, panel, min_names=min_names, progress=progress,
        )
    finally:
        del panel
    return rows, "built"


def prepare_rank_ic_runtime(config, *, load_panel, manifest=None):
    """Validate the frozen window. Rank 0 also materializes the cross-section.

    ``load_panel=False`` is the cheap path: manifest, panel file SHA, and a
    well-formed ``rows_sha256`` if that field is present. It does not unpickle
    the panel or build rows. ``load_panel=True`` loads a pinned rows artifact
    when one is expected, otherwise builds rows and drops the panel.
    """
    if not temporal_rank_ic_enabled(config):
        return None
    if manifest is None:
        manifest = load_temporal_selection_manifest(
            config["temporal_rank_ic_manifest_path"],
            panel_path=config["temporal_rank_ic_panel_path"],
            val_data_paths=config.get("val_data_paths") or (),
            train_signal_end=str(config.get("train_signal_end") or ""),
        )
    pinned_rows_sha(manifest)
    rows = None
    source = ""
    if load_panel:
        rows, source = materialize_evaluation_rows(manifest)
    return {
        "manifest": manifest,
        "panel": None,
        "rows": rows,
        "rows_source": source,
    }


def format_prepare_error(exc) -> str:
    """One log line. The type and message stay inside the wrapper RuntimeError."""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) > 3500:
        text = text[:3500] + "..."
    return text


def rank_ic_prepare_run_id() -> str:
    """Identity shared by every worker of this launch, and by no previous one."""
    run = os.environ.get("TORCHELASTIC_RUN_ID", "").strip()
    restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0").strip() or "0"
    if run:
        return f"{run}-r{restart}"
    explicit = os.environ.get("KRONOS_TEMPORAL_RANK_IC_PREPARE_ID", "").strip()
    if explicit:
        return explicit
    parts = [
        os.environ.get("MASTER_ADDR", "").strip(),
        os.environ.get("MASTER_PORT", "").strip(),
        os.environ.get("WORLD_SIZE", "").strip(),
        restart,
    ]
    if any(parts[:3]):
        return "-".join(parts)
    return "local"


def prepare_timeout_seconds() -> float:
    raw = os.environ.get(
        "KRONOS_TEMPORAL_RANK_IC_PREPARE_TIMEOUT_SECONDS", ""
    ).strip()
    if not raw:
        return float(PREPARE_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except ValueError as exc:
        raise TemporalRankICSelectionError(
            "KRONOS_TEMPORAL_RANK_IC_PREPARE_TIMEOUT_SECONDS must be a number "
            "of seconds"
        ) from exc
    if value < 60:
        raise TemporalRankICSelectionError(
            "KRONOS_TEMPORAL_RANK_IC_PREPARE_TIMEOUT_SECONDS must be at least 60"
        )
    return value


def score_timeout_seconds() -> float:
    """How long peers may poll while rank 0 scores ~300k windows.

    Default is 16 hours so a full single-GPU pass over ~300k windows
    cannot trip the peer wait. Override with
    ``KRONOS_TEMPORAL_RANK_IC_SCORE_TIMEOUT_SECONDS``.
    """
    raw = os.environ.get(
        "KRONOS_TEMPORAL_RANK_IC_SCORE_TIMEOUT_SECONDS", ""
    ).strip()
    if not raw:
        return float(SCORE_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except ValueError as exc:
        raise TemporalRankICSelectionError(
            "KRONOS_TEMPORAL_RANK_IC_SCORE_TIMEOUT_SECONDS must be a number "
            "of seconds"
        ) from exc
    if value < 60:
        raise TemporalRankICSelectionError(
            "KRONOS_TEMPORAL_RANK_IC_SCORE_TIMEOUT_SECONDS must be at least 60"
        )
    return value


def _safe_run_id(run_id) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "local"))
    return text[:96] or "local"


def prepare_status_path(directory, run_id, rank, kind) -> Path:
    safe = _safe_run_id(run_id)
    return (
        Path(directory)
        / f"temporal_rank_ic_prepare_{safe}_rank{int(rank)}_{kind}.json"
    )


def _atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    os.replace(temporary, path)


def wait_for_status_file(
    path,
    *,
    timeout,
    role,
    poll_seconds=2.0,
    heartbeat_seconds=30.0,
    sleep=time.sleep,
    clock=time.monotonic,
    phase="prepare",
):
    """Block in Python until ``path`` is ready. This is not a collective."""
    label = str(phase or "prepare")
    deadline = clock() + float(timeout)
    started = clock()
    next_heartbeat = started
    while True:
        if Path(path).is_file():
            try:
                document = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                document = None
            if isinstance(document, dict) and document.get("state") == "ready":
                return document
        current = clock()
        if current >= deadline:
            raise TemporalRankICSelectionError(
                f"Timed out waiting for Temporal Rank IC {label} "
                f"({role}) after {float(timeout):.0f}s at {path}. "
                "This wait does not enter NCCL, broadcast, or all_reduce. "
                "Refusing to select best_model by val loss."
            )
        if current >= next_heartbeat:
            print(
                f"Temporal Rank IC {label}: "
                f"{role} elapsed={current - started:.0f}s "
                f"timeout={float(timeout):.0f}s collective=none "
                f"path={Path(path).name}",
                flush=True,
            )
            next_heartbeat = current + float(heartbeat_seconds)
        remaining = deadline - clock()
        sleep(min(float(poll_seconds), max(0.0, remaining)))


def _raise_prepare_failure(detail, exc=None):
    text = str(detail or "").strip() or "unknown prepare failure"
    error = RuntimeError(
        "Temporal Rank IC manifest or panel failed validation: "
        f"{text}. Refusing to select best_model by val loss or symbol holdout."
    )
    if exc is None:
        raise error
    raise error from exc


def coordinate_rank_ic_prepare(
    *,
    rank,
    world_size,
    directory,
    run_id,
    cheap_check,
    materialize,
    exchange_errors,
    timeout=PREPARE_TIMEOUT_SECONDS,
    poll_seconds=2.0,
    heartbeat_seconds=30.0,
):
    """Cheap checks on every rank, then rank-0 row materialization.

    Non-zero ranks wait on a status file while rank 0 builds. They must not
    call ``exchange_errors`` (the only collective) until that file exists.
    ``exchange_errors`` then shares the underlying exception text so every
    worker raises the same fail-closed ``RuntimeError``.
    """
    rank = int(rank)
    world_size = int(world_size)
    local_error = ""
    runtime = None
    try:
        runtime = cheap_check()
    except Exception as exc:
        local_error = format_prepare_error(exc)
        print(
            "Temporal Rank IC prepare failed during manifest check: "
            f"{local_error}",
            flush=True,
        )
        import traceback
        traceback.print_exc()

    if world_size > 1:
        _atomic_write_json(
            prepare_status_path(directory, run_id, rank, "cheap"),
            {
                "state": "ready",
                "ok": not local_error,
                "error": local_error,
                "rank": rank,
                "run_id": str(run_id),
            },
        )

    if rank == 0 and not local_error:
        print(
            "Temporal Rank IC prepare: rank0 materializing rows; "
            "other ranks wait outside NCCL",
            flush=True,
        )
        try:
            runtime = materialize(runtime)
            rows = runtime.get("rows") if isinstance(runtime, dict) else None
            if not rows:
                raise TemporalRankICSelectionError(
                    "Temporal Rank IC produced no evaluation rows"
                )
        except Exception as exc:
            local_error = format_prepare_error(exc)
            print(
                "Temporal Rank IC prepare failed during row build: "
                f"{local_error}",
                flush=True,
            )
            import traceback
            traceback.print_exc()
            runtime = None

    if world_size > 1:
        try:
            if rank == 0:
                # Peers write their cheap file before waiting, so this poll
                # finishes as soon as their manifest check does. It is still
                # outside NCCL: rank 0 must not publish "done" and enter the
                # collective while another rank is hashing the panel.
                for peer in range(1, world_size):
                    wait_for_status_file(
                        prepare_status_path(directory, run_id, peer, "cheap"),
                        timeout=timeout,
                        role=f"rank0 waiting for rank {peer} manifest check",
                        poll_seconds=poll_seconds,
                        heartbeat_seconds=heartbeat_seconds,
                    )
                _atomic_write_json(
                    prepare_status_path(directory, run_id, 0, "done"),
                    {
                        "state": "ready",
                        "ok": not local_error,
                        "error": local_error,
                        "n_rows": (
                            len(runtime["rows"])
                            if runtime and runtime.get("rows") else 0
                        ),
                        "rows_source": (
                            str(runtime.get("rows_source") or "") if runtime else ""
                        ),
                        "run_id": str(run_id),
                    },
                )
            else:
                wait_for_status_file(
                    prepare_status_path(directory, run_id, 0, "done"),
                    timeout=timeout,
                    role=f"rank {rank} waiting for rank0 row build",
                    poll_seconds=poll_seconds,
                    heartbeat_seconds=heartbeat_seconds,
                )
        except Exception as exc:
            text = format_prepare_error(exc)
            print(
                f"Temporal Rank IC prepare wait failed: {text}",
                flush=True,
            )
            _raise_prepare_failure(text, exc)
        try:
            errors = exchange_errors(local_error)
        except Exception as exc:
            text = format_prepare_error(exc)
            print(
                f"Temporal Rank IC prepare collective failed: {text}",
                flush=True,
            )
            _raise_prepare_failure(text, exc)
    else:
        errors = [local_error] if local_error else []

    problems = [item for item in (errors or []) if item]
    if local_error and local_error not in problems:
        problems.append(local_error)
    if problems:
        _raise_prepare_failure("; ".join(problems))
    if rank == 0 and (not isinstance(runtime, dict) or not runtime.get("rows")):
        _raise_prepare_failure("rank0 finished without evaluation rows")
    return runtime


@contextlib.contextmanager
def isolated_path_seed():
    """Seed path sampling, then put the training RNGs back.

    ``torch.multinomial`` reads the global generator. Seeding it in the middle
    of a run must not stick, or rank 0's next segment diverges from the others.
    """
    import random

    import numpy as np
    import torch

    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(PATH_SEED)
        np.random.seed(PATH_SEED)
        torch.manual_seed(PATH_SEED)
        if saved_cuda is not None:
            torch.cuda.manual_seed_all(PATH_SEED)
        yield
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)


def _predict_temporal_d10_batch(model, tokenizer, rows, device, batch_size, manifest):
    """One seeded pass. Caller owns eval mode and the path-seed context."""
    import numpy as np
    import torch
    from model.kronos import auto_regressive_inference

    predictions = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        x = torch.as_tensor(np.stack([row["x"] for row in batch]), device=device)
        stamp = torch.as_tensor(np.stack([row["stamp"] for row in batch]), device=device)
        sector = torch.as_tensor(
            [row["sector_id"] for row in batch], dtype=torch.long, device=device,
        )
        size = torch.as_tensor(
            [row["size_percentile"] for row in batch], dtype=torch.float32, device=device,
        )
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda" else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            forecasts = auto_regressive_inference(
                tokenizer, model, x, stamp[:, :LOOKBACK], stamp[:, LOOKBACK:WINDOW],
                max_context=512, pred_len=PREDICT, clip=5,
                T=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
                sample_count=N_PATHS, verbose=False,
                sector_id=sector, size_percentile=size, return_samples=True,
            )
        for index, row in enumerate(batch):
            predicted_close = (
                forecasts[index, :, -PREDICT:, 3].astype(np.float64)
                * (float(row["std"][3]) + 1e-5)
                + float(row["mean"][3])
            )
            predicted = median_path_d10_return(predicted_close[:, -1], row["last_close"])
            predictions.append({
                "asof_date": row["asof_date"],
                "symbol": row["symbol"],
                "predicted": predicted,
                "actual": float(row["actual"]),
            })
        done = min(offset + batch_size, len(rows))
        if offset == 0 or done == len(rows) or (offset // batch_size) % 20 == 0:
            print(
                f"Temporal Rank IC windows {done}/{len(rows)} "
                f"segment_dates={len(manifest['signal_dates'])}",
                flush=True,
            )
    return predictions


def evaluate_temporal_d10_rank_ic(model, tokenizer, runtime, device):
    """Single-path D10 Rank IC on the frozen temporal rows.

    The path seed applies only inside this call. Training RNGs and train/eval
    mode are restored before the next segment.
    """
    manifest = runtime["manifest"]
    rows = runtime.get("rows")
    if not rows:
        if runtime.get("panel") is None:
            raise TemporalRankICSelectionError(
                "Temporal Rank IC rows are not loaded on this rank"
            )
        rows = build_evaluation_rows(manifest, runtime["panel"])
    was_training = bool(model.training)
    model.eval()
    try:
        with isolated_path_seed():
            predictions = _predict_temporal_d10_batch(
                model, tokenizer, rows, device, INFERENCE_BATCH, manifest,
            )
    finally:
        if was_training:
            model.train()
    summary = daily_d10_rank_ic_mean(predictions)
    summary.update({
        "n_paths": N_PATHS,
        "path_agg": PATH_AGG,
        "manifest_sha256": manifest["sha256"],
        "samples": len(predictions),
        "selection_protocol": PROTOCOL_ID,
    })
    return summary



def _raise_score_failure(detail, exc=None):
    text = str(detail or "").strip() or "unknown score failure"
    error = RuntimeError(
        "Temporal Rank IC failed: "
        f"{text}. Refusing to keep a val-loss best_model for this segment."
    )
    if exc is None:
        raise error
    raise error from exc


def coordinate_rank_ic_score(
    *,
    rank,
    world_size,
    directory,
    run_id,
    segment,
    score_fn,
    timeout=SCORE_TIMEOUT_SECONDS,
    poll_seconds=2.0,
    heartbeat_seconds=30.0,
):
    """Rank 0 runs the long temporal forward; peers wait outside NCCL.

    Dual-GPU e01c died because rank 1 entered ``broadcast`` for the score
    float while rank 0 was still scanning ~300k windows under the default
    600s watchdog. Peers must poll a status file until rank 0 publishes the
    measured Rank IC (or an error). No collective runs during the forward.
    The metric is unchanged: rank 0 still owns the single-path path seed.
    """
    rank = int(rank)
    world_size = int(world_size)
    segment = int(segment)
    if world_size <= 1:
        try:
            summary = score_fn()
        except Exception as exc:
            text = format_prepare_error(exc)
            print(
                f"Temporal Rank IC score failed: {text}",
                flush=True,
            )
            import traceback
            traceback.print_exc()
            _raise_score_failure(text, exc)
        if not isinstance(summary, dict) or "daily_rank_ic_mean" not in summary:
            _raise_score_failure("score_fn returned no daily_rank_ic_mean")
        measured = unpack_rank_ic_broadcast(summary.get("daily_rank_ic_mean"))
        if measured is None:
            _raise_score_failure("score_fn returned a non-finite Rank IC")
        return summary

    done_path = prepare_status_path(
        directory, run_id, 0, f"score_seg{segment}_done",
    )
    local_error = ""
    summary = None
    if rank == 0:
        print(
            "Temporal Rank IC score: rank0 evaluating; "
            "other ranks wait outside NCCL "
            f"(segment={segment}, timeout={float(timeout):.0f}s)",
            flush=True,
        )
        try:
            summary = score_fn()
            if not isinstance(summary, dict) or "daily_rank_ic_mean" not in summary:
                raise TemporalRankICSelectionError(
                    "Temporal Rank IC score returned no daily_rank_ic_mean"
                )
            measured = unpack_rank_ic_broadcast(summary.get("daily_rank_ic_mean"))
            if measured is None:
                raise TemporalRankICSelectionError(
                    "Temporal Rank IC score returned a non-finite Rank IC"
                )
        except Exception as exc:
            local_error = format_prepare_error(exc)
            print(
                f"Temporal Rank IC score failed during forward: {local_error}",
                flush=True,
            )
            import traceback
            traceback.print_exc()
            summary = None
        payload = {
            "state": "ready",
            "ok": not local_error and summary is not None,
            "error": local_error,
            "segment": segment,
            "run_id": str(run_id),
        }
        if summary is not None and not local_error:
            payload.update({
                "daily_rank_ic_mean": float(summary["daily_rank_ic_mean"]),
                "n_dates": int(summary.get("n_dates") or 0),
                "n_paths": int(summary.get("n_paths") or N_PATHS),
                "path_agg": str(summary.get("path_agg") or PATH_AGG),
                "manifest_sha256": str(summary.get("manifest_sha256") or ""),
                "samples": int(summary.get("samples") or 0),
                "selection_protocol": str(
                    summary.get("selection_protocol") or PROTOCOL_ID
                ),
            })
        _atomic_write_json(done_path, payload)
    else:
        try:
            document = wait_for_status_file(
                done_path,
                timeout=timeout,
                role=f"rank {rank} waiting for rank0 score segment={segment}",
                poll_seconds=poll_seconds,
                heartbeat_seconds=heartbeat_seconds,
                phase="score",
            )
        except Exception as exc:
            text = format_prepare_error(exc)
            print(f"Temporal Rank IC score wait failed: {text}", flush=True)
            _raise_score_failure(text, exc)
        if not document.get("ok"):
            detail = str(document.get("error") or "").strip() or (
                "rank0 score failed without an error payload"
            )
            _raise_score_failure(detail)
        measured = unpack_rank_ic_broadcast(document.get("daily_rank_ic_mean"))
        if measured is None:
            _raise_score_failure(
                "rank0 score status had no finite daily_rank_ic_mean"
            )
        summary = {
            "daily_rank_ic_mean": float(measured),
            "n_dates": int(document.get("n_dates") or 0),
            "n_paths": int(document.get("n_paths") or N_PATHS),
            "path_agg": str(document.get("path_agg") or PATH_AGG),
            "manifest_sha256": str(document.get("manifest_sha256") or ""),
            "samples": int(document.get("samples") or 0),
            "selection_protocol": str(
                document.get("selection_protocol") or PROTOCOL_ID
            ),
        }

    if rank == 0:
        if local_error or summary is None:
            _raise_score_failure(local_error or "rank0 finished without a score")
        measured = unpack_rank_ic_broadcast(summary.get("daily_rank_ic_mean"))
        if measured is None:
            _raise_score_failure("rank0 score lost its finite Rank IC")
    return summary



def pack_rank_ic_broadcast(value, *, is_source):
    if is_source and value is not None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return _BROADCAST_EMPTY
        if math.isfinite(number):
            return number
    return _BROADCAST_EMPTY


def unpack_rank_ic_broadcast(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= -1.0e29:
        return None
    return number
