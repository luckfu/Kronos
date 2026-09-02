import hashlib
import json
import pickle
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
try:
    from .config import Config
    from .asset_metadata import AssetMetadata
except ImportError:
    from config import Config
    from asset_metadata import AssetMetadata


def build_beta_v21_labels(raw_window, lookback_window, asof_date):
    """Build causal next-open return and path-event labels from one raw window."""
    raw_window = np.asarray(raw_window, dtype=np.float64)
    lookback_window = int(lookback_window)
    horizons = np.asarray([1, 3, 5, 10], dtype=np.int64)
    future = raw_window[lookback_window:lookback_window + 10]
    if future.shape[0] != 10:
        raise ValueError("Beta v2.1 labels require ten future trading days")

    entry_price = max(float(future[0, 0]), 1e-8)
    horizon_closes = np.maximum(future[horizons - 1, 3], 1e-8)
    raw_returns = np.log(horizon_closes / entry_price)
    past_closes = np.maximum(raw_window[:lookback_window, 3], 1e-8)
    daily_returns = np.diff(np.log(past_closes))
    sigma20 = max(float(np.std(daily_returns[-20:])), 0.005)
    return_scales = sigma20 * np.sqrt(horizons.astype(np.float64))
    return_targets = np.clip(raw_returns / return_scales, -3.0, 3.0)

    barrier_target = 2
    barrier_valid = True
    for high, low in zip(future[:, 1], future[:, 2]):
        hit_take_profit = float(high) / entry_price - 1.0 >= 0.05
        hit_stop_loss = float(low) / entry_price - 1.0 <= -0.03
        if hit_take_profit and hit_stop_loss:
            barrier_target = 1
            barrier_valid = False
            break
        if hit_take_profit:
            barrier_target = 0
            break
        if hit_stop_loss:
            barrier_target = 1
            break

    if barrier_target == 0:
        utility = 0.046
    elif barrier_target == 1:
        utility = -0.034
    else:
        utility = float(horizon_closes[-1] / entry_price - 1.0 - 0.004)

    return {
        'return_targets': torch.tensor(return_targets, dtype=torch.float32),
        'raw_returns': torch.tensor(raw_returns, dtype=torch.float32),
        'return_scales': torch.tensor(return_scales, dtype=torch.float32),
        'barrier_target': torch.tensor(barrier_target, dtype=torch.long),
        'barrier_valid': torch.tensor(barrier_valid, dtype=torch.bool),
        'utility': torch.tensor(utility, dtype=torch.float32),
        'date_id': torch.tensor(
            int(np.datetime64(asof_date, 'D').astype(np.int64)), dtype=torch.long
        ),
    }


def build_balanced_coverage_order(bucket_ids, segment_size, seed):
    """Interleave shuffled buckets per segment while using every index once."""
    bucket_ids = np.asarray(bucket_ids, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    unknown_positions = np.flatnonzero(bucket_ids == 255)
    rng.shuffle(unknown_positions)
    known_mask = bucket_ids != 255
    known_positions = np.flatnonzero(known_mask)
    known_bucket_ids = bucket_ids[known_mask]
    groups = {}
    for bucket in np.unique(known_bucket_ids):
        positions = known_positions[np.flatnonzero(known_bucket_ids == bucket)]
        rng.shuffle(positions)
        groups[int(bucket)] = positions
    cursors = {bucket: 0 for bucket in groups}
    order = np.empty(len(bucket_ids), dtype=np.int64)
    output_start = 0
    known_total = len(known_positions)
    while output_start < known_total:
        active = [
            bucket for bucket, positions in groups.items()
            if cursors[bucket] < len(positions)
        ]
        target = min(int(segment_size), known_total - output_start)
        allocations = {bucket: 0 for bucket in active}
        remaining = target
        while remaining > 0:
            available = [
                bucket for bucket in active
                if cursors[bucket] + allocations[bucket] < len(groups[bucket])
            ]
            if not available:
                break
            share = max(1, remaining // len(available))
            progressed = 0
            for bucket in available:
                capacity = len(groups[bucket]) - cursors[bucket] - allocations[bucket]
                take = min(share, capacity, remaining)
                allocations[bucket] += take
                remaining -= take
                progressed += take
                if remaining == 0:
                    break
            if progressed == 0:
                raise RuntimeError('Unable to allocate a balanced coverage segment')
        segment_parts = []
        for bucket in active:
            take = allocations[bucket]
            if take <= 0:
                continue
            start = cursors[bucket]
            end = start + take
            segment_parts.append(groups[bucket][start:end])
            cursors[bucket] = end
        segment = np.concatenate(segment_parts)
        rng.shuffle(segment)
        order[output_start:output_start + len(segment)] = segment
        output_start += len(segment)
    order[output_start:] = unknown_positions
    return order


def load_merged_panels(paths):
    """Load one or more panel pickles and join each symbol chronologically."""
    merged = {}
    for path in paths:
        with open(path, 'rb') as handle:
            panel = pickle.load(handle)
        for symbol, frame in panel.items():
            if symbol not in merged:
                merged[symbol] = frame
                continue
            joined = pd.concat([merged[symbol], frame]).sort_index()
            merged[symbol] = joined[~joined.index.duplicated(keep='last')]
    return merged


_SHA256_CACHE = {}


def sha256_file(path):
    path = Path(path).resolve()
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    if cache_key in _SHA256_CACHE:
        return _SHA256_CACHE[cache_key]
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    result = digest.hexdigest()
    _SHA256_CACHE[cache_key] = result
    return result


def load_fixed_validation_artifacts(config):
    """Load signed fixed-validation metadata and its ordered sample records."""
    manifest_path = Path(config.fixed_validation_manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f'Fixed validation manifest does not exist: {manifest_path}')
    actual_manifest_sha = sha256_file(manifest_path)
    expected_manifest_sha = str(config.fixed_validation_manifest_sha256)
    if not expected_manifest_sha:
        raise ValueError('Fixed validation requires KRONOS_FIXED_VALIDATION_MANIFEST_SHA256')
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            'Fixed validation manifest SHA mismatch: '
            f'{actual_manifest_sha} != {expected_manifest_sha}'
        )

    manifest = json.loads(manifest_path.read_text())
    source = manifest.get('source', {})
    selection = manifest.get('selection', {})
    if source.get('data_manifest_sha256') != config.dataset_manifest_sha256:
        raise ValueError('Fixed validation source data manifest SHA does not match training data')
    if int(source.get('lookback', -1)) != int(config.lookback_window):
        raise ValueError('Fixed validation lookback does not match training config')
    if int(source.get('predict', -1)) != int(config.predict_window):
        raise ValueError('Fixed validation horizon does not match training config')
    samples_path = manifest_path.parent / str(selection.get('samples_file', ''))
    if not samples_path.is_file():
        raise ValueError(f'Fixed validation sample file does not exist: {samples_path}')
    if sha256_file(samples_path) != selection.get('samples_file_sha256'):
        raise ValueError('Fixed validation sample file SHA mismatch')
    records = [
        json.loads(line)
        for line in samples_path.read_text().splitlines()
        if line.strip()
    ]
    quick_count = int(selection.get('quick_samples', -1))
    large_count = int(selection.get('large_samples', -1))
    if quick_count != int(config.validation_quick_samples):
        raise ValueError('Fixed validation quick sample count does not match config')
    if large_count != int(config.validation_large_samples) or len(records) != large_count:
        raise ValueError('Fixed validation large sample count does not match config')
    quick_flags = [bool(record.get('quick', False)) for record in records]
    if quick_flags != [True] * quick_count + [False] * (large_count - quick_count):
        raise ValueError('Fixed validation quick samples must be the ordered large-set prefix')
    return manifest, records, actual_manifest_sha


def load_fixed_validation_selection(config, indices, timestamps_by_symbol):
    """Validate and map a signed fixed-validation manifest to dataset positions."""
    manifest, records, actual_manifest_sha = load_fixed_validation_artifacts(config)
    if len(config.val_data_paths) != 1:
        raise ValueError('Fixed validation currently requires exactly one validation panel')
    actual_val_sha = sha256_file(config.val_data_paths[0])
    if actual_val_sha != manifest['source'].get('val_data_sha256'):
        raise ValueError('Fixed validation panel SHA does not match its manifest')

    position_by_identity = {
        (str(symbol), int(start)): position
        for position, (symbol, start) in enumerate(indices)
    }
    mapped = []
    identities = set()
    for record in records:
        identity = (str(record['symbol']), int(record['start_index']))
        if identity in identities:
            raise ValueError(f'Duplicate fixed validation identity: {identity}')
        identities.add(identity)
        if identity not in position_by_identity:
            raise ValueError(f'Fixed validation sample is absent from the panel: {identity}')
        timestamps = timestamps_by_symbol[identity[0]]
        asof_position = identity[1] + int(config.lookback_window) - 1
        target_position = asof_position + int(config.predict_window)
        asof_date = str(pd.Timestamp(timestamps.iloc[asof_position]).date())
        target_date = str(pd.Timestamp(timestamps.iloc[target_position]).date())
        if asof_date != record.get('asof_date') or target_date != record.get('target_date'):
            raise ValueError(f'Fixed validation dates do not match the panel: {identity}')
        mapped.append(position_by_identity[identity])
    periods = list(manifest.get('selection', {}).get('periods', []))
    period_codes = {name: code for code, name in enumerate(periods)}
    record_period_codes = []
    for record in records:
        period = str(record.get('period', ''))
        if period not in period_codes:
            raise ValueError(
                f'Fixed validation record has unknown period {period!r}; '
                f'expected one of {periods}'
            )
        record_period_codes.append(period_codes[period])
    return (
        np.asarray(mapped, dtype=np.int64), manifest, actual_manifest_sha,
        np.asarray(record_period_codes, dtype=np.int16), period_codes,
    )


def balanced_stratum_quotas(strata, target):
    """Allocate a replay target evenly across available year/size strata."""
    strata = np.asarray(strata, dtype=np.int32)
    target = min(max(0, int(target)), len(strata))
    unique, counts = np.unique(strata, return_counts=True)
    quotas = np.zeros(len(unique), dtype=np.int64)
    remaining = target
    while remaining > 0:
        available = np.flatnonzero(quotas < counts)
        if len(available) == 0:
            break
        share = max(1, remaining // len(available))
        consumed = 0
        for position in available:
            take = min(
                share,
                int(counts[position] - quotas[position]),
                remaining,
            )
            quotas[position] += take
            remaining -= take
            consumed += take
            if remaining == 0:
                break
        if not consumed:
            break
    return unique, quotas


def select_stratified_replay(strata, target, seed):
    """Select deterministic replay positions without replacement."""
    strata = np.asarray(strata, dtype=np.int32)
    unique, quotas = balanced_stratum_quotas(strata, target)
    if not len(strata) or not int(quotas.sum()):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(strata, kind='stable')
    sorted_strata = strata[order]
    values, starts, counts = np.unique(
        sorted_strata, return_index=True, return_counts=True
    )
    quota_by_value = dict(zip(unique.tolist(), quotas.tolist()))
    rng = np.random.default_rng(seed)
    selected = []
    for value, start, count in zip(values, starts, counts):
        quota = quota_by_value.get(int(value), 0)
        if quota <= 0:
            continue
        candidates = order[start:start + count]
        selected.append(rng.choice(candidates, size=quota, replace=False))
    result = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    rng.shuffle(result)
    return result


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and exposes deterministic, non-repeating coverage segments.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_paths = self.config.train_data_paths
            self.configured_samples = self.config.n_train_iter
        else:
            self.data_paths = self.config.val_data_paths
            self.configured_samples = self.config.n_val_iter

        self.data = load_merged_panels(self.data_paths)

        fixed_manifest_path = str(
            getattr(self.config, 'fixed_validation_manifest_path', '')
        ).strip()
        excluded_validation_dates = {}
        expected_training_exclusions = 0
        if (
            data_type == 'train'
            and fixed_manifest_path
            and bool(self.config.exclude_fixed_validation_from_training)
        ):
            fixed_manifest, fixed_records, _ = load_fixed_validation_artifacts(
                self.config
            )
            for record in fixed_records:
                excluded_validation_dates.setdefault(
                    str(record['symbol']), set()
                ).add(np.datetime64(record['asof_date'], 'D'))
            isolation = fixed_manifest.get('training_isolation', {})
            expected_training_exclusions = int(
                isolation.get('training_candidate_overlap_samples', -1)
            )
            absent_training_samples = int(
                isolation.get('not_in_training_candidate_samples', -1)
            )
            if (
                expected_training_exclusions < 0
                or absent_training_samples < 0
                or expected_training_exclusions + absent_training_samples
                != len(fixed_records)
                or not isolation.get('all_training_overlaps_must_be_excluded')
            ):
                raise ValueError('Fixed validation training-isolation audit is invalid')

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list
        self.use_context_features = bool(getattr(self.config, 'use_context_features', False))
        self.use_sector_features = bool(getattr(self.config, 'use_sector_features', True))
        self.use_size_features = bool(getattr(self.config, 'use_size_features', True))
        self.use_size_percentile = bool(getattr(self.config, 'use_size_percentile', False))
        self.use_beta_v21_auxiliary = bool(
            getattr(self.config, 'use_beta_v21_auxiliary', False)
        )
        self.has_inline_size = self.use_size_features and any('size_bucket' in frame.columns for frame in self.data.values())
        self.has_inline_percentile = self.use_size_percentile and any(
            'size_percentile' in frame.columns for frame in self.data.values()
        )
        metadata_path = getattr(self.config, 'asset_metadata_path', '')
        if self.has_inline_size and (self.has_inline_percentile or not self.use_size_percentile) and not self.use_sector_features:
            metadata_path = ''
        self.asset_metadata = AssetMetadata(
            metadata_path,
            getattr(self.config, 'num_sectors', 0),
            getattr(self.config, 'num_size_buckets', 0),
        )
        self.timestamps_by_symbol = {}
        self.size_by_symbol = {}
        self.size_percentile_by_symbol = {}

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        bucket_ids = bytearray()
        replay_symbols = []
        replay_starts = []
        replay_buckets = []
        replay_strata = []
        replay_dates = []
        signal_date_ids = []
        excluded_fixed_validation_samples = 0
        signal_dates_seen = set()
        replay_ratio = (
            float(self.config.history_replay_ratio) if data_type == 'train' else 0.0
        )
        if not 0.0 <= replay_ratio < 1.0:
            raise ValueError('KRONOS_HISTORY_REPLAY_RATIO must be in [0, 1)')
        signal_start = getattr(self.config, f'{data_type}_signal_start') or None
        signal_end = getattr(self.config, f'{data_type}_signal_end') or None
        signal_start = np.datetime64(signal_start, 'D') if signal_start else None
        signal_end = np.datetime64(signal_end, 'D') if signal_end else None
        replay_start = self.config.replay_signal_start or None
        replay_end = self.config.replay_signal_end or None
        replay_start = np.datetime64(replay_start, 'D') if replay_start else None
        replay_end = np.datetime64(replay_end, 'D') if replay_end else None
        if replay_ratio and (signal_start is None or signal_end is None):
            raise ValueError('Historical replay requires an explicit train signal range')
        if replay_ratio and (replay_start is None or replay_end is None):
            raise ValueError('Historical replay requires an explicit replay signal range')
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol_position, symbol in enumerate(self.symbols):
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                self.timestamps_by_symbol[str(symbol)] = df['datetime'].reset_index(drop=True)
                if 'size_bucket' in df.columns:
                    self.size_by_symbol[str(symbol)] = pd.to_numeric(df['size_bucket'], errors='coerce').reset_index(drop=True)
                if 'size_percentile' in df.columns:
                    self.size_percentile_by_symbol[str(symbol)] = pd.to_numeric(
                        df['size_percentile'], errors='coerce'
                    ).reset_index(drop=True)
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                starts = np.arange(num_samples, dtype=np.int32)
                asof_positions = starts + self.config.lookback_window - 1
                dates = df['datetime'].to_numpy(dtype='datetime64[D]')[asof_positions]
                buckets = np.full(num_samples, 255, dtype=np.uint8)
                if str(symbol) in self.size_by_symbol:
                    raw_buckets = pd.to_numeric(
                        self.size_by_symbol[str(symbol)].iloc[asof_positions],
                        errors='coerce',
                    ).to_numpy(dtype=np.float64)
                    known = np.isfinite(raw_buckets)
                    buckets[known] = raw_buckets[known].astype(np.uint8)

                eligible = np.ones(num_samples, dtype=bool)
                if signal_start is not None:
                    eligible &= dates >= signal_start
                if signal_end is not None:
                    eligible &= dates <= signal_end
                symbol_exclusions = excluded_validation_dates.get(str(symbol))
                if symbol_exclusions:
                    excluded = np.isin(
                        dates,
                        np.asarray(sorted(symbol_exclusions), dtype='datetime64[D]'),
                    )
                    excluded_fixed_validation_samples += int(
                        np.count_nonzero(eligible & excluded)
                    )
                    eligible &= ~excluded
                eligible_positions = np.flatnonzero(eligible)
                for position in eligible_positions:
                    self.indices.append((symbol, int(starts[position])))
                    bucket_ids.append(int(buckets[position]))
                    signal_date_ids.append(int(dates[position].astype(np.int64)))
                signal_dates_seen.update(dates[eligible].tolist())

                if replay_ratio:
                    replay_eligible = ~eligible
                    replay_eligible &= dates >= replay_start
                    replay_eligible &= dates <= replay_end
                    replay_positions = np.flatnonzero(replay_eligible)
                    if len(replay_positions):
                        replay_symbols.append(np.full(
                            len(replay_positions), symbol_position, dtype=np.int32
                        ))
                        replay_starts.append(starts[replay_positions])
                        replay_buckets.append(buckets[replay_positions])
                        replay_dates.append(dates[replay_positions])
                        years = pd.DatetimeIndex(
                            dates[replay_positions]
                        ).year.to_numpy(dtype=np.int32)
                        replay_strata.append(
                            years * 256 + buckets[replay_positions].astype(np.int32)
                        )

        if (
            expected_training_exclusions
            and excluded_fixed_validation_samples != expected_training_exclusions
        ):
            raise ValueError(
                'Fixed validation training exclusion count mismatch: '
                f'{excluded_fixed_validation_samples} != '
                f'{expected_training_exclusions}'
            )
        recent_samples = len(self.indices)
        replay_samples = 0
        if replay_ratio:
            candidate_symbols = np.concatenate(replay_symbols)
            candidate_starts = np.concatenate(replay_starts)
            candidate_buckets = np.concatenate(replay_buckets)
            candidate_strata = np.concatenate(replay_strata)
            candidate_dates = np.concatenate(replay_dates)
            replay_target = round(recent_samples * replay_ratio / (1.0 - replay_ratio))
            selected = select_stratified_replay(
                candidate_strata, replay_target, self.config.seed + 2026
            )
            for candidate in selected:
                symbol = self.symbols[int(candidate_symbols[candidate])]
                self.indices.append((symbol, int(candidate_starts[candidate])))
                bucket_ids.append(int(candidate_buckets[candidate]))
                signal_date_ids.append(
                    int(candidate_dates[candidate].astype(np.int64))
                )
            replay_samples = len(selected)

        self.total_samples = len(self.indices)
        self.signal_date_ids = np.asarray(signal_date_ids, dtype=np.int64)
        if len(self.signal_date_ids) != self.total_samples:
            raise ValueError('Signal-date index is not aligned with dataset indices')
        requested = int(self.configured_samples)
        self.n_samples = self.total_samples if requested <= 0 else min(requested, self.total_samples)
        permutation_seed = self.config.seed + (0 if data_type == 'train' else 1)
        self.quick_validation_count = self.n_samples
        self.fixed_validation_manifest_sha256 = None
        self.validation_period_codes = None
        self.validation_period_names = {}
        if data_type == 'val' and fixed_manifest_path:
            (
                self.coverage_order, fixed_manifest, manifest_sha,
                self.validation_period_codes, period_codes,
            ) = (
                load_fixed_validation_selection(
                    self.config, self.indices, self.timestamps_by_symbol
                )
            )
            self.validation_period_names = {
                code: name for name, code in period_codes.items()
            }
            self.n_samples = len(self.coverage_order)
            self.quick_validation_count = int(
                fixed_manifest['selection']['quick_samples']
            )
            self.fixed_validation_manifest_sha256 = manifest_sha
            print(
                '[VAL] Fixed manifest validation enabled: '
                f'{self.quick_validation_count:,} quick / {self.n_samples:,} large; '
                f'manifest_sha256={manifest_sha}'
            )
        elif (
            data_type == 'train'
            and bool(getattr(self.config, 'balance_size_buckets', False))
        ):
            self.coverage_order = build_balanced_coverage_order(
                np.frombuffer(bucket_ids, dtype=np.uint8),
                self.n_samples,
                permutation_seed,
            )
            print('[TRAIN] Size-balanced no-replacement coverage order enabled.')
        else:
            self.coverage_order = np.random.default_rng(permutation_seed).permutation(
                self.total_samples
            )
        self.active_positions = self.coverage_order[:self.n_samples]
        self.coverage_start = 0
        self.selection_report = {
            'recent_samples': recent_samples,
            'replay_samples': replay_samples,
            'replay_ratio': replay_samples / self.total_samples if self.total_samples else 0.0,
            'signal_days': len(signal_dates_seen),
            'signal_start': str(min(signal_dates_seen)) if signal_dates_seen else None,
            'signal_end': str(max(signal_dates_seen)) if signal_dates_seen else None,
            'fixed_validation_manifest_sha256': self.fixed_validation_manifest_sha256,
            'quick_validation_samples': self.quick_validation_count,
            'large_validation_samples': self.n_samples if data_type == 'val' else None,
            'excluded_fixed_validation_samples': excluded_fixed_validation_samples,
            'fixed_validation_samples_absent_from_training': (
                absent_training_samples if data_type == 'train' and fixed_manifest_path
                else 0
            ),
            'validation_period_counts': (
                {
                    self.validation_period_names[int(code)]: int(count)
                    for code, count in zip(
                        *np.unique(self.validation_period_codes, return_counts=True)
                    )
                }
                if self.validation_period_codes is not None else {}
            ),
        }
        print(
            f"[{data_type.upper()}] Found {self.total_samples} possible samples. "
            f"Using {self.n_samples} unique samples per segment."
        )
        if signal_start is not None or signal_end is not None or replay_samples:
            print(f"[{data_type.upper()}] Selection: {self.selection_report}")

    def set_epoch_seed(self, epoch: int):
        """
        Advance the training coverage cursor without replacement.

        Args:
            epoch (int): The current epoch number.
        """
        if self.total_samples == 0:
            return
        if self.data_type == 'val':
            self.coverage_start = 0
            self.active_positions = self.coverage_order[:self.n_samples]
            return

        segments_per_pass = math.ceil(self.total_samples / self.n_samples)
        segment_in_pass = int(epoch) % segments_per_pass
        start = segment_in_pass * self.n_samples
        end = min(start + self.n_samples, self.total_samples)
        self.coverage_start = start
        self.active_positions = self.coverage_order[start:end]
        if self.use_beta_v21_auxiliary:
            date_order = np.argsort(
                self.signal_date_ids[self.active_positions], kind='stable'
            )
            self.active_positions = self.active_positions[date_order]

    def coverage_state(self, segment_index: int = 0) -> dict:
        consumed = min((int(segment_index) + 1) * self.n_samples, self.total_samples)
        return {
            'total_samples': self.total_samples,
            'samples_per_segment': len(self.active_positions),
            'segment_start': self.coverage_start,
            'unique_samples_covered': consumed,
            'coverage_fraction': consumed / self.total_samples if self.total_samples else 0.0,
        }
    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return len(self.active_positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        source_position = int(self.active_positions[idx])
        symbol, start_idx = self.indices[source_position]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)
        raw_x = x.copy()

        # Normalize the window. Mean and std are calculated strictly on the
        # lookback window (past data) to prevent future data leakage.
        past_len = self.config.lookback_window
        past_x = x[:past_len]

        x_mean = np.mean(past_x, axis=0)
        x_std  = np.std(past_x, axis=0)

        # Apply normalization and robust clipping to the entire sequence
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        asof = self.timestamps_by_symbol[str(symbol)].iloc[
            start_idx + self.config.lookback_window - 1
        ]
        auxiliary_labels = None
        if self.use_beta_v21_auxiliary:
            auxiliary_labels = build_beta_v21_labels(raw_x, past_len, asof)
            auxiliary_labels['feature_means'] = torch.from_numpy(
                x_mean.astype(np.float32)
            )
            auxiliary_labels['feature_stds'] = torch.from_numpy(
                x_std.astype(np.float32)
            )

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        if self.use_context_features:
            sector_value, size_value, size_percentile_value = self.asset_metadata.get_conditions(
                symbol, asof=asof
            )
            if str(symbol) in self.size_by_symbol:
                inline_size = self.size_by_symbol[str(symbol)].iloc[start_idx + self.config.lookback_window - 1]
                if pd.notna(inline_size):
                    size_value = int(inline_size)
            if str(symbol) in self.size_percentile_by_symbol:
                inline_percentile = self.size_percentile_by_symbol[str(symbol)].iloc[
                    start_idx + self.config.lookback_window - 1
                ]
                if pd.notna(inline_percentile):
                    size_percentile_value = float(inline_percentile)
            sector_id = torch.tensor(sector_value, dtype=torch.long)
            size_bucket = torch.tensor(size_value, dtype=torch.long)
            if self.use_size_percentile:
                size_percentile = torch.tensor(size_percentile_value, dtype=torch.float32)
                result = (
                    x_tensor, x_stamp_tensor, sector_id, size_bucket,
                    size_percentile,
                )
                if self.data_type == 'val' and self.validation_period_codes is not None:
                    period_code = torch.tensor(
                        int(self.validation_period_codes[idx]), dtype=torch.int16
                    )
                    result = (*result, period_code)
                if auxiliary_labels is not None:
                    result = (*result, auxiliary_labels)
                return result
            result = (x_tensor, x_stamp_tensor, sector_id, size_bucket)
            if auxiliary_labels is not None:
                result = (*result, auxiliary_labels)
            return result
        result = (x_tensor, x_stamp_tensor)
        if auxiliary_labels is not None:
            result = (*result, auxiliary_labels)
        return result


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")
