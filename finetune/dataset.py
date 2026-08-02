import pickle
import math
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
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.configured_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.configured_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list
        self.use_context_features = bool(getattr(self.config, 'use_context_features', False))
        self.use_sector_features = bool(getattr(self.config, 'use_sector_features', True))
        self.use_size_features = bool(getattr(self.config, 'use_size_features', True))
        self.use_size_percentile = bool(getattr(self.config, 'use_size_percentile', False))
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
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
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

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        self.total_samples = len(self.indices)
        requested = int(self.configured_samples)
        self.n_samples = self.total_samples if requested <= 0 else min(requested, self.total_samples)
        permutation_seed = self.config.seed + (0 if data_type == 'train' else 1)
        self.coverage_order = np.random.default_rng(permutation_seed).permutation(
            self.total_samples
        )
        self.active_positions = self.coverage_order[:self.n_samples]
        self.coverage_start = 0
        print(
            f"[{data_type.upper()}] Found {self.total_samples} possible samples. "
            f"Using {self.n_samples} unique samples per segment."
        )

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

        # Normalize the window. Mean and std are calculated strictly on the
        # lookback window (past data) to prevent future data leakage.
        past_len = self.config.lookback_window
        past_x = x[:past_len]

        x_mean = np.mean(past_x, axis=0)
        x_std  = np.std(past_x, axis=0)

        # Apply normalization and robust clipping to the entire sequence
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        if self.use_context_features:
            asof = self.timestamps_by_symbol[str(symbol)].iloc[start_idx + self.config.lookback_window - 1]
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
                return x_tensor, x_stamp_tensor, sector_id, size_bucket, size_percentile
            return x_tensor, x_stamp_tensor, sector_id, size_bucket
        return x_tensor, x_stamp_tensor


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
