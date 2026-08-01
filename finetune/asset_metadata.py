"""Point-in-time sector and size metadata for asset-conditioned training."""

import os
import numpy as np
import pandas as pd


class AssetMetadata:
    """Resolve sector and size IDs as of a sample's last observed timestamp."""

    def __init__(self, path: str, num_sectors: int, num_size_buckets: int):
        self.num_sectors = int(num_sectors)
        self.num_size_buckets = int(num_size_buckets)
        self.unknown_sector = self.num_sectors
        self.unknown_size = self.num_size_buckets
        self.rows_by_symbol = {}
        self.sector_map = {}

        path = str(path or '').strip()
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"Asset metadata file not found: {path}")

        metadata = pd.read_csv(path)
        if 'symbol' not in metadata.columns:
            raise ValueError("Asset metadata must contain a 'symbol' column")
        date_col = 'datetime' if 'datetime' in metadata.columns else ('date' if 'date' in metadata.columns else None)
        if date_col is not None:
            metadata['_metadata_datetime'] = pd.to_datetime(metadata[date_col], errors='coerce')
            metadata = metadata.dropna(subset=['_metadata_datetime'])
        else:
            metadata['_metadata_datetime'] = pd.Timestamp.min
        metadata['symbol'] = metadata['symbol'].astype(str)

        if self.num_sectors > 0 and 'sector' in metadata.columns:
            sector_values = metadata['sector'].dropna().astype(str).unique()
            self.sector_map = {value: idx for idx, value in enumerate(sorted(sector_values))}
            if len(self.sector_map) > self.num_sectors:
                raise ValueError(
                    f"Found {len(self.sector_map)} sectors but num_sectors={self.num_sectors}"
                )

        if 'size_bucket' not in metadata.columns and 'market_cap' in metadata.columns:
            metadata['market_cap'] = pd.to_numeric(metadata['market_cap'], errors='coerce')
            if date_col is None:
                ranks = metadata['market_cap'].rank(method='first', pct=True)
            else:
                ranks = metadata.groupby('_metadata_datetime')['market_cap'].rank(method='first', pct=True)
            metadata['size_bucket'] = np.minimum(
                (ranks * self.num_size_buckets).fillna(-1).astype(int),
                self.num_size_buckets - 1,
            )

        for symbol, rows in metadata.groupby('symbol', sort=False):
            self.rows_by_symbol[symbol] = rows.sort_values('_metadata_datetime').reset_index(drop=True)

    def get(self, symbol: str, asof=None):
        rows = self.rows_by_symbol.get(str(symbol))
        if rows is None or rows.empty:
            return self.unknown_sector, self.unknown_size

        if asof is not None and rows['_metadata_datetime'].iloc[0] != pd.Timestamp.min:
            asof = pd.Timestamp(asof)
            valid = rows[rows['_metadata_datetime'] <= asof]
            row = valid.iloc[-1] if not valid.empty else None
        else:
            row = rows.iloc[-1]
        if row is None:
            return self.unknown_sector, self.unknown_size

        sector_id = self.unknown_sector
        if 'sector' in rows.columns and pd.notna(row.get('sector')):
            sector_id = self.sector_map.get(str(row['sector']), self.unknown_sector)
        elif 'sector_id' in rows.columns and pd.notna(row.get('sector_id')):
            value = int(row['sector_id'])
            if 0 <= value < self.num_sectors:
                sector_id = value

        size_bucket = self.unknown_size
        if 'size_bucket' in rows.columns and pd.notna(row.get('size_bucket')):
            value = int(row['size_bucket'])
            if 0 <= value < self.num_size_buckets:
                size_bucket = value
        return sector_id, size_bucket
