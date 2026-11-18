"""G1: Full-universe multi-layer & multi-lookback frozen-C2 representation + Ridge heads.

Strictly observes temporal isolation:
- Train: <= 2026-04-30 (train_data.pkl, strided dates for speed & diversity)
- Embargo: 2026-05-01 .. 2026-05-15 (never used)
- Validation selection: 2026-05-18 .. 2026-07-02 (val_data.pkl, unseen stocks)
- Final refit: train (<= 2026-04-30) + full val (<= 2026-07-02) using val-selected alpha
- OOS inference: 2026-07-17 .. 2026-10-23 (65d, 335,312 windows); 2026-10-24+ untouched.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
ALPHAS = (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7)
BLOCK_SLICES = {
    "L120_LAST": slice(0, 512),
    "L120_PRE": slice(512, 1024),
    "L120_M5": slice(1024, 1536),
    "L120_DIFF20": slice(1536, 2048),
    "L60_LAST": slice(2048, 2560),
    "L60_PRE": slice(2560, 3072),
}
TOTAL_DIM = 3072

TRAIN_FIT_END = "2026-04-30"
TRAIN_START = os.environ.get("G1_TRAIN_START", "2021-01-01")
TRAIN_DATE_STRIDE = int(os.environ.get("G1_TRAIN_STRIDE", "3") or "3")
VAL_SEL_START = "2026-05-18"
VAL_SEL_END = "2026-07-02"
OOS_START = "2026-07-17"
OOS_END = "2026-10-23"


def phase(name: str, **values) -> None:
    print(json.dumps({"time": time.time(), "phase": name, **values}, ensure_ascii=False), flush=True)


def rank_pct_centered(values: np.ndarray) -> np.ndarray:
    """Rank-percentile in [-0.5, 0.5] within a single cross-section."""
    s = pd.Series(values)
    r = s.rank(pct=True, method="average").to_numpy(dtype=np.float64) - 0.5
    return np.nan_to_num(r, nan=0.0)


def standardize_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - arr.mean()
    std = float(arr.std())
    return arr / max(std, 1e-6)


def orthogonalize_target(
    ret10: np.ndarray,
    size_pct: np.ndarray,
    rev10_past: np.ndarray,
    sector_ids: np.ndarray,
    num_sectors: int = 86,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_raw_rank_z, y_orth_rank_z) for one cross-section."""
    y_raw = standardize_1d(rank_pct_centered(ret10))
    n = len(y_raw)
    if n < 20:
        return y_raw, y_raw.copy()
    sz = rank_pct_centered(size_pct)
    rv = rank_pct_centered(rev10_past)
    sec = np.zeros((n, num_sectors), dtype=np.float64)
    valid = (sector_ids >= 0) & (sector_ids < num_sectors)
    sec[np.arange(n)[valid], sector_ids[valid]] = 1.0
    # Drop empty sector columns cleanly with lstsq
    design = np.column_stack([np.ones(n, dtype=np.float64), sz, rv, sec])
    coef, _, _, _ = np.linalg.lstsq(design, y_raw, rcond=None)
    resid = y_raw - design @ coef
    return y_raw, standardize_1d(resid)


def zscore_cross_section_torch(x: torch.Tensor, clip_val: float = 5.0) -> torch.Tensor:
    """Cross-sectionally z-score [N, D] tensor per column on GPU."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return torch.clamp((x - mean) / std, -clip_val, clip_val)


def normalize_window_np(raw_values: np.ndarray, lookback: int) -> np.ndarray:
    """Take trailing `lookback` bars of [W, 6], normalize by their own mean/std, clip [-5, 5]."""
    sub = raw_values[-lookback:].astype(np.float32)
    mean = sub.mean(axis=0, keepdims=True)
    std = sub.std(axis=0, keepdims=True)
    return np.clip((sub - mean) / (std + 1e-5), -5.0, 5.0)


@torch.inference_mode()
def extract_multiscale_batch(
    model,
    tokenizer,
    raw_hist: torch.Tensor,
    stamp_hist: torch.Tensor,
    sector_id: torch.Tensor,
    size_pct: torch.Tensor,
) -> torch.Tensor:
    """Extract [B, 3072] multi-layer + L120/L60 representations from frozen C2.

    raw_hist: [B, 120, 6] raw OHLCVA on GPU.
    stamp_hist: [B, 120, 5] float32 timestamps on GPU.
    """
    device = raw_hist.device

    def _run_lookback(lb: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sub = raw_hist[:, -lb:, :]
        mean = sub.mean(dim=1, keepdim=True)
        std = sub.std(dim=1, unbiased=False, keepdim=True)
        x_norm = torch.clamp((sub - mean) / (std + 1e-5), -5.0, 5.0)
        st = stamp_hist[:, -lb:, :]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            tokens = tokenizer.encode(x_norm, half=True)
            h = model.embedding([tokens[0], tokens[1]])
            if st is not None:
                h = h + model.time_emb(st)
            if model.context_layer == 0:
                h = model._add_context(h, sector_id=sector_id, size_percentile=size_pct)
            h = model.token_drop(h)
            h_pre = None
            for layer_idx, layer in enumerate(model.transformer):
                if model.context_layer > 0 and layer_idx == model.context_layer:
                    h_pre = h[:, -1, :].float()
                    h = model._add_context(h, sector_id=sector_id, size_percentile=size_pct)
                h = layer(h, key_padding_mask=None)
            h_out = model.norm(h).float()
        h_last = h_out[:, -1, :]
        h_m5 = h_out[:, -5:, :].mean(dim=1)
        h_diff20 = h_last - h_out[:, -20:, :].mean(dim=1)
        return h_last, h_pre, h_m5, h_diff20

    l120_last, l120_pre, l120_m5, l120_diff20 = _run_lookback(120)
    l60_last, l60_pre, _, _ = _run_lookback(60)
    return torch.cat([l120_last, l120_pre, l120_m5, l120_diff20, l60_last, l60_pre], dim=1)


def solve_ridge_subblock(
    gram: np.ndarray,
    rhs: np.ndarray,
    cols: slice | list[int],
    alpha: float,
) -> np.ndarray:
    """Closed-form Ridge solve restricted to a column subset of global Gram matrix."""
    sub_g = gram[np.ix_(cols, cols)] if isinstance(cols, list) else gram[cols, cols]
    sub_b = rhs[cols]
    dim = sub_g.shape[0]
    return np.linalg.solve(sub_g + float(alpha) * np.eye(dim, dtype=np.float64), sub_b)


def build_fast_metadata_lookup(meta_path: Path, sector_map: dict[str, int]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build per-symbol (date_int64_ns, sector_id, size_percentile) sorted arrays for O(log T) lookup."""
    df = pd.read_csv(meta_path, usecols=lambda c: c in {"symbol", "datetime", "date", "sector", "market_cap", "size_percentile"})
    date_col = "datetime" if "datetime" in df.columns else "date"
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_dt"]).sort_values(["symbol", "_dt"])
    df["symbol"] = df["symbol"].astype(str)
    if "size_percentile" not in df.columns and "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        df["size_percentile"] = df.groupby("_dt")["market_cap"].rank(pct=True, method="average")
    unk = sector_map.get("unknown", 85)
    df["sec_id"] = df["sector"].fillna("unknown").astype(str).map(lambda s: sector_map.get(s, unk)).astype(np.int32)
    df["pct"] = pd.to_numeric(df["size_percentile"], errors="coerce").fillna(0.5).clip(0.0, 1.0).astype(np.float32)
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym, grp in df.groupby("symbol", sort=False):
        lookup[sym] = (
            grp["_dt"].to_numpy(dtype="datetime64[ns]").view(np.int64),
            grp["sec_id"].to_numpy(dtype=np.int32),
            grp["pct"].to_numpy(dtype=np.float32),
        )
    return lookup


def lookup_conditions_for_symbol(
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    symbol: str,
    dates_ns: np.ndarray,
    default_sector: int = 85,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized as-of lookup matching AssetMetadata.get_conditions <= asof."""
    entry = lookup.get(symbol)
    n = len(dates_ns)
    if entry is None or len(entry[0]) == 0:
        return np.full(n, default_sector, dtype=np.int64), np.full(n, 0.5, dtype=np.float32)
    m_dates, m_sec, m_pct = entry
    pos = np.searchsorted(m_dates, dates_ns, side="right") - 1
    valid = pos >= 0
    sec_out = np.full(n, default_sector, dtype=np.int64)
    pct_out = np.full(n, 0.5, dtype=np.float32)
    sec_out[valid] = m_sec[pos[valid]]
    pct_out[valid] = m_pct[pos[valid]]
    return sec_out, pct_out


def build_panel_date_batches(
    pkl_path: Path,
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    date_from: str,
    date_to: str,
    date_stride: int = 1,
    default_sector: int = 85,
) -> tuple[list[dict], list[tuple]]:
    """Convert panel pickle into compact per-symbol arrays + per-date window index lists."""
    with pkl_path.open("rb") as f:
        raw_panel: dict[str, pd.DataFrame] = pickle.load(f)

    sym_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    date_to_records: dict[str, list[tuple[int, int, int, float, float, float]]] = {}

    for sym_idx, (symbol, df) in enumerate(raw_panel.items()):
        df = df.sort_index()
        if len(df) < 131:
            continue
        idx = pd.DatetimeIndex(df.index)
        raw_vals = df[FEATURES].to_numpy(dtype=np.float32)
        stamps = np.column_stack([idx.minute, idx.hour, idx.weekday, idx.day, idx.month]).astype(np.float32)
        dates_str = idx.strftime("%Y-%m-%d").to_numpy()
        dates_ns = idx.to_numpy(dtype="datetime64[ns]").view(np.int64)
        sec_arr, pct_arr = lookup_conditions_for_symbol(lookup, str(symbol), dates_ns, default_sector)
        if "size_percentile" in df.columns:
            inline_pct = pd.to_numeric(df["size_percentile"], errors="coerce").to_numpy(dtype=np.float32)
            good = np.isfinite(inline_pct)
            pct_arr = np.where(good, inline_pct, pct_arr)
        sym_arrays.append((raw_vals, stamps, sec_arr, pct_arr))
        s_i = len(sym_arrays) - 1
        closes = raw_vals[:, 3].astype(np.float64)

        # Anchor t is index of asof_date (requires t >= 119 and t + 10 < len(df))
        valid_t = np.where((dates_str >= date_from) & (dates_str <= date_to))[0]
        for t in valid_t:
            if t < 119 or t + 10 >= len(df):
                continue
            d_str = str(dates_str[t])
            c_asof = closes[t]
            c_fut = closes[t + 10]
            c_past10 = closes[t - 10]
            if c_asof <= 1e-8 or c_past10 <= 1e-8 or not np.isfinite(c_fut):
                continue
            ret10 = float(c_fut / c_asof - 1.0)
            rev10_past = float(-(c_asof / c_past10 - 1.0))
            date_to_records.setdefault(d_str, []).append(
                (s_i, int(t), int(sec_arr[t]), float(pct_arr[t]), ret10, rev10_past)
            )

    all_dates = sorted(date_to_records.keys())
    selected_dates = all_dates[::max(1, date_stride)]
    batches = []
    for d_str in selected_dates:
        recs = date_to_records[d_str]
        if len(recs) < 50:
            continue
        batches.append((d_str, recs))
    return sym_arrays, batches


def daily_spearman_ic(dates: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> float:
    df = pd.DataFrame({"d": dates, "p": pred, "a": actual})
    return float(df.groupby("d", sort=False).apply(lambda g: g["p"].corr(g["a"], method="spearman"), include_groups=False).mean())


def run_gpu_extract_stats(args: tuple) -> dict:
    """Worker running on GPU `rank` processing assigned train/val/oos dates."""
    (
        rank,
        weights_dir,
        tokenizer_dir,
        sym_arrays_train,
        train_batches,
        sym_arrays_val,
        val_batches,
        oos_mode_data,
        batch_size,
    ) = args

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = Kronos.from_pretrained(
        weights_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()

    gram_tr = torch.zeros((TOTAL_DIM, TOTAL_DIM), dtype=torch.float64, device=device)
    rhs_raw_tr = torch.zeros(TOTAL_DIM, dtype=torch.float64, device=device)
    rhs_orth_tr = torch.zeros(TOTAL_DIM, dtype=torch.float64, device=device)
    n_tr_rows = 0

    def _forward_date(sym_arrays, recs) -> torch.Tensor:
        chunks = []
        for off in range(0, len(recs), batch_size):
            sl = recs[off:off + batch_size]
            raw_b = np.stack([sym_arrays[r[0]][0][r[1] - 119:r[1] + 1] for r in sl])
            st_b = np.stack([sym_arrays[r[0]][1][r[1] - 119:r[1] + 1] for r in sl])
            sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
            feats = extract_multiscale_batch(
                model,
                tokenizer,
                torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                torch.as_tensor(st_b, device=device, dtype=torch.float32),
                torch.as_tensor(sec_b, device=device, dtype=torch.long),
                torch.as_tensor(pct_b, device=device, dtype=torch.float32),
            )
            chunks.append(feats)
        x_d = torch.cat(chunks, dim=0)
        return zscore_cross_section_torch(x_d, clip_val=5.0)

    # 1. Train accumulation
    my_tr = [b for i, b in enumerate(train_batches) if i % 2 == rank]
    for bi, (d_str, recs) in enumerate(my_tr):
        x_d = _forward_date(sym_arrays_train, recs)
        sec_np = np.asarray([r[2] for r in recs], dtype=np.int64)
        pct_np = np.asarray([r[3] for r in recs], dtype=np.float64)
        ret_np = np.asarray([r[4] for r in recs], dtype=np.float64)
        rev_np = np.asarray([r[5] for r in recs], dtype=np.float64)
        y_raw, y_orth = orthogonalize_target(ret_np, pct_np, rev_np, sec_np)
        x64 = x_d.to(torch.float64)
        yr_t = torch.as_tensor(y_raw, device=device, dtype=torch.float64)
        yo_t = torch.as_tensor(y_orth, device=device, dtype=torch.float64)
        gram_tr.addmm_(x64.T, x64)
        rhs_raw_tr.addmv_(x64.T, yr_t)
        rhs_orth_tr.addmv_(x64.T, yo_t)
        n_tr_rows += len(recs)
        if (bi + 1) % 30 == 0 or bi + 1 == len(my_tr):
            phase("train_stats_progress", rank=rank, done=bi + 1, total=len(my_tr), rows=n_tr_rows)

    # 2. Val extraction (compact float16 vectors + labels — only ~124k rows total across 2 GPUs!)
    my_val = [b for i, b in enumerate(val_batches) if i % 2 == rank]
    val_feats, val_dates, val_ret, val_y_raw, val_y_orth = [], [], [], [], []
    for d_str, recs in my_val:
        x_d = _forward_date(sym_arrays_val, recs)
        sec_np = np.asarray([r[2] for r in recs], dtype=np.int64)
        pct_np = np.asarray([r[3] for r in recs], dtype=np.float64)
        ret_np = np.asarray([r[4] for r in recs], dtype=np.float64)
        rev_np = np.asarray([r[5] for r in recs], dtype=np.float64)
        y_raw, y_orth = orthogonalize_target(ret_np, pct_np, rev_np, sec_np)
        val_feats.append(x_d.half().cpu().numpy())
        val_dates.extend([d_str] * len(recs))
        val_ret.append(ret_np.astype(np.float32))
        val_y_raw.append(y_raw.astype(np.float32))
        val_y_orth.append(y_orth.astype(np.float32))

    val_out = None
    if val_feats:
        val_out = {
            "x": np.concatenate(val_feats, axis=0),
            "dates": np.asarray(val_dates),
            "ret10": np.concatenate(val_ret),
            "y_raw": np.concatenate(val_y_raw),
            "y_orth": np.concatenate(val_y_orth),
        }

    return {
        "rank": rank,
        "gram_tr": gram_tr.cpu().numpy(),
        "rhs_raw_tr": rhs_raw_tr.cpu().numpy(),
        "rhs_orth_tr": rhs_orth_tr.cpu().numpy(),
        "n_tr_rows": n_tr_rows,
        "val": val_out,
    }


def score_oos_on_gpu(args: tuple) -> pd.DataFrame:
    """Score OOS dates assigned to `rank` with solved weight vectors directly on GPU."""
    rank, weights_dir, tokenizer_dir, sym_arrays_oos, oos_batches, heads_dict, batch_size = args
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = Kronos.from_pretrained(
        weights_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()

    head_tensors = {
        name: (sl, torch.as_tensor(w, device=device, dtype=torch.float32))
        for name, (sl, w) in heads_dict.items()
    }
    my_oos = [b for i, b in enumerate(oos_batches) if i % 2 == rank]
    out_frames = []
    for bi, (d_str, recs) in enumerate(my_oos):
        chunks = []
        for off in range(0, len(recs), batch_size):
            sl = recs[off:off + batch_size]
            raw_b = np.stack([sym_arrays_oos[r[0]][0][r[1] - 119:r[1] + 1] for r in sl])
            st_b = np.stack([sym_arrays_oos[r[0]][1][r[1] - 119:r[1] + 1] for r in sl])
            sec_b = np.asarray([r[2] for r in sl], dtype=np.int64)
            pct_b = np.asarray([r[3] for r in sl], dtype=np.float32)
            feats = extract_multiscale_batch(
                model,
                tokenizer,
                torch.as_tensor(raw_b, device=device, dtype=torch.float32),
                torch.as_tensor(st_b, device=device, dtype=torch.float32),
                torch.as_tensor(sec_b, device=device, dtype=torch.long),
                torch.as_tensor(pct_b, device=device, dtype=torch.float32),
            )
            chunks.append(feats)
        x_d = zscore_cross_section_torch(torch.cat(chunks, dim=0), clip_val=5.0)
        df_d = pd.DataFrame({
            "identity": [r[4] for r in recs],
            "symbol": [r[5] for r in recs],
            "asof_date": d_str,
        })
        for name, (sl_cols, w_t) in head_tensors.items():
            df_d[f"p_{name}"] = (x_d[:, sl_cols] @ w_t).cpu().numpy().astype(np.float32)
        out_frames.append(df_d)
        if (bi + 1) % 10 == 0 or bi + 1 == len(my_oos):
            phase("oos_score_progress", rank=rank, done=bi + 1, total=len(my_oos))
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
