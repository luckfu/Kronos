"""Kairos tokenizer expressiveness ablations for y=1{mfe10>=0.10}.

Three cheap CPU/GPU-optional probes on the validation panel:
  1) Linear/logistic probe on frozen Kronos tokenizer codes / quantized bit
     embeddings / reconstruction-error features.
  2) Raw OHLCV window features (bypass tokenizer) logistic + ridge.
  3) Mutual information / association: codes & recon error vs label;
     neighboring-code stability.

Compare Δ vs Phase G2 handcrafted (~−0.034) and failed deep sidecar (~+0.008).
Verdict: tokenizer bottleneck yes / no / unclear.

No R2 restart, no Kaggle long train, no TPU WIP edits.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modernbert_finance.ablations.buy_profit_mfe_ablations import (
    PROFIT_THRESHOLD,
    SEED,
    build_mfe_buy_labels,
)
from modernbert_finance.ablations.continuous_xsection_ablations import (
    _fit_eval_logistic,
    _prior_ll,
)
from modernbert_finance.ablations.simple_baseline import summarize_history_windows
from modernbert_finance.build_dataset import FEATURES, LOOKBACK, prepare_frame
from modernbert_finance.build_targets import WINDOW
from modernbert_finance.tokenizer import FrozenKronosTokenizer

EPS = 1e-4
CLIP = 5.0
BATCH = 256
# Comparators (recorded from prior Kairos memos).
G2_COMB_DELTA = -0.034130436131804
G2_BASE_DELTA = -0.020109348008008898
SIDECAR_BEST_DELTA = 0.008022
GATE_DELTA = -0.04


def _collect_normalized_histories(
    panel: Any,
    targets: pd.DataFrame,
    *,
    signal_start: str | None = "2025-07-03",
    signal_end: str | None = "2026-07-02",
    clip: float = CLIP,
) -> dict[str, Any]:
    """Aligned (N, 120, 6) z-scored windows + mfe10 labels (same bounds as G2)."""
    from modernbert_finance.ablations._panel_io import load_panel

    if isinstance(panel, (str, Path)):
        panel = load_panel(Path(panel))
    start_date = pd.Timestamp(signal_start).date() if signal_start else None
    end_date = pd.Timestamp(signal_end).date() if signal_end else None

    histories_raw: list[np.ndarray] = []
    histories_norm: list[np.ndarray] = []
    for symbol in sorted(panel):
        frame = prepare_frame(panel[symbol], symbol)
        if len(frame) < WINDOW:
            continue
        values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
        dates = frame.index
        max_start = len(frame) - WINDOW + 1
        for start in range(max_start):
            asof_pos = start + LOOKBACK - 1
            asof = pd.Timestamp(dates[asof_pos]).date()
            if start_date and asof < start_date:
                continue
            if end_date and asof > end_date:
                continue
            hist = values[start : start + LOOKBACK]
            mean = hist.mean(axis=0)
            std = hist.std(axis=0)
            norm = (hist - mean) / (std + 1e-5)
            norm = np.clip(norm, -clip, clip).astype(np.float32)
            histories_raw.append(hist.astype(np.float32))
            histories_norm.append(norm)

    if len(histories_raw) != len(targets):
        raise ValueError(
            f"windows {len(histories_raw)} != targets {len(targets)}"
        )
    raw = np.stack(histories_raw, axis=0)
    norm = np.stack(histories_norm, axis=0)
    mfe = targets["mfe10"].to_numpy(dtype=np.float64)
    y = (mfe >= PROFIT_THRESHOLD).astype(np.int64)
    return {
        "raw": raw,
        "norm": norm,
        "mfe10": mfe,
        "y": y,
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
    }


def _encode_tokenizer(
    tokenizer: FrozenKronosTokenizer,
    hist_norm: np.ndarray,
    *,
    batch_size: int = BATCH,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Encode frozen s1/s2 + quantized bit embeddings + reconstruction error."""
    n = hist_norm.shape[0]
    s1_all = np.empty((n, LOOKBACK), dtype=np.int64)
    s2_all = np.empty((n, LOOKBACK), dtype=np.int64)
    # codebook_dim = s1_bits + s2_bits = 20
    bit_dim = int(tokenizer.tokenizer.s1_bits + tokenizer.tokenizer.s2_bits)
    # Per-sample summaries collected after loop
    bit_mean = np.empty((n, bit_dim), dtype=np.float32)
    bit_std = np.empty((n, bit_dim), dtype=np.float32)
    bit_last = np.empty((n, bit_dim), dtype=np.float32)
    recon_mse = np.empty((n,), dtype=np.float32)
    recon_mse_ohlc = np.empty((n,), dtype=np.float32)

    tok = tokenizer.tokenizer  # underlying KronosTokenizer
    tok.eval()

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(hist_norm[start:end]).to(device=device, dtype=torch.float32)
            s1, s2 = tokenizer(x)
            s1_all[start:end] = s1.cpu().numpy()
            s2_all[start:end] = s2.cpu().numpy()

            # Quantized bipolar bits from indices (frozen embedding of codes)
            bits = tok.indices_to_bits((s1, s2), half=True)  # (B, T, 20)
            bit_mean[start:end] = bits.mean(dim=1).cpu().numpy()
            bit_std[start:end] = bits.std(dim=1).cpu().numpy()
            bit_last[start:end] = bits[:, -1, :].cpu().numpy()

            # Reconstruction via decode(codes)
            recon = tok.decode((s1, s2), half=True)  # (B, T, 6)
            err = (recon - x).pow(2)
            recon_mse[start:end] = err.mean(dim=(1, 2)).cpu().numpy()
            recon_mse_ohlc[start:end] = err[:, :, :4].mean(dim=(1, 2)).cpu().numpy()

            if start == 0 or (end % (batch_size * 20) == 0) or end == n:
                elapsed = time.time() - t0
                rate = end / max(elapsed, 1e-6)
                print(
                    f"[encode] {end}/{n} ({100.0 * end / n:.1f}%) "
                    f"{rate:.1f} samp/s elapsed={elapsed:.1f}s",
                    flush=True,
                )

    return {
        "s1": s1_all,
        "s2": s2_all,
        "bit_mean": bit_mean,
        "bit_std": bit_std,
        "bit_last": bit_last,
        "recon_mse": recon_mse,
        "recon_mse_ohlc": recon_mse_ohlc,
        "encode_seconds": float(time.time() - t0),
        "bit_dim": bit_dim,
    }


def _code_stability_features(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Neighboring-code stability + diversity summaries → (N, D)."""
    # equality with previous step
    eq1 = (s1[:, 1:] == s1[:, :-1]).astype(np.float64)
    eq2 = (s2[:, 1:] == s2[:, :-1]).astype(np.float64)
    both = ((s1[:, 1:] == s1[:, :-1]) & (s2[:, 1:] == s2[:, :-1])).astype(np.float64)

    # unique counts (approx diversity)
    # vectorized unique is awkward; use a cheap hash-bucket occupancy
    def _approx_unique(codes: np.ndarray, buckets: int = 64) -> np.ndarray:
        # codes in [0, 1024); bucket occupancy fraction
        b = codes % buckets
        n, t = b.shape
        out = np.zeros((n, buckets), dtype=np.float64)
        for i in range(t):
            rows = np.arange(n)
            out[rows, b[:, i]] = 1.0
        return out.sum(axis=1) / buckets

    uniq1 = _approx_unique(s1)
    uniq2 = _approx_unique(s2)

    # last-k change rates
    def _tail_change(eq: np.ndarray, k: int) -> np.ndarray:
        return 1.0 - eq[:, -k:].mean(axis=1)

    feats = np.column_stack(
        [
            eq1.mean(axis=1),
            eq2.mean(axis=1),
            both.mean(axis=1),
            eq1[:, -20:].mean(axis=1),
            eq2[:, -20:].mean(axis=1),
            both[:, -20:].mean(axis=1),
            _tail_change(eq1, 5),
            _tail_change(eq2, 5),
            _tail_change(both, 5),
            uniq1,
            uniq2,
            (s1[:, -1] / 1024.0),
            (s2[:, -1] / 1024.0),
            (s1[:, -5:].astype(np.float64).mean(axis=1) / 1024.0),
            (s2[:, -5:].astype(np.float64).mean(axis=1) / 1024.0),
        ]
    )
    return feats.astype(np.float64)


def _code_histogram_features(s1: np.ndarray, s2: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Coarse bag-of-codes histograms for s1 and s2 (normalized)."""
    n = s1.shape[0]
    h1 = np.zeros((n, n_bins), dtype=np.float64)
    h2 = np.zeros((n, n_bins), dtype=np.float64)
    b1 = (s1 % n_bins).astype(np.int64)
    b2 = (s2 % n_bins).astype(np.int64)
    for t in range(s1.shape[1]):
        rows = np.arange(n)
        # bincount per row via advanced indexing accumulation
        np.add.at(h1, (rows, b1[:, t]), 1.0)
        np.add.at(h2, (rows, b2[:, t]), 1.0)
    h1 /= max(s1.shape[1], 1)
    h2 /= max(s2.shape[1], 1)
    return np.concatenate([h1, h2], axis=1)


def _fit_eval_logistic_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Logistic + RidgeClassifier probe with same split as G2."""
    y = y.astype(np.int64).reshape(-1)
    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "single_class"}
    x = np.asarray(x, dtype=np.float64)
    # replace non-finite
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    prior = _prior_ll(y_te)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed),
    )
    clf.fit(x_tr, y_tr)
    proba = clf.predict_proba(x_te)[:, 1]
    ll = float(log_loss(y_te, np.column_stack([1 - proba, proba]), labels=[0, 1]))

    ridge = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0, random_state=seed),
    )
    ridge.fit(x_tr, y_tr)
    # decision_function → sigmoid-ish for log_loss proxy
    dec = ridge.decision_function(x_te)
    # map to [0,1] via logistic
    ridge_proba = 1.0 / (1.0 + np.exp(-dec))
    ridge_ll = float(
        log_loss(y_te, np.column_stack([1 - ridge_proba, ridge_proba]), labels=[0, 1])
    )

    return {
        "skipped": False,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_dim": int(x.shape[1]),
        "prior_log_loss": prior,
        "logistic": {
            "model_log_loss": ll,
            "delta_model_minus_prior": ll - prior,
            "beats_prior": bool(ll < prior - EPS),
        },
        "ridge": {
            "model_log_loss": ridge_ll,
            "delta_model_minus_prior": ridge_ll - prior,
            "beats_prior": bool(ridge_ll < prior - EPS),
        },
        "positive_rate_test": float(np.mean(y_te)),
    }


def _association_report(
    encoded: dict[str, np.ndarray],
    y: np.ndarray,
    tok_feats: np.ndarray,
    raw_feats: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """MI / point-biserial style associations + neighboring-code stability."""
    rng = np.random.default_rng(seed)
    y = y.astype(np.int64)
    n = len(y)
    # subsample for MI speed if needed
    mi_n = min(n, 40000)
    idx = rng.choice(n, size=mi_n, replace=False) if n > mi_n else np.arange(n)

    s1_last = encoded["s1"][idx, -1]
    s2_last = encoded["s2"][idx, -1]
    y_sub = y[idx]

    mi_s1 = float(mutual_info_score(s1_last, y_sub))
    mi_s2 = float(mutual_info_score(s2_last, y_sub))

    # continuous MI on feature packs (sklearn)
    def _mi_pack(feats: np.ndarray, name: str, k: int = 16) -> dict[str, Any]:
        f = np.nan_to_num(feats[idx], nan=0.0, posinf=0.0, neginf=0.0)
        # cap dims for speed
        if f.shape[1] > 64:
            # take first 64 after random projection-ish: just slice
            f = f[:, :64]
        mi = mutual_info_classif(f, y_sub, discrete_features=False, random_state=seed, n_neighbors=3)
        return {
            "name": name,
            "n": int(mi_n),
            "n_features_used": int(f.shape[1]),
            "mi_mean": float(np.mean(mi)),
            "mi_max": float(np.max(mi)),
            "mi_sum": float(np.sum(mi)),
            "mi_top5": [float(v) for v in sorted(mi, reverse=True)[:5]],
        }

    # stability vs label
    eq1 = (encoded["s1"][:, 1:] == encoded["s1"][:, :-1]).mean(axis=1)
    eq2 = (encoded["s2"][:, 1:] == encoded["s2"][:, :-1]).mean(axis=1)
    both = (
        (encoded["s1"][:, 1:] == encoded["s1"][:, :-1])
        & (encoded["s2"][:, 1:] == encoded["s2"][:, :-1])
    ).mean(axis=1)

    def _pb(x: np.ndarray, yy: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        yy = np.asarray(yy, dtype=np.float64)
        if x.std() < 1e-12 or yy.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(x, yy)[0, 1])

    recon = encoded["recon_mse"]
    recon_ohlc = encoded["recon_mse_ohlc"]

    # pos vs neg mean stability
    pos = y == 1
    neg = ~pos

    return {
        "mi_last_s1_vs_y": mi_s1,
        "mi_last_s2_vs_y": mi_s2,
        "mi_tok_feature_pack": _mi_pack(tok_feats, "tokenizer_probe_features"),
        "mi_raw_feature_pack": _mi_pack(raw_feats, "raw_ohlcv_features"),
        "recon_mse_vs_y_corr": _pb(recon, y),
        "recon_mse_ohlc_vs_y_corr": _pb(recon_ohlc, y),
        "recon_mse_mean": float(np.mean(recon)),
        "recon_mse_std": float(np.std(recon)),
        "recon_mse_pos_mean": float(np.mean(recon[pos])) if pos.any() else None,
        "recon_mse_neg_mean": float(np.mean(recon[neg])) if neg.any() else None,
        "neighbor_code_stability": {
            "s1_stay_rate_mean": float(np.mean(eq1)),
            "s2_stay_rate_mean": float(np.mean(eq2)),
            "both_stay_rate_mean": float(np.mean(both)),
            "s1_stay_vs_y_corr": _pb(eq1, y),
            "s2_stay_vs_y_corr": _pb(eq2, y),
            "both_stay_vs_y_corr": _pb(both, y),
            "s1_stay_pos_mean": float(np.mean(eq1[pos])) if pos.any() else None,
            "s1_stay_neg_mean": float(np.mean(eq1[neg])) if neg.any() else None,
            "both_stay_pos_mean": float(np.mean(both[pos])) if pos.any() else None,
            "both_stay_neg_mean": float(np.mean(both[neg])) if neg.any() else None,
        },
        "mi_subsample_n": int(mi_n),
    }


def _verdict(
    tok_delta: float,
    raw_delta: float,
    g2_delta: float = G2_COMB_DELTA,
    sidecar_delta: float = SIDECAR_BEST_DELTA,
    assoc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether frozen tokenizer is the bottleneck for mfe10≥10%."""
    # More negative Δ = better.
    gap_raw_minus_tok = float(raw_delta - tok_delta)  # negative ⇒ raw better
    gap_tok_vs_g2 = float(tok_delta - g2_delta)
    gap_raw_vs_g2 = float(raw_delta - g2_delta)

    # Heuristics:
    # YES bottleneck: raw clearly beats tokenizer (raw Δ more negative by ≥0.01),
    #   and tokenizer probe is weak vs G2 / near prior.
    # NO bottleneck: tokenizer probe ≈ raw (within 0.01) — codes preserve the
    #   linearly available signal; deep sidecar failure is elsewhere.
    # UNCLEAR: otherwise.
    raw_beats_tok = (raw_delta < tok_delta - 0.01)
    tok_near_raw = abs(tok_delta - raw_delta) <= 0.01
    tok_weak = tok_delta > -0.01  # barely beats prior
    tok_competitive_with_g2 = tok_delta <= g2_delta + 0.01  # within 0.01 of G2

    mi_last = 0.0
    if assoc is not None:
        mi_last = max(
            float(assoc.get("mi_last_s1_vs_y") or 0.0),
            float(assoc.get("mi_last_s2_vs_y") or 0.0),
        )

    if raw_beats_tok and tok_weak:
        label = "yes"
        rationale = (
            "Raw OHLCV linear probe clearly beats frozen-tokenizer probe "
            f"(raw Δ={raw_delta:.6f} vs tok Δ={tok_delta:.6f}); "
            "codes discard linearly usable signal → tokenizer bottleneck likely."
        )
    elif tok_near_raw and (tok_competitive_with_g2 or tok_delta <= -0.015):
        label = "no"
        rationale = (
            "Frozen-tokenizer probe matches raw OHLCV "
            f"(tok Δ={tok_delta:.6f}, raw Δ={raw_delta:.6f}, G2 comb Δ={g2_delta:.6f}); "
            "codes preserve the cheap signal, so deep sidecar Δ=+0.008 is not "
            "explained by tokenizer information loss."
        )
    elif tok_near_raw and tok_weak:
        label = "unclear"
        rationale = (
            "Both tokenizer and raw probes are weak vs prior; bottleneck may be "
            "label/feature content rather than quantization alone."
        )
    else:
        label = "unclear"
        rationale = (
            f"Mixed: tok Δ={tok_delta:.6f}, raw Δ={raw_delta:.6f}, "
            f"G2 Δ={g2_delta:.6f}, sidecar Δ={sidecar_delta:.6f}."
        )

    return {
        "tokenizer_bottleneck": label,
        "rationale": rationale,
        "tok_delta": float(tok_delta),
        "raw_delta": float(raw_delta),
        "g2_handcrafted_comb_delta": float(g2_delta),
        "g2_handcrafted_base_delta": float(G2_BASE_DELTA),
        "sidecar_best_delta": float(sidecar_delta),
        "gap_raw_minus_tok": gap_raw_minus_tok,
        "gap_tok_vs_g2": gap_tok_vs_g2,
        "gap_raw_vs_g2": gap_raw_vs_g2,
        "raw_beats_tok_by_ge_0_01": bool(raw_beats_tok),
        "tok_near_raw_within_0_01": bool(tok_near_raw),
        "tok_weak_delta_gt_minus_0_01": bool(tok_weak),
        "mi_last_code_max": mi_last,
        "comparators": {
            "phase_g2_comb_delta": G2_COMB_DELTA,
            "phase_g2_base_delta": G2_BASE_DELTA,
            "deep_sidecar_best_delta": SIDECAR_BEST_DELTA,
            "gate_delta": GATE_DELTA,
        },
    }


def run_tokenizer_expressiveness(
    *,
    val_panel: Path,
    val_targets: Path,
    tokenizer_path: Path,
    seed: int = SEED,
    batch_size: int = BATCH,
    max_samples: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    t_wall0 = time.time()
    targets = pd.read_parquet(val_targets)
    # Reuse G2 label builder for combined/base features (handcrafted baseline refresh)
    packed = build_mfe_buy_labels(val_panel, targets)
    y_g2 = packed["buy_labels"]["buy_worth_mfe10pct"]
    base_g2 = packed["base_features"]
    comb_g2 = packed["combined_features"]

    windows = _collect_normalized_histories(val_panel, targets)
    assert np.array_equal(windows["y"], y_g2)

    if max_samples is not None and windows["n"] > max_samples:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(windows["n"], size=max_samples, replace=False))
        for key in ("raw", "norm", "mfe10", "y"):
            windows[key] = windows[key][keep]
        windows["n"] = int(len(windows["y"]))
        windows["pos_rate"] = float(np.mean(windows["y"]))
        base_g2 = base_g2[keep]
        comb_g2 = comb_g2[keep]
        y_g2 = y_g2[keep]
        print(f"[subsample] n={windows['n']}", flush=True)

    y = windows["y"]
    device_t = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device_t}", flush=True)
    tokenizer = FrozenKronosTokenizer.from_pretrained(tokenizer_path, device=device_t)

    encoded = _encode_tokenizer(
        tokenizer, windows["norm"], batch_size=batch_size, device=device_t
    )

    # --- Feature packs ---
    stability = _code_stability_features(encoded["s1"], encoded["s2"])
    hist_codes = _code_histogram_features(encoded["s1"], encoded["s2"], n_bins=32)
    bit_pack = np.concatenate(
        [encoded["bit_mean"], encoded["bit_std"], encoded["bit_last"]], axis=1
    ).astype(np.float64)
    recon_pack = np.column_stack(
        [
            encoded["recon_mse"],
            encoded["recon_mse_ohlc"],
            np.log1p(encoded["recon_mse"]),
            np.log1p(encoded["recon_mse_ohlc"]),
        ]
    ).astype(np.float64)

    tok_bits_only = bit_pack
    tok_codes_summary = np.concatenate([stability, hist_codes, recon_pack], axis=1)
    tok_full = np.concatenate([bit_pack, stability, hist_codes, recon_pack], axis=1)

    raw_summary = summarize_history_windows(windows["raw"]).astype(np.float64)
    # Last 10 bars flattened (bypass tokenizer) — cheap local shape
    last10 = windows["raw"][:, -10:, :].reshape(windows["n"], -1).astype(np.float64)
    # Per-window z-scored last10 from already-normalized hist
    last10_norm = windows["norm"][:, -10:, :].reshape(windows["n"], -1).astype(np.float64)
    raw_full = np.concatenate([raw_summary, last10_norm], axis=1)

    print("[probe] fitting logistics...", flush=True)
    probes = {
        "tok_bit_embed": _fit_eval_logistic_ridge(tok_bits_only, y, seed=seed),
        "tok_codes_summary": _fit_eval_logistic_ridge(tok_codes_summary, y, seed=seed),
        "tok_full": _fit_eval_logistic_ridge(tok_full, y, seed=seed),
        "raw_summary": _fit_eval_logistic_ridge(raw_summary, y, seed=seed),
        "raw_last10_norm_flat": _fit_eval_logistic_ridge(last10_norm, y, seed=seed),
        "raw_full": _fit_eval_logistic_ridge(raw_full, y, seed=seed),
        # refresh G2-style handcrafted on same split/seed for sanity
        "g2_base_refresh": _fit_eval_logistic(base_g2, y, seed=seed),
        "g2_comb_refresh": _fit_eval_logistic(comb_g2, y, seed=seed),
    }

    assoc = _association_report(
        encoded, y, tok_full, raw_full, seed=seed
    )

    def _best_delta(block: dict[str, Any]) -> float:
        if block.get("skipped"):
            return float("nan")
        if "logistic" in block:
            return float(block["logistic"]["delta_model_minus_prior"])
        return float(block["delta_model_minus_prior"])

    tok_delta = _best_delta(probes["tok_full"])
    # also consider best among tok packs
    tok_deltas = {
        k: _best_delta(probes[k])
        for k in ("tok_bit_embed", "tok_codes_summary", "tok_full")
    }
    tok_best_name = min(tok_deltas, key=lambda k: tok_deltas[k])
    tok_best_delta = tok_deltas[tok_best_name]

    raw_deltas = {
        k: _best_delta(probes[k])
        for k in ("raw_summary", "raw_last10_norm_flat", "raw_full")
    }
    raw_best_name = min(raw_deltas, key=lambda k: raw_deltas[k])
    raw_best_delta = raw_deltas[raw_best_name]

    verdict = _verdict(tok_best_delta, raw_best_delta, assoc=assoc)

    payload: dict[str, Any] = {
        "phase": "tokenizer_expressiveness_mfe10",
        "n_samples": int(windows["n"]),
        "pos_rate": float(windows["pos_rate"]),
        "seed": int(seed),
        "threshold": float(PROFIT_THRESHOLD),
        "definition": (
            f"y = 1{{mfe10 >= {PROFIT_THRESHOLD}}} where "
            "mfe10 = max(high[T+1:T+10]) / close[T] - 1"
        ),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_s1_bits": int(tokenizer.s1_bits) if hasattr(tokenizer, "s1_bits") else 10,
        "tokenizer_s2_vocab": int(tokenizer.s2_vocab_size),
        "tokenizer_s1_vocab": int(tokenizer.s1_vocab_size),
        "device": str(device_t),
        "encode_seconds": encoded["encode_seconds"],
        "wall_seconds": float(time.time() - t_wall0),
        "feature_dims": {
            "tok_bit_embed": int(tok_bits_only.shape[1]),
            "tok_codes_summary": int(tok_codes_summary.shape[1]),
            "tok_full": int(tok_full.shape[1]),
            "raw_summary": int(raw_summary.shape[1]),
            "raw_last10_norm_flat": int(last10_norm.shape[1]),
            "raw_full": int(raw_full.shape[1]),
            "g2_base": int(base_g2.shape[1]),
            "g2_comb": int(comb_g2.shape[1]),
        },
        "probes": probes,
        "tok_deltas": tok_deltas,
        "raw_deltas": raw_deltas,
        "tok_best": {"name": tok_best_name, "delta": tok_best_delta},
        "raw_best": {"name": raw_best_name, "delta": raw_best_delta},
        "association": assoc,
        "comparators": {
            "phase_g2_handcrafted_comb_delta": G2_COMB_DELTA,
            "phase_g2_handcrafted_base_delta": G2_BASE_DELTA,
            "deep_sidecar_best_delta": SIDECAR_BEST_DELTA,
            "gate_delta_le": GATE_DELTA,
            "g2_base_refresh_delta": probes["g2_base_refresh"].get(
                "delta_model_minus_prior"
            ),
            "g2_comb_refresh_delta": probes["g2_comb_refresh"].get(
                "delta_model_minus_prior"
            ),
        },
        "decision": verdict,
        "notes": [
            "Val panel only (train panel not on disk); 75/25 stratified split seed=20261001.",
            "Tokenizer frozen NeoQuasar/Kronos-Tokenizer-base local weights.",
            "No R2 / Kaggle long train / TPU WIP.",
            "Deep sidecar comparator from kairos_mfe10_sidecar_short_results (best Δ=+0.008022).",
        ],
    }
    return payload


def write_cn_memo(payload: dict[str, Any], out_md: Path) -> str:
    dec = payload["decision"]
    probes = payload["probes"]
    assoc = payload["association"]
    stab = assoc["neighbor_code_stability"]
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
    except Exception:
        now = time.strftime("%Y-%m-%d %H:%M CST")

    def _d(block: dict[str, Any]) -> str:
        if block.get("skipped"):
            return "skipped"
        if "logistic" in block:
            return f"{block['logistic']['delta_model_minus_prior']:.6f}"
        return f"{block['delta_model_minus_prior']:.6f}"

    lines = [
        "# Kairos Tokenizer 表达力消融（mfe10≥10%）",
        "",
        f"日期：{now}。",
        "",
        "## 一句话结论",
        "",
        f"**Tokenizer bottleneck = `{dec['tokenizer_bottleneck']}`。** "
        f"最佳 tok Δ=`{payload['tok_best']['delta']:.6f}`（{payload['tok_best']['name']}）；"
        f"最佳 raw Δ=`{payload['raw_best']['delta']:.6f}`（{payload['raw_best']['name']}）；"
        f"对照 Phase G2 comb Δ≈`{G2_COMB_DELTA:.6f}`，失败 deep sidecar Δ≈`+{SIDECAR_BEST_DELTA:.6f}`。",
        "",
        f"- 理由：{dec['rationale']}",
        "",
        "## 设定",
        "",
        f"- n_samples=`{payload['n_samples']}`，pos_rate=`{payload['pos_rate']:.4f}`，seed=`{payload['seed']}`",
        f"- 定义：`{payload['definition']}`",
        f"- tokenizer：`{payload['tokenizer_path']}`（s1/s2 vocab={payload['tokenizer_s1_vocab']}/{payload['tokenizer_s2_vocab']}）",
        f"- device=`{payload['device']}`；encode_seconds=`{payload['encode_seconds']:.1f}`；wall=`{payload['wall_seconds']:.1f}`s",
        "- **未**重启 R2 / **未** Kaggle 长训 / **未**动 TPU WIP",
        "",
        "## 1) 冻结 Tokenizer 线性探针",
        "",
        "| Pack | dim | prior LL | logistic Δ | ridge Δ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("tok_bit_embed", "tok_codes_summary", "tok_full"):
        b = probes[name]
        if b.get("skipped"):
            lines.append(f"| {name} | — | — | skipped | skipped |")
            continue
        lines.append(
            f"| {name} | {b['feature_dim']} | {b['prior_log_loss']:.6f} | "
            f"{b['logistic']['delta_model_minus_prior']:.6f} | "
            f"{b['ridge']['delta_model_minus_prior']:.6f} |"
        )

    lines += [
        "",
        "## 2) Raw OHLCV（绕过 Tokenizer）",
        "",
        "| Pack | dim | logistic Δ | ridge Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("raw_summary", "raw_last10_norm_flat", "raw_full"):
        b = probes[name]
        lines.append(
            f"| {name} | {b['feature_dim']} | "
            f"{b['logistic']['delta_model_minus_prior']:.6f} | "
            f"{b['ridge']['delta_model_minus_prior']:.6f} |"
        )

    g2b = probes["g2_base_refresh"]
    g2c = probes["g2_comb_refresh"]
    lines += [
        "",
        "## 3) 对照刷新（同 seed 切分）",
        "",
        f"- G2 base refresh Δ=`{g2b.get('delta_model_minus_prior')}`",
        f"- G2 comb refresh Δ=`{g2c.get('delta_model_minus_prior')}`",
        f"- 记录对照：G2 comb=`{G2_COMB_DELTA}`；sidecar best=`+{SIDECAR_BEST_DELTA}`",
        "",
        "## 4) 互信息 / 码稳定性",
        "",
        f"- MI(last s1, y)=`{assoc['mi_last_s1_vs_y']:.6f}`；MI(last s2, y)=`{assoc['mi_last_s2_vs_y']:.6f}`",
        f"- tok pack MI mean/max=`{assoc['mi_tok_feature_pack']['mi_mean']:.6f}`/`{assoc['mi_tok_feature_pack']['mi_max']:.6f}`",
        f"- raw pack MI mean/max=`{assoc['mi_raw_feature_pack']['mi_mean']:.6f}`/`{assoc['mi_raw_feature_pack']['mi_max']:.6f}`",
        f"- recon_mse vs y corr=`{assoc['recon_mse_vs_y_corr']:.6f}`（mean=`{assoc['recon_mse_mean']:.6f}`）",
        f"- 邻码停留率 s1/s2/both=`{stab['s1_stay_rate_mean']:.4f}`/`{stab['s2_stay_rate_mean']:.4f}`/`{stab['both_stay_rate_mean']:.4f}`",
        f"- 停留率 vs y corr s1/both=`{stab['s1_stay_vs_y_corr']:.6f}`/`{stab['both_stay_vs_y_corr']:.6f}`",
        "",
        "## 5) 判决",
        "",
        f"| 项 | 值 |",
        f"| --- | --- |",
        f"| tokenizer_bottleneck | **{dec['tokenizer_bottleneck']}** |",
        f"| tok_best Δ | {payload['tok_best']['delta']:.6f} ({payload['tok_best']['name']}) |",
        f"| raw_best Δ | {payload['raw_best']['delta']:.6f} ({payload['raw_best']['name']}) |",
        f"| gap(raw−tok) | {dec['gap_raw_minus_tok']:.6f} |",
        f"| vs G2 comb | tok gap={dec['gap_tok_vs_g2']:.6f}；raw gap={dec['gap_raw_vs_g2']:.6f} |",
        "",
        "### 建议",
        "",
    ]
    if dec["tokenizer_bottleneck"] == "yes":
        lines += [
            "1. Tokenizer 像瓶颈：优先考虑更强/可微调的离散化，或决策头直接吃 raw/连续表征。",
            "2. **不要**在同一冻结 tokenizer + ModernBERT 路径上加长 sidecar。",
            "3. 仍维持停 R2 八头长训。",
        ]
    elif dec["tokenizer_bottleneck"] == "no":
        lines += [
            "1. Tokenizer **不是**主瓶颈：冻结码已保住廉价线性信号。",
            "2. Deep sidecar 失败应查优化/容量/标签协议，而非先换 tokenizer。",
            "3. **不要**加长训；可回到特征/标签消融。",
        ]
    else:
        lines += [
            "1. 证据不足以下定论：tok/raw 差距或绝对 Δ 均不清晰。",
            "2. **不要**扩 sidecar 预算；可加小规模连续表征对照。",
            "3. 维持停 R2。",
        ]

    lines += [
        "",
        "## 用户要点（中文）",
        "",
        f"- **判决**：tokenizer bottleneck = `{dec['tokenizer_bottleneck']}`",
        f"- **tok 最佳 Δ**：`{payload['tok_best']['delta']:.6f}`（{payload['tok_best']['name']}）",
        f"- **raw 最佳 Δ**：`{payload['raw_best']['delta']:.6f}`（{payload['raw_best']['name']}）",
        f"- **对照**：G2 comb Δ≈`{G2_COMB_DELTA:.6f}`；deep sidecar Δ≈`+{SIDECAR_BEST_DELTA:.6f}`",
        f"- **邻码稳定性**：both stay≈`{stab['both_stay_rate_mean']:.4f}`；vs y corr=`{stab['both_stay_vs_y_corr']:.6f}`",
        f"- **下一步**：{'查 tokenizer/表征' if dec['tokenizer_bottleneck']=='yes' else ('别扩 sidecar，查优化/标签' if dec['tokenizer_bottleneck']=='no' else '证据不足，勿扩训')}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-panel", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "modernbert_finance/ablations/kairos_tokenizer_expressiveness_mfe10.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("modernbert_finance/kairos_tokenizer_expressiveness_mfe10_cn.md"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    payload = run_tokenizer_expressiveness(
        val_panel=args.val_panel,
        val_targets=args.val_targets,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_cn_memo(payload, args.out_md)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "decision": payload["decision"],
                "tok_best": payload["tok_best"],
                "raw_best": payload["raw_best"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
