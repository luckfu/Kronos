"""Pre-registered study: Non-linear cross-sectional Cosine-IC MLP on frozen C2 hidden states.

Trains strictly on val_hidden (2025-07-03..2026-07-02) with internal split
fit=2025-07-03..2026-04-30, embargo=2026-05-01..2026-05-15, val=2026-05-18..2026-07-02,
refits on 2025-07-03..2026-07-02, and evaluates on the 65-day OOS (2026-07-17..2026-10-23)
aligned with C2 (16-path T=0.65, p=0.80) while keeping 2026-10-24+ sealed.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import broad_regime_gate_study as brg  # noqa: E402
import few_names_backtest as fnb  # noqa: E402
import fusion_analysis as fa  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402

TRAIN_FIT_END = "2026-04-30"
TRAIN_VAL_START = "2026-05-18"
TRAIN_END = "2026-07-02"
OOS_START = "2026-07-17"
OOS_END = "2026-10-23"


def zscore_hidden_by_date(dates: np.ndarray, matrix: np.ndarray, clip_val: float = 5.0) -> np.ndarray:
    """Cross-sectionally z-score hidden vectors within each trading date."""
    out = np.empty_like(matrix, dtype=np.float32)
    dates_s = pd.Series(np.arange(len(dates)), index=dates)
    for _, idx in dates_s.groupby(level=0, sort=False):
        rows = idx.to_numpy()
        block = matrix[rows].astype(np.float32)
        mean = block.mean(axis=0, keepdims=True)
        std = block.std(axis=0, keepdims=True)
        z = (block - mean) / np.maximum(std, 1e-6)
        out[rows] = np.clip(z, -clip_val, clip_val)
    return out


def date_standardized_targets(dates: np.ndarray, ret10: np.ndarray) -> np.ndarray:
    """Within-date percentile rank target standardized to zero mean, unit norm/std per date."""
    df = pd.DataFrame({"d": dates, "r": ret10})
    ranked = df.groupby("d", sort=False)["r"].rank(pct=True, method="average").to_numpy(dtype=np.float32) - 0.5
    out = np.empty_like(ranked, dtype=np.float32)
    for _, idx in pd.Series(np.arange(len(dates)), index=dates).groupby(level=0, sort=False):
        rows = idx.to_numpy()
        v = ranked[rows]
        v = v - v.mean()
        out[rows] = v / max(float(v.std()), 1e-6)
    return out


def build_date_batches(dates: np.ndarray) -> list[np.ndarray]:
    """Return list of row-index arrays, one per unique date."""
    return [idx.to_numpy() for _, idx in pd.Series(np.arange(len(dates)), index=dates).groupby(level=0, sort=False)]


class CosineICResidualMLP(nn.Module):
    """Linear skip + regularized hidden bottleneck for cross-sectional ranking."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.skip = nn.Linear(in_dim, 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.skip(x) + self.net(x)).squeeze(-1)


def cosine_ic_loss(pred: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity across a single date cross-section + mild scale MSE."""
    p = pred - pred.mean()
    t = target_z - target_z.mean()
    cos = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
    p_norm = p / (p.std(unbiased=False) + 1e-6)
    return (1.0 - cos) + 0.1 * F.mse_loss(p_norm, t)


def daily_pearson_on_z(dates: np.ndarray, pred: np.ndarray, target_z: np.ndarray) -> float:
    """Mean per-date Pearson correlation against ranked target (= Spearman Rank IC)."""
    corrs = []
    for rows in build_date_batches(dates):
        p = pred[rows]
        t = target_z[rows]
        pc = p - p.mean()
        tc = t - t.mean()
        denom = float(np.sqrt((pc ** 2).sum() * (tc ** 2).sum()))
        if denom > 1e-9:
            corrs.append(float((pc * tc).sum() / denom))
    return float(np.mean(corrs))


def fit_ridge_benchmark(x: np.ndarray, dates: np.ndarray, ret10: np.ndarray) -> tuple[np.ndarray, dict]:
    """Select Ridge alpha on internal val (2026-05-18..2026-07-02) and refit on all val_hidden."""
    y_rank = pd.Series(ret10).groupby(dates).rank(pct=True).to_numpy(dtype=np.float64) - 0.5
    fit_mask = dates <= TRAIN_FIT_END
    val_mask = (dates >= TRAIN_VAL_START) & (dates <= TRAIN_END)

    best_alpha, best_ic = None, -1e9
    grid = []
    xf = x[fit_mask].astype(np.float64)
    yf = y_rank[fit_mask]
    gram = xf.T @ xf
    rhs = xf.T @ yf
    xv = x[val_mask].astype(np.float64)
    for alpha in (1e4, 1e5, 3e5):
        w = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), rhs)
        ic = daily_pearson_on_z(dates[val_mask], xv @ w, date_standardized_targets(dates[val_mask], ret10[val_mask]))
        grid.append({"alpha": alpha, "val_ic": ic})
        if ic > best_ic:
            best_ic, best_alpha = ic, alpha
    xf_all = x.astype(np.float64)
    w_full = np.linalg.solve(xf_all.T @ xf_all + best_alpha * np.eye(x.shape[1]), xf_all.T @ y_rank)
    return w_full, {"selected_alpha": best_alpha, "val_ic": best_ic, "grid": grid}


def train_mlp_fold(
    x_train: np.ndarray,
    batches_train: list[np.ndarray],
    y_train_z: np.ndarray,
    x_val: np.ndarray | None,
    dates_val: np.ndarray | None,
    y_val_z: np.ndarray | None,
    weight_decay: float,
    dropout: float,
    epochs: int,
    seed: int,
) -> tuple[CosineICResidualMLP, list[float]]:
    """Train one CosineICResidualMLP on CPU with date-grouped batches."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CosineICResidualMLP(in_dim=x_train.shape[1], hidden_dim=128, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train_z)
    val_ics: list[float] = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(batches_train))
        for bi in order:
            idx = torch.from_numpy(batches_train[bi])
            opt.zero_grad(set_to_none=True)
            pred = model(xt[idx])
            loss = cosine_ic_loss(pred, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if x_val is not None and dates_val is not None and y_val_z is not None:
            model.eval()
            with torch.inference_mode():
                vp = model(torch.from_numpy(x_val)).numpy()
            val_ics.append(daily_pearson_on_z(dates_val, vp, y_val_z))
    return model, val_ics


def fit_and_select_mlp(x: np.ndarray, dates: np.ndarray, ret10: np.ndarray) -> tuple[list[CosineICResidualMLP], dict]:
    """Select (weight_decay, dropout, epoch) strictly pre-OOS, then refit 3-seed ensemble."""
    fit_mask = dates <= TRAIN_FIT_END
    val_mask = (dates >= TRAIN_VAL_START) & (dates <= TRAIN_END)
    y_fit_z = date_standardized_targets(dates[fit_mask], ret10[fit_mask])
    y_val_z = date_standardized_targets(dates[val_mask], ret10[val_mask])
    batches_fit = build_date_batches(dates[fit_mask])
    xf = x[fit_mask]
    xv = x[val_mask]
    dv = dates[val_mask]

    max_epochs = 20
    candidates = []
    best = None
    for wd in (1e-2, 5e-2):
        for drop in (0.3, 0.5):
            # Average validation curve across 2 seeds during selection for stability
            curves = []
            for s in (20261101, 20261102):
                _, curve = train_mlp_fold(xf, batches_fit, y_fit_z, xv, dv, y_val_z, wd, drop, max_epochs, s)
                curves.append(curve)
            mean_curve = np.mean(curves, axis=0)
            ep_idx = int(np.argmax(mean_curve))
            ic = float(mean_curve[ep_idx])
            rec = {"weight_decay": wd, "dropout": drop, "best_epoch": ep_idx + 1, "val_ic": ic}
            candidates.append(rec)
            if best is None or ic > best["val_ic"]:
                best = rec

    y_all_z = date_standardized_targets(dates, ret10)
    batches_all = build_date_batches(dates)
    models = []
    for seed in (20261101, 20261102, 20261103):
        m, _ = train_mlp_fold(
            x, batches_all, y_all_z, None, None, None,
            best["weight_decay"], best["dropout"], best["best_epoch"], seed,
        )
        m.eval()
        models.append(m)
    return models, {"selected": best, "candidates": candidates, "fit_rows": int(fit_mask.sum()), "val_rows": int(val_mask.sum())}


def load_val_hidden(hidden_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-OOS validation hidden states strictly <= 2026-07-02."""
    parts = [np.load(p, allow_pickle=True) for p in sorted(glob.glob(str(hidden_dir / "val_hidden_*.npz")))]
    if not parts:
        raise FileNotFoundError(f"No val_hidden_*.npz under {hidden_dir}")
    dates = np.concatenate([p["asof_date"] for p in parts]).astype(str)
    max_date = max(set(dates))
    assert max_date <= TRAIN_END, f"Validation hidden exceeds {TRAIN_END}: {max_date}"
    ret10 = np.concatenate([p["ret10"] for p in parts]).astype(np.float64)
    hidden = np.concatenate([p["hidden"] for p in parts]).astype(np.float32)
    return zscore_hidden_by_date(dates, hidden), dates, ret10


def score_oos_hidden(hidden_dir: Path, c2_dir: Path, ridge_w: np.ndarray, mlp_models: list[CosineICResidualMLP]) -> pd.DataFrame:
    """Align 65d oos_hidden_*.npz to C2 p16 shards and score RIDGE_BASE & MLP_COSIC."""
    c2 = fa.load_model(c2_dir)
    c2["asof_date"] = c2["asof_date"].astype(str)
    c2 = c2[(c2["asof_date"] >= OOS_START) & (c2["asof_date"] <= OOS_END)].reset_index(drop=True)
    assert c2["asof_date"].max() <= OOS_END, "Sealed window violation!"

    parts = [np.load(p, allow_pickle=True) for p in sorted(glob.glob(str(hidden_dir / "oos_hidden_*.npz")))]
    ids = np.concatenate([p["identity"] for p in parts])
    hidden = np.concatenate([p["hidden"] for p in parts]).astype(np.float32)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    if not pos.index.is_unique or not c2["identity"].isin(pos.index).all():
        raise RuntimeError("OOS hidden states do not cleanly cover C2 shards")

    aligned_h = hidden[pos.loc[c2["identity"]].to_numpy()]
    xz = zscore_hidden_by_date(c2["asof_date"].to_numpy(), aligned_h)
    c2["p_ridge"] = xz.astype(np.float64) @ ridge_w
    with torch.inference_mode():
        preds = [m(torch.from_numpy(xz)).numpy() for m in mlp_models]
    c2["p_mlp"] = np.mean(preds, axis=0)
    return c2


def evaluate_all(frame: pd.DataFrame, raw_path: Path, non_pv_dir: Path | None = None) -> dict:
    """Evaluate Rank IC (overall + A/B splits + neutralized + heads) and k=10/20 portfolios."""
    frame = frame.copy()
    frame["r_c2"] = fa.ranks(frame, "predicted_return_d10")
    frame["r_ridge"] = fa.ranks(frame, "p_ridge")
    frame["r_mlp"] = fa.ranks(frame, "p_mlp")
    frame["p_fuse_ridge"] = 0.5 * frame["r_c2"] + 0.5 * frame["r_ridge"]
    frame["p_fuse_mlp"] = 0.5 * frame["r_c2"] + 0.5 * frame["r_mlp"]

    frame["rev10"] = fa.reversal_10d(raw_path, frame)
    cols = {
        "C2": "predicted_return_d10",
        "RIDGE_BASE": "p_ridge",
        "MLP_COSIC": "p_mlp",
        "FUSE_RIDGE_C2": "p_fuse_ridge",
        "FUSE_MLP_C2": "p_fuse_mlp",
    }
    raw_df = pd.read_csv(raw_path, dtype={"symbol": str})
    raw_df["date"] = raw_df["date"].astype(str)
    traded = raw_df[raw_df["volume"] > 0].sort_values(["symbol", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("symbol")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    liq_map = traded[["symbol", "date", "med_amt_20d"]].rename(columns={"date": "asof_date"})
    frame = frame.merge(liq_map, on=["symbol", "asof_date"], how="left")
    liq = frame[frame["med_amt_20d"] >= 5e7].copy()

    base_ic = fa.daily_ic(frame, cols["C2"])
    # Residualized neutralized target
    frame["resid_actual"] = fa.residualize(frame, "actual_return_d10", ["size_decile", "rev10"], sector=True)

    seg_a = frame["asof_date"] <= "2026-09-10"
    seg_b = frame["asof_date"] >= "2026-09-15"
    cross_section = {}
    for name, col in cols.items():
        ic_s = fa.daily_ic(frame, col)
        frame[f"res_{name}"] = fa.residualize(frame, col, ["size_decile", "rev10"], sector=True)
        nic_s = frame.groupby("asof_date", sort=False).apply(
            lambda g: g[f"res_{name}"].corr(g["resid_actual"], method="spearman"), include_groups=False
        )
        # Liquid top10/top20 mean excess vs frame universe on same dates
        univ_daily = frame.groupby("asof_date", sort=False)["actual_return_d10"].mean()
        t10 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(10)
        t20 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(20)
        ex10 = float((t10.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean())
        ex20 = float((t20.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean())
        entry = {
            **fa.ic_stats(ic_s),
            "ic_seg_a": float(ic_s.loc[ic_s.index <= "2026-09-10"].mean()),
            "ic_seg_b": float(ic_s.loc[ic_s.index >= "2026-09-15"].mean()),
            "neutral_ic": float(nic_s.mean()),
            "neutral_icir": float(nic_s.mean() / nic_s.std(ddof=1)),
            "top10pct_excess": fa.top_bottom(frame, col, 0.1),
            "liq_top10_excess": ex10,
            "liq_top20_excess": ex20,
        }
        if name != "C2":
            entry["paired_vs_C2"] = fa.paired_bootstrap(base_ic, ic_s, seed=20261115)
        cross_section[name] = entry

    # Rank correlation with C2
    c2_mlp_corr = float(frame.groupby("asof_date").apply(lambda g: g["r_c2"].corr(g["r_mlp"]), include_groups=False).mean())
    c2_ridge_corr = float(frame.groupby("asof_date").apply(lambda g: g["r_c2"].corr(g["r_ridge"]), include_groups=False).mean())

    # Part B: few-name executable books (k=10, 20; step=10; 10 phases)
    market = sub.Market(raw_path)
    last_day = len(market.dates) - 1
    signal_dates = sorted(frame["asof_date"].unique())
    signal_days = [market.date_index[d] for d in signal_dates]
    phases = [signal_days[p::10] for p in range(10)]
    score_mats = {
        name: market.score_matrix(frame[["symbol", "asof_date", col]].rename(columns={col: "predicted_return_d10"}))
        for name, col in cols.items()
    }
    bench = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases]
    dummy_gate = np.full(len(market.dates), 2, dtype=int)
    books = {}
    c2_daily_by_k = {}
    for k in (10, 20):
        for name in ("C2", "RIDGE_BASE", "MLP_COSIC", "FUSE_RIDGE_C2", "FUSE_MLP_C2"):
            runs = [
                nrs.run_gated_slots(market, nrs.gated_plan(market, score_mats[name], k, dummy_gate, False), days, k, last_day)
                for days in phases
            ]
            nets = [r["daily"]["ret"] for r in runs]
            if name == "C2":
                c2_daily_by_k[k] = nets
            excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
            drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
            diff_univ = pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()
            diff_c2 = pd.concat([r - c for r, c in zip(nets, c2_daily_by_k[k])], axis=1).mean(axis=1).to_numpy()
            books[f"{name}_k{k}"] = {
                "k": k, "model": name,
                "mean_net": float(np.mean([fnb.total(r) for r in nets])),
                "mean_universe_net": float(np.mean([fnb.total(b) for b in bench])),
                "mean_net_excess": float(np.mean(excess)),
                "beat_pool_phases": int(sum(x > 0 for x in excess)),
                "phases": len(phases),
                "mean_max_drawdown": float(np.mean(drawdowns)),
                "calmar_like": float(np.mean([fnb.total(r) for r in nets]) / abs(float(np.mean(drawdowns)))) if abs(float(np.mean(drawdowns))) > 1e-9 else 0.0,
                "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
                "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                "vs_universe_ci95": sub.block_ci(diff_univ),
                "vs_c2_ci95": sub.block_ci(diff_c2),
            }
    if non_pv_dir is not None:
        regime = brg.build_broad_regime_table(raw_path, non_pv_dir)
        for gname in ("G_EW", "G_BREADTH"):
            g_arr = np.full(len(market.dates), 2, dtype=int)
            for d, idx in market.date_index.items():
                if d in regime.index:
                    g_arr[idx] = int(regime.loc[d, gname])
            for k in (10, 20):
                runs = [
                    nrs.run_gated_slots(market, nrs.gated_plan(market, score_mats["FUSE_RIDGE_C2"], k, g_arr, True), days, k, last_day)
                    for days in phases
                ]
                nets = [r["daily"]["ret"] for r in runs]
                excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
                drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
                mean_net = float(np.mean([fnb.total(r) for r in nets]))
                mean_dd = float(np.mean(drawdowns))
                diff_univ = pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()
                diff_c2 = pd.concat([r - c for r, c in zip(nets, c2_daily_by_k[k])], axis=1).mean(axis=1).to_numpy()
                books[f"FUSE_RIDGE_C2_{gname}_k{k}"] = {
                    "k": k, "model": f"FUSE_RIDGE_C2+{gname}",
                    "mean_net": mean_net,
                    "mean_universe_net": float(np.mean([fnb.total(b) for b in bench])),
                    "mean_net_excess": float(np.mean(excess)),
                    "beat_pool_phases": int(sum(x > 0 for x in excess)),
                    "phases": len(phases),
                    "mean_max_drawdown": mean_dd,
                    "calmar_like": float(mean_net / abs(mean_dd)) if abs(mean_dd) > 1e-9 else 0.0,
                    "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
                    "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                    "vs_universe_ci95": sub.block_ci(diff_univ),
                    "vs_c2_ci95": sub.block_ci(diff_c2),
                }
    return {
        "c2_mlp_rank_corr": c2_mlp_corr,
        "c2_ridge_rank_corr": c2_ridge_corr,
        "seg_a_dates": int(seg_a.loc[seg_a].shape[0] > 0 and frame.loc[seg_a, "asof_date"].nunique()),
        "seg_b_dates": int(frame.loc[seg_b, "asof_date"].nunique()),
        "cross_section": cross_section,
        "books": books,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dir", default="scratch/c2_ceiling/hidden")
    parser.add_argument("--c2-dir", default="scratch/non_pv/c2_p16_65d")
    parser.add_argument("--raw", default="scratch/oos65/combined_raw.csv")
    parser.add_argument("--nonpv-dir", default="scratch/non_pv")
    parser.add_argument("--save-ridge", default="finetune/c2_ceiling/frozen/head_val_alpha1e4.npy")
    parser.add_argument("--out", default="finetune/reports/20261115_hidden_ranknet_results.json")
    args = parser.parse_args()
    torch.set_num_threads(1)

    x_val, dates_val, ret10_val = load_val_hidden(ROOT / args.hidden_dir)
    ridge_w, ridge_info = fit_ridge_benchmark(x_val, dates_val, ret10_val)
    if args.save_ridge:
        rw_path = ROOT / args.save_ridge
        rw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rw_path, ridge_w.astype(np.float32))
    mlp_models, mlp_info = fit_and_select_mlp(x_val, dates_val, ret10_val)
    frame = score_oos_hidden(ROOT / args.hidden_dir, ROOT / args.c2_dir, ridge_w, mlp_models)
    eval_res = evaluate_all(frame, ROOT / args.raw, ROOT / args.nonpv_dir)

    result = {"ridge_train": ridge_info, "mlp_train": mlp_info, **eval_res}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
