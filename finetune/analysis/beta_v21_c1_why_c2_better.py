"""Zero-GPU decomposition: why Small C2 beats Beta v2.1 Best@475 on the sealed 18d OOS.

Reads ONLY already-saved per-sample predictions (no decoding, no training):
  * sealed OOS kronos_beta_v2_time_oos_through_20260903 (RETIRED for selection):
      Best@475 / Seg9 shards, Seg155 prod/rank results, C2 prod/rank predictions
      -> descriptive analysis only, nothing here selects a post-processing.
  * validation temporal_symbol_validation_v1 (24-date subsample, 12,256 windows):
      Best@475 / Seg155 / Seg8 per-sample predictions
      -> the only place where post-processing (neutralization etc.) is compared.
Conditioning/style features (sector, size_percentile, lookback momentum/vol/liquidity)
are rebuilt from the dataset panels via each record's (symbol, start_index).

Score = predicted_return_d10 (mean over N AR samples of close_d10/last_close-1).
Label = package return_10d. Main metric = daily Spearman rank IC.

Usage (paths are args; defaults are the box paths used on 2026-10-07):
  python3 finetune/analysis/beta_v21_c1_why_c2_better.py --out-json finetune/reports/beta_v21_c1_why_c2_better_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOOKBACK, PREDICT, WINDOW = 120, 10, 130
TAIL = ["2026-09-01", "2026-09-02", "2026-09-03"]
FEATS = ["size_pct", "mom5", "mom20", "mom60", "mom120", "vol20", "liq20"]

P = "/workspace/kronos_patrol_out"
DEFAULTS = dict(
    seg9_shards=f"{P}/oos_seg9_1007_0345/output/beta_v2_1_c1_gen_return_oos_pilot_seg9/shards",
    seg155_results=f"{P}/genret_A_final_1006_1057/output/beta_v2_1_c1_gen_return_oos/results",
    c2_dir="/workspace/kaggle_c2_18d_alpha_oos/kronos_c2_18d_alpha_oos",
    oos_package="/workspace/scratch/why_c2/data",
    val_results=f"{P}/valgenic_final_1006_1425/output/beta_v2_1_c1_val_gen_ic/results",
    val_records=f"{P}/valgenic_final_1006_1425/output/beta_v2_1_c1_val_gen_ic/val_subsample_records.json",
    val_data="/workspace/kairos/ref/val_data.pkl",
)


# ----------------------------------------------------------------------------- utils
def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def gauss_rank(x):
    r = stats.rankdata(x)
    return stats.norm.ppf((r - 0.5) / len(r))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def paired(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    return dict(mean=d.mean(), se=se, t=d.mean() / se if se > 0 else np.nan, wins=int((d > 0).sum()), n=n)


def merge_small(labels, min_n=5):
    """sector code -> fine (CSRC 2-digit); merge <min_n groups into letter section, then OTHER."""
    s = pd.Series(labels).astype(str)
    cnt = s.map(s.value_counts())
    letter = s.str[0]
    out = s.where(cnt >= min_n, letter)
    cnt2 = out.map(out.value_counts())
    return out.where(cnt2 >= min_n, "OTHER").values


def coarse_sector(labels, min_n=5):
    s = pd.Series(labels).astype(str).str[0]
    cnt = s.map(s.value_counts())
    return s.where(cnt >= min_n, "OTHER").values


def residualize(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def dummies(lab):
    u, inv = np.unique(lab, return_inverse=True)
    D = np.zeros((len(lab), len(u)))
    D[np.arange(len(lab)), inv] = 1
    return [D[:, j] for j in range(1, len(u))]  # drop first (intercept added)


# ------------------------------------------------------------------- feature builder
def build_features(panel: dict, records: pd.DataFrame) -> pd.DataFrame:
    """Per-record features at asof (window position LOOKBACK-1), lookback-only info."""
    out = []
    for sym, g in records.groupby("symbol"):
        f = panel[sym]
        c = f["close"].astype(float).values
        lr = np.diff(np.log(np.maximum(c, 1e-9)), prepend=np.nan)
        amt = f["amount"].astype(float).values
        vol20 = pd.Series(lr).rolling(20).std().values
        liq20 = np.log1p(pd.Series(amt).rolling(20).mean().values)
        sp = f["size_percentile"].astype(float).values
        sec = f["sector"].astype(str).values
        for start, asof, ret in zip(g.start_index.values, g.asof_date.values, g.ret_pkg.values):
            a = int(start) + LOOKBACK - 1
            row = dict(symbol=sym, asof_date=asof, sector=sec[a], size_pct=sp[a],
                       mom5=c[a] / c[a - 5] - 1, mom20=c[a] / c[a - 20] - 1, mom60=c[a] / c[a - 60] - 1,
                       mom120=c[a] / c[int(start)] - 1, vol20=vol20[a], liq20=liq20[a],
                       ret_check=c[a + PREDICT] / c[a] - 1, ret_pkg=ret)
            out.append(row)
    df = pd.DataFrame(out)
    df["size_pct"] = df.size_pct.fillna(0.5)
    return df


# -------------------------------------------------------------------------- loaders
def load_oos(a):
    keys = ["symbol", "asof_date"]
    def rd(path, label, score="predicted_return_d10"):
        cols = keys + [score, "predicted_terminal_dispersion", "sample_count"] if "c2_" not in label else \
            keys + [score, "predicted_terminal_dispersion", "sample_count"]
        f = pd.concat([pd.read_csv(p, usecols=lambda c: c in set(cols + ["return_10d", "utility"])) for p in path])
        f = f.rename(columns={score: f"s_{label}", "predicted_terminal_dispersion": f"disp_{label}", "sample_count": f"n_{label}"})
        return f
    b = rd(sorted(glob.glob(a.seg9_shards + "/seg000_beta_v21_release_best475__prod_t065_p80_n5_*.csv.gz")), "b475")
    s9 = rd(sorted(glob.glob(a.seg9_shards + "/pilot_seg009__prod_t065_p80_n5_*.csv.gz")), "seg9")
    s155 = rd([a.seg155_results + "/seg155_forecast_best_prod_t065_p80_n5_predictions.csv.gz"], "seg155")
    s155r = rd([a.seg155_results + "/seg155_forecast_best_rank_t060_p90_n16_predictions.csv.gz"], "seg155r")
    c2p = rd([a.c2_dir + "/predictions_prod_t065_p80_n5.csv.gz"], "c2p", "predicted_return_10d")
    c2r = rd([a.c2_dir + "/predictions_rank_t060_p90_n16.csv.gz"], "c2r", "predicted_return_10d")
    m = b.rename(columns={"return_10d": "ret"})
    for f in [s9, s155, s155r, c2p, c2r]:
        m = m.merge(f.drop(columns=[c for c in ["return_10d", "utility"] if c in f]), on=keys, how="inner")
    assert len(m) == len(b) == 92751, (len(m), len(b))
    recs = pd.read_json(Path(a.oos_package) / "evaluation_samples.jsonl", lines=True)
    recs = recs[recs.set.isin(["incremental_future_all", "future_all"])]
    recs = recs[(recs.asof_date >= "2026-08-11") & (recs.asof_date <= "2026-09-03")]
    recs["asof_date"] = recs.asof_date.astype(str)
    recs = recs.rename(columns={"return_10d": "ret_pkg"})
    panel = pickle.load(open(Path(a.oos_package) / "evaluation_panel.pkl", "rb"))
    feats = build_features(panel, recs[["symbol", "asof_date", "start_index", "ret_pkg"]])
    m = m.merge(feats, on=keys, how="left")
    assert m.sector.notna().all()
    return m


def load_val(a):
    keys = ["symbol", "asof_date"]
    names = {"b475": "beta_v21_release_best475", "seg155": "seg155_forecast_best", "seg8": "seg8_rank_unfreeze_best"}
    m = None
    for lab, fn in names.items():
        f = pd.read_csv(f"{a.val_results}/{fn}_val_predictions.csv.gz",
                        usecols=keys + ["return_10d", "predicted_return_d10", "predicted_terminal_dispersion"])
        f = f.rename(columns={"predicted_return_d10": f"s_{lab}", "predicted_terminal_dispersion": f"disp_{lab}"})
        f[f"n_{lab}"] = 5
        m = f.rename(columns={"return_10d": "ret"}) if m is None else m.merge(f.drop(columns="return_10d"), on=keys)
    recs = pd.DataFrame(json.load(open(a.val_records)))
    panel = pickle.load(open(a.val_data, "rb"))
    recs = recs.merge(m[keys + ["ret"]].rename(columns={"ret": "ret_pkg"}), on=keys)
    feats = build_features(panel, recs[["symbol", "asof_date", "start_index", "ret_pkg"]])
    m = m.merge(feats, on=keys, how="left")
    assert m.sector.notna().all() and len(m) == 12256, len(m)
    return m


# ------------------------------------------------------------------------- analyses
def item1_shape_noise(m, models, n_map):
    per = []
    for d, g in m.groupby("asof_date"):
        r = g.ret.values
        row = dict(asof_date=d, ret_std=r.std(), ret_mean=r.mean())
        for k in models:
            s = g[f"s_{k}"].values
            z = (s - s.mean()) / s.std()
            srt = np.sort(s)
            gaps = np.diff(srt)
            disp = g[f"disp_{k}"].values
            se2 = (disp ** 2) / n_map[k]
            rel = 1 - se2.mean() / s.var()
            row.update({
                f"{k}_ic": spearman(s, r),
                f"{k}_pearson_raw": float(np.corrcoef(s, r)[0, 1]),
                f"{k}_pearson_gauss": float(np.corrcoef(gauss_rank(s), r)[0, 1]),
                f"{k}_std": s.std(), f"{k}_mean": s.mean(), f"{k}_skew": stats.skew(s), f"{k}_exkurt": stats.kurtosis(s),
                f"{k}_share_absz3": float((np.abs(z) > 3).mean()),
                f"{k}_exact_ties": int(len(s) - len(np.unique(s))),
                f"{k}_near_ties": int((gaps < 1e-4 * s.std()).sum()),
                f"{k}_std_over_retstd": s.std() / r.std(),
                f"{k}_mean_disp": disp.mean(),
                f"{k}_reliability": rel,
            })
            # IC within terciles of sampling SE (noise) — descriptive
            se = np.sqrt(se2)
            q = pd.qcut(stats.rankdata(se, method="ordinal"), 3, labels=False)
            for t in range(3):
                mask = q == t
                row[f"{k}_ic_se_t{t}"] = spearman(s[mask], r[mask])
        per.append(row)
    D = pd.DataFrame(per)
    summ = {}
    for k in models:
        cols = ["ic", "pearson_raw", "pearson_gauss", "std", "mean", "skew", "exkurt", "share_absz3", "exact_ties", "near_ties",
                "std_over_retstd", "mean_disp", "reliability", "ic_se_t0", "ic_se_t1", "ic_se_t2"]
        s = {c: float(D[f"{k}_{c}"].mean()) for c in cols}
        rel = D[f"{k}_reliability"].clip(lower=1e-3)
        s["ic_disattenuated_inf_N"] = float((D[f"{k}_ic"] / np.sqrt(rel)).mean())
        # across-day relation of shape/noise to IC (18 or 24 points; weak)
        for c in ["skew", "exkurt", "share_absz3", "reliability", "std_over_retstd"]:
            s[f"acrossday_spearman_{c}_vs_ic"] = spearman(D[f"{k}_{c}"], D[f"{k}_ic"])
        summ[k] = s
    return D, summ


def cross_decode(m, a, b):
    return float(np.mean([spearman(g[f"s_{a}"], g[f"s_{b}"]) for _, g in m.groupby("asof_date")]))


def item2_deciles(m, models, head_dates):
    prof, per = {}, []
    for d, g in m.groupby("asof_date"):
        r = g.ret.values
        mu = r.mean()
        row = dict(asof_date=d)
        for k in models:
            b = pd.qcut(stats.rankdata(g[f"s_{k}"].values, method="ordinal"), 10, labels=False)
            means = pd.Series(r).groupby(b).mean().values
            prof.setdefault(k, {})[d] = (means - mu).tolist()
            row[f"{k}_mono"] = spearman(means, np.arange(10))
            row[f"{k}_top_excess"] = means[-1] - mu
            row[f"{k}_bottom_short_excess"] = mu - means[0]
            row[f"{k}_tb"] = means[-1] - means[0]
            row[f"{k}_mid_slope"] = float(np.polyfit(np.arange(1, 9), means[1:9], 1)[0])  # D2..D9 slope / decile
        per.append(row)
    D = pd.DataFrame(per).set_index("asof_date")
    summ = {}
    for k in models:
        P = np.array([prof[k][d] for d in D.index])
        H = np.array([prof[k][d] for d in head_dates])
        summ[k] = dict(
            mono_all18=D[f"{k}_mono"].mean(), mono_first15=D.loc[head_dates, f"{k}_mono"].mean(),
            pooled_decile_excess_all18=P.mean(0).tolist(), pooled_decile_excess_first15=H.mean(0).tolist(),
            pooled_mono_all18=spearman(P.mean(0), np.arange(10)), pooled_mono_first15=spearman(H.mean(0), np.arange(10)),
            top_excess_all18=D[f"{k}_top_excess"].mean(), bottom_short_excess_all18=D[f"{k}_bottom_short_excess"].mean(),
            top_excess_first15=D.loc[head_dates, f"{k}_top_excess"].mean(),
            bottom_short_excess_first15=D.loc[head_dates, f"{k}_bottom_short_excess"].mean(),
            tb_all18=D[f"{k}_tb"].mean(), mid_slope_D2_D9_all18=D[f"{k}_mid_slope"].mean(),
            mid_slope_D2_D9_first15=D.loc[head_dates, f"{k}_mid_slope"].mean(),
        )
    return D, prof, summ


def item3_calibration(m, models):
    out = {}
    for k in models:
        s, r = m[f"s_{k}"].values, m.ret.values
        sl, ic_, *_ = stats.linregress(s, r)
        # winsorized pooled (1/99) to reduce tail leverage
        lo, hi = np.percentile(s, [1, 99])
        sw = np.clip(s, lo, hi)
        slw, icw, *_ = stats.linregress(sw, r)
        days = []
        for d, g in m.groupby("asof_date"):
            x, y = g[f"s_{k}"].values, g.ret.values
            xw = np.clip(x, *np.percentile(x, [1, 99]))
            b1, b0, *_ = stats.linregress(xw, y)
            days.append(dict(asof_date=d, slope_w=b1, intercept_w=b0, mean_pred=x.mean(), mean_ret=y.mean()))
        Dd = pd.DataFrame(days)
        q = pd.qcut(stats.rankdata(s, method="ordinal"), 10, labels=False)
        bins = pd.DataFrame(dict(q=q, s=s, r=r)).groupby("q").agg(mean_pred=("s", "mean"), mean_ret=("r", "mean")).reset_index()
        out[k] = dict(pooled_slope=sl, pooled_intercept=ic_, pooled_slope_w1_99=slw, pooled_intercept_w1_99=icw,
                      daily_slope_w_mean=Dd.slope_w.mean(), daily_slope_w_median=Dd.slope_w.median(),
                      daily_intercept_w_mean=Dd.intercept_w.mean(),
                      mean_pred=s.mean(), mean_ret=r.mean(), std_pred=s.std(), std_ret=r.std(),
                      pooled_decile_bins=bins.to_dict(orient="records"))
    return out


def neutral_variants(g, k, fine, coarse, extra_style=True):
    y = gauss_rank(g[f"s_{k}"].values)
    sp = g.size_pct.values - g.size_pct.mean()
    v = {"raw": y}
    v["sector_fine"] = residualize(y, dummies(fine))
    v["sector_coarse"] = residualize(y, dummies(coarse))
    v["size"] = residualize(y, [sp])
    v["sector_fine+size"] = residualize(y, dummies(fine) + [sp])
    v["sector_coarse+size"] = residualize(y, dummies(coarse) + [sp])
    if extra_style:
        st = [gauss_rank(g[c].fillna(g[c].median()).values) for c in ["mom5", "mom20", "mom60", "vol20", "liq20"]]
        v["sector_fine+size+style"] = residualize(y, dummies(fine) + [sp] + st)
    return v


def item5_exposures(m, models, pair_main=("c2p", "b475")):
    per, neut = [], []
    for d, g in m.groupby("asof_date"):
        r = g.ret.values
        fine = merge_small(g.sector.values)
        coarse = coarse_sector(g.sector.values)
        row = dict(asof_date=d)
        Dfine = np.column_stack([np.ones(len(g))] + dummies(fine))
        for f in FEATS:
            x = g[f].fillna(g[f].median()).values
            row[f"factor_ic_{f}"] = spearman(x, r)
        # sector factor: IC of sector-mean return predictor is look-ahead; report R^2 of returns on sectors instead
        rr = gauss_rank(r)
        row["ret_sector_r2"] = 1 - residualize(rr, dummies(fine)).var() / rr.var()
        for k in models:
            s = g[f"s_{k}"].values
            gs = gauss_rank(s)
            for f in FEATS:
                row[f"{k}_expo_{f}"] = spearman(s, g[f].fillna(g[f].median()).values)
            row[f"{k}_sector_r2"] = 1 - residualize(gs, dummies(fine)).var() / gs.var()
            nr = {"asof_date": d, "model": k}
            for name, vec in neutral_variants(g, k, fine, coarse).items():
                nr[name] = spearman(vec, r)
            # sector-demeaned IC: within-sector ranking only (demean both score-rank and return-rank)
            nr["within_sector_ic"] = float(np.corrcoef(residualize(gs, dummies(fine)), residualize(rr, dummies(fine)))[0, 1])
            neut.append(nr)
        # C2 vs ours
        a, b = pair_main
        sa, sb = g[f"s_{a}"].values, g[f"s_{b}"].values
        row["corr_c2p_b475"] = spearman(sa, sb)
        row["corr_c2p_seg9"] = spearman(sa, g.s_seg9.values)
        row["corr_c2p_seg155"] = spearman(sa, g.s_seg155.values)
        row["corr_c2r_b475"] = spearman(g.s_c2r.values, sb)
        row["corr_b475_seg9"] = spearman(sb, g.s_seg9.values)
        row["corr_b475_seg155"] = spearman(sb, g.s_seg155.values)
        ra, rb = stats.rankdata(sa), stats.rankdata(sb)
        row["blend_c2p_b475_ic"] = spearman(ra + rb, r)
        row["blend_c2r_b475_ic"] = spearman(stats.rankdata(g.s_c2r.values) + rb, r)
        ga, gb = gauss_rank(sa), gauss_rank(sb)
        row["c2_minus_ours_rankdiff_ic"] = spearman(ra - rb, r)
        row["c2_resid_on_ours_ic"] = spearman(residualize(ga, [gb]), r)
        row["ours_resid_on_c2_ic"] = spearman(residualize(gb, [ga]), r)
        st = [gauss_rank(g[c].fillna(g[c].median()).values) for c in ["mom5", "mom20", "mom60", "vol20", "liq20"]]
        X = dummies(fine) + [g.size_pct.values] + st
        row["c2_resid_on_ours_and_style_ic"] = spearman(residualize(ga, [gb] + X), r)
        row["ours_resid_on_c2_and_style_ic"] = spearman(residualize(gb, [ga] + X), r)
        # how much of C2 score is explained by sector+size+style alone, and by ours
        row["c2_r2_style"] = 1 - residualize(ga, X).var() / ga.var()
        row["b475_r2_style"] = 1 - residualize(gb, X).var() / gb.var()
        row["c2_r2_ours_plus_style"] = 1 - residualize(ga, [gb] + X).var() / ga.var()
        # style-only composite IC (in-sample daily OLS of C2/ours on style -> fitted part) = exposure-implied IC
        for lab, gv in [("c2p", ga), ("b475", gb)]:
            fitted = gv - residualize(gv, X)
            row[f"{lab}_style_fitted_ic"] = spearman(fitted, r)
        per.append(row)
    return pd.DataFrame(per), pd.DataFrame(neut)


def item4_tail(m, models, prof, D2):
    out = {}
    for d in TAIL:
        g = m[m.asof_date == d]
        r = g.ret.values
        day = {"mkt_mean": r.mean(), "pct_up": float((r > 0).mean())}
        for k in models:
            day[k] = dict(ic=spearman(g[f"s_{k}"], r), decile_excess=prof[k][d], mono=D2.loc[d, f"{k}_mono"],
                          tb=D2.loc[d, f"{k}_tb"])
            day[k]["expo"] = {f: spearman(g[f"s_{k}"], g[f].fillna(g[f].median())) for f in FEATS}
        day["factor_ic"] = {f: spearman(g[f].fillna(g[f].median()), r) for f in FEATS}
        # who differs: C2-prod vs Best475 bottom/top deciles
        n = len(g); k10 = int(round(n * 0.1))
        rc = stats.rankdata(g.s_c2p.values); rb = stats.rankdata(g.s_b475.values)
        c_top, c_bot = rc > n - k10, rc <= k10
        b_top, b_bot = rb > n - k10, rb <= k10
        ex = r - r.mean()
        day["overlap"] = dict(
            top_overlap_share=float((c_top & b_top).sum() / k10), bottom_overlap_share=float((c_bot & b_bot).sum() / k10),
            c2_top_only_excess=float(ex[c_top & ~b_top].mean()), b475_top_only_excess=float(ex[b_top & ~c_top].mean()),
            both_top_excess=float(ex[c_top & b_top].mean()),
            c2_bottom_only_excess=float(ex[c_bot & ~b_bot].mean()), b475_bottom_only_excess=float(ex[b_bot & ~c_bot].mean()),
            both_bottom_excess=float(ex[c_bot & b_bot].mean()),
            c2_top_only_feat_median={f: float(g[f].values[c_top & ~b_top].mean()) for f in ["size_pct", "mom5", "mom20", "vol20"]},
            b475_top_only_feat_median={f: float(g[f].values[b_top & ~c_top].mean()) for f in ["size_pct", "mom5", "mom20", "vol20"]},
            c2_bottom_only_feat_mean={f: float(g[f].values[c_bot & ~b_bot].mean()) for f in ["size_pct", "mom5", "mom20", "vol20"]},
            b475_bottom_only_feat_mean={f: float(g[f].values[b_bot & ~c_bot].mean()) for f in ["size_pct", "mom5", "mom20", "vol20"]},
        )
        out[d] = day
    return out


def item6_val(v):
    models = ["b475", "seg155", "seg8"]
    rows = []
    sanity = []
    for d, g in v.groupby("asof_date"):
        r = g.ret.values
        fine = merge_small(g.sector.values)
        coarse = coarse_sector(g.sector.values)
        for k in models:
            s = g[f"s_{k}"].values
            nr = dict(asof_date=d, model=k)
            for name, vec in neutral_variants(g, k, fine, coarse).items():
                nr[name] = spearman(vec, r)
            disp = g[f"disp_{k}"].values
            nr["snr_t_score"] = spearman((s - np.median(s)) / (disp / np.sqrt(5) + 1e-4), r)
            rows.append(nr)
            w = np.clip(s, *np.percentile(s, [1, 99]))
            sanity.append(dict(asof_date=d, model=k, raw=spearman(s, r), winsor_1_99=spearman(w, r),
                               rank_std=spearman((stats.rankdata(s) - 0.5) / len(s), r), gauss_rank=spearman(gauss_rank(s), r)))
    N = pd.DataFrame(rows)
    S = pd.DataFrame(sanity)
    summ = {}
    for k in models:
        x = N[N.model == k].set_index("asof_date")
        summ[k] = {c: dict(ic=x[c].mean(), **{f"diff_vs_raw_{kk}": vv for kk, vv in paired(x[c] - x["raw"]).items()})
                   for c in x.columns if c not in ("model",)}
    ss = S.groupby("model")[["raw", "winsor_1_99", "rank_std", "gauss_rank"]].agg(lambda c: c.mean())
    maxdev = float(max((S.winsor_1_99 - S.raw).abs().max(), (S.rank_std - S.raw).abs().max(), (S.gauss_rank - S.raw).abs().max()))
    return N, summ, ss.to_dict(orient="index"), maxdev


def simex(m, models, n_map, ks=(1.0, 1.5, 2.0, 3.0), reps=8, seed=0):
    """SIMEX on the N-sample-mean noise: add N(0,(k-1)*disp^2/N) to the score, track daily Spearman IC,
    extrapolate (quadratic and linear in k) to k=0 (= infinite samples). Assumes Gaussian sampling noise."""
    rng = np.random.default_rng(seed)
    out = {}
    groups = [(d, g) for d, g in m.groupby("asof_date")]
    for k in models:
        curve = {}
        for kk in ks:
            vals = []
            for _ in range(reps if kk > 1 else 1):
                ics = []
                for d, g in groups:
                    s = g[f"s_{k}"].values
                    se = g[f"disp_{k}"].values / np.sqrt(n_map[k])
                    s2 = s + rng.normal(size=len(s)) * se * np.sqrt(kk - 1) if kk > 1 else s
                    ics.append(spearman(s2, g.ret.values))
                vals.append(np.mean(ics))
            curve[kk] = float(np.mean(vals))
        x = np.array(list(curve)); y = np.array(list(curve.values()))
        q = np.polyfit(x, y, 2); l = np.polyfit(x, y, 1)
        out[k] = dict(curve={str(a): b for a, b in curve.items()}, ic_k0_quadratic=float(np.polyval(q, 0)),
                      ic_k0_linear=float(np.polyval(l, 0)),
                      ic_at_N16_quadratic=float(np.polyval(q, n_map[k] / 16)))
    return out


def decile_anatomy(m, models, feats=("size_pct", "vol20", "mom5", "mom20")):
    out = {}
    for k in models:
        rows = []
        for d, g in m.groupby("asof_date"):
            b = pd.qcut(stats.rankdata(g[f"s_{k}"].values, method="ordinal"), 10, labels=False)
            t = pd.DataFrame({"b": b, "pred": g[f"s_{k}"].values, "disp": g[f"disp_{k}"].values,
                              **{f: g[f].values for f in feats}})
            agg = t.groupby("b").mean()
            top = b == 9
            agg["within_decile_ic"] = [spearman(g[f"s_{k}"].values[b == j], g.ret.values[b == j]) for j in range(10)]
            rows.append(agg)
        A = sum(rows) / len(rows)
        out[k] = {c: A[c].tolist() for c in A.columns}
    return out


def val_postproc(v, models=("b475", "seg155", "seg8")):
    """Validation-only post-processing candidates (beyond neutralization)."""
    rows = []
    for d, g in v.groupby("asof_date"):
        r = g.ret.values
        fine = merge_small(g.sector.values)
        st = [gauss_rank(g[c].fillna(g[c].median()).values) for c in ["mom5", "mom20", "mom60", "vol20", "liq20"]]
        X = dummies(fine) + [g.size_pct.values - g.size_pct.mean()] + st
        for k in models:
            s = g[f"s_{k}"].values
            disp = g[f"disp_{k}"].values
            gs = gauss_rank(s)
            res = residualize(gs, X)
            fit = gs - res
            row = dict(asof_date=d, model=k, raw=spearman(s, r),
                       disp_pen_0p5=spearman(s - 0.5 * disp, r), disp_pen_1p0=spearman(s - 1.0 * disp, r),
                       style_fitted_only=spearman(fit, r), fitted_plus_half_resid=spearman(fit + 0.5 * res, r))
            rows.append(row)
    N = pd.DataFrame(rows)
    summ = {}
    for k in models:
        x = N[N.model == k].set_index("asof_date").drop(columns="model")
        summ[k] = {c: dict(ic=x[c].mean(), **{f"diff_vs_raw_{a}": b for a, b in paired(x[c] - x["raw"]).items()}) for c in x.columns}
    return summ


def top_disp(m, models):
    """Within each model's daily top decile: Spearman(dispersion, return) and Spearman(score, return)."""
    out = {}
    for k in models:
        a, b, sh = [], [], []
        for d, g in m.groupby("asof_date"):
            rk = stats.rankdata(g[f"s_{k}"].values, method="ordinal")
            top = rk > len(g) * 0.9
            a.append(spearman(g[f"disp_{k}"].values[top], g.ret.values[top]))
            b.append(spearman(g[f"s_{k}"].values[top], g.ret.values[top]))
            hi = g[f"disp_{k}"].values[top] > np.median(g[f"disp_{k}"].values[top])
            ex = g.ret.values[top] - g.ret.values.mean()
            sh.append(ex[hi].mean() - ex[~hi].mean())
        out[k] = dict(disp_vs_ret_in_top=float(np.mean(a)), score_vs_ret_in_top=float(np.mean(b)),
                      hi_minus_lo_disp_excess_in_top=float(np.mean(sh)))
    return out


def val_shape(v):
    D, summ = item1_shape_noise(v, ["b475", "seg155", "seg8"], {"b475": 5, "seg155": 5, "seg8": 5})
    return summ


def main():
    ap = argparse.ArgumentParser()
    for k, val in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=val)
    ap.add_argument("--out-json", default="finetune/reports/beta_v21_c1_why_c2_better_analysis.json")
    a = ap.parse_args()

    m = load_oos(a)
    lab_check = float((m.ret - m.ret_pkg).abs().max())
    close_check = float((m.ret - m.ret_check).abs().quantile(0.99))
    models = ["c2p", "c2r", "b475", "seg9", "seg155", "seg155r"]
    n_map = {"c2p": 5, "c2r": 16, "b475": 5, "seg9": 5, "seg155": 5, "seg155r": 16}
    dates = sorted(m.asof_date.unique())
    head = [d for d in dates if d not in TAIL]

    D1, S1 = item1_shape_noise(m, models, n_map)
    xdec = {"c2p_vs_c2r": cross_decode(m, "c2p", "c2r"), "seg155_vs_seg155r": cross_decode(m, "seg155", "seg155r"),
            "b475_vs_seg9": cross_decode(m, "b475", "seg9")}
    D2, prof, S2 = item2_deciles(m, models, head)
    S3 = item3_calibration(m, models)
    D5, N5 = item5_exposures(m, models)
    S4 = item4_tail(m, models, prof, D2)

    exp_summ = {k: {f: float(D5[f"{k}_expo_{f}"].mean()) for f in FEATS} | {"sector_r2": float(D5[f"{k}_sector_r2"].mean())}
                for k in models}
    fac = {f: float(D5[f"factor_ic_{f}"].mean()) for f in FEATS} | {"ret_sector_r2": float(D5.ret_sector_r2.mean())}
    neut_summ = {}
    for k in models:
        x = N5[N5.model == k].set_index("asof_date").drop(columns="model")
        neut_summ[k] = {c: float(x[c].mean()) for c in x.columns}
    gap = {}
    for var in N5.columns.drop(["asof_date", "model"]):
        for ours in ["b475", "seg9", "seg155"]:
            for c2 in ["c2p", "c2r"]:
                d = N5[N5.model == ours].set_index("asof_date")[var] - N5[N5.model == c2].set_index("asof_date")[var]
                gap[f"{ours}_minus_{c2}__{var}"] = paired(d)
    pair_cols = [c for c in D5.columns if c.startswith(("corr_", "blend_", "c2_", "ours_", "b475_", "c2p_"))]
    pairs = {c: float(D5[c].mean()) for c in pair_cols}
    pairs["blend_c2p_b475_minus_c2p"] = paired(D5.blend_c2p_b475_ic.values - D1.c2p_ic.values)

    SIM = simex(m, models, n_map)
    ANAT = decile_anatomy(m, ["c2p", "b475", "seg9", "seg155"])
    v = load_val(a)
    vlab = float((v.ret - v.ret_check).abs().quantile(0.99))
    N6, S6, sanity, maxdev = item6_val(v)
    V1 = val_shape(v)
    vnm = {"b475": 5, "seg155": 5, "seg8": 5}
    VSIM = simex(v, ["b475", "seg155", "seg8"], vnm)
    vdates = sorted(v.asof_date.unique())
    _, _, VDEC = item2_deciles(v, ["b475", "seg155", "seg8"], vdates)
    VANAT = decile_anatomy(v, ["b475", "seg155"])
    VPP = val_postproc(v)
    vexp = {}
    for k in ["b475", "seg155"]:
        vexp[k] = {f: float(np.mean([spearman(g[f"s_{k}"], g[f].fillna(g[f].median())) for _, g in v.groupby("asof_date")])) for f in FEATS}
    vexp["factor_ic"] = {f: float(np.mean([spearman(g[f].fillna(g[f].median()), g.ret) for _, g in v.groupby("asof_date")])) for f in FEATS}

    out = dict(
        meta=dict(script="finetune/analysis/beta_v21_c1_why_c2_better.py", oos_rows=len(m), oos_dates=len(dates), val_rows=len(v),
                  val_dates=int(v.asof_date.nunique()), oos_label_maxabs_vs_package=lab_check,
                  oos_label_vs_panel_close_p99=close_check, val_label_vs_panel_close_p99=vlab,
                  models={"c2p": "Small C2 prod T0.65/p0.8/N5", "c2r": "Small C2 rank T0.6/p0.9/N16",
                          "b475": "Beta v2.1 Best@475 (Seg0) prod N5", "seg9": "cosine pilot Seg9 prod N5",
                          "seg155": "Seg155 forecast-best prod N5", "seg155r": "Seg155 rank N16", "seg8": "Seg8 rank-unfreeze (val only)"},
                  note="Sealed OOS is retired for selection: everything under oos_* is descriptive. Only val_* is used to compare post-processing."),
        oos_item1_shape_noise=dict(summary=S1, cross_decode_rank_corr=xdec, simex=SIM, per_date=D1.to_dict(orient="records")),
        oos_decile_anatomy=ANAT,
        oos_item2_deciles=dict(summary=S2, per_date=D2.reset_index().to_dict(orient="records")),
        oos_item3_calibration=S3,
        oos_item4_tail=S4,
        oos_item5_exposures=dict(score_exposure_mean_spearman=exp_summ, factor_ic=fac, neutralized_ic=neut_summ,
                                 neutralized_gap_vs_c2=gap, pairs=pairs, per_date=D5.to_dict(orient="records"),
                                 neutralized_per_date=N5.to_dict(orient="records")),
        oos_top_decile_disp=top_disp(m, ["c2p", "b475", "seg9", "seg155"]),
        val_item6=dict(neutralization=S6, sanity_monotone_transforms=sanity, sanity_max_abs_daily_ic_dev=maxdev,
                       shape_noise=V1, simex=VSIM, deciles=VDEC, decile_anatomy=VANAT, postproc_candidates=VPP,
                       top_decile_disp=top_disp(v, ["b475", "seg155", "seg8"]),
                       score_exposure=vexp, per_date=N6.to_dict(orient="records")),
    )
    g = lambda var: gap[f"b475_minus_c2p__{var}"]["mean"]
    out["headline"] = dict(
        ic_gap_b475_minus_c2p_raw=g("raw"),
        ic_gap_noise_free_simex=SIM["b475"]["ic_k0_quadratic"] - SIM["c2p"]["ic_k0_quadratic"],
        ic_gap_after_sector_fine=g("sector_fine"), ic_gap_after_sector_fine_size=g("sector_fine+size"),
        ic_gap_after_sector_size_style=g("sector_fine+size+style"),
        style_fitted_ic=dict(c2p=pairs["c2p_style_fitted_ic"], b475=pairs["b475_style_fitted_ic"]),
        top_decile_excess=dict(c2p=S2["c2p"]["top_excess_all18"], b475=S2["b475"]["top_excess_all18"]),
        bottom_short_excess=dict(c2p=S2["c2p"]["bottom_short_excess_all18"], b475=S2["b475"]["bottom_short_excess_all18"]),
        decile_mono_first15=dict(c2p=S2["c2p"]["mono_first15"], b475=S2["b475"]["mono_first15"], seg155=S2["seg155"]["mono_first15"]),
        corr_c2p_b475=pairs["corr_c2p_b475"], blend_ic_descriptive=pairs["blend_c2p_b475_ic"],
        val_b475_neutralization_delta={c: S6["b475"][c]["diff_vs_raw_mean"] for c in S6["b475"] if c != "raw"},
        val_b475_postproc_delta={c: VPP["b475"][c]["diff_vs_raw_mean"] for c in VPP["b475"] if c != "raw"},
        verdict=("Not sampling noise/shape (SIMEX noise-free gap ~= raw gap). ~Half of the 0.022 gap is sector/size/style exposure "
                 "(gap -> ~0.011 after full neutralization; 09-01 gap vanishes), ~half is structural within-sector style-neutral "
                 "stock signal (residual IC 0.025 vs 0.036) incl. a broken top decile filled with explosive-AR-path names. "
                 "Calibration is irrelevant to rank IC. On validation, neutralization hurts (-0.02..-0.15); the only significant "
                 "candidate is style-projection + 0.5*residual (+0.024, t 5.1) which must be confirmed on a fresh post-09-03 window."),
    )
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(clean(out), open(a.out_json, "w"), indent=1, ensure_ascii=False)
    print("wrote", a.out_json)


if __name__ == "__main__":
    main()
