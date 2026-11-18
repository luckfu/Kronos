"""Pre-registered study: Point-in-time non-PV features + C2 fusion + market regime gate.

Evaluates two hypotheses on the clean 65-day OOS (2026-07-17..2026-10-23) while
keeping 2026-10-24+ sealed and training NONPV strictly pre-OOS (2026-02-02..2026-07-02):
  H1: Point-in-time fundamentals, valuation, earnings pre-announcements, and LHB flows
      add cross-sectional IC and improve long-side head returns when combined with C2.
  H2: A causal market-regime gate (CSI300 > MA20 + market breadth >= 0.35) improves
      few-name after-cost net return and drawdown across up/down regimes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import few_names_backtest as fnb  # noqa: E402
import fusion_analysis as fa  # noqa: E402
import rolling_compare as rc  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402

TRAIN_START = "2026-02-02"
TRAIN_FIT_END = "2026-05-15"
TRAIN_VAL_START = "2026-05-29"
TRAIN_END = "2026-07-02"
OOS_START = "2026-07-17"
OOS_END = "2026-10-23"

FEATURES = [
    "ep_ann", "sp_ann", "bp_implied", "log_mcap",
    "roe", "gross_margin", "cash_quality",
    "rev_yoy", "rev_qoq", "np_yoy", "np_qoq", "days_since_ann",
    "yjyg_growth_mid", "yjyg_sign",
    "lhb_net_20d", "lhb_count_20d",
]

YJBB_QUARTERS = [
    ("20250930", 3, True),
    ("20251231", 4, False),
    ("20260331", 1, False),
    ("20260630", 2, False),
    ("20260930", 3, False),
]
YJYG_QUARTERS = ["20251231", "20260331", "20260630", "20260930"]
POS_YJYG = {"预增", "略增", "扭亏", "续盈"}
NEG_YJYG = {"首亏", "增亏", "预减", "略减", "续亏"}
BOOTSTRAP_SEED = 20261115


def load_yjbb(non_pv_dir: Path) -> pd.DataFrame:
    """Load quarterly fundamentals with strict point-in-time announcement dates."""
    frames = []
    for q_str, q_num, is_legacy in YJBB_QUARTERS:
        df = pd.read_parquet(non_pv_dir / f"yjbb_{q_str}.parquet").copy()
        df["code"] = df["股票代码"].astype(str).str.zfill(6)
        df = df[df["code"].str.startswith(("0", "3", "6"))].drop_duplicates("code")
        # Eastmoney overwrote 20250930's 最新公告日期 when 20260930 released;
        # Q3 2025 statutory filing deadline was 2025-10-31 (public before 2026).
        ann = pd.Series("2025-11-01", index=df.index) if is_legacy else pd.to_datetime(df["最新公告日期"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = pd.DataFrame({
            "code": df["code"],
            "period": q_str,
            "q_num": q_num,
            "ann_date": ann,
            "eps": pd.to_numeric(df["每股收益"], errors="coerce"),
            "rev": pd.to_numeric(df["营业总收入-营业总收入"], errors="coerce"),
            "rev_yoy": pd.to_numeric(df["营业总收入-同比增长"], errors="coerce"),
            "rev_qoq": pd.to_numeric(df["营业总收入-季度环比增长"], errors="coerce"),
            "np": pd.to_numeric(df["净利润-净利润"], errors="coerce"),
            "np_yoy": pd.to_numeric(df["净利润-同比增长"], errors="coerce"),
            "np_qoq": pd.to_numeric(df["净利润-季度环比增长"], errors="coerce"),
            "bvps": pd.to_numeric(df["每股净资产"], errors="coerce"),
            "roe": pd.to_numeric(df["净资产收益率"], errors="coerce"),
            "ocf_ps": pd.to_numeric(df["每股经营现金流量"], errors="coerce"),
            "gross_margin": pd.to_numeric(df["销售毛利率"], errors="coerce"),
        })
        frames.append(out.dropna(subset=["ann_date"]))
    return pd.concat(frames, ignore_index=True)


def load_yjyg(non_pv_dir: Path) -> pd.DataFrame:
    """Load earnings pre-announcements deduplicated per (code, period, ann_date)."""
    frames = []
    for q_str in YJYG_QUARTERS:
        df = pd.read_parquet(non_pv_dir / f"yjyg_{q_str}.parquet").copy()
        df["code"] = df["股票代码"].astype(str).str.zfill(6)
        df = df[df["code"].str.startswith(("0", "3", "6"))]
        df["ann_date"] = pd.to_datetime(df["公告日期"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["prio"] = np.where(df["预测指标"].astype(str).str.contains("归属于上市公司股东的净利润"), 0, 1)
        df = df.sort_values(["code", "ann_date", "prio"]).drop_duplicates(["code", "ann_date"])
        sign = np.where(df["预告类型"].isin(POS_YJYG), 1.0, np.where(df["预告类型"].isin(NEG_YJYG), -1.0, 0.0))
        frames.append(pd.DataFrame({
            "code": df["code"],
            "period": q_str,
            "ann_date": df["ann_date"],
            "yjyg_growth_mid": pd.to_numeric(df["业绩变动幅度"], errors="coerce").fillna(0.0),
            "yjyg_sign": sign,
        }).dropna(subset=["ann_date"]))
    return pd.concat(frames, ignore_index=True)


def load_lhb_daily(non_pv_dir: Path, trade_dates: list[str]) -> pd.DataFrame:
    """Aggregate Dragon-Tiger list net buy & counts by (code, date)."""
    lhb = pd.read_parquet(non_pv_dir / "lhb_20251201_20261025.parquet").copy()
    lhb["code"] = lhb["代码"].astype(str).str.zfill(6)
    lhb["date"] = pd.to_datetime(lhb["上榜日"]).dt.strftime("%Y-%m-%d")
    lhb["net_buy"] = pd.to_numeric(lhb["龙虎榜净买额"], errors="coerce").fillna(0.0)
    lhb = lhb[lhb["date"].isin(set(trade_dates))]
    agg = lhb.groupby(["code", "date"], as_index=False).agg(lhb_net=("net_buy", "sum"), lhb_cnt=("net_buy", "size"))
    return agg


def build_panel(raw_path: Path, non_pv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build point-in-time non-PV feature panel for 2026-02-02..2026-10-23."""
    raw = pd.read_csv(raw_path, dtype={"symbol": str})
    raw["date"] = raw["date"].astype(str)
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    raw["code"] = raw["symbol"].str[-6:]

    # Forward D10 return matching evaluator's volume>0 definition
    traded = raw[raw["volume"] > 0].copy()
    traded["actual_return_d10"] = traded.groupby("symbol")["close"].shift(-10) / traded["close"] - 1.0
    traded["avg_amt_20d"] = traded.groupby("symbol")["amount"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    traded["med_amt_20d"] = traded.groupby("symbol")["amount"].transform(
        lambda s: s.rolling(20, min_periods=20).median()
    )
    traded["ma20"] = traded.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    traded["above_ma20"] = (traded["close"] > traded["ma20"]).astype(float)

    # Market breadth strictly <= signal date
    breadth = traded.dropna(subset=["ma20"]).groupby("date")["above_ma20"].mean().rename("breadth_ma20")
    idx300 = pd.read_parquet(non_pv_dir / "idx_sh000300.parquet").copy()
    idx300["date"] = idx300["date"].astype(str)
    idx300 = idx300.sort_values("date")
    idx300["ma20"] = idx300["close"].rolling(20, min_periods=20).mean()
    idx300["csi300_above_ma20"] = (idx300["close"] > idx300["ma20"]).astype(int)
    regime = idx300.set_index("date")[["close", "ma20", "csi300_above_ma20"]].join(breadth, how="inner")
    regime["score_mkt"] = regime["csi300_above_ma20"] + (regime["breadth_ma20"] >= 0.35).astype(int)

    # LHB rolling 20-bar net buy normalized by avg_amt_20d
    trade_dates = sorted(traded["date"].unique())
    lhb = load_lhb_daily(non_pv_dir, trade_dates)
    traded = traded.merge(lhb, on=["code", "date"], how="left")
    traded["lhb_net"] = traded["lhb_net"].fillna(0.0)
    traded["lhb_cnt"] = traded["lhb_cnt"].fillna(0.0)
    traded["lhb_net_20d"] = (
        traded.groupby("symbol")["lhb_net"].transform(lambda s: s.rolling(20, min_periods=1).sum())
        / traded["avg_amt_20d"].clip(lower=1e6)
    )
    traded["lhb_count_20d"] = traded.groupby("symbol")["lhb_cnt"].transform(lambda s: s.rolling(20, min_periods=1).sum())

    # Filter to signal window 2026-02-02..2026-10-23 (never touch 2026-10-24+)
    panel = traded[(traded["date"] >= TRAIN_START) & (traded["date"] <= OOS_END)].copy()
    assert panel["date"].max() <= OOS_END, "Sealed window violation!"

    # Attach latest announced yjbb (strictly ann_date <= date, preferring latest report period)
    yjbb = load_yjbb(non_pv_dir)
    yjyg = load_yjyg(non_pv_dir)

    panel["date_dt"] = pd.to_datetime(panel["date"])
    yjbb["ann_dt"] = pd.to_datetime(yjbb["ann_date"])
    yjyg["ann_dt"] = pd.to_datetime(yjyg["ann_date"])

    # For each (code, date), keep yjbb with ann_dt <= date_dt and max (period, ann_dt)
    panel_parts = []
    yjbb_by_code = {c: g.sort_values(["period", "ann_dt"]) for c, g in yjbb.groupby("code")}
    yjyg_by_code = {c: g.sort_values(["ann_dt", "period"]) for c, g in yjyg.groupby("code")}

    for code, grp in panel.groupby("code", sort=False):
        dates = grp["date_dt"].to_numpy(dtype="datetime64[ns]")
        g = grp.copy()
        yb = yjbb_by_code.get(code)
        if yb is not None and len(yb):
            yb_ann = yb["ann_dt"].to_numpy(dtype="datetime64[ns]")
            # For each date pick latest announced report (highest period among announced)
            idx_list = []
            for d in dates:
                valid = np.flatnonzero(yb_ann <= d)
                idx_list.append(valid[-1] if len(valid) else -1)
            idx_arr = np.array(idx_list)
            has = idx_arr >= 0
            for col in ["q_num", "eps", "rev", "rev_yoy", "rev_qoq", "np", "np_yoy", "np_qoq", "bvps", "roe", "ocf_ps", "gross_margin"]:
                vals = np.full(len(g), np.nan, dtype=np.float64)
                vals[has] = yb[col].to_numpy()[idx_arr[has]]
                g[col] = vals
            p_arr = np.full(len(g), "", dtype=object)
            p_arr[has] = yb["period"].to_numpy()[idx_arr[has]]
            g["yb_period"] = p_arr
            ann_arr = np.full(len(g), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
            ann_arr[has] = yb_ann[idx_arr[has]]
            days_arr = np.full(len(g), np.nan, dtype=np.float64)
            days_arr[has] = ((dates[has] - ann_arr[has]) // np.timedelta64(1, "D")).astype(np.float64)
            g["days_since_ann"] = days_arr
        else:
            for col in ["q_num", "eps", "rev", "rev_yoy", "rev_qoq", "np", "np_yoy", "np_qoq", "bvps", "roe", "ocf_ps", "gross_margin", "days_since_ann"]:
                g[col] = np.nan
            g["yb_period"] = ""

        yg = yjyg_by_code.get(code)
        if yg is not None and len(yg):
            yg_ann = yg["ann_dt"].to_numpy(dtype="datetime64[ns]")
            yg_per = yg["period"].to_numpy()
            yb_p = g["yb_period"].to_numpy()
            gm = np.zeros(len(g), dtype=np.float64)
            gs = np.zeros(len(g), dtype=np.float64)
            for i, d in enumerate(dates):
                days_diff = (d - yg_ann) // np.timedelta64(1, "D")
                valid = np.flatnonzero((yg_ann <= d) & (days_diff <= 90) & (yg_per > yb_p[i]))
                if len(valid):
                    gm[i] = float(yg["yjyg_growth_mid"].iloc[valid[-1]])
                    gs[i] = float(yg["yjyg_sign"].iloc[valid[-1]])
            g["yjyg_growth_mid"] = gm
            g["yjyg_sign"] = gs
        else:
            g["yjyg_growth_mid"] = 0.0
            g["yjyg_sign"] = 0.0
        panel_parts.append(g)

    out = pd.concat(panel_parts, ignore_index=True)
    mcap = out["market_cap"].where(out["market_cap"] > 1e6, np.nan)
    scale = 4.0 / out["q_num"]
    out["ep_ann"] = (out["np"] * scale) / mcap
    out["sp_ann"] = (out["rev"] * scale) / mcap
    out["bp_implied"] = np.where(out["roe"] > 1.0, out["ep_ann"] / (out["roe"] / 100.0), np.nan)
    out["log_mcap"] = np.log(mcap)
    out["cash_quality"] = (out["ocf_ps"] - out["eps"]) / (out["bvps"].abs() + 1.0)
    out = out.rename(columns={"date": "asof_date"})

    # Cross-sectional rank-normalize features per asof_date into [-0.5, 0.5], NaN -> 0.0
    for feat in FEATURES:
        if feat in {"yjyg_sign", "lhb_count_20d"}:
            # Sparse event features: center around 0 via cross-sectional rank of non-zero or plain rank
            out[f"x_{feat}"] = out.groupby("asof_date")[feat].rank(pct=True, method="average") - 0.5
        else:
            ranked = out.groupby("asof_date")[feat].rank(pct=True, method="average") - 0.5
            out[f"x_{feat}"] = ranked.fillna(0.0)

    return out, regime


def fit_and_select_nonpv(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train & select NONPV strictly inside 2026-02-02..2026-07-02, then score 65-day OOS."""
    train_all = panel[(panel["asof_date"] >= TRAIN_START) & (panel["asof_date"] <= TRAIN_END)].dropna(
        subset=["actual_return_d10"]
    ).copy()
    train_all["y"] = train_all.groupby("asof_date")["actual_return_d10"].rank(pct=True, method="average") - 0.5

    fit_df = train_all[train_all["asof_date"] <= TRAIN_FIT_END]
    val_df = train_all[train_all["asof_date"] >= TRAIN_VAL_START]
    x_cols = [f"x_{f}" for f in FEATURES]

    X_fit, y_fit = fit_df[x_cols].to_numpy(dtype=np.float32), fit_df["y"].to_numpy(dtype=np.float32)
    X_val = val_df[x_cols].to_numpy(dtype=np.float32)

    candidates = []
    for alpha in [10.0, 100.0, 1000.0, 10000.0, 50000.0]:
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_fit, y_fit)
        pred = mdl.predict(X_val)
        ic = float(val_df.assign(_p=pred).groupby("asof_date").apply(
            lambda d: d["_p"].corr(d["actual_return_d10"], method="spearman"), include_groups=False
        ).mean())
        candidates.append({"name": f"ridge_{int(alpha)}", "kind": "ridge", "params": {"alpha": alpha}, "val_ic": ic})

    for depth in [2, 3]:
        for reg in [10.0, 50.0, 200.0]:
            mdl = XGBRegressor(
                max_depth=depth, reg_lambda=reg, n_estimators=150, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=20261115, n_jobs=4, tree_method="hist",
            )
            mdl.fit(X_fit, y_fit)
            pred = mdl.predict(X_val)
            ic = float(val_df.assign(_p=pred).groupby("asof_date").apply(
                lambda d: d["_p"].corr(d["actual_return_d10"], method="spearman"), include_groups=False
            ).mean())
            candidates.append({
                "name": f"xgb_d{depth}_l{int(reg)}", "kind": "xgb",
                "params": {"max_depth": depth, "reg_lambda": reg}, "val_ic": ic,
            })

    best = max(candidates, key=lambda c: c["val_ic"])
    X_full, y_full = train_all[x_cols].to_numpy(dtype=np.float32), train_all["y"].to_numpy(dtype=np.float32)
    if best["kind"] == "ridge":
        final_model = Ridge(alpha=best["params"]["alpha"]).fit(X_full, y_full)
        weights = dict(zip(FEATURES, [float(w) for w in final_model.coef_]))
    else:
        final_model = XGBRegressor(
            max_depth=best["params"]["max_depth"], reg_lambda=best["params"]["reg_lambda"],
            n_estimators=150, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            random_state=20261115, n_jobs=4, tree_method="hist",
        ).fit(X_full, y_full)
        weights = dict(zip(FEATURES, [float(w) for w in final_model.feature_importances_]))

    oos = panel[(panel["asof_date"] >= OOS_START) & (panel["asof_date"] <= OOS_END)].copy()
    oos["p_nonpv"] = final_model.predict(oos[x_cols].to_numpy(dtype=np.float32))
    info = {
        "fit_range": f"{TRAIN_START}..{TRAIN_FIT_END}", "fit_rows": int(len(fit_df)),
        "val_range": f"{TRAIN_VAL_START}..{TRAIN_END}", "val_rows": int(len(val_df)),
        "full_train_rows": int(len(train_all)),
        "candidates": candidates, "selected": best, "feature_weights_or_importance": weights,
    }
    return oos[["symbol", "asof_date", "p_nonpv", "med_amt_20d"]], info


def eval_cross_section(c2_dir: Path, nonpv_oos: pd.DataFrame, raw_path: Path) -> tuple[dict, pd.DataFrame]:
    """Part A: Paired 65-day OOS evaluation against C2 (16-path T=0.65, p=0.80)."""
    c2 = rc.load(c2_dir, OOS_START, OOS_END)
    frame = c2.merge(nonpv_oos, on=["symbol", "asof_date"], how="inner").dropna(
        subset=["predicted_return_d10", "p_nonpv", "actual_return_d10"]
    ).reset_index(drop=True)
    frame["p_c2"] = frame["predicted_return_d10"]
    frame["r_c2"] = fa.ranks(frame, "p_c2")
    frame["r_nonpv"] = fa.ranks(frame, "p_nonpv")
    frame["p_fuse"] = 0.5 * frame["r_c2"] + 0.5 * frame["r_nonpv"]
    frame["p_vet"] = np.where(frame["r_c2"] > 0.20, frame["r_nonpv"], frame["r_c2"] - 1.0)

    # Controls for Neutral IC
    frame["r_size"] = frame.groupby("asof_date")["size_decile"].rank(pct=True, method="average")
    frame["ret10"] = fa.reversal_10d(raw_path, frame)
    frame["r_rev"] = fa.ranks(frame, "ret10")

    # Market direction per signal day
    mkt_d10 = frame.groupby("asof_date")["actual_return_d10"].mean()
    up_dates = set(mkt_d10[mkt_d10 > 0].index)
    down_dates = set(mkt_d10[mkt_d10 <= 0].index)

    models = {"C2": "p_c2", "NONPV": "p_nonpv", "FUSE_50_50": "p_fuse", "NONPV_VET_C2": "p_vet"}
    ic_series = {}
    summary = {}
    for name, col in models.items():
        ic = fa.daily_ic(frame, col)
        ic_series[name] = ic
        nic = rc.neutral_daily_ic(frame, col)
        frame["_r"] = fa.ranks(frame, col)
        frame["_ex"] = frame["actual_return_d10"] - frame.groupby("asof_date")["actual_return_d10"].transform("mean")

        def slice_excess(sub_df: pd.DataFrame) -> dict:
            if len(sub_df) == 0:
                return {}
            top10pct = sub_df[sub_df["_r"] > 0.9].groupby("asof_date")["_ex"].mean().mean()
            bot10pct = sub_df[sub_df["_r"] <= 0.1].groupby("asof_date")["_ex"].mean().mean()
            liq = sub_df[sub_df["med_amt_20d"] >= 5e7].copy()
            liq["_rk"] = liq.groupby("asof_date")[col].rank(ascending=False, method="first")
            top10n = liq[liq["_rk"] <= 10].groupby("asof_date")["_ex"].mean().mean()
            top20n = liq[liq["_rk"] <= 20].groupby("asof_date")["_ex"].mean().mean()
            return {
                "top10pct_excess": float(top10pct), "bottom10pct_excess": float(bot10pct),
                "liq_top10_excess": float(top10n), "liq_top20_excess": float(top20n),
            }

        row = {
            **fa.ic_stats(ic),
            "neutral_ic": fa.ic_stats(nic)["ic"],
            "neutral_icir": fa.ic_stats(nic)["icir"],
            "all_days": slice_excess(frame),
            "up_days": slice_excess(frame[frame["asof_date"].isin(up_dates)]),
            "down_days": slice_excess(frame[frame["asof_date"].isin(down_dates)]),
        }
        if name != "C2":
            row["paired_vs_C2"] = fa.paired_bootstrap(ic_series["C2"], ic, seed=BOOTSTRAP_SEED)
        summary[name] = row

    corr_c2_nonpv = float(frame.groupby("asof_date").apply(
        lambda d: d["r_c2"].corr(d["r_nonpv"]), include_groups=False
    ).mean())
    out = {
        "aligned_rows": int(len(frame)), "aligned_dates": int(frame["asof_date"].nunique()),
        "up_market_days": len(up_dates), "down_market_days": len(down_dates),
        "c2_nonpv_mean_daily_rank_corr": corr_c2_nonpv,
        "models": summary,
    }
    return out, frame


def gated_plan(market: sub.Market, scores: np.ndarray, k: int, gate_scores: np.ndarray, use_gate: bool):
    """Few-name plan with 30% buffer and optional causal 3-tier regime gate (0 / k//2 / k slots)."""
    def plan(signal: int):
        order, pct = fnb.ranked_pool(market, scores, signal)
        if not use_gate:
            keep = ~np.isfinite(scores[signal])
            keep |= pct > 1.0 - fnb.BUFFER_KEEP_FRACTION
            return order, keep, k
        g = int(gate_scores[signal])
        if g == 0:
            return np.empty(0, dtype=int), np.zeros(scores.shape[1], dtype=bool), 0
        target_k = k if g == 2 else max(1, k // 2)
        keep = ~np.isfinite(scores[signal])
        keep |= pct > 1.0 - fnb.BUFFER_KEEP_FRACTION
        return order, keep, target_k
    return plan


def run_gated_slots(market: sub.Market, plan, rebalance_days: list[int], k: int, last_day: int) -> dict:
    """Extension of few_names_backtest.run_slots allowing target_k <= k (holding cash when gated)."""
    n_symbols = len(market.symbols)
    value = np.zeros(n_symbols)
    cash, previous_equity = 1.0, 1.0
    rebalance = set(rebalance_days)
    rows, orders, cash_rebalances = [], [], 0
    for day in range(rebalance_days[0] + 1, last_day + 1):
        held = value > 0
        if held.any():
            value[held] *= market.open[day, held] / market.close[day - 1, held]
        traded_value = cost = 0.0
        if day - 1 in rebalance:
            order, keep, target_k = plan(day - 1)
            if target_k == 0:
                cash_rebalances += 1
            leaving = held & ~keep
            # If target_k shrunk (e.g. half-risk), trim worst-ranked held names beyond target_k
            staying = np.flatnonzero(held & keep)
            if len(staying) > target_k:
                pos_in_order = {sym: i for i, sym in enumerate(order)}
                staying_sorted = sorted(staying, key=lambda s: pos_in_order.get(s, 10**9))
                for s in staying_sorted[target_k:]:
                    leaving[s] = True
            sells = leaving & market.traded[day] & ~market.one_down[day]
            if sells.any():
                proceeds = float(value[sells].sum())
                fee = float((value[sells] * market.side_cost[day, sells]).sum())
                cash += proceeds - fee
                cost += fee
                traded_value += proceeds
                value[sells] = 0.0
            held = value > 0
            free = max(target_k - int(held.sum()), 0)
            buyable = order[~held[order] & market.traded[day, order] & ~market.one_up[day, order]]
            buys = buyable[:free]
            equity = cash + float(value.sum())
            if len(buys) and cash > 1e-9 * equity:
                each = min(equity / k, cash / len(buys))
                rate = market.side_cost[day, buys]
                value[buys] = each * (1.0 - rate)
                cash -= each * len(buys)
                cost += float((each * rate).sum())
                traded_value += each * len(buys)
            orders.append(int(sells.sum()) + len(buys))
        held = value > 0
        if held.any():
            value[held] *= market.close[day, held] / market.open[day, held]
        equity = cash + float(value.sum())
        rows.append((market.dates[day], equity / previous_equity - 1.0, int(held.sum()),
                     float(value.sum()) / equity if equity > 0 else 0.0, cost))
        previous_equity = equity
    daily = pd.DataFrame(rows, columns=["date", "ret", "names", "exposure", "cost"]).set_index("date")
    return {"daily": daily, "orders": orders, "cash_rebalances": cash_rebalances}


def eval_portfolios(raw_path: Path, frame: pd.DataFrame, regime: pd.DataFrame) -> dict:
    """Part B: Few-name executable books (k=10, 20; step=10; 10 phases) +/- regime gate."""
    market = sub.Market(raw_path)
    last_day = len(market.dates) - 1
    signal_dates = sorted(frame["asof_date"].unique())
    signal_days = [market.date_index[d] for d in signal_dates]
    gate_scores = np.full(len(market.dates), 2, dtype=int)
    for d, idx in market.date_index.items():
        if d in regime.index:
            gate_scores[idx] = int(regime.loc[d, "score_mkt"])

    step = 10
    phases = [signal_days[p::step] for p in range(step)]
    score_cols = {"C2": "p_c2", "NONPV": "p_nonpv", "FUSE_50_50": "p_fuse", "NONPV_VET_C2": "p_vet"}
    score_mats = {}
    for name, col in score_cols.items():
        sc = frame[["symbol", "asof_date", col]].rename(columns={col: "predicted_return_d10"})
        score_mats[name] = market.score_matrix(sc)

    bench = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases]
    results = {}
    c2_daily_by_k = {}

    for k in (10, 20):
        for use_gate in (False, True):
            for name in ("C2", "NONPV", "FUSE_50_50", "NONPV_VET_C2"):
                if use_gate and name == "NONPV":
                    continue
                tag = f"{name}_k{k}_{'GATED' if use_gate else 'RAW'}"
                runs = [
                    run_gated_slots(market, gated_plan(market, score_mats[name], k, gate_scores, use_gate), days, k, last_day)
                    for days in phases
                ]
                nets = [r["daily"]["ret"] for r in runs]
                if tag == f"C2_k{k}_RAW":
                    c2_daily_by_k[k] = nets
                excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
                drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
                diff_univ = pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()
                diff_c2 = pd.concat([r - c for r, c in zip(nets, c2_daily_by_k[k])], axis=1).mean(axis=1).to_numpy()
                results[tag] = {
                    "k": k, "model": name, "gated": use_gate,
                    "mean_net": float(np.mean([fnb.total(r) for r in nets])),
                    "mean_universe_net": float(np.mean([fnb.total(b) for b in bench])),
                    "mean_net_excess": float(np.mean(excess)),
                    "net_excess_range": [float(min(excess)), float(max(excess))],
                    "beat_pool_phases": int(sum(x > 0 for x in excess)),
                    "phases": len(phases),
                    "mean_max_drawdown": float(np.mean(drawdowns)),
                    "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
                    "mean_orders": float(np.mean([np.mean(r["orders"][1:]) for r in runs if len(r["orders"]) > 1])),
                    "vs_universe_ci95": sub.block_ci(diff_univ),
                    "vs_c2_raw_ci95": sub.block_ci(diff_c2),
                    "vs_c2_mean_total_diff": float(np.mean([fnb.total(r) - fnb.total(c) for r, c in zip(nets, c2_daily_by_k[k])])),
                }
    gate_oos_dist = regime.loc[signal_dates, "score_mkt"].value_counts().sort_index().to_dict()
    return {"gate_oos_day_counts": {int(k): int(v) for k, v in gate_oos_dist.items()}, "books": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="scratch/oos65/combined_raw.csv")
    parser.add_argument("--nonpv-dir", default="scratch/non_pv")
    parser.add_argument("--c2-dir", default="scratch/non_pv/c2_p16_65d")
    parser.add_argument("--out", default="finetune/reports/20261115_nonpv_and_regime_results.json")
    args = parser.parse_args()

    panel, regime = build_panel(Path(args.raw), Path(args.nonpv_dir))
    nonpv_oos, train_info = fit_and_select_nonpv(panel)
    cs_res, aligned_frame = eval_cross_section(Path(args.c2_dir), nonpv_oos, Path(args.raw))
    port_res = eval_portfolios(Path(args.raw), aligned_frame, regime)

    report = {
        "train_nonpv": train_info,
        "part_a_cross_section": cs_res,
        "part_b_portfolios": port_res,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
