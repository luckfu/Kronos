"""Study E2: No-G32 Crash-Shield Re-Composition, Hybrid Gate, and Mid-Cycle Daily Circuit Breaker (MCB).

Strictly evaluates on 65-day OOS (2026-07-17 .. 2026-10-23, 335,312 rows), keeping 2026-10-24+ untouched.
All gates, ensembles (equal-weight cross-sectional rank average), and mid-cycle triggers are strictly causal
(computed <= decision_date <= 2026-10-23).
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

for _mod in ("sklearn", "sklearn.linear_model", "xgboost"):
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        if _mod == "sklearn.linear_model":
            _m.Ridge = object
        if _mod == "xgboost":
            _m.XGBRegressor = object
        sys.modules[_mod] = _m

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OOS_MOD = ROOT / "finetune" / "oos_eval" / "base_clean_sealed_oos"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OOS_MOD))

import eval_adaptive_execution_shield as e1  # noqa: E402
import eval_wave4_master_ensembles as ew4  # noqa: E402
import few_names_backtest as fnb  # noqa: E402
import fusion_analysis as fa  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402

OOS_START = "2026-07-17"
OOS_END = "2026-10-23"

SOURCES: dict[str, tuple[str, list[str]]] = {
    "g27": ("scratch/gpu_alpha_g27_out/oos_g27_scores.csv.gz", ["p_g27_resid_mlp_cosic"]),
    "g29": ("scratch/gpu_alpha_g29_out/oos_g29_scores.csv.gz", ["p_g29_mini_l45_pool_regime"]),
    "g30": ("scratch/gpu_alpha_g30_out/oos_g30_scores.csv.gz", ["p_g30_quad_resid_cosic", "p_g30_quad_xstock_attn"]),
    "g31": (
        "scratch/gpu_alpha_g31_out/oos_g31_scores.csv.gz",
        ["p_g31_c2_l45_rich_regime", "p_g31_minic2_joint_regime", "p_g31_mini_l45_dep_s2"],
    ),
    "g32": ("scratch/gpu_alpha_g32_out/oos_g32_scores.csv.gz", ["p_g32_c2_l45_ar16_d10_mean"]),
    "g33": ("scratch/gpu_alpha_g33_out/oos_g33_scores.csv.gz", ["p_g33_base_l52_rich_ret20"]),
    "g34": ("scratch/gpu_alpha_g34_out/oos_g34_scores.csv.gz", ["p_g34_dep_c2_raw"]),
    "g35": (
        "scratch/gpu_alpha_g35_out/oos_g35_scores.csv.gz",
        [
            "p_g35_lgbm_cvar10",
            "p_g35_lgbm_nowhipsaw",
            "p_g35_ridge_nowhipsaw",
            "p_g35_ridge_crashentry",
            "p_g35_tailshield_cvar_shield35",
            "p_g35_tailshield_nowhipsaw_shield35",
        ],
    ),
    "g36": ("scratch/gpu_alpha_g36_out/oos_g36_scores.csv.gz", ["p_g36_quad_liq_ridge"]),
    "g37": (
        "scratch/gpu_alpha_g37_out/oos_g37_scores.csv.gz",
        [
            "p_g37_c2only_distill_resid",
            "p_g37_c2only_distill_xstock",
            "p_g37_minic2_distill_resid",
            "p_g37_minic2_distill_xstock",
        ],
    ),
    "g38": ("scratch/gpu_alpha_g38_out/oos_g38_scores.csv.gz", ["p_g38_ridge_fwd5_10", "p_g38_lgbm_fwd5_10", "p_g38_consistent_d5_d10"]),
}

E2_ENSEMBLES: dict[str, list[str]] = {
    "ENS_W2F_C2_MINI29": ["r_c2", "r_p_g29_mini_l45_pool_regime"],
    "W3_C2ONLY_FULL4": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_c2_l45_rich_regime",
        "r_p_g34_dep_c2_raw",
    ],
    "NOG32_C2ONLY_DISTILL5": [
        "r_c2",
        "r_p_g31_c2_l45_rich_regime",
        "r_p_g34_dep_c2_raw",
        "r_p_g37_c2only_distill_resid",
        "r_p_g37_c2only_distill_xstock",
    ],
    "W4_LIGHT_MINIC2_DISTILL": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g37_minic2_distill_resid",
        "r_p_g37_minic2_distill_xstock",
    ],
    "NOG32_LIGHT_MINIC2_DISTILL6": [
        "r_c2",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_mini_l45_dep_s2",
        "r_p_g37_minic2_distill_resid",
        "r_p_g37_minic2_distill_xstock",
    ],
    "W4_CRASH_SHIELD_5PILLAR": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g35_tailshield_cvar_shield35",
        "r_p_g35_tailshield_nowhipsaw_shield35",
        "r_p_g35_ridge_crashentry",
        "r_p_g38_consistent_d5_d10",
        "r_p_g30_quad_xstock_attn",
    ],
    "WITHG32_CRASH_SHIELD_CVAR10": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g30_quad_xstock_attn",
        "r_p_g35_lgbm_cvar10",
        "r_p_g35_tailshield_cvar_shield35",
        "r_p_g35_tailshield_nowhipsaw_shield35",
        "r_p_g38_ridge_fwd5_10",
    ],
    "NOG32_CRASH_SHIELD_CORE": [
        "r_c2",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g30_quad_resid_cosic",
        "r_p_g35_lgbm_cvar10",
        "r_p_g35_tailshield_cvar_shield35",
        "r_p_g35_tailshield_nowhipsaw_shield35",
        "r_p_g35_ridge_crashentry",
        "r_p_g38_ridge_fwd5_10",
    ],
    "NOG32_TAIL_CONT_APEX": [
        "r_c2",
        "r_p_g27_resid_mlp_cosic",
        "r_p_g30_quad_resid_cosic",
        "r_p_g35_lgbm_cvar10",
        "r_p_g35_lgbm_nowhipsaw",
        "r_p_g35_ridge_nowhipsaw",
        "r_p_g36_quad_liq_ridge",
        "r_p_g38_ridge_fwd5_10",
        "r_p_g38_lgbm_fwd5_10",
    ],
}

MCB_MODELS = [
    "C2",
    "ENS_W2F_C2_MINI29",
    "NOG32_C2ONLY_DISTILL5",
    "W4_LIGHT_MINIC2_DISTILL",
    "NOG32_LIGHT_MINIC2_DISTILL6",
    "W4_CRASH_SHIELD_5PILLAR",
    "WITHG32_CRASH_SHIELD_CVAR10",
    "NOG32_CRASH_SHIELD_CORE",
    "NOG32_TAIL_CONT_APEX",
]


def compute_e2_gates(raw_path: Path, haz_path: Path) -> pd.DataFrame:
    """Compute strictly causal E2 gates including G_HYBRID_MIN and MCB_PANIC_TIER."""
    reg, _ = e1.compute_causal_breadth_and_vol_mask(raw_path)
    reg["delta_b20_3d"] = reg["breadth_ma20"] - reg["breadth_ma20"].shift(3)
    haz_df = pd.read_csv(haz_path)
    haz_df["asof_date"] = haz_df["asof_date"].astype(str)
    haz_idx = haz_df.set_index("asof_date")
    aligned_b20 = reg["breadth_ma20"].reindex(haz_idx.index).ffill().fillna(0.5)
    reg.loc[haz_idx.index, "G_HAZARD_G38"] = ew4.compute_hazard_gate_tiers(aligned_b20, haz_idx["p_crash_hazard"])
    reg["G_HAZARD_G38"] = reg["G_HAZARD_G38"].fillna(reg["G_BREADTH"]).astype(int)
    reg["G_HYBRID_MIN"] = np.minimum(reg["G_BREADTH_VEL"], reg["G_HAZARD_G38"]).astype(int)
    panic = (reg["breadth_ma20"] < 0.28) | (reg["delta_b20_3d"] <= -0.15) | (reg["delta_b20_5d"] <= -0.18)
    reg["MCB_PANIC_TIER"] = np.where(panic, 0, np.minimum(2, reg["G_BREADTH_VEL"])).astype(int)
    return reg


def run_mcb_gated_slots(
    market: sub.Market,
    plan,
    rebalance_days: list[int],
    k: int,
    last_day: int,
    *,
    mcb_tier: np.ndarray | None = None,
    enable_mcb: bool = False,
    max_decision_day: int | None = None,
    extra_side_cost: float = 0.0,
) -> dict:
    """Extension of nrs.run_gated_slots adding strictly causal downward-only Mid-Cycle Circuit Breaker."""
    n_symbols = len(market.symbols)
    value = np.zeros(n_symbols)
    cash, previous_equity = 1.0, 1.0
    rebalance = set(rebalance_days)
    limit_day = rebalance_days[-1] if max_decision_day is None else max_decision_day
    rows, orders, cash_rebalances = [], [], 0
    mcb_triggers, mcb_sells = 0, 0
    last_reb_order = np.empty(0, dtype=int)
    active_tier = 2

    for day in range(rebalance_days[0] + 1, last_day + 1):
        held = value > 0
        if held.any():
            value[held] *= market.open[day, held] / market.close[day - 1, held]
        cost = 0.0
        if day - 1 in rebalance:
            order, keep, target_k = plan(day - 1)
            if len(order):
                last_reb_order = order
            active_tier = 0 if target_k <= 0 else (1 if target_k < k else 2)
            if target_k == 0:
                cash_rebalances += 1
            leaving = held & ~keep
            staying = np.flatnonzero(held & keep)
            if len(staying) > target_k:
                pos_in_order = {sym: i for i, sym in enumerate(order)}
                staying_sorted = sorted(staying, key=lambda s: pos_in_order.get(s, 10**9))
                for s in staying_sorted[target_k:]:
                    leaving[s] = True
            sells = leaving & market.traded[day] & ~market.one_down[day]
            if sells.any():
                proceeds = float(value[sells].sum())
                fee = float((value[sells] * (market.side_cost[day, sells] + extra_side_cost)).sum())
                cash += proceeds - fee
                cost += fee
                value[sells] = 0.0
            held = value > 0
            free = max(target_k - int(held.sum()), 0)
            buyable = order[~held[order] & market.traded[day, order] & ~market.one_up[day, order]]
            buys = buyable[:free]
            equity = cash + float(value.sum())
            if len(buys) and cash > 1e-9 * equity:
                each = min(equity / k, cash / len(buys))
                rate = market.side_cost[day, buys] + extra_side_cost
                value[buys] = each * (1.0 - rate)
                cash -= each * len(buys)
                cost += float((each * rate).sum())
            orders.append(int(sells.sum()) + len(buys))
        elif enable_mcb and mcb_tier is not None and (day - 1 <= limit_day):
            cand_tier = int(mcb_tier[day - 1])
            if cand_tier < active_tier:
                active_tier = cand_tier
                mcb_triggers += 1
            if active_tier < 2:
                target_k = 0 if active_tier == 0 else max(1, k // 2)
                if int(held.sum()) > target_k:
                    leaving = np.zeros(n_symbols, dtype=bool)
                    if target_k == 0:
                        leaving = held.copy()
                    else:
                        staying = np.flatnonzero(held)
                        pos_in_order = {sym: i for i, sym in enumerate(last_reb_order)}
                        staying_sorted = sorted(staying, key=lambda s: pos_in_order.get(s, 10**9))
                        for s in staying_sorted[target_k:]:
                            leaving[s] = True
                    sells = leaving & market.traded[day] & ~market.one_down[day]
                    if sells.any():
                        proceeds = float(value[sells].sum())
                        fee = float((value[sells] * (market.side_cost[day, sells] + extra_side_cost)).sum())
                        cash += proceeds - fee
                        cost += fee
                        value[sells] = 0.0
                        mcb_sells += int(sells.sum())
        held = value > 0
        if held.any():
            value[held] *= market.close[day, held] / market.open[day, held]
        equity = cash + float(value.sum())
        rows.append((
            market.dates[day],
            equity / previous_equity - 1.0,
            int(held.sum()),
            float(value.sum()) / equity if equity > 0 else 0.0,
            cost,
        ))
        previous_equity = equity
    daily = pd.DataFrame(rows, columns=["date", "ret", "names", "exposure", "cost"]).set_index("date")
    return {
        "daily": daily,
        "orders": orders,
        "cash_rebalances": cash_rebalances,
        "mcb_triggers": mcb_triggers,
        "mcb_sells": mcb_sells,
    }


def _summarize_runs(runs: list[dict], bench: list[pd.Series], c2_nets: list[pd.Series] | None = None) -> dict:
    nets = [r["daily"]["ret"] for r in runs]
    excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
    drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
    mean_net = float(np.mean([fnb.total(r) for r in nets]))
    mean_dd = float(np.mean(drawdowns))
    diff_univ = pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()
    out = {
        "mean_net": mean_net,
        "mean_universe_net": float(np.mean([fnb.total(b) for b in bench])),
        "mean_net_excess": float(np.mean(excess)),
        "beat_pool_phases": int(sum(x > 0 for x in excess)),
        "phases": len(runs),
        "mean_max_drawdown": mean_dd,
        "calmar_like": float(mean_net / abs(mean_dd)) if abs(mean_dd) > 1e-9 else 0.0,
        "mean_exposure": float(np.mean([r["daily"]["exposure"].mean() for r in runs])),
        "mean_total_cost": float(np.mean([r["daily"]["cost"].sum() for r in runs])),
        "mean_mcb_triggers": float(np.mean([r["mcb_triggers"] for r in runs])),
        "mean_mcb_sells": float(np.mean([r["mcb_sells"] for r in runs])),
        "vs_universe_ci95": sub.block_ci(diff_univ),
    }
    if c2_nets is not None:
        diff_c2 = pd.concat([r - c for r, c in zip(nets, c2_nets)], axis=1).mean(axis=1).to_numpy()
        out["vs_c2_ci95"] = sub.block_ci(diff_c2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Study E2: No-G32 + Mid-Cycle Circuit Breaker")
    parser.add_argument("--raw", default="scratch/oos65/combined_raw.csv")
    parser.add_argument("--c2-dir", default="scratch/non_pv/c2_p16_65d")
    parser.add_argument("--haz", default="scratch/gpu_alpha_g38_out/oos_g38_regime_hazard.csv")
    parser.add_argument("--out", default="finetune/reports/20261128_e2_nog32_midcycle_shield_results.json")
    args = parser.parse_args()

    raw_path = ROOT / args.raw
    c2_dir = ROOT / args.c2_dir
    haz_path = ROOT / args.haz
    out_json = ROOT / args.out

    c2 = fa.load_model(c2_dir)
    c2["asof_date"] = c2["asof_date"].astype(str)
    c2 = c2[(c2["asof_date"] >= OOS_START) & (c2["asof_date"] <= OOS_END)].reset_index(drop=True)
    assert c2["asof_date"].max() <= OOS_END, "Sealed window violation!"

    merged = c2.copy()
    merged["r_c2"] = fa.ranks(merged, "predicted_return_d10")
    for tag, (rel_path, req_cols) in SOURCES.items():
        df = pd.read_csv(ROOT / rel_path, usecols=["identity", "asof_date"] + req_cols)
        df["asof_date"] = df["asof_date"].astype(str)
        assert df["asof_date"].max() <= OOS_END, f"Sealed violation in {tag}"
        merged = merged.merge(df[["identity"] + req_cols], on="identity", how="inner", validate="one_to_one")
        if len(merged) != len(c2):
            raise RuntimeError(f"Alignment mismatch after {tag}: {len(merged)} vs {len(c2)}")
        for sc in req_cols:
            merged[f"r_{sc}"] = fa.ranks(merged, sc)

    cols: dict[str, str] = {"C2": "predicted_return_d10"}
    ens_def: dict[str, list[str]] = {"C2": ["predicted_return_d10"]}
    for name, members in E2_ENSEMBLES.items():
        merged[name] = merged[members].mean(axis=1)
        cols[name] = name
        ens_def[name] = list(members)

    merged["rev10"] = fa.reversal_10d(raw_path, merged)
    raw_df = pd.read_csv(raw_path, dtype={"symbol": str})
    raw_df["date"] = raw_df["date"].astype(str)
    traded = raw_df[raw_df["volume"] > 0].sort_values(["symbol", "date"]).copy()
    traded["med_amt_20d"] = traded.groupby("symbol")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    liq_map = traded[["symbol", "date", "med_amt_20d"]].rename(columns={"date": "asof_date"})
    merged = merged.merge(liq_map, on=["symbol", "asof_date"], how="left")
    liq = merged[merged["med_amt_20d"] >= 5e7].copy()

    base_ic = fa.daily_ic(merged, "predicted_return_d10")
    merged["resid_actual"] = fa.residualize(merged, "actual_return_d10", ["size_decile", "rev10"], sector=True)

    cross_section = {}
    for name, col in cols.items():
        ic_s = fa.daily_ic(merged, col)
        res_col = fa.residualize(merged, col, ["size_decile", "rev10"], sector=True)
        tmp = pd.DataFrame({"asof_date": merged["asof_date"], "res": res_col, "act": merged["resid_actual"]})
        nic_s = tmp.groupby("asof_date", sort=False).apply(
            lambda g: g["res"].corr(g["act"], method="spearman"), include_groups=False
        )
        univ_daily = merged.groupby("asof_date", sort=False)["actual_return_d10"].mean()
        t10 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(10)
        t20 = liq.sort_values(["asof_date", col], ascending=[True, False]).groupby("asof_date", sort=False).head(20)
        entry = {
            **fa.ic_stats(ic_s),
            "ic_seg_a": float(ic_s.loc[ic_s.index <= "2026-09-10"].mean()),
            "ic_seg_b": float(ic_s.loc[ic_s.index >= "2026-09-15"].mean()),
            "neutral_ic": float(nic_s.mean()),
            "neutral_icir": float(nic_s.mean() / nic_s.std(ddof=1)),
            "top10pct_excess": fa.top_bottom(merged, col, 0.1),
            "liq_top10_excess": float((t10.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean()),
            "liq_top20_excess": float((t20.groupby("asof_date")["actual_return_d10"].mean() - univ_daily).mean()),
        }
        if name != "C2":
            entry["paired_vs_C2"] = fa.paired_bootstrap(base_ic, ic_s, seed=20261128)
        cross_section[name] = entry

    market = sub.Market(raw_path)
    last_day = len(market.dates) - 1
    signal_dates = sorted(merged["asof_date"].unique())
    signal_days = [market.date_index[d] for d in signal_dates]
    max_dec_day = max(signal_days)
    assert market.dates[max_dec_day] <= OOS_END

    reg = compute_e2_gates(raw_path, haz_path)
    gate_arrays = {}
    for gname in ("RAW", "G_BREADTH", "G_BREADTH_VEL", "G_HAZARD_G38", "G_HYBRID_MIN", "MCB_PANIC_TIER"):
        arr = np.full(len(market.dates), 2, dtype=int)
        if gname != "RAW":
            for d, idx in market.date_index.items():
                if d in reg.index:
                    arr[idx] = int(reg.loc[d, gname])
        gate_arrays[gname] = arr

    score_mats = {
        name: market.score_matrix(merged[["symbol", "asof_date", col]].rename(columns={col: "predicted_return_d10"}))
        for name, col in cols.items()
    }

    books = {}
    c2_raw_nets = {}
    phases10 = [signal_days[p::10] for p in range(10)]
    bench10 = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases10]
    for k in (10, 20):
        for name in cols:
            runs = [
                run_mcb_gated_slots(market, nrs.gated_plan(market, score_mats[name], k, gate_arrays["RAW"], False), days, k, last_day)
                for days in phases10
            ]
            if name == "C2":
                c2_raw_nets[k] = [r["daily"]["ret"] for r in runs]
            books[f"{name}_k{k}"] = {"k": k, "model": name, **_summarize_runs(runs, bench10, c2_raw_nets[k])}

    shield_specs = [
        ("RAW", "RAW", False, 0.0),
        ("G_BREADTH", "G_BREADTH", False, 0.0),
        ("G_BREADTH_VEL", "G_BREADTH_VEL", False, 0.0),
        ("G_HAZARD_G38", "G_HAZARD_G38", False, 0.0),
        ("G_HYBRID_MIN", "G_HYBRID_MIN", False, 0.0),
        ("RAW_MCB", "RAW", True, 0.0),
        ("G_BREADTH_VEL_MCB", "G_BREADTH_VEL", True, 0.0),
        ("G_HAZARD_G38_MCB", "G_HAZARD_G38", True, 0.0),
        ("G_HYBRID_MIN_MCB", "G_HYBRID_MIN", True, 0.0),
        ("STRESS25BP_RAW_MCB", "RAW", True, 0.0010),
        ("STRESS25BP_G_HYBRID_MIN_MCB", "G_HYBRID_MIN", True, 0.0010),
    ]
    shield_books = {}
    for step in (10, 5):
        phases = [signal_days[p::step] for p in range(step)]
        bench = [fnb.universe(market, score_mats["C2"], days[0], last_day) for days in phases]
        for k in (10, 20):
            for sname, g_key, enable_mcb, extra_cost in shield_specs:
                use_gate = g_key != "RAW"
                for mname in MCB_MODELS:
                    runs = [
                        run_mcb_gated_slots(
                            market,
                            nrs.gated_plan(market, score_mats[mname], k, gate_arrays[g_key], use_gate),
                            days,
                            k,
                            last_day,
                            mcb_tier=gate_arrays["MCB_PANIC_TIER"],
                            enable_mcb=enable_mcb,
                            max_decision_day=max_dec_day,
                            extra_side_cost=extra_cost,
                        )
                        for days in phases
                    ]
                    shield_books[f"step{step}_k{k}_{sname}_{mname}"] = {
                        "step": step,
                        "k": k,
                        "shield": sname,
                        "base_gate": g_key,
                        "enable_mcb": enable_mcb,
                        "extra_side_cost": extra_cost,
                        "model": mname,
                        **_summarize_runs(runs, bench),
                    }

    gate_counts = {
        g: {int(k): int(v) for k, v in reg.loc[signal_dates, g].value_counts().sort_index().items()}
        for g in ("G_BREADTH", "G_BREADTH_VEL", "G_HAZARD_G38", "G_HYBRID_MIN", "MCB_PANIC_TIER")
    }
    payload = {
        "oos_range": [OOS_START, OOS_END],
        "ensembles": ens_def,
        "cross_section": cross_section,
        "books": books,
        "gate_oos_day_counts": gate_counts,
        "shield_books": shield_books,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{'NAME':<32} {'IC':>7} {'ICIR':>5} {'POS':>6} {'SEGA/B':>11} {'NIC':>7} {'LIQ10':>7} {'K10_NET':>8} {'K10_DD':>8} {'K10_CAL':>7}")
    for name in cols:
        v = cross_section[name]
        b10 = books[f"{name}_k10"]
        print(
            f"{name:<32} {v['ic']:7.4f} {v['icir']:5.2f} {v['positive_ratio']*100:5.1f}% "
            f"{v['ic_seg_a']:.3f}/{v['ic_seg_b']:.3f} {v['neutral_ic']:7.4f} {v['liq_top10_excess']*100:+6.2f}% "
            f"{b10['mean_net']*100:+7.2f}% {b10['mean_max_drawdown']*100:+7.2f}% {b10['calmar_like']:7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
