"""Study E1: Adaptive Execution Shield (Rebalance Cadence step=5 vs 10, Breadth-Velocity Gate, LowVol-85% Filter).

Strictly evaluates on the 65-day clean OOS window (2026-07-17 .. 2026-10-23), keeping 2026-10-24+ sealed untouched.
All gate thresholds and filters are pre-registered and strictly causal (computed <= asof_date).
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

import few_names_backtest as fnb  # noqa: E402
import fusion_analysis as fa  # noqa: E402
import nonpv_and_regime_study as nrs  # noqa: E402
import signal_usage_backtest as sub  # noqa: E402

OOS_START = "2026-07-17"
OOS_END = "2026-10-23"

E1_MODELS: dict[str, list[str]] = {
    "C2": ["r_c2"],
    "ENS_W2F_C2_MINI29": ["r_c2", "r_p_g29_mini_l45_pool_regime"],
    "W3_LIGHT_C2AR_MINIC2_5WAY": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g31_minic2_joint_ret5",
        "r_p_g31_mini_l45_dep_s2",
    ],
    "W3_FLAGSHIP_5PILLAR_WITH_S1S2": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g31_minic2_joint_regime",
        "r_p_g33_base_l52_rich_ret20",
        "r_p_g34_dep_plus_stats_regime",
        "r_p_g34_s1s2_stats_quad40",
        "r_p_g30_quad_film_rankw",
        "r_p_g30_quad_xstock_attn",
        "r_p_g27_film_regime_mlp",
    ],
    "W3_SUPREME_AR_W2_LGBMRET5": [
        "r_c2",
        "r_p_g32_c2_l45_ar16_d10_mean",
        "r_p_g29_mini_l45_pool_regime",
        "r_p_g25_base_l52_pool",
        "r_p_g27_film_regime_mlp",
        "r_p_g30_quad_lgbm_ret5",
    ],
}

SOURCES: dict[str, tuple[str, list[str]]] = {
    "g25": ("scratch/gpu_alpha_g25_out/oos_g25_scores.csv.gz", ["p_g25_base_l52_pool"]),
    "g27": ("scratch/gpu_alpha_g27_out/oos_g27_scores.csv.gz", ["p_g27_film_regime_mlp"]),
    "g29": ("scratch/gpu_alpha_g29_out/oos_g29_scores.csv.gz", ["p_g29_mini_l45_pool_regime"]),
    "g30": ("scratch/gpu_alpha_g30_out/oos_g30_scores.csv.gz", ["p_g30_quad_film_rankw", "p_g30_quad_xstock_attn", "p_g30_quad_lgbm_ret5"]),
    "g31": ("scratch/gpu_alpha_g31_out/oos_g31_scores.csv.gz", ["p_g31_minic2_joint_regime", "p_g31_minic2_joint_ret5", "p_g31_mini_l45_dep_s2"]),
    "g32": ("scratch/gpu_alpha_g32_out/oos_g32_scores.csv.gz", ["p_g32_c2_l45_ar16_d10_mean"]),
    "g33": ("scratch/gpu_alpha_g33_out/oos_g33_scores.csv.gz", ["p_g33_base_l52_rich_ret20"]),
    "g34": ("scratch/gpu_alpha_g34_out/oos_g34_scores.csv.gz", ["p_g34_dep_plus_stats_regime", "p_g34_s1s2_stats_quad40"]),
}


def compute_causal_breadth_and_vol_mask(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute strictly causal daily breadth_ma20, breadth_ma5, delta_b20_5d, and per-stock 20d realized vol."""
    raw = pd.read_csv(raw_path, dtype={"symbol": str})
    raw["date"] = raw["date"].astype(str)
    traded = raw[raw["volume"] > 0].sort_values(["symbol", "date"]).copy()
    traded["ma20"] = traded.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    traded["ma5"] = traded.groupby("symbol")["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    traded["lret"] = np.log(np.maximum(traded["close"], 1e-8) / np.maximum(traded.groupby("symbol")["close"].shift(1), 1e-8))
    traded["vol20"] = traded.groupby("symbol")["lret"].transform(lambda s: s.rolling(20, min_periods=20).std())
    traded["vol20_pct"] = traded.groupby("date")["vol20"].rank(pct=True, method="average")

    b20 = traded.dropna(subset=["ma20"]).groupby("date").apply(lambda g: float((g["close"] > g["ma20"]).mean()), include_groups=False).rename("breadth_ma20")
    b5 = traded.dropna(subset=["ma5"]).groupby("date").apply(lambda g: float((g["close"] > g["ma5"]).mean()), include_groups=False).rename("breadth_ma5")
    reg = pd.concat([b20, b5], axis=1).sort_index()
    reg["delta_b20_5d"] = reg["breadth_ma20"] - reg["breadth_ma20"].shift(5)
    reg["G_BREADTH"] = np.where(reg["breadth_ma20"] < 0.30, 0, np.where(reg["breadth_ma20"] < 0.45, 1, 2))
    vel_warn = (reg["delta_b20_5d"] <= -0.12) | (reg["breadth_ma5"] < (reg["breadth_ma20"] - 0.10))
    reg["G_BREADTH_VEL"] = np.where(
        reg["breadth_ma20"] < 0.30,
        0,
        np.where(vel_warn, np.maximum(reg["G_BREADTH"] - 1, 0), reg["G_BREADTH"]),
    )
    vol_df = traded[["symbol", "date", "vol20_pct"]].rename(columns={"date": "asof_date"})
    return reg, vol_df


def run_shield_suite(raw_path: Path, c2_dir: Path, out_json: Path) -> dict:
    c2 = fa.load_model(c2_dir)
    c2["asof_date"] = c2["asof_date"].astype(str)
    c2 = c2[(c2["asof_date"] >= OOS_START) & (c2["asof_date"] <= OOS_END)].reset_index(drop=True)
    assert c2["asof_date"].max() <= OOS_END

    merged = c2.copy()
    merged["r_c2"] = fa.ranks(merged, "predicted_return_d10")
    for tag, (rel_path, cols) in SOURCES.items():
        df = pd.read_csv(ROOT / rel_path, usecols=["identity", "asof_date"] + cols)
        df["asof_date"] = df["asof_date"].astype(str)
        assert df["asof_date"].max() <= OOS_END
        merged = merged.merge(df[["identity"] + cols], on="identity", how="inner", validate="one_to_one")
        for sc in cols:
            merged[f"r_{sc}"] = fa.ranks(merged, sc)

    reg, vol_df = compute_causal_breadth_and_vol_mask(raw_path)
    merged = merged.merge(vol_df, on=["symbol", "asof_date"], how="left")
    is_lowvol85 = merged["vol20_pct"].fillna(0.5) <= 0.85

    market = sub.Market(raw_path)
    last_day = len(market.dates) - 1
    signal_dates = sorted(merged["asof_date"].unique())
    signal_days = [market.date_index[d] for d in signal_dates]

    gate_arrays = {}
    for gname in ("RAW", "G_BREADTH", "G_BREADTH_VEL"):
        arr = np.full(len(market.dates), 2, dtype=int)
        if gname != "RAW":
            for d, idx in market.date_index.items():
                if d in reg.index:
                    arr[idx] = int(reg.loc[d, gname])
        gate_arrays[gname] = arr

    score_mats_raw = {}
    score_mats_lv85 = {}
    for name, members in E1_MODELS.items():
        merged[name] = merged[members].mean(axis=1)
        score_mats_raw[name] = market.score_matrix(merged[["symbol", "asof_date", name]].rename(columns={name: "predicted_return_d10"}))
        lv_col = f"{name}_LV85"
        merged[lv_col] = np.where(is_lowvol85, merged[name], merged[name] - 10.0)
        score_mats_lv85[name] = market.score_matrix(merged[["symbol", "asof_date", lv_col]].rename(columns={lv_col: "predicted_return_d10"}))

    results = {}
    for step in (10, 5):
        phases = [signal_days[p::step] for p in range(step)]
        bench = [fnb.universe(market, score_mats_raw["C2"], days[0], last_day) for days in phases]
        for k in (10, 20):
            for shield in ("RAW", "G_BREADTH", "G_BREADTH_VEL", "G_BREADTH_VEL_LV85"):
                g_key = "G_BREADTH_VEL" if shield == "G_BREADTH_VEL_LV85" else shield
                use_gate = shield != "RAW"
                s_mats = score_mats_lv85 if shield == "G_BREADTH_VEL_LV85" else score_mats_raw
                for mname in E1_MODELS:
                    runs = [
                        nrs.run_gated_slots(market, nrs.gated_plan(market, s_mats[mname], k, gate_arrays[g_key], use_gate), days, k, last_day)
                        for days in phases
                    ]
                    nets = [r["daily"]["ret"] for r in runs]
                    excess = [fnb.total(r) - fnb.total(b) for r, b in zip(nets, bench)]
                    drawdowns = [float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()) for r in nets]
                    mean_net = float(np.mean([fnb.total(r) for r in nets]))
                    mean_dd = float(np.mean(drawdowns))
                    diff_univ = pd.concat([r - b for r, b in zip(nets, bench)], axis=1).mean(axis=1).to_numpy()
                    key = f"step{step}_k{k}_{shield}_{mname}"
                    results[key] = {
                        "step": step,
                        "k": k,
                        "shield": shield,
                        "model": mname,
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
                    }

    gate_oos_counts = {
        g: {int(k): int(v) for k, v in reg.loc[signal_dates, g].value_counts().sort_index().items()}
        for g in ("G_BREADTH", "G_BREADTH_VEL")
    }
    payload = {"oos_range": [OOS_START, OOS_END], "gate_oos_day_counts": gate_oos_counts, "books": results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="scratch/oos65/combined_raw.csv")
    parser.add_argument("--c2-dir", default="scratch/non_pv/c2_p16_65d")
    parser.add_argument("--out", default="finetune/reports/20261127_e1_execution_shield_results.json")
    args = parser.parse_args()
    payload = run_shield_suite(ROOT / args.raw, ROOT / args.c2_dir, ROOT / args.out)
    print("GATE COUNTS:", json.dumps(payload["gate_oos_day_counts"]))
    for key, v in sorted(payload["books"].items()):
        if v["k"] == 10 and v["model"] in ("C2", "W3_LIGHT_C2AR_MINIC2_5WAY", "W3_FLAGSHIP_5PILLAR_WITH_S1S2"):
            print(
                f"{key:<58} net={v['mean_net']*100:+6.2f}% dd={v['mean_max_drawdown']*100:+6.2f}% "
                f"cal={v['calmar_like']:5.3f} exp={v['mean_exposure']*100:4.1f}% beat={v['beat_pool_phases']}/{v['phases']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
