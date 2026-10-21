"""Direction + amplitude panel on the 34-date OOS, for c2_best vs clean_best (C03).

Adds a halt-gap subset: the OOS labels are the next 10 *actually traded* rows, so a
stock that was suspended has its label dates pushed out relative to the market.
Rows whose label lag exceeds the cross-sectional norm are the halt-gap population.
"""
import glob
import numpy as np
import pandas as pd

SHARDS = "artifacts/small_clean_oos_shards/small_clean_oos/shards/*.csv.gz"
COST = 0.004
H = list(range(1, 11))


def spearman(x, y):
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else np.nan


frames = {}
for path in sorted(glob.glob(SHARDS)):
    df = pd.read_csv(path, compression="gzip")
    label = df["model"].iloc[0]
    frames.setdefault(label, []).append(df)
data = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
for k, v in data.items():
    print(f"{k}: rows={len(v)} dates={v.asof_date.nunique()} symbols={v.symbol.nunique()}")


def halt_flag(df):
    """Label rows whose 10-day label horizon is pushed out by suspension."""
    df = df.copy()
    for h in H:
        df[f"lag{h}"] = (pd.to_datetime(df[f"label_date_d{h}"]) - pd.to_datetime(df["asof_date"])).dt.days
    med = df.groupby("asof_date")[[f"lag{h}" for h in H]].transform("median")
    out = pd.Series(False, index=df.index)
    for h in H:
        out |= df[f"lag{h}"] > med[f"lag{h}"] + 3
    df["halt_gap"] = out
    return df


def panel(df, title):
    p10 = df["predicted_return_d10"].to_numpy()
    a10 = df["actual_return_d10"].to_numpy()
    pred = df[[f"predicted_return_d{h}" for h in H]].to_numpy()
    act = df[[f"actual_return_d{h}" for h in H]].to_numpy()
    d = df["asof_date"].to_numpy()
    dir_h = (np.sign(pred) == np.sign(act))
    print(f"\n{'=' * 92}\n{title}   (n={len(df)}, dates={df.asof_date.nunique()})")

    ric, amp, pnl = {}, {}, {}
    for day in sorted(set(d)):
        m = d == day
        ric[day] = spearman(p10[m], a10[m])
        amp[day] = spearman(np.abs(p10[m]), np.abs(a10[m]))
        pnl[day] = float(np.mean(np.sign(p10[m]) * a10[m]))
    ric_s, amp_s, pnl_s = pd.Series(ric), pd.Series(amp), pd.Series(pnl)

    err = p10 - a10
    print(f"  [方向] d10 方向准确率            : {dir_h[:, -1].mean():.4f}")
    print(f"  [方向] 10日路径全对率             : {dir_h.mean(axis=1).mean():.4f}")
    print(f"  [幅度] d10 RMSE / MAE             : {np.sqrt((err**2).mean()):.5f} / {np.abs(err).mean():.5f}")
    print(f"  [幅度] |pred| vs |actual| 秩相关   : 日均 {amp_s.mean():.4f}  "
          f"(ICIR {amp_s.mean()/amp_s.std(ddof=1):.2f})   池化 {spearman(np.abs(p10), np.abs(a10)):.4f}")
    print(f"  [幅度] 平均 |pred_d10| / |actual_d10|: {np.abs(p10).mean():.4f} / {np.abs(a10).mean():.4f}")
    print(f"  [决策] 单笔方向性收益(毛)         : {pnl_s.mean():+.5f}")
    print(f"  [决策] 扣0.4%成本后               : {pnl_s.mean() - COST:+.5f}")
    print(f"  [决策] 日度P&L Sharpe(年化)       : {pnl_s.mean()/pnl_s.std(ddof=1)*np.sqrt(252):+.2f}   "
          f"P&L>0 天数 {int((pnl_s>0).sum())}/{len(pnl_s)}")
    print(f"  [排序] d10 Rank IC 日均/池化      : {ric_s.mean():.4f} / {spearman(p10, a10):.4f}  "
          f"(ICIR {ric_s.mean()/ric_s.std(ddof=1):.2f}, IC+ {int((ric_s>0).sum())}/{len(ric_s)})")
    return {"dir_d10": dir_h[:, -1].mean(), "path_dir": dir_h.mean(axis=1).mean(),
            "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "amp_rank_ic": float(amp_s.mean()), "gross_pnl": float(pnl_s.mean()),
            "net_pnl": float(pnl_s.mean() - COST), "pnl_sharpe": float(pnl_s.mean() / pnl_s.std(ddof=1) * np.sqrt(252)),
            "rank_ic": float(ric_s.mean()), "pooled_rank_ic": float(spearman(p10, a10)),
            "amp_rank_ic_pooled": float(spearman(np.abs(p10), np.abs(a10)))}


def decile_table(df, title):
    d = df[["asof_date", "predicted_return_d10", "actual_return_d10"]].copy()
    d["ap"] = d["predicted_return_d10"].abs()
    d["dec"] = d.groupby("asof_date")["ap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    d["signed"] = np.sign(d["predicted_return_d10"]) * d["actual_return_d10"]
    g = d.groupby("dec").agg(n=("actual_return_d10", "size"),
                             mean_abs_pred=("ap", "mean"),
                             mean_abs_actual=("actual_return_d10", lambda s: s.abs().mean()),
                             mean_signed=("signed", "mean"),
                             dir_acc=("signed", lambda s: (s > 0).mean()))
    g["net"] = g["mean_signed"] - COST
    g["calib_ratio"] = g["mean_abs_actual"] / g["mean_abs_pred"]
    print(f"\n--- {title}: 按 |predicted_return_d10| 日内十分位 ---")
    print(g.to_string(float_format=lambda v: f"{v:.5f}"))


results = {}
for label, df in data.items():
    df = halt_flag(df)
    data[label] = df
    results[label] = panel(df, f"{label}")
    decile_table(df, label)

print("\n" + "=" * 92)
print("停牌跳空子集 (halt_gap=True)")
for label, df in data.items():
    sub = df[df["halt_gap"]]
    other = df[~df["halt_gap"]]
    print(f"\n  [{label}]  halt 行数={len(sub)} ({len(sub)/len(df):.2%})  "
          f"非halt={len(other)}")
    for tag, part in (("HALT", sub), ("NORMAL", other)):
        p10 = part["predicted_return_d10"].to_numpy()
        a10 = part["actual_return_d10"].to_numpy()
        d = part["asof_date"].to_numpy()
        dir10 = (np.sign(p10) == np.sign(a10)).mean()
        pnl = pd.Series({day: float(np.mean(np.sign(p10[d == day]) * a10[d == day]))
                          for day in sorted(set(d))})
        amp = pd.Series({day: spearman(np.abs(p10[d == day]), np.abs(a10[d == day]))
                          for day in sorted(set(d))})
        print(f"    {tag:<7} d10方向={dir10:.4f}  单笔毛收益={pnl.mean():+.5f}  "
              f"净={pnl.mean()-COST:+.5f}  |pred|秩相关={amp.mean():.4f}  "
              f"RMSE={np.sqrt(((p10-a10)**2).mean()):.5f}  mean|actual|={np.abs(a10).mean():.4f}")

print("\n" + "=" * 92)
print("C2 vs C03 (clean) 汇总差值")
if "c2_best" in results and "clean_best" in results:
    for k in results["c2_best"]:
        c2v, cv = results["c2_best"][k], results["clean_best"][k]
        print(f"  {k:<20} C2={c2v:>10.5f}   C03={cv:>10.5f}   delta={cv-c2v:>+10.5f}")
