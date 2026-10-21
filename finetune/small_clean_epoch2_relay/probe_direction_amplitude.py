"""Direction + amplitude metric panel for the online C2 model (18-day sealed package).

Establishes the baseline for the criterion the project actually cares about:
predict the sign AND the amplitude of the move, in return units, net of cost.
"""
import numpy as np
import pandas as pd

SRC = "finetune/kaggle_c2_top20_consensus_oos/predictions_prod_t065_p80_n5.csv.gz"
COST = 0.004  # round trip, same constant as the Beta utility contract

df = pd.read_csv(SRC, compression="gzip")
print(f"rows={len(df)}  dates={df.asof_date.nunique()}  "
      f"range={df.asof_date.min()}..{df.asof_date.max()}  model={df.model.iloc[0]}")

H = list(range(1, 11))
pred = df[[f"predicted_return_d{h}" for h in H]].to_numpy()
act = df[[f"actual_return_d{h}" for h in H]].to_numpy()
p10 = df["predicted_return_d10"].to_numpy()
a10 = df["actual_return_d10"].to_numpy()
dates = df["asof_date"].to_numpy()


def daily(fn):
    """Apply fn to each asof_date and return the per-day series."""
    out = {}
    for d in sorted(set(dates)):
        m = dates == d
        out[d] = fn(m)
    return out


def spearman(x, y):
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else np.nan


# ---------------------------------------------------------------- 1. direction
dir_h = (np.sign(pred) == np.sign(act))
path_dir = dir_h.mean(axis=1)                     # per-window: all 10 horizons correct
d10_dir = dir_h[:, -1]
print("\n=== 方向 (direction) ===")
print(f"  d10 方向准确率            : {d10_dir.mean():.4f}")
print(f"  10日路径全对率             : {path_dir.mean():.4f}")
print(f"  单 horizon 方向准确率范围  : {dir_h.mean(axis=0).min():.4f} (d{dir_h.mean(axis=0).argmin()+1})"
      f" .. {dir_h.mean(axis=0).max():.4f} (d{dir_h.mean(axis=0).argmax()+1})")

# ---------------------------------------------------------------- 2. amplitude fidelity
print("\n=== 幅度 (amplitude) ===")
amp_daily = daily(lambda m: spearman(np.abs(p10[m]), np.abs(a10[m])))
amp_series = pd.Series(amp_daily)
print(f"  |pred_d10| vs |actual_d10| 日内秩相关 均值 : {amp_series.mean():.4f}  "
      f"(std {amp_series.std(ddof=1):.4f}, ICIR {amp_series.mean()/amp_series.std(ddof=1):.2f})")
print(f"  池化 |pred| vs |actual| 秩相关             : {spearman(np.abs(p10), np.abs(a10)):.4f}")
print(f"  d10 预测幅度均值 |pred_d10|                : {np.abs(p10).mean():.4f}")
print(f"  d10 实际幅度均值 |actual_d10|              : {np.abs(a10).mean():.4f}")
print(f"  幅度比 |pred|/|actual| 中位数              : {np.median(np.abs(p10)/np.maximum(np.abs(a10),1e-9)):.4f}")
print(f"  预测幅度与实际幅度 符号一致率              : {(np.sign(p10)==np.sign(a10)).mean():.4f}")

pv = df["predicted_path_vol"].to_numpy()
rv = df["realized_path_vol"].to_numpy()
pv_d = daily(lambda m: spearman(pv[m], rv[m]))
pvs = pd.Series(pv_d)
print(f"  predicted_path_vol vs realized_path_vol 秩相关: {pvs.mean():.4f}  (ICIR {pvs.mean()/pvs.std(ddof=1):.2f})")

# ---------------------------------------------------------------- 3. calibration by |pred| decile
print("\n=== 幅度标定：按 |predicted_return_d10| 日内十分位 ===")
work = df[["asof_date", "predicted_return_d10", "actual_return_d10"]].copy()
work["dec"] = work.groupby("asof_date")["predicted_return_d10"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
work["signed"] = np.sign(work["predicted_return_d10"]) * work["actual_return_d10"]
g = work.groupby("dec").agg(n=("actual_return_d10", "size"),
                            mean_abs_pred=("predicted_return_d10", lambda s: s.abs().mean()),
                            mean_actual=("actual_return_d10", "mean"),
                            mean_signed=("signed", "mean"),
                            dir_acc=("signed", lambda s: ((np.sign(s) == 1)).mean()))
print(g.to_string(float_format=lambda v: f"{v:.5f}"))

# ---------------------------------------------------------------- 4. money metric
print("\n=== 决策口径：按预测方向下单（毛/净，单位：单笔收益） ===")
gross = float(np.mean(np.sign(p10) * a10))
print(f"  单笔平均方向性收益(毛)   : {gross:+.5f}")
print(f"  扣 0.4% 成本后            : {gross - COST:+.5f}")
daily_pnl = daily(lambda m: float(np.mean(np.sign(p10[m]) * a10[m])))
pnl_s = pd.Series(daily_pnl)
print(f"  日度 P&L 均值/标准差      : {pnl_s.mean():+.5f} / {pnl_s.std(ddof=1):.5f}"
      f"  ->  年化 Sharpe {pnl_s.mean()/pnl_s.std(ddof=1)*np.sqrt(252):+.2f}")
print(f"  P&L>0 的天数              : {int((pnl_s > 0).sum())}/{len(pnl_s)}")

# magnitude-sized variant: only trade when |pred| is large enough to clear the cost
print("\n=== 幅度门槛：只在 |pred_d10| > 阈值时按方向下单 ===")
for thr in (0.0, 0.005, 0.01, 0.02, 0.03):
    m = np.abs(p10) > thr
    if m.sum() == 0:
        continue
    net = float(np.mean(np.sign(p10[m]) * a10[m])) - COST
    print(f"  thr={thr:.3f}  覆盖率={m.mean():.3f}  笔数={m.sum():>6}  "
          f"单笔净收益={net:+.5f}")

# ---------------------------------------------------------------- 5. ranking (the current fallback)
print("\n=== 排序 (rank IC) — 现有用法 ===")
ric = daily(lambda m: spearman(p10[m], a10[m]))
ric_s = pd.Series(ric)
print(f"  d10 Rank IC 日均 {ric_s.mean():.4f}  (std {ric_s.std(ddof=1):.4f}, "
      f"ICIR {ric_s.mean()/ric_s.std(ddof=1):.2f}, IC+天数 {int((ric_s>0).sum())}/{len(ric_s)})")
print(f"  池化 d10 Rank IC {spearman(p10, a10):.4f}")

# ---------------------------------------------------------------- 6. top-bottom spread
print("\n=== 多空 10% 分位组合 ===")
spreads = []
for d in sorted(set(dates)):
    m = dates == d
    sub = pd.DataFrame({"p": p10[m], "a": a10[m]}).sort_values("p")
    k = max(1, int(np.ceil(len(sub) * 0.10)))
    spreads.append(sub["a"].tail(k).mean() - sub["a"].head(k).mean())
sp = pd.Series(spreads)
print(f"  top-bottom 10% spread 日均 {sp.mean():+.5f}  (std {sp.std(ddof=1):.5f}, "
      f"ICIR {sp.mean()/sp.std(ddof=1):.2f}, 正天数 {int((sp>0).sum())}/{len(sp)})")
