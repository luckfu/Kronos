"""Sealed OOS inference for C2 best vs bigbatch e03b best on the 42-day panel."""
from __future__ import annotations
import gc, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

LOOKBACK, PREDICT, WINDOW = 120, 10, 130
FEATURES = ["open", "high", "low", "close", "volume", "amount"]
OUTPUT = Path("/kaggle/working/small_sealed_oos")
LIMIT = os.environ.get("SMALL_OOS_LIMIT", "").strip()
SECTOR_LABELS = ['A01农业','A02林业','A03畜牧业','A04渔业','A05农、林、牧、渔专业及辅助性活动','B06煤炭开采和洗选业','B07石油和天然气开采业','B08黑色金属矿采选业','B09有色金属矿采选业','B10非金属矿采选业','B11开采专业及辅助性活动','C13农副食品加工业','C14食品制造业','C15酒、饮料和精制茶制造业','C17纺织业','C18纺织服装、服饰业','C19皮革、毛皮、羽毛及其制品和制鞋业','C20木材加工和木、竹、藤、棕、草制品业','C21家具制造业','C22造纸和纸制品业','C23印刷和记录媒介复制业','C24文教、工美、体育和娱乐用品制造业','C25石油、煤炭及其他燃料加工业','C26化学原料和化学制品制造业','C27医药制造业','C28化学纤维制造业','C29橡胶和塑料制品业','C30非金属矿物制品业','C31黑色金属冶炼和压延加工业','C32有色金属冶炼和压延加工业','C33金属制品业','C34通用设备制造业','C35专用设备制造业','C36汽车制造业','C37铁路、船舶、航空航天和其他运输设备制造业','C38电气机械和器材制造业','C39计算机、通信和其他电子设备制造业','C40仪器仪表制造业','C41其他制造业','C42废弃资源综合利用业','C43金属制品、机械和设备修理业','D44电力、热力生产和供应业','D45燃气生产和供应业','D46水的生产和供应业','E47房屋建筑业','E48土木工程建筑业','E49建筑安装业','E50建筑装饰、装修和其他建筑业','F51批发业','F52零售业','G53铁路运输业','G54道路运输业','G55水上运输业','G56航空运输业','G57管道运输业','G58多式联运和运输代理业','G59装卸搬运和仓储业','G60邮政业','H61住宿业','H62餐饮业','I63电信、广播电视和卫星传输服务','I64互联网和相关服务','I65软件和信息技术服务业','J66货币金融服务','J67资本市场服务','J68保险业','J69其他金融业','K70房地产业','L71租赁业','L72商务服务业','M73研究和试验发展','M74专业技术服务业','M75科技推广和应用服务业','N76水利管理业','N77生态保护和环境治理业','N78公共设施管理业','N79土地管理业','O81机动车、电子产品和日用产品修理业','P83教育','Q84卫生','R86新闻和出版业','R87广播、电视、电影和录音制作业','R88文化艺术业','R89体育','S91综合','unknown']
SECTOR_MAP = {label: i for i, label in enumerate(SECTOR_LABELS)}

def phase(name, **values):
    print(json.dumps(dict(time=time.time(), phase=name, **values)), flush=True)

def find_one(root, name):
    paths = sorted(Path(root).glob("**/" + name))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {name}; found {paths[:5]}")
    return paths[0]

def load_dataset(root):
    raw_path = find_one(root, "combined_raw.csv")
    samples_path = find_one(root, "evaluation_samples.jsonl")
    raw = pd.read_csv(raw_path)
    raw["date"] = raw["date"].astype(str)
    raw = raw[raw["volume"] > 0].sort_values(["symbol", "date"])
    samples = pd.read_json(samples_path, lines=True)
    samples["asof_date"] = samples["asof_date"].astype(str)
    samples["target_date"] = samples["target_date"].astype(str)
    if len(samples) != 213564 or samples.asof_date.nunique() != 42:
        raise RuntimeError(f"Unexpected samples: {len(samples)} / {samples.asof_date.nunique()}")
    if (samples.asof_date.min(), samples.asof_date.max()) != ("2026-08-03", "2026-09-30"):
        raise RuntimeError("Date drift")
    if samples.target_date.max() != "2026-10-21":
        raise RuntimeError("Target drift")
    raw["size_percentile"] = raw.groupby("date")["market_cap"].rank(pct=True, method="average")
    sector_by_key = {(r.symbol, r.asof_date): r.sector for r in samples.itertuples()}
    panel = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.set_index(pd.to_datetime(group["date"]))[FEATURES + ["size_percentile"]].copy()
        frame["sector"] = "unknown"
        panel[symbol] = frame
    for r in samples.itertuples():
        frame = panel.get(r.symbol)
        if frame is None:
            continue
        ts = pd.Timestamp(r.asof_date)
        if ts in frame.index:
            frame.at[ts, "sector"] = sector_by_key[(r.symbol, r.asof_date)]
    return panel, samples

class WindowStore:
    def __init__(self, panel):
        self.panel = panel
    def prepare(self, record):
        frame = self.panel[str(record["symbol"])]
        # The public dataset stores the as-of index; the window starts 120 rows earlier.
        start = int(frame.index.get_loc(pd.Timestamp(record["asof_date"]))) - (LOOKBACK - 1)
        window = frame.iloc[start:start + WINDOW]
        if len(window) != WINDOW:
            raise RuntimeError(f"Incomplete window: {record['symbol']} {record['asof_date']}")
        dates = pd.DatetimeIndex(window.index)
        if (str(dates[119].date()), str(dates[129].date())) != (record["asof_date"], record["target_date"]):
            raise RuntimeError("Window identity drift")
        values = window[FEATURES].to_numpy(dtype=np.float32)
        raw_close = window.close.to_numpy(dtype=np.float64)
        mean, std = values[:120].mean(0), values[:120].std(0)
        x = np.clip((values[:120] - mean) / (std + 1e-5), -5, 5)
        actual = raw_close[120:130] / raw_close[119] - 1
        sector = str(window.sector.iloc[119])
        percentile = float(window.size_percentile.iloc[119])
        return dict(identity="|".join(str(record[k]) for k in ("symbol","start_index","asof_date","target_date")),
                    symbol=record["symbol"], asof_date=record["asof_date"], target_date=record["target_date"],
                    sector=sector, size_decile=int(record["size_decile"]), future_dates=[str(d.date()) for d in dates[120:130]],
                    x=x.astype(np.float32), mean=mean, std=std,
                    stamp=np.column_stack([dates.minute, dates.hour, dates.weekday, dates.day, dates.month]).astype(np.float32),
                    last_close=raw_close[119], actual=actual, percentile=percentile,
                    sector_id=SECTOR_MAP.get(sector, SECTOR_MAP["unknown"]))

def decode(records, store, model, tokenizer, device):
    from model.kronos import auto_regressive_inference
    rows = []
    batch_size = 24
    for offset in range(0, len(records), batch_size):
        items = [store.prepare(r) for r in records[offset:offset + batch_size]]
        x = torch.as_tensor(np.stack([i["x"] for i in items]), device=device)
        stamp = torch.as_tensor(np.stack([i["stamp"] for i in items]), device=device)
        sectors = torch.as_tensor([i["sector_id"] for i in items], device=device)
        sizes = torch.as_tensor([i["percentile"] for i in items], dtype=torch.float32, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            forecasts = auto_regressive_inference(
                tokenizer, model, x, stamp[:, :120], stamp[:, 120:130],
                max_context=512, pred_len=10, clip=5, T=0.65, top_k=0, top_p=0.8,
                sample_count=1, verbose=False, sector_id=sectors, size_percentile=sizes,
                return_samples=True)
        for i, item in enumerate(items):
            predicted_close = (forecasts[i, :, -10:, 3].astype(np.float64) * (float(item["std"][3]) + 1e-5) + float(item["mean"][3])).mean(0)
            predicted = predicted_close / item["last_close"] - 1
            if not np.isfinite(predicted).all():
                raise RuntimeError("Nonfinite prediction")
            row = {k: item[k] for k in ("identity","symbol","asof_date","target_date","sector","size_decile","last_close")}
            for h in range(10):
                row[f"predicted_return_d{h+1}"] = predicted[h]
                row[f"actual_return_d{h+1}"] = item["actual"][h]
                row[f"label_date_d{h+1}"] = item["future_dates"][h]
            rows.append(row)
        if offset % (batch_size * 20) == 0:
            phase("prediction", date=records[0]["asof_date"], done=min(offset + batch_size, len(records)), total=len(records))
    return pd.DataFrame(rows)

def score(frame):
    p = frame[[f"predicted_return_d{h}" for h in range(1, 11)]].to_numpy()
    a = frame[[f"actual_return_d{h}" for h in range(1, 11)]].to_numpy()
    e = p - a; d = np.sign(p) == np.sign(a)
    out = dict(samples=len(frame), path_direction=float(d.mean()), path_return_mae=float(np.abs(e).mean()),
               path_return_rmse=float(np.sqrt(np.square(e).mean())), horizons={})
    for h in range(10):
        out["horizons"][str(h+1)] = dict(direction=float(d[:,h].mean()), return_mae=float(np.abs(e[:,h]).mean()), return_rmse=float(np.sqrt(np.square(e[:,h]).mean())))
    out["d10_daily_rank_ic"] = float(frame.groupby("asof_date").apply(lambda day: day.predicted_return_d10.corr(day.actual_return_d10, method="spearman"), include_groups=False).mean())
    return out

def worker(rank):
    from model import Kronos, KronosTokenizer
    label = ("c2_best", "bigbatch_best")[rank]
    device = torch.device("cuda:0")
    root = Path("/kaggle/input")
    dataset_root = find_one(root, "combined_raw.csv").parent
    panel, samples = load_dataset(dataset_root)
    store = WindowStore(panel)
    patterns = dict(c2_best="**/kronos-small-c2-best-checkpoint/model.safetensors",
                    bigbatch_best="**/small_0.1_clean_bigbatch_20261022/checkpoints/best_model/model.safetensors")
    weights = find_one(root, "**/" + patterns[label].split("**/",1)[1]) if False else None
    # find by glob relative to /kaggle/input
    matches = sorted(root.glob(patterns[label]))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {label}: {matches[:5]}")
    weights = matches[0]
    model = Kronos.from_pretrained(str(weights.parent), num_sectors=86, num_size_buckets=0,
                                   context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(str(find_one(root, "Kronos-Tokenizer-base/model.safetensors").parent)).to(device).eval()
    phase("model_loaded", label=label, weights=str(weights))
    dates = sorted(samples.asof_date.unique())
    if LIMIT:
        dates = dates[-int(LIMIT):]
    OUTPUT.joinpath("shards").mkdir(parents=True, exist_ok=True)
    for date in dates:
        selected = samples[samples.asof_date == date].to_dict("records")
        frame = decode(selected, store, model, tokenizer, device)
        frame.to_csv(OUTPUT / "shards" / f"{label}_{date}.csv.gz", index=False, compression="gzip")
        phase("date_done", label=label, date=date, windows=len(frame), metrics=score(frame))
    phase("worker_done", label=label)

def daily_ic(frame):
    return frame.groupby("asof_date").apply(
        lambda day: day.predicted_return_d10.corr(day.actual_return_d10, method="spearman"),
        include_groups=False)

def paired(c2, big, seed=20261022, block=5, replicates=2000):
    c2 = c2.sort_values("identity").reset_index(drop=True)
    big = big.sort_values("identity").reset_index(drop=True)
    if not c2.identity.equals(big.identity):
        raise RuntimeError("Unpaired predictions")
    a = c2[[f"actual_return_d{h}" for h in range(1,11)]].to_numpy()
    b = big[[f"actual_return_d{h}" for h in range(1,11)]].to_numpy()
    if not np.array_equal(a, b):
        raise RuntimeError("Paired labels differ")
    ic2 = daily_ic(c2); icb = daily_ic(big)
    dates = sorted(set(ic2.index) & set(icb.index))
    d = (icb.reindex(dates) - ic2.reindex(dates)).to_numpy()
    rng = np.random.default_rng(seed); samples = []
    for _ in range(replicates):
        idx = []
        while len(idx) < len(dates):
            start = rng.integers(0, max(1, len(dates) - block + 1))
            idx.extend(range(start, min(start + block, len(dates))))
        idx = idx[:len(dates)]
        samples.append(float(np.nanmean(d[idx])))
    return dict(dates=len(dates), c2_ic=float(np.nanmean(ic2.reindex(dates))),
                bigbatch_ic=float(np.nanmean(icb.reindex(dates))),
                delta_ic=float(np.nanmean(d)), ci95=[float(np.nanquantile(samples,.025)), float(np.nanquantile(samples,.975))],
                block=block, replicates=replicates, seed=seed)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SMALL_OOS_RANK") is not None:
        worker(int(os.environ["SMALL_OOS_RANK"]))
        return
    phase("parent_start")
    procs = []
    for rank in range(2):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), SMALL_OOS_RANK=str(rank))
        procs.append(subprocess.Popen([sys.executable, "-u", __file__], env=env))
    for p in procs:
        if p.wait():
            raise RuntimeError("worker failed")
    frames = {label: pd.concat([pd.read_csv(p) for p in sorted((OUTPUT/"shards").glob(f"{label}_*.csv.gz"))], ignore_index=True)
              for label in ("c2_best", "bigbatch_best")}
    summary = {label: score(frame) for label, frame in frames.items()}
    first_use = {label: score(frame[frame.asof_date >= "2026-09-18"]) for label, frame in frames.items()}
    pair_full = paired(frames["c2_best"], frames["bigbatch_best"], block=10)
    pair_first = paired(frames["c2_best"][frames["c2_best"].asof_date >= "2026-09-18"],
                        frames["bigbatch_best"][frames["bigbatch_best"].asof_date >= "2026-09-18"], block=5)
    (OUTPUT / "summary.json").write_text(json.dumps(dict(full_42d=summary, first_use_9d=first_use,
        paired_full=pair_full, paired_first_use=pair_first), indent=2))
    phase("finished", summary=summary, first_use_9d=first_use, paired_full=pair_full, paired_first_use=pair_first)

if __name__ == "__main__":
    main()
