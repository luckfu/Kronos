"""Phase Z2: materialize C2 Best Seg@179 prod scores on Kairos val dates.

Decode locked: T=0.65 top_p=0.8 N=5 seed=20260906.
Panel: symbol-holdout val_data.pkl (516 syms). Signal: 2025-07-03..2026-07-02.
Purpose: decision-only features for y=1{mfe10>=0.10}; NOT ranking; NOT TPU WIP.
Note: these dates fall inside C2 training window (latest target 2026-07-31);
this run is for Kairos decision-head val_temporal / join, not Kronos OOS claim.
"""
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

OUTPUT = Path("/kaggle/working/kronos_c2_kairos_val_prod_scores")
INPUT = Path("/kaggle/input")
REPO_DIR = Path("/kaggle/working/kronos_repo")
HARD_LIMIT_SECONDS = 39600
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_CACHE = Path("/kaggle/working/kronos_tokenizer_base")
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
EXPECTED_C2_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
EXPECTED_SEGMENT = 179
SIGNAL_START = "2025-07-03"
SIGNAL_END = "2026-07-02"
EXPECTED_ROWS = 123836
EXPECTED_DATES = 242
EFFECTIVE_BATCH = 64
WORLD_SIZE = 2
DECODE_SEED = 20260906
ARM = {
    "name": "prod_t065_p80_n5",
    "checkpoint": "c2_best",
    "sample_count": 5,
    "temperature": 0.65,
    "top_p": 0.8,
    "seed": DECODE_SEED,
    "protocol": "production",
}

SECTOR_LABELS_EMBEDDED = ["A01农业", "A02林业", "A03畜牧业", "A04渔业", "A05农、林、牧、渔专业及辅助性活动", "B06煤炭开采和洗选业", "B07石油和天然气开采业", "B08黑色金属矿采选业", "B09有色金属矿采选业", "B10非金属矿采选业", "B11开采专业及辅助性活动", "C13农副食品加工业", "C14食品制造业", "C15酒、饮料和精制茶制造业", "C17纺织业", "C18纺织服装、服饰业", "C19皮革、毛皮、羽毛及其制品和制鞋业", "C20木材加工和木、竹、藤、棕、草制品业", "C21家具制造业", "C22造纸和纸制品业", "C23印刷和记录媒介复制业", "C24文教、工美、体育和娱乐用品制造业", "C25石油、煤炭及其他燃料加工业", "C26化学原料和化学制品制造业", "C27医药制造业", "C28化学纤维制造业", "C29橡胶和塑料制品业", "C30非金属矿物制品业", "C31黑色金属冶炼和压延加工业", "C32有色金属冶炼和压延加工业", "C33金属制品业", "C34通用设备制造业", "C35专用设备制造业", "C36汽车制造业", "C37铁路、船舶、航空航天和其他运输设备制造业", "C38电气机械和器材制造业", "C39计算机、通信和其他电子设备制造业", "C40仪器仪表制造业", "C41其他制造业", "C42废弃资源综合利用业", "C43金属制品、机械和设备修理业", "D44电力、热力生产和供应业", "D45燃气生产和供应业", "D46水的生产和供应业", "E47房屋建筑业", "E48土木工程建筑业", "E49建筑安装业", "E50建筑装饰、装修和其他建筑业", "F51批发业", "F52零售业", "G53铁路运输业", "G54道路运输业", "G55水上运输业", "G56航空运输业", "G57管道运输业", "G58多式联运和运输代理业", "G59装卸搬运和仓储业", "G60邮政业", "H61住宿业", "H62餐饮业", "I63电信、广播电视和卫星传输服务", "I64互联网和相关服务", "I65软件和信息技术服务业", "J66货币金融服务", "J67资本市场服务", "J68保险业", "J69其他金融业", "K70房地产业", "L71租赁业", "L72商务服务业", "M73研究和试验发展", "M74专业技术服务业", "M75科技推广和应用服务业", "N76水利管理业", "N77生态保护和环境治理业", "N78公共设施管理业", "N79土地管理业", "O81机动车、电子产品和日用产品修理业", "P83教育", "Q84卫生", "R86新闻和出版业", "R87广播、电视、电影和录音制作业", "R88文化艺术业", "R89体育", "S91综合", "unknown"]
SECTOR_LABELS = list(SECTOR_LABELS_EMBEDDED)



def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def clone_source():
    last_err = None
    for attempt in range(1, 4):
        repo = REPO_DIR if attempt == 1 else Path(tempfile.mkdtemp(prefix="kronos-z2-")) / "repo"
        try:
            if repo.exists():
                subprocess.run(["rm", "-rf", str(repo)], check=True)
            run(["git", "clone", "--depth", "30", "--branch", "master",
                 "https://github.com/luckfu/Kronos.git", str(repo)])
            # Use whatever master tip provides evaluate_v1_beta_checkpoints
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            print({"cloned_head": head}, flush=True)
            return repo, head
        except Exception as exc:
            last_err = exc
            time.sleep(3 * attempt)
    raise RuntimeError(f"clone failed: {last_err}")


def resolve_tokenizer():
    matches = [
        path for path in INPUT.glob("**/Kronos-Tokenizer-base/model.safetensors")
    ]
    if len(matches) == 1:
        tokenizer_dir = matches[0].parent
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
            from huggingface_hub import snapshot_download
        tokenizer_dir = Path(
            snapshot_download(TOKENIZER_REPO, local_dir=str(TOKENIZER_CACHE))
        )
    tokenizer_sha = sha256_file(tokenizer_dir / "model.safetensors")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer sha mismatch: {tokenizer_sha}")
    return tokenizer_dir


def resolve_c2_weights():
    candidates = list(INPUT.glob("**/model.safetensors"))
    matched = None
    for path in candidates:
        name = str(path).lower()
        if path.stat().st_size != 99293144:
            continue
        if "c1" in name and "beta" in name:
            continue  # never touch colleague TPU WIP
        if sha256_file(path) == EXPECTED_C2_SHA:
            matched = path
            break
    if matched is None:
        raise RuntimeError(
            f"C2 Seg@179 weights not found; scanned={[str(c) for c in candidates]}"
        )
    metric_path = matched.parent / "best_metric.json"
    if metric_path.exists():
        metric = json.loads(metric_path.read_text())
        if int(metric.get("segment", -1)) != EXPECTED_SEGMENT:
            raise RuntimeError(f"unexpected segment: {metric}")
    print({"c2_weights": str(matched)}, flush=True)
    return matched.parent


def resolve_val_panel():
    panel_path = find_one("**/processed_datasets/val_data.pkl")
    with panel_path.open("rb") as handle:
        panel = pickle.load(handle)
    return panel_path, panel


def load_sector_labels(repo):
    if SECTOR_LABELS and len(SECTOR_LABELS) == 86:
        return SECTOR_LABELS
    manifests = list(INPUT.glob("**/evaluation_manifest.json"))
    for path in manifests:
        manifest = json.loads(path.read_text())
        labels = (manifest.get("model_contract") or {}).get("sector_labels")
        if labels and len(labels) == 86:
            return labels
    raise RuntimeError("sector_labels not found")


def to_daily(cumulative):
    import numpy as np

    previous = np.concatenate(
        [np.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], axis=-1
    )
    return (1.0 + cumulative) / (1.0 + previous) - 1.0


def decode_date(arm, model, tokenizer, records, store, device):
    import numpy as np
    import torch
    from finetune.evaluate_v1_beta_checkpoints import (
        FEATURES, LOOKBACK, PREDICT, batches, stack_batch,
    )
    from model.kronos import auto_regressive_inference

    sample_count = int(arm["sample_count"])
    batch_size = max(1, EFFECTIVE_BATCH // sample_count)
    close_index = FEATURES.index("close")
    torch.manual_seed(arm["seed"])
    np.random.seed(arm["seed"])

    rows = []
    for items in batches(records, store, batch_size):
        batch = stack_batch(items, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            forecast = auto_regressive_inference(
                tokenizer,
                model,
                batch["x"][:, :LOOKBACK],
                batch["stamp"][:, :LOOKBACK],
                batch["stamp"][:, LOOKBACK: LOOKBACK + PREDICT],
                max_context=512,
                pred_len=PREDICT,
                clip=5,
                T=float(arm["temperature"]),
                top_k=0,
                top_p=float(arm["top_p"]),
                sample_count=sample_count,
                verbose=False,
                sector_id=batch["sector"],
                size_percentile=batch["percentile"],
                return_samples=True,
            )
        future = forecast[:, :, -PREDICT:, close_index]
        for index, item in enumerate(items):
            scale = float(item["std"][close_index]) + 1e-5
            shift = float(item["mean"][close_index])
            last_close = float(item["x"][LOOKBACK - 1, close_index]) * scale + shift
            paths = future[index].astype(np.float64) * scale + shift
            cumulative = paths / last_close - 1.0
            mean_cumulative = cumulative.mean(axis=0)
            actual = (
                item["x"][LOOKBACK: LOOKBACK + PREDICT, close_index].astype(np.float64)
                * scale + shift
            ) / last_close - 1.0
            predicted_daily = to_daily(cumulative)
            actual_daily = to_daily(actual)
            row = {
                "model": arm["name"],
                "checkpoint": arm["checkpoint"],
                "sample_count": sample_count,
                "temperature": float(arm["temperature"]),
                "top_p": float(arm["top_p"]),
                "seed": int(arm["seed"]),
                **{
                    key: item[key]
                    for key in (
                        "identity", "symbol", "asof_date", "target_date",
                        "direction", "sector", "size_decile", "return_10d",
                    )
                },
                "predicted_return_10d": float(mean_cumulative[-1]),
                "predicted_path_vol": float(predicted_daily.std(axis=1).mean()),
                "predicted_terminal_dispersion": (
                    float(cumulative[:, -1].std(ddof=1)) if sample_count > 1 else None
                ),
                "realized_path_vol": float(actual_daily.std()),
            }
            for horizon in range(PREDICT):
                row[f"predicted_return_d{horizon + 1}"] = float(mean_cumulative[horizon])
                row[f"actual_return_d{horizon + 1}"] = float(actual[horizon])
            rows.append(row)
    return rows


def make_relaxed_store(panel, sector_labels):
    """WindowStore allowing panel sector subset of training vocabulary."""
    from finetune.evaluate_v1_beta_checkpoints import WindowStore

    class RelaxedWindowStore(WindowStore):
        def __init__(self, panel, sector_labels):
            self.panel = panel
            self.sector_map = {value: index for index, value in enumerate(sector_labels)}
            observed = sorted(
                {
                    str(value)
                    for frame in panel.values()
                    for value in frame["sector"].dropna().unique()
                }
            )
            missing = [x for x in observed if x not in self.sector_map]
            if missing:
                raise RuntimeError(f"Unknown sectors in panel: {missing}")
            # subset OK for symbol-holdout val panel (73/86)

    return RelaxedWindowStore(panel, sector_labels)


def worker(rank, plan):
    import pandas as pd
    import torch
    from model import Kronos, KronosTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    if not torch.cuda.is_available():
        raise RuntimeError(f"worker {rank} has no GPU")
    device = torch.device("cuda:0")

    tokenizer = (
        KronosTokenizer.from_pretrained(Path(plan["tokenizer_dir"])).to(device).eval()
    )
    model = Kronos.from_pretrained(
        Path(plan["checkpoint_dir"]),
        num_sectors=86,
        num_size_buckets=0,
        context_layer=6,
        use_size_percentile=True,
        size_mlp_hidden_dim=64,
    ).to(device).eval()

    with open(plan["panel_path"], "rb") as handle:
        panel = pickle.load(handle)
    store = make_relaxed_store(panel, plan["sector_labels"])

    dates = plan["dates"]
    my_dates = [d for i, d in enumerate(dates) if i % plan["world_size"] == rank]
    records_by_date = plan["records_by_date"]

    shard_dir = OUTPUT / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for date in my_dates:
        if time.time() > plan["deadline"]:
            raise RuntimeError(f"worker {rank} hit deadline before {date}")
        rows = decode_date(ARM, model, tokenizer, records_by_date[date], store, device)
        out = shard_dir / f"{ARM['name']}_{date}.csv.gz"
        pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
        print({"rank": rank, "date": date, "rows": len(rows), "path": str(out)}, flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print({"rank": rank, "status": "done", "dates": len(my_dates)}, flush=True)


def main():
    import multiprocessing as mp
    import pandas as pd
    import torch

    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    repo, head = clone_source()
    sys.path.insert(0, str(repo))

    tokenizer_dir = resolve_tokenizer()
    checkpoint_dir = resolve_c2_weights()
    panel_path, panel = resolve_val_panel()
    sector_labels = load_sector_labels(repo)

    from finetune.prepare_v1_beta_evaluation import build_candidates

    records = build_candidates(panel, SIGNAL_START, SIGNAL_END)
    if len(records) != EXPECTED_ROWS:
        print({"warn_rows": len(records), "expected": EXPECTED_ROWS}, flush=True)
    records_by_date = defaultdict(list)
    for record in records:
        records_by_date[record["asof_date"]].append(record)
    dates = sorted(records_by_date)
    if len(dates) != EXPECTED_DATES:
        print({"warn_dates": len(dates), "expected": EXPECTED_DATES}, flush=True)

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    world = min(WORLD_SIZE, max(1, len(gpu_names)))
    print({"gpus": gpu_names, "world": world}, flush=True)

    plan = {
        "tokenizer_dir": str(tokenizer_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "panel_path": str(panel_path),
        "sector_labels": sector_labels,
        "dates": dates,
        "records_by_date": dict(records_by_date),
        "world_size": world,
        "deadline": started + HARD_LIMIT_SECONDS,
        "source_head": head,
    }
    (OUTPUT / "plan.json").write_text(json.dumps({
        k: (v if k != "records_by_date" else {d: len(rows) for d, rows in v.items()})
        for k, v in plan.items()
    }, ensure_ascii=False, indent=2) + "\n")

    if world == 1:
        worker(0, plan)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(rank, plan)) for rank in range(world)]
        for proc in procs:
            proc.start()
        failures = []
        for rank, proc in enumerate(procs):
            proc.join()
            if proc.exitcode != 0:
                failures.append((rank, proc.exitcode))
        if failures:
            raise RuntimeError(f"Worker exit codes: {failures}")

    shard_dir = OUTPUT / "shards"
    frames = []
    for date in dates:
        path = shard_dir / f"{ARM['name']}_{date}.csv.gz"
        if not path.exists():
            raise RuntimeError(f"missing shard {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != len(records):
        raise RuntimeError(f"row mismatch: {len(frame)} != {len(records)}")
    pred_path = OUTPUT / f"predictions_{ARM['name']}.csv.gz"
    frame.to_csv(pred_path, index=False, compression="gzip")

    summary = {
        "purpose": "phase_z2_c2_seg179_prod_scores_on_kairos_val",
        "checkpoint": "cosine_c2_best",
        "segment": EXPECTED_SEGMENT,
        "sha256": EXPECTED_C2_SHA,
        "decode": ARM,
        "signal_start": SIGNAL_START,
        "signal_end": SIGNAL_END,
        "n_rows": int(len(frame)),
        "n_dates": int(frame["asof_date"].nunique()),
        "n_symbols": int(frame["symbol"].nunique()),
        "source_head": head,
        "elapsed_sec": time.time() - started,
        "pred_path": str(pred_path),
        "contamination_note": (
            "Kairos val dates are inside C2 training window (latest target 2026-07-31); "
            "scores are for decision-head join/val_temporal only, not Kronos time-OOS."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
