"""Fetch validation/forecast_loss for a SwanLab run and summarise its trend.

Usage:
    python finetune/small_clean_epoch2_relay/probe_swanlab_loss.py [run_slug ...]

The SwanLab credential is read from SWANLAB_API_KEY or Kaggle Secrets, so it
never has to appear in a shell command or in the transcript.
"""
import argparse
import os
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = "roc_fu"
PROJECT = "finance"
DEFAULT_RUNS = ("smce2c3_long", "smce2c3_probe", "smce2c3")


def local_credential() -> str:
    """Read SWANLAB_API_KEY from the environment or Kaggle Secrets."""
    value = os.environ.get("SWANLAB_API_KEY", "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        value = (UserSecretsClient().get_secret("SWANLAB_API_KEY") or "").strip()
    except Exception:
        value = ""
    if not value:
        raise SystemExit("SWANLAB_API_KEY is empty")
    return value


def series(exp, key, sample=1500):
    """Return the (step, value) points of one scalar metric, full or sampled."""
    points = []
    for mode in ({"all": True}, {"sample": sample}):
        resp = exp.metrics(keys=[key], **mode)
        for entry in resp.get("list", []) or []:
            if entry.get("key") != key:
                continue
            for p in entry.get("metrics", []) or []:
                value = p.get("data", p.get("value"))
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                points.append((int(p.get("index", p.get("step", 0))), float(value)))
        if points:
            break
    return sorted(set(points))


def regress(points):
    """Slope of value against step, plus the correlation coefficient."""
    xs = [float(s) for s, _ in points]
    ys = [v for _, v in points]
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    syy = sum((y - ybar) ** 2 for y in ys)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    corr = sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    return slope, corr


def report(exp, slug, keys):
    info = exp.json()
    print("=" * 100)
    print(f"RUN {slug}  name={info.get('name')}  state={info.get('state')}")
    print(f"  created={info.get('created_at')}  finished={info.get('finished_at')}")
    print(f"  url={info.get('url')}")
    for key in keys:
        points = series(exp, key)
        if not points:
            print(f"  {key}: (no data)")
            continue
        steps = [p[0] for p in points]
        vals = [p[1] for p in points]
        n = len(vals)
        slope, corr = regress(points)
        imin = min(range(n), key=lambda i: vals[i])
        print(f"  {key}")
        print(f"    n={n}  step [{steps[0]} .. {steps[-1]}]")
        print(f"    first={vals[0]:.6f}  last={vals[-1]:.6f}  "
              f"delta={vals[-1] - vals[0]:+.6f} ({(vals[-1] / vals[0] - 1) * 100:+.3f}%)")
        print(f"    min={vals[imin]:.6f} @ step {steps[imin]}   max={max(vals):.6f}   "
              f"mean={statistics.fmean(vals):.6f}   std={statistics.pstdev(vals):.6f}")
        print(f"    full-run slope={slope:+.3e}/step   corr(step,value)={corr:+.4f}")
        for k in (30, 60, 100, 300):
            if n >= k:
                s, c = regress(points[-k:])
                print(f"    last {k:>3} pts: slope={s:+.3e}/step  corr={c:+.3f}  "
                      f"mean={statistics.fmean(vals[-k:]):.6f}")
        sixth = max(1, n // 6)
        chunks = [vals[i:i + sixth] for i in range(0, n, sixth)]
        print("    sixths mean: " + " -> ".join(f"{statistics.fmean(c):.6f}" for c in chunks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=list(DEFAULT_RUNS))
    parser.add_argument("--keys", nargs="*",
                        default=["validation/forecast_loss", "validation/full_loss",
                                 "train/forecast_loss", "segment"])
    args = parser.parse_args()
    from swanlab import Api
    api = Api(api_key=local_credential())
    print("username:", api.username)
    for slug in args.runs:
        report(api.run(f"{WORKSPACE}/{PROJECT}/{slug}"), slug, args.keys)


if __name__ == "__main__":
    main()
