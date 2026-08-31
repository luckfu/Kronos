"""Relay an A800 JSONL training log to SwanLab from a networked local host."""

import argparse
import base64
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


REMOTE_READ_CODE = """\
import base64, json, sys
from pathlib import Path
path = Path(sys.argv[1])
requested = int(sys.argv[2])
size = path.stat().st_size
start = requested if requested <= size else 0
with path.open('rb') as handle:
    handle.seek(start)
    data = handle.read()
print(json.dumps({'start': start, 'end': start + len(data), 'data': base64.b64encode(data).decode('ascii')}))
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="A800")
    parser.add_argument("--remote-metrics", required=True)
    parser.add_argument("--remote-baseline")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--project", default="finance")
    parser.add_argument("--workspace", default="roc_fu")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps-per-segment", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def fetch_remote_bytes(host, path, offset):
    command = "python3 -c {} {} {}".format(
        shlex.quote(REMOTE_READ_CODE), shlex.quote(path), int(offset)
    )
    result = subprocess.run(
        [
            "tssh", "-o", "ConnectionAttempts=3", "-o", "ConnectTimeout=30",
            host, command,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    payload = json.loads(result.stdout)
    return int(payload["start"]), base64.b64decode(payload["data"])


def global_step(record, steps_per_segment=0):
    segment = int(record.get("segment", 0))
    total_steps = max(
        1,
        int(steps_per_segment or record.get("total_steps", 625)),
    )
    if record.get("type") == "train":
        within_segment = int(record.get("step", 0))
    else:
        within_segment = total_steps
    return max(0, segment - 1) * total_steps + within_segment


def flatten_numeric(prefix, value, output):
    if isinstance(value, bool):
        output[prefix] = int(value)
    elif isinstance(value, (int, float)):
        output[prefix] = value
    elif isinstance(value, dict):
        for key, nested in value.items():
            flatten_numeric(f"{prefix}/{key}", nested, output)


def metric_payload(record):
    record_type = str(record.get("type", ""))
    if record_type not in {"train", "validation", "validation_large"}:
        return None
    prefix = {
        "train": "train",
        "validation": "validation/quick",
        "validation_large": "validation/full",
    }[record_type]
    ignored = {"type", "updated_at", "total_segments", "total_steps", "step"}
    payload = {}
    for key, value in record.items():
        if key not in ignored:
            flatten_numeric(f"{prefix}/{key}", value, payload)
    payload["progress/segment"] = int(record.get("segment", 0))
    return payload


def baseline_payload(document):
    payload = {}
    flatten_numeric("baseline/quick", document.get("quick", {}), payload)
    full = document.get("large", document.get("validation_large", {}))
    flatten_numeric("baseline/full", full, payload)
    return payload


def load_state(path):
    if not path.exists():
        return {"offset": 0, "baseline_logged": False, "records_logged": 0}
    return json.loads(path.read_text())


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def login_swanlab(swanlab, api_key):
    """Use an explicit key when supplied, otherwise use the local CLI profile."""
    if api_key:
        swanlab.login(api_key=api_key)
    else:
        swanlab.login()


def log_complete_lines(
    run, state_path, state, start, data, steps_per_segment=0
):
    if start != int(state["offset"]):
        state["offset"] = start
        save_state(state_path, state)
    consumed = 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        consumed += len(line)
        if line.strip():
            record = json.loads(line)
            payload = metric_payload(record)
            if payload:
                run.log(
                    payload,
                    step=global_step(record, steps_per_segment),
                )
                state["records_logged"] = int(state["records_logged"]) + 1
        state["offset"] = start + consumed
        save_state(state_path, state)
    return consumed


def main():
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    api_key = os.getenv("SWANLAB_API_KEY", "").strip()

    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.state.with_suffix(args.state.suffix + ".lock")
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another relay owns {lock_path}") from exc

    import swanlab

    login_swanlab(swanlab, api_key)
    run = swanlab.init(
        project=args.project,
        workspace=args.workspace,
        experiment_name=args.experiment_name,
        id=args.run_id,
        resume="allow",
        tags=["kronos", "v1-beta", "natural-validation", "a800-relay"],
        config={
            "source_host": args.host,
            "source_metrics": args.remote_metrics,
            "poll_seconds": args.poll_seconds,
            "steps_per_segment": args.steps_per_segment,
            "transport": "local_tssh_relay",
            "validation_periods": ["2025H2", "2026H1"],
        },
    )
    state = load_state(args.state)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        json.dumps({
            "status": "started", "run_id": args.run_id,
            "offset": state["offset"], "poll_seconds": args.poll_seconds,
        }),
        flush=True,
    )
    try:
        while not stop_requested:
            try:
                if args.remote_baseline and not state.get("baseline_logged", False):
                    _, raw = fetch_remote_bytes(args.host, args.remote_baseline, 0)
                    run.log(baseline_payload(json.loads(raw)), step=0)
                    state["baseline_logged"] = True
                    save_state(args.state, state)
                    print("baseline uploaded", flush=True)

                start, data = fetch_remote_bytes(
                    args.host, args.remote_metrics, int(state["offset"])
                )
                before = int(state["records_logged"])
                log_complete_lines(
                    run, args.state, state, start, data,
                    args.steps_per_segment,
                )
                added = int(state["records_logged"]) - before
                if added:
                    print(
                        f"uploaded {added} rows; total={state['records_logged']} "
                        f"offset={state['offset']}",
                        flush=True,
                    )
            except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
                print(f"relay retry after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
