#!/usr/bin/env python3
"""Watch a Base-clean relay chunk while it is still running.

Why this exists
---------------
`kaggle kernels output` only materialises `/kaggle/working` when the session
ends, so it returns nothing mid-run — that is the "训练途中获取不了日志" trap.
Kaggle does stream the console, but through two places the relay never used:

  * `POST /v1/kernels.KernelsApiService/ListKernelSessionOutput` returns the
    session console in its `log` field (a JSON list of SSE events, not plain
    text — which is why pasting `kaggle kernels logs` looked unreadable);
  * `GET /v1/kernels/logs/stream/<owner>/<slug>` is a real SSE feed while the
    session runs, and falls back to the persisted blob once it ends.

This script speaks those endpoints directly (no `kaggle` CLI in the path, since
the 2.2.4 client intermittently dies with `SSLEOFError` on api.kaggle.com) and
folds the trainer's own summary block into one line per coverage segment plus
an ETA.

It only reads logs. Attaching or detaching never touches the training process.

Usage
-----
    # live: one line per segment (Ctrl-C detaches; training is unaffected)
    python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --follow

    # right after a push: wait for the session, then attach
    python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --follow --wait

    # replay the last session and show the last 12 segments
    python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --tail 12

    # machine-readable snapshot (one JSON object per segment while following)
    python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --json

    # also echo the raw trainer lines (debugging the log format itself)
    python3 finetune/base_clean_bootstrap_relay/monitor_chunk.py --follow --raw

Credentials: `KAGGLE_API_TOKEN` (access token) or `~/.kaggle/credentials.json`
(written by `kaggle auth login`), or `KAGGLE_USERNAME` + `KAGGLE_KEY`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover - requests ships with kaggle
    requests = None

try:
    import relay_auth
except ImportError:  # run from another cwd
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import relay_auth

DEFAULT_KERNEL = "luckfu/kronos-base-clean-0-1-bootstrap"
API_ROOT = os.environ.get("KRONOS_KAGGLE_API", "https://api.kaggle.com")
RPC_ROOT = f"{API_ROOT}/v1/kernels.KernelsApiService"

PROGRESS_RE = re.compile(r"\[PROGRESS\]\s+(\{.*\})\s*$")
SUMMARY_RE = re.compile(r"---\s*Coverage Segment (\d+)/(\d+)\s*Summary\s*---")
TRAIN_STEP_RE = re.compile(r"Segment (\d+)/(\d+), Step (\d+)/(\d+)")
VAL_LOSS_RE = re.compile(
    r"Validation Forecast/History/Full: ([0-9.]+) / ([0-9.]+) / ([0-9.]+)"
)
VAL_WEIGHTED_RE = re.compile(r"Validation Weighted Forecast: ([0-9.]+)")
VAL_BEST_RE = re.compile(r"Validation Train Average/Best: ([0-9.]+) / ([0-9.]+)")
BEST_METRIC_RE = re.compile(r"Best selection metric: forecast=([0-9.]+|not evaluated)")
ABLATION_RE = re.compile(
    r"Validation Condition Full/None/Shuffled Forecast: "
    r"([0-9.]+) / ([0-9.]+) / ([0-9.]+); "
    r"Delta Full-None/Full-Shuffled: ([+-]?[0-9.eE+-]+) / ([+-]?[0-9.eE+-]+)"
)
CUDA_RE = re.compile(r"CUDA Peak Allocated/Reserved: ([0-9.]+) / ([0-9.]+) GB")
SEG_TIME_RE = re.compile(r"Time This Epoch: (\d+):(\d+):(\d+)")
ELAPSED_RE = re.compile(r"Total Time Elapsed: (\d+):(\d+):(\d+)")
CHUNK_END_RE = re.compile(
    r"Chunk limit reached after (\d+) segment\(s\); resume from coverage segment (\d+)"
)
RESUME_RE = re.compile(r"resume from coverage segment (\d+)")
FINISHED_RE = re.compile(r"Training finished\. Summary file saved\.")


def _hms(text: str) -> int:
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _duration(seconds) -> str:
    seconds = int(max(0, seconds or 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


# --------------------------------------------------------------------------
# credentials / transport
# --------------------------------------------------------------------------
def _session():
    """Shared REST session; refreshes the 30-min OAuth token on its own."""
    try:
        return relay_auth.session()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def kernel_status(session, ref: str) -> str:
    owner, slug = ref.split("/", 1)
    try:
        response = session.post(
            f"{RPC_ROOT}/GetKernelSessionStatus",
            json={"userName": owner, "kernelSlug": slug},
            timeout=30,
        )
        response.raise_for_status()
        return str(response.json().get("status", "UNKNOWN"))
    except Exception as exc:  # noqa: BLE001 - status is best-effort
        return f"unknown ({type(exc).__name__})"


def fetch_log_text(session, ref: str) -> str:
    owner, slug = ref.split("/", 1)
    response = session.post(
        f"{RPC_ROOT}/ListKernelSessionOutput",
        json={"userName": owner, "kernelSlug": slug},
        timeout=120,
    )
    response.raise_for_status()
    return str(response.json().get("log") or "")


def events_from_text(text: str):
    """Yield (event_time, console payload) from the persisted blob.

    The blob is a JSON list of `{stream_name,time,data}` events, but older
    sessions may come back as plain text or one JSON object per line.
    """
    text = (text or "").strip()
    if not text:
        return
    if text.startswith("["):
        try:
            for event in json.loads(text):
                if not isinstance(event, dict):
                    continue
                data = event.get("data")
                if data:
                    yield event.get("time"), data
            return
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if stripped.startswith("data:"):
            stripped = stripped[5:].strip()
        if stripped == "END_OF_LOG":
            return
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            yield None, stripped + "\n"
            continue
        if isinstance(event, dict) and event.get("data"):
            yield event.get("time"), str(event["data"])
        else:
            yield None, stripped + "\n"


def stream_events(session, ref: str):
    """Yield (event_time, console payload) from the SSE feed or the blob."""
    owner, slug = ref.split("/", 1)
    url = f"{API_ROOT}/v1/kernels/logs/stream/{owner}/{slug}"
    headers = {"Accept": "text/event-stream, */*", "Content-Type": None}
    # Read timeout matters: a stream attached before the session emits its first
    # line never delivers anything, so reconnect (the server replays from the
    # start) instead of hanging for minutes.
    with session.get(url, headers=headers, stream=True, timeout=(20, 75)) as response:
        response.raise_for_status()
        if "event-stream" not in response.headers.get("content-type", ""):
            for event in events_from_text(response.text):
                yield event
            return
        for raw in response.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "END_OF_LOG":
                return
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                yield None, line + "\n"
                continue
            if isinstance(event, dict) and event.get("data"):
                yield event.get("time"), str(event["data"])


# --------------------------------------------------------------------------
# log -> state
# --------------------------------------------------------------------------
class ChunkState:
    """Fold trainer log lines into one dict per finished coverage segment."""

    WINDOW = 20  # trailing window for the speed / ETA estimate

    def __init__(self) -> None:
        self.current: dict = {}
        self.total_segments: int | None = None
        self.completed: int = 0
        self.segments: list[dict] = []
        self.chunk_limit: int | None = None
        self.resume_segment: int | None = None
        self.chunk_end_segment: int | None = None
        self.finished: bool = False

    # -- feeding ---------------------------------------------------------
    def feed(self, line: str):
        match = PROGRESS_RE.search(line)
        if match:
            try:
                return self._accept(json.loads(match.group(1)))
            except json.JSONDecodeError:
                return None

        match = SUMMARY_RE.search(line)
        if match:
            self.current = {"segment": int(match.group(1))}
            self.total_segments = int(match.group(2))
            return None

        match = TRAIN_STEP_RE.search(line)
        if match:
            self.completed = max(self.completed, int(match.group(1)) - 1)
            self.total_segments = self.total_segments or int(match.group(2))
            return None

        if not self.current:
            return self._outside_summary(line)

        match = VAL_LOSS_RE.search(line)
        if match:
            forecast, history, full = (float(value) for value in match.groups())
            self.current.update(forecast=forecast, history=history, full=full)
            return None
        match = VAL_WEIGHTED_RE.search(line)
        if match:
            self.current["selection"] = float(match.group(1))
            return None
        match = VAL_BEST_RE.search(line)
        if match:
            self.current["train_average"] = float(match.group(1))
            self.current["best"] = float(match.group(2))
            return None
        match = BEST_METRIC_RE.search(line)
        if match:
            value = match.group(1)
            self.current["best_metric"] = None if value == "not evaluated" else float(value)
            return None
        match = ABLATION_RE.search(line)
        if match:
            full, none, shuffled, delta_none, delta_shuffled = match.groups()
            self.current.update(
                condition_full=float(full),
                condition_none=float(none),
                condition_shuffled=float(shuffled),
                delta_full_none=float(delta_none),
                delta_full_shuffled=float(delta_shuffled),
            )
            return None
        match = CUDA_RE.search(line)
        if match:
            self.current["cuda_peak_gb"] = float(match.group(1))
            return None
        match = SEG_TIME_RE.search(line)
        if match:
            self.current["segment_seconds"] = _hms(":".join(match.groups()))
            return None
        match = ELAPSED_RE.search(line)
        if match:
            self.current["run_elapsed_seconds"] = _hms(":".join(match.groups()))
            return self._accept(self.current)
        return None

    def _outside_summary(self, line: str):
        match = CHUNK_END_RE.search(line)
        if match:
            self.chunk_limit = int(match.group(1))
            self.resume_segment = int(match.group(2))
            self.chunk_end_segment = self.resume_segment + self.chunk_limit - 1
            return None
        match = RESUME_RE.search(line)
        if match:
            self.resume_segment = self.resume_segment or int(match.group(1))
            return None
        if FINISHED_RE.search(line):
            self.finished = True
        return None

    def _accept(self, record: dict):
        self.current = {}
        segment = int(record.get("segment") or 0)
        if not segment:
            return None
        if self.total_segments is None and record.get("total"):
            self.total_segments = int(record["total"])
        if self.segments and self.segments[-1].get("segment") == segment:
            self.segments[-1] = record
        else:
            self.segments.append(record)
        self.completed = max(self.completed, segment)
        if record.get("resume_segment"):
            self.resume_segment = int(record["resume_segment"])
        if record.get("chunk_end_segment"):
            self.chunk_end_segment = int(record["chunk_end_segment"])
        return record

    # -- derived ---------------------------------------------------------
    @property
    def average_segment_seconds(self):
        """Trailing window: the first segments of a fresh chunk carry startup noise."""
        values = [
            float(s["segment_seconds"])
            for s in self.segments[-self.WINDOW:]
            if s.get("segment_seconds")
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _eta(self, target):
        average = self.average_segment_seconds
        if average is None or not target or not self.completed:
            return None
        return max(0, target - self.completed) * average

    @property
    def resolved_chunk_end(self):
        if self.chunk_end_segment:
            return self.chunk_end_segment
        if self.chunk_limit and self.resume_segment:
            return self.resume_segment + self.chunk_limit - 1
        return None

    def eta_chunk_seconds(self):
        return self._eta(self.resolved_chunk_end)

    def eta_total_seconds(self):
        return self._eta(self.total_segments)

    def snapshot(self) -> dict:
        return {
            "completed_segments": self.completed,
            "total_segments": self.total_segments,
            "avg_segment_seconds": (
                round(self.average_segment_seconds, 1) if self.average_segment_seconds else None
            ),
            "chunk_end_segment": self.resolved_chunk_end,
            "eta_chunk_seconds": (
                round(self.eta_chunk_seconds(), 1) if self.eta_chunk_seconds() else None
            ),
            "eta_total_seconds": (
                round(self.eta_total_seconds(), 1) if self.eta_total_seconds() else None
            ),
            "chunk_limit": self.chunk_limit,
            "resume_segment": self.resume_segment,
            "finished": self.finished,
            "last_segment": self.segments[-1] if self.segments else None,
        }


def format_record(record: dict, state: ChunkState) -> str:
    total = record.get("total") or state.total_segments or "?"
    parts = [f"seg {record.get('segment', '?')}/{total}"]

    if record.get("forecast") is not None:
        parts.append(f"fcast {record['forecast']:.5f}")
    elif record.get("best") is not None:
        parts.append("val skipped")
    if record.get("selection") is not None:
        parts.append(f"sel {record['selection']:.5f}")
    if record.get("best") is not None:
        parts.append(f"best {record['best']:.5f}")
    if record.get("delta_full_none") is not None:
        parts.append(
            f"Δfn {record['delta_full_none']:+.2e} Δfs {record['delta_full_shuffled']:+.2e}"
        )

    seconds = record.get("segment_seconds")
    if seconds:
        parts.append(f"{_duration(seconds)}/seg")
    average = record.get("avg_segment_seconds") or state.average_segment_seconds
    if average:
        parts.append(f"avg{ChunkState.WINDOW} {_duration(average)}")
    if record.get("run_elapsed_seconds"):
        parts.append(f"run {_duration(record['run_elapsed_seconds'])}")

    if state.finished:
        parts.append("finished")
    else:
        eta_chunk = record.get("eta_chunk_seconds") or state.eta_chunk_seconds()
        target = record.get("chunk_end_segment") or state.resolved_chunk_end
        eta_total = record.get("eta_total_seconds") or state.eta_total_seconds()
        if eta_chunk is not None:
            parts.append(f"ETAchunk {_duration(eta_chunk)}{f'->{target}' if target else ''}")
        elif eta_total is not None:
            parts.append(f"ETAtotal {_duration(eta_total)}")
    if record.get("cuda_peak_gb") is not None:
        parts.append(f"peak {record['cuda_peak_gb']:.2f}GB")
    return " | ".join(parts)


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def _print_record(record: dict, state: ChunkState, as_json: bool) -> None:
    stamp = time.strftime("%H:%M:%SZ", time.gmtime())
    if as_json:
        print(json.dumps(state.snapshot(), ensure_ascii=False), flush=True)
    else:
        print(f"[{stamp}] {format_record(record, state)}", flush=True)


def one_shot(ref: str, tail: int, as_json: bool, raw: bool) -> int:
    session = _session()
    status = kernel_status(session, ref)
    text = fetch_log_text(session, ref)
    state = ChunkState()
    for _event_time, payload in events_from_text(text):
        for line in payload.splitlines():
            record = state.feed(line)
            if record and raw:
                print(line, flush=True)
    if as_json:
        print(json.dumps(state.snapshot(), indent=2, ensure_ascii=False))
        return 0
    if not state.segments:
        print(f"[{ref}] status={status}; console log has no segment summaries yet.")
        print("  A queued/running session keeps its progress in the live stream: --follow --wait")
        return 0
    for record in state.segments[-tail:]:
        print(format_record(record, state))
    snapshot = state.snapshot()
    tail_text = (
        " [finished]"
        if snapshot["finished"]
        else f" eta_chunk={_duration(snapshot['eta_chunk_seconds'])}"
    )
    print(
        f"-- status={status} completed={snapshot['completed_segments']}"
        f"/{snapshot['total_segments']}"
        f" avg={_duration(snapshot['avg_segment_seconds'])}"
        + tail_text
    )
    return 0


def wait_for_session(session, ref: str, timeout_seconds: int) -> str:
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        status = kernel_status(session, ref)
        if status != last:
            print(f"# {ref} status={status}", flush=True)
            last = status
        if status.startswith("unknown"):
            return status
        if status in {"RUNNING", "COMPLETE", "ERROR", "CANCELED"}:
            return status
        time.sleep(10)
    return last or "TIMEOUT"


def follow(ref: str, tail: int, as_json: bool, raw: bool, wait: bool) -> int:
    session = _session()
    status = kernel_status(session, ref)
    if wait:
        status = wait_for_session(session, ref, 30 * 60)
    print(f"# attaching to {ref} (status={status}); Ctrl-C detaches, training keeps running", flush=True)

    state = ChunkState()
    last_event_time = None
    failures = 0
    last_beat = time.time()
    last_line = ""
    pending = {"record": None, "at": 0.0}

    def flush_pending(force: bool = False) -> None:
        """Print a held segment record unless a [PROGRESS] line superseded it."""
        record = pending["record"]
        if record is None:
            return
        if not force and time.time() - pending["at"] < 4.0:
            return
        pending["record"] = None
        _print_record(record, state, as_json)

    def beat(force: bool = False) -> None:
        """One trainer line a minute, so "alive" is visible between segments.

        Trainer lines carry their own UTC stamp (added by the runner), so the
        client clock is not repeated here.
        """
        nonlocal last_beat
        if not last_line:
            return
        if force or time.time() - last_beat >= 60:
            print(f"· {last_line[:170]}", flush=True)
            last_beat = time.time()

    while True:
        progressed = False
        finished_stream = False
        try:
            for event_time, payload in stream_events(session, ref):
                # A reconnect replays from the beginning (or from "now"); dedupe
                # on the server timestamp instead of a fragile event count.
                if event_time is not None:
                    if last_event_time is not None and event_time <= last_event_time:
                        continue
                    last_event_time = event_time
                progressed = True
                for line in payload.splitlines():
                    last_line = line.strip() or last_line
                    record = state.feed(line)
                    if raw:
                        print(line, flush=True)
                    if record:
                        held = pending["record"]
                        # [PROGRESS] records always carry an averaged segment
                        # time; the legacy summary block never does.
                        is_progress = record.get("type") == "progress" or (
                            "avg_segment_seconds" in record
                        )
                        if is_progress and held and held.get("segment") == record.get("segment"):
                            pending["record"] = None  # superseded, do not print twice
                            _print_record(record, state, as_json)
                        else:
                            flush_pending(force=True)
                            pending["record"] = record
                            pending["at"] = time.time()
                    elif line.lstrip().startswith('{"phase"'):
                        # GPU heartbeats are already on SwanLab; they would
                        # otherwise print twice a minute for hours.
                        if '"phase": "gpu"' in line:
                            continue
                        # Server-relative elapsed is stable across replays; the
                        # client clock is not.
                        when = f"+{_duration(event_time)}" if event_time else "phase"
                        print(f"[{when}] # {line.strip()[:220]}", flush=True)
                    else:
                        flush_pending()
                        beat()
            finished_stream = True
        except KeyboardInterrupt:
            print("\n# detached (training unaffected)", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 - SSE drops are routine
            failures = 0 if progressed else failures + 1
            if failures:
                print(
                    f"# stream idle/failed ({failures}/10): {type(exc).__name__}; "
                    f"session={kernel_status(session, ref)}",
                    flush=True,
                )
                beat(force=True)
            # Kaggle's SSE endpoint and this network both flap; a 10h run will
            # see multi-minute outages, so back off for ~20 minutes before
            # giving up instead of ~30 seconds.
            if failures >= 40:
                print(
                    "# giving up on the live stream after "
                    f"{failures} consecutive failures; re-run with --follow later",
                    flush=True,
                )
                break
            time.sleep(min(30, 3 * failures))
        if finished_stream:
            status = kernel_status(session, ref)
            if state.finished or state.chunk_limit:
                break
            if status in {"COMPLETE", "ERROR", "CANCELED"}:
                break
            # The SSE feed can close while the session keeps running: reattach.
            print(f"# stream closed but session={status}; reattaching", flush=True)
            time.sleep(2)

    flush_pending(force=True)
    snapshot = state.snapshot()
    print(
        f"-- stream ended: completed={snapshot['completed_segments']}"
        f"/{snapshot['total_segments']} chunk_end={snapshot['chunk_end_segment']}"
        f" resume={snapshot['resume_segment']}"
        + (" [finished]" if snapshot["finished"] else ""),
        flush=True,
    )
    for record in state.segments[-tail:]:
        print(format_record(record, state))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kernel", default=DEFAULT_KERNEL, help="owner/slug (default: %(default)s)")
    parser.add_argument("--follow", action="store_true", help="stream live logs while the session runs")
    parser.add_argument("--wait", action="store_true", help="wait up to 30 min for the session to start")
    parser.add_argument("--tail", type=int, default=10, help="segments to print at the end (default: 10)")
    parser.add_argument("--json", action="store_true", help="print machine-readable snapshots")
    parser.add_argument("--raw", action="store_true", help="also echo original trainer lines")
    args = parser.parse_args()
    if args.follow:
        return follow(args.kernel, args.tail, args.json, args.raw, args.wait)
    return one_shot(args.kernel, args.tail, args.json, args.raw)


if __name__ == "__main__":
    raise SystemExit(main())
