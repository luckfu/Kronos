#!/usr/bin/env python
"""Run read-only, resumable full-market inference from PostgreSQL to Modal.

The same program runs locally or on Oracle. It reads Supabase PostgreSQL through
DB_URL/DATABASE_URL, writes only filesystem artifacts, and never mutates source
market data or production prediction tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


LOOKBACK = 120
HORIZON = 10
MAX_BATCH_SIZE = 12
DEFAULT_INFERENCE_URL = "https://luckfu--kronos-beta-v1-2-inference-web.modal.run"
FEATURES = ("open", "high", "low", "close", "volume", "amount")
DEFAULT_TEMPERATURE = 0.60
DEFAULT_TOP_P = 0.90
DEFAULT_TOP_K = 0
DEFAULT_SAMPLE_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict A-share universe and predict it in resumable batches."
    )
    parser.add_argument("--asof", required=True, type=date.fromisoformat)
    parser.add_argument("--sector-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--inference-url",
        default=os.getenv("KRONOS_INFERENCE_URL", DEFAULT_INFERENCE_URL),
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Predict only the first N eligible symbols; 0 means the full universe.",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument(
        "--future-dates",
        help="Comma-separated exchange trading dates. If omitted, use AkShare's calendar.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Refuse to reuse completed batch files from the same run fingerprint.",
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.max_retries < 1:
        parser.error("--max-retries must be positive")
    if args.request_timeout < 1:
        parser.error("--request-timeout must be positive")
    if not 1 <= args.sample_count <= 50:
        parser.error("--sample-count must be between 1 and 50")
    return args


def database_url() -> str:
    value = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("DB_URL or DATABASE_URL is required")
    return value


def load_sector_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        raise RuntimeError("sector map does not contain a symbols object")
    return symbols


def load_audit_rows(connection, asof: date) -> tuple[pd.DataFrame, list[date]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT date
            FROM public.ashare_daily_k
            WHERE date <= %s
            ORDER BY date DESC
            LIMIT %s
            """,
            (asof, LOOKBACK),
        )
        dates = sorted(row[0] for row in cursor.fetchall())
        if len(dates) != LOOKBACK or dates[-1] != asof:
            raise RuntimeError(
                f"expected {LOOKBACK} market dates ending at {asof}, got {len(dates)}"
            )
        cursor.execute(
            """
            WITH raw_window_rows AS (
                SELECT k.code, k.date, k.open, k.high, k.low, k.close,
                       k.volume, k.amount, f.adj_factor, f.source
                FROM public.ashare_daily_k k
                LEFT JOIN public.ashare_adj_factor f
                  ON f.code = k.code AND f.date = k.date
                WHERE k.date = ANY(%s)
            ), window_rows AS (
                SELECT *,
                       lag(adj_factor) OVER (PARTITION BY code ORDER BY date) AS previous_factor,
                       lag(source) OVER (PARTITION BY code ORDER BY date) AS previous_source
                FROM raw_window_rows
            ), history AS (
                SELECT code,
                       count(*) AS history_rows,
                       count(*) FILTER (
                           WHERE open IS NULL OR high IS NULL OR low IS NULL
                              OR close IS NULL OR volume IS NULL OR amount IS NULL
                              OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                              OR high < greatest(open, close, low)
                              OR low > least(open, close, high)
                              OR volume < 0 OR amount < 0
                       ) AS invalid_ohlcva_rows,
                       count(adj_factor) AS factor_rows,
                       count(*) FILTER (
                           WHERE adj_factor IS NULL OR adj_factor <= 0
                       ) AS invalid_factor_rows,
                       count(DISTINCT source) FILTER (WHERE source IS NOT NULL)
                           AS factor_source_count,
                       count(*) FILTER (
                           WHERE previous_source IS NOT NULL AND source <> previous_source
                       ) AS factor_source_switches,
                       coalesce(max(abs(ln(adj_factor / previous_factor))) FILTER (
                           WHERE previous_source IS NOT NULL AND source <> previous_source
                             AND adj_factor > 0 AND previous_factor > 0
                       ), 0) AS max_source_switch_log_jump
                FROM window_rows
                GROUP BY code
            )
            SELECT a.code, a.trade_status, a.is_st, a.close, a.amount, a.turnover,
                   h.history_rows, h.invalid_ohlcva_rows, h.factor_rows,
                   h.invalid_factor_rows, h.factor_source_count,
                   h.factor_source_switches, h.max_source_switch_log_jump
            FROM public.ashare_daily_k a
            LEFT JOIN history h ON h.code = a.code
            WHERE a.date = %s
            ORDER BY a.code
            """,
            (dates, asof),
        )
        columns = [item.name for item in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns), dates


def classify(rows: pd.DataFrame, sectors: dict[str, dict]) -> tuple[pd.DataFrame, Counter]:
    numeric = ["close", "amount", "turnover", "history_rows", "invalid_ohlcva_rows",
               "factor_rows", "invalid_factor_rows", "factor_source_count",
               "factor_source_switches", "max_source_switch_log_jump"]
    for column in numeric:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    valid_cap = rows["amount"].gt(0) & rows["turnover"].gt(0)
    rows["market_cap_proxy"] = np.where(
        valid_cap, rows["amount"] / (rows["turnover"] / 100.0), np.nan
    )
    reference = rows.loc[valid_cap].sort_values("code").copy()
    reference["size_percentile"] = reference["market_cap_proxy"].rank(
        method="first", pct=True
    )
    rows = rows.merge(reference[["code", "size_percentile"]], on="code", how="left")

    reasons: list[str] = []
    sector_ids: list[int | None] = []
    sector_labels: list[str | None] = []
    for row in rows.itertuples(index=False):
        sector = sectors.get(row.code)
        sector_label = None if sector is None else str(sector.get("sector_label", ""))
        sector_id = None if sector is None else int(sector.get("sector_id", -1))
        sector_ids.append(sector_id)
        sector_labels.append(sector_label)
        if row.trade_status != "1" or int(row.is_st) != 0:
            reason = "st_or_not_tradeable"
        elif int(row.history_rows or 0) != LOOKBACK:
            reason = "missing_history"
        elif int(row.invalid_ohlcva_rows or 0) != 0:
            reason = "invalid_ohlcva"
        elif int(row.factor_rows or 0) != LOOKBACK:
            reason = "missing_adj_factor"
        elif int(row.invalid_factor_rows or 0) != 0:
            reason = "invalid_adj_factor"
        elif float(row.max_source_switch_log_jump or 0) > 1e-6:
            reason = "discontinuous_factor_source"
        elif sector is None or sector_label.lower() == "unknown" or not 0 <= sector_id < 86:
            reason = "unknown_sector"
        elif not math.isfinite(float(row.market_cap_proxy)):
            reason = "invalid_market_cap"
        else:
            reason = "eligible"
        reasons.append(reason)
    rows["sector_id"] = sector_ids
    rows["sector_label"] = sector_labels
    rows["reason"] = reasons
    return rows, Counter(reasons)


def load_histories(connection, codes: list[str], dates: list[date]) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT k.code, k.date, k.open, k.high, k.low, k.close,
                   k.volume, k.amount, f.adj_factor
            FROM public.ashare_daily_k k
            JOIN public.ashare_adj_factor f
              ON f.code = k.code AND f.date = k.date
            WHERE k.code = ANY(%s) AND k.date = ANY(%s)
            ORDER BY k.code, k.date
            """,
            (codes, dates),
        )
        columns = [item.name for item in cursor.description]
        frame = pd.DataFrame(cursor.fetchall(), columns=columns)
    for column in (*FEATURES, "adj_factor"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def future_trading_dates(asof: date, supplied: str | None = None) -> tuple[list[str], str]:
    if supplied:
        values = [item.strip() for item in supplied.split(",") if item.strip()]
        source = "command_line"
    else:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError(
                "AkShare is required to resolve future A-share trading dates; "
                "install prediction-requirements.txt or pass --future-dates"
            ) from exc
        calendar = ak.tool_trade_date_hist_sina()
        parsed = pd.to_datetime(calendar["trade_date"], errors="coerce").dt.date
        values = [item.isoformat() for item in parsed if item > asof][:HORIZON]
        source = "akshare.tool_trade_date_hist_sina"
    parsed_values = [date.fromisoformat(item) for item in values]
    if len(parsed_values) != HORIZON:
        raise RuntimeError(f"expected exactly {HORIZON} future trading dates")
    if parsed_values != sorted(set(parsed_values)) or parsed_values[0] <= asof:
        raise RuntimeError("future trading dates must be unique, increasing, and after asof")
    return [item.isoformat() for item in parsed_values], source


def make_payload(
    histories: pd.DataFrame,
    eligible: pd.DataFrame,
    future_dates: list[str],
    sample_count: int,
) -> dict:
    metadata = eligible.set_index("code")
    items = []
    for code, frame in histories.groupby("code", sort=True):
        latest_factor = float(frame.iloc[-1]["adj_factor"])
        adjusted = frame.copy()
        ratio = adjusted["adj_factor"] / latest_factor
        for column in ("open", "high", "low", "close"):
            adjusted[column] *= ratio
        data = []
        for row in adjusted.itertuples(index=False):
            values = {column: float(getattr(row, column)) for column in FEATURES}
            data.append({"timestamp": row.date.isoformat(), **values})
        row = metadata.loc[code]
        items.append({
            "id": code,
            "data": data,
            "sector_id": int(row["sector_id"]),
            "size_percentile": float(row["size_percentile"]),
        })
    return {
        "items": items,
        "future_timestamps": future_dates,
        "pred_len": HORIZON,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "sample_count": sample_count,
    }


def validate_response(
    response: dict, expected_codes: list[str], current_closes: dict[str, float]
) -> dict:
    results = response.get("results")
    if not isinstance(results, list) or len(results) != len(expected_codes):
        raise RuntimeError("Modal response has an unexpected result count")
    summaries = []
    for result, code in zip(results, expected_codes):
        predictions = result.get("predictions")
        if result.get("id") != code or not isinstance(predictions, list) or len(predictions) != HORIZON:
            raise RuntimeError(f"invalid prediction shape for {code}")
        closes = np.asarray([row["close_p50"] for row in predictions], dtype=float)
        if not np.isfinite(closes).all():
            raise RuntimeError(f"non-finite close_p50 for {code}")
        summaries.append({
            "code": code,
            "close_asof": current_closes[code],
            "close_p50_d10": float(closes[-1]),
            "predicted_return_d10": float(closes[-1] / current_closes[code] - 1.0),
        })
    meta = response.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError("Modal response is missing metadata")
    if meta.get("model_release") != "small-0.1-cosine-c2":
        raise RuntimeError(f"unexpected model release: {meta.get('model_release')}")
    if meta.get("model_checkpoint") != "Segment@179":
        raise RuntimeError(f"unexpected model checkpoint: {meta.get('model_checkpoint')}")
    if int(meta.get("sample_count", -1)) <= 0:
        raise RuntimeError("Modal response has an invalid sample count")
    return {"meta": meta, "symbols": summaries}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def chunks(values: list[str], size: int):
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def run_fingerprint(
    args: argparse.Namespace,
    dates: list[date],
    future_dates: list[str],
    codes: list[str],
) -> str:
    identity = {
        "asof": args.asof.isoformat(),
        "lookback_dates": [item.isoformat() for item in dates],
        "future_dates": future_dates,
        "codes": codes,
        "sector_map_sha256": hashlib.sha256(args.sector_map.read_bytes()).hexdigest(),
        "inference_url": args.inference_url.rstrip("/"),
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "sample_count": args.sample_count,
        "batch_size": args.batch_size,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def post_batch(
    session: requests.Session,
    endpoint: str,
    payload: dict,
    max_retries: int,
    timeout: int,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(endpoint, json=payload, timeout=timeout)
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response.json()
            response.raise_for_status()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            wait_seconds = min(2 ** attempt, 30)
            print(
                f"request attempt {attempt}/{max_retries} failed; "
                f"retrying in {wait_seconds}s: {type(exc).__name__}",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Modal batch failed after {max_retries} attempts") from last_error


def validate_histories(histories: pd.DataFrame, codes: list[str], dates: list[date]) -> None:
    expected_dates = [item.isoformat() for item in dates]
    observed_codes = histories["code"].drop_duplicates().tolist()
    if observed_codes != codes:
        raise RuntimeError(f"history code mismatch: expected {codes}, got {observed_codes}")
    for code, frame in histories.groupby("code", sort=False):
        observed_dates = [item.isoformat() for item in frame["date"]]
        if observed_dates != expected_dates:
            raise RuntimeError(f"history dates are incomplete for {code}")


def completed_batch(path: Path, fingerprint: str, codes: list[str]) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("fingerprint") != fingerprint or payload.get("codes") != codes:
        raise RuntimeError(f"completed batch does not match this run: {path}")
    return payload


def build_outputs(
    output_dir: Path,
    selected: pd.DataFrame,
    batch_payloads: list[dict],
    fingerprint: str,
) -> dict:
    metadata = selected.set_index("code")
    prediction_records = []
    ranking_rows = []
    model_meta = None
    for batch in batch_payloads:
        response = batch["response"]
        model_meta = response["meta"]
        for result in response["results"]:
            code = str(result["id"])
            row = metadata.loc[code]
            closes = [float(item["close_p50"]) for item in result["predictions"]]
            close_asof = float(row["close"])
            predicted_return = closes[-1] / close_asof - 1.0
            prediction_records.append({
                "asof": batch["asof"],
                "code": code,
                "sector_id": int(row["sector_id"]),
                "sector_label": str(row["sector_label"]),
                "size_percentile": float(row["size_percentile"]),
                "close_asof": close_asof,
                "predictions": result["predictions"],
                "samples": result.get("samples"),
            })
            ranking_rows.append({
                "asof": batch["asof"],
                "code": code,
                "sector_id": int(row["sector_id"]),
                "sector_label": str(row["sector_label"]),
                "size_percentile": float(row["size_percentile"]),
                "close_asof": close_asof,
                "close_p50_d1": closes[0],
                "close_p50_d10": closes[-1],
                "predicted_return_d1": closes[0] / close_asof - 1.0,
                "predicted_return_d10": predicted_return,
            })
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["predicted_return_d10", "code"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking["rank_d10"] = np.arange(1, len(ranking) + 1)
    ranking["rank_percentile_d10"] = ranking["predicted_return_d10"].rank(
        method="first", pct=True
    )
    atomic_csv(output_dir / "ranking.csv", ranking)
    jsonl_tmp = output_dir / "predictions.jsonl.tmp"
    with jsonl_tmp.open("w") as stream:
        for record in prediction_records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    jsonl_tmp.replace(output_dir / "predictions.jsonl")
    summary = {
        "status": "complete",
        "fingerprint": fingerprint,
        "prediction_count": len(prediction_records),
        "batch_count": len(batch_payloads),
        "model": model_meta,
        "ranking_file": "ranking.csv",
        "predictions_file": "predictions.jsonl",
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    sectors = load_sector_map(args.sector_map)
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError(
            "psycopg2 is required; install deploy/oracle-kronos/prediction-requirements.txt"
        ) from exc

    future_dates, calendar_source = future_trading_dates(args.asof, args.future_dates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with psycopg2.connect(database_url()) as connection:
        rows, dates = load_audit_rows(connection, args.asof)
        classified, counts = classify(rows, sectors)
        eligible = classified[classified["reason"] == "eligible"].sort_values("code")
        selected = eligible if args.limit == 0 else eligible.head(args.limit)
        if selected.empty:
            raise RuntimeError("no symbols passed the strict eligibility filters")
        codes = selected["code"].tolist()
        fingerprint = run_fingerprint(args, dates, future_dates, codes)
        batches = list(chunks(codes, args.batch_size))
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "output directory belongs to a different run; choose another directory"
                )
        manifest = {
            "schema_version": 1,
            "status": "running",
            "fingerprint": fingerprint,
            "asof": args.asof.isoformat(),
            "lookback_dates": [item.isoformat() for item in dates],
            "future_dates": future_dates,
            "calendar_source": calendar_source,
            "eligible_count": int(len(eligible)),
            "selected_count": int(len(selected)),
            "batch_size": args.batch_size,
            "batch_count": len(batches),
            "sample_count": args.sample_count,
            "inference_url": args.inference_url.rstrip("/"),
        }
        atomic_json(manifest_path, manifest)
        audit = {
            "asof": args.asof.isoformat(),
            "lookback": LOOKBACK,
            "market_date_start": dates[0].isoformat(),
            "market_date_end": dates[-1].isoformat(),
            "future_dates": future_dates,
            "calendar_source": calendar_source,
            "asof_universe": int(len(classified)),
            "market_cap_reference_count": int(classified["size_percentile"].notna().sum()),
            "factor_source_count_distribution": {
                str(int(key)): int(value)
                for key, value in classified["factor_source_count"].value_counts().sort_index().items()
            },
            "factor_source_switches": int(classified["factor_source_switches"].sum()),
            "max_source_switch_log_jump": float(
                classified["max_source_switch_log_jump"].max()
            ),
            "reason_counts": dict(sorted(counts.items())),
            "eligible_count": int(len(eligible)),
            "selected_count": int(len(selected)),
        }
        atomic_json(args.output_dir / "audit.json", audit)
        atomic_csv(args.output_dir / "universe.csv", classified)

        session = requests.Session()
        api_key = os.getenv("KRONOS_API_KEY", "").strip()
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        endpoint = args.inference_url.rstrip("/") + "/predict-batch"
        batch_payloads = []
        batch_dir = args.output_dir / "batches"
        for batch_index, batch_codes in enumerate(batches, start=1):
            batch_path = batch_dir / f"batch_{batch_index:04d}.json"
            existing = None if args.no_resume else completed_batch(
                batch_path, fingerprint, batch_codes
            )
            if existing is not None:
                batch_payloads.append(existing)
                print(
                    f"batch {batch_index}/{len(batches)} resumed "
                    f"({len(batch_codes)} symbols)", flush=True,
                )
                continue
            if args.no_resume and batch_path.exists():
                raise RuntimeError(f"batch exists and --no-resume was supplied: {batch_path}")
            histories = load_histories(connection, batch_codes, dates)
            validate_histories(histories, batch_codes, dates)
            batch_rows = selected[selected["code"].isin(batch_codes)].copy()
            payload = make_payload(histories, batch_rows, future_dates, args.sample_count)
            response = post_batch(
                session, endpoint, payload, args.max_retries, args.request_timeout
            )
            current_closes = {
                str(row.code): float(row.close)
                for row in batch_rows[["code", "close"]].itertuples(index=False)
            }
            validate_response(response, batch_codes, current_closes)
            completed = {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "asof": args.asof.isoformat(),
                "batch_index": batch_index,
                "codes": batch_codes,
                "response": response,
            }
            atomic_json(batch_path, completed)
            batch_payloads.append(completed)
            atomic_json(args.output_dir / "progress.json", {
                "status": "running",
                "fingerprint": fingerprint,
                "completed_batches": batch_index,
                "batch_count": len(batches),
                "completed_symbols": sum(len(item["codes"]) for item in batch_payloads),
                "selected_count": len(codes),
            })
            print(
                f"batch {batch_index}/{len(batches)} completed "
                f"({len(batch_codes)} symbols)", flush=True,
            )

    summary = build_outputs(args.output_dir, selected, batch_payloads, fingerprint)
    manifest["status"] = "complete"
    atomic_json(manifest_path, manifest)
    atomic_json(args.output_dir / "progress.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
