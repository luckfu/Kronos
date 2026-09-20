import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from webui import app as web_app


def published_run(tmp_path, names=None):
    run = tmp_path / "2026-09-18" / "full_market"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "status": "complete",
        "prediction_count": 2,
        "batch_count": 1,
        "model": {"sample_count": 5, "model_release": "small-0.1-cosine-c2"},
    }))
    rows = [
        {
            "asof": "2026-09-18", "code": "sh.600000", "sector_id": 63,
            "sector_label": "J66货币金融服务", "size_percentile": 0.9,
            "close_asof": 10.0, "close_p50_d1": 10.1, "close_p50_d10": 11.0,
            "predicted_return_d1": 0.01, "predicted_return_d10": 0.1,
            "rank_d10": 1, "rank_percentile_d10": 1.0,
        },
        {
            "asof": "2026-09-18", "code": "sz.000001", "sector_id": 63,
            "sector_label": "J66货币金融服务", "size_percentile": 0.7,
            "close_asof": 20.0, "close_p50_d1": 19.8, "close_p50_d10": 19.0,
            "predicted_return_d1": -0.01, "predicted_return_d10": -0.05,
            "rank_d10": 2, "rank_percentile_d10": 0.5,
        },
        {
            "asof": "2026-09-18", "code": "sz.000063", "sector_id": 39,
            "sector_label": "C39计算机、通信和其他电子设备制造业",
            "size_percentile": 0.85,
            "close_asof": 30.0, "close_p50_d1": 30.2, "close_p50_d10": 31.0,
            "predicted_return_d1": 0.007, "predicted_return_d10": 0.033,
            "rank_d10": 11, "rank_percentile_d10": 0.01,
        },
    ]
    if names:
        for row in rows:
            if row["code"] in names:
                row["name"] = names[row["code"]]
    pd.DataFrame(rows).to_csv(run / "ranking.csv", index=False)
    detail = {
        "asof": "2026-09-18", "code": "sh.600000", "sector_id": 63,
        "sector_label": "J66货币金融服务", "size_percentile": 0.9,
        "close_asof": 10.0,
        "predictions": [{"timestamp": "2026-09-21", "close_p10": 9.8, "close_p50": 10.1, "close_p90": 10.4}],
        "samples": {"close": [[10.1], [10.2], [10.0], [9.9], [10.3]]},
    }
    extra = []
    for row in rows:
        if row["code"] == "sh.600000":
            continue
        extra.append({
            "asof": row["asof"], "code": row["code"], "sector_id": row["sector_id"],
            "sector_label": row["sector_label"], "size_percentile": row["size_percentile"],
            "close_asof": row["close_asof"],
            "predictions": [
                {
                    "timestamp": "2026-09-21",
                    "close_p10": row["close_p50_d1"] - 0.3,
                    "close_p50": row["close_p50_d1"],
                    "close_p90": row["close_p50_d1"] + 0.3,
                },
                {
                    "timestamp": "2026-10-09",
                    "close_p10": row["close_p50_d10"] - 0.4,
                    "close_p50": row["close_p50_d10"],
                    "close_p90": row["close_p50_d10"] + 0.4,
                },
            ],
            "samples": {"close": [[row["close_p50_d1"]], [row["close_p50_d10"]]]},
        })
    (run / "predictions.jsonl").write_text(
        json.dumps(detail) + "\n"
        + "".join(json.dumps(item) + "\n" for item in extra)
    )
    return run


def reset_stock_name_cache(monkeypatch, tmp_path, names=None, updated_at=None):
    cache_path = tmp_path / "stock_name_cache.json"
    monkeypatch.setattr(web_app, "STOCK_NAME_CACHE_PATH", str(cache_path))
    web_app.stock_name_cache.clear()
    web_app.stock_name_cache_loaded = False
    web_app.stock_name_cache_loaded_path = None
    web_app.stock_name_cache_updated_at = None
    web_app.stock_name_cache_mtime = None
    if names is not None:
        payload = {
            "updated_at": (
                updated_at
                if isinstance(updated_at, str)
                else (updated_at or datetime.now(timezone.utc)).isoformat()
            ),
            "names": names,
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return cache_path


def forbid_live_name_lookups(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("live stock-name lookup should not run")

    monkeypatch.setattr(web_app, "fetch_baostock_stock_names", explode)
    monkeypatch.setattr(web_app, "query_baostock_stock_name", explode)
    monkeypatch.setattr(web_app, "query_eastmoney_stock_name", explode)


@pytest.fixture(autouse=True)
def no_live_ranking_history(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "_fetch_ranking_history_remote",
        lambda *args, **kwargs: (pd.DataFrame(columns=web_app.MARKET_DATA_COLUMNS), None),
    )


def history_frame(asof="2026-09-18", rows=90, close=10.0, extra_after=0):
    dates = list(pd.bdate_range(end=asof, periods=rows))
    closes = list(close * (1 + np.linspace(-0.05, 0.0, rows)))
    if extra_after:
        dates.extend(pd.bdate_range(
            start=pd.Timestamp(dates[-1]) + pd.Timedelta(days=1),
            periods=extra_after,
        ))
        closes.extend(close * (1 + np.linspace(0.01, 0.08, extra_after)))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamps": dates,
        "open": closes - 0.12,
        "high": closes + 0.25,
        "low": closes - 0.22,
        "close": closes,
        "volume": np.full(len(dates), 1000.0),
        "amount": np.full(len(dates), 10000.0),
        "turn": np.full(len(dates), 1.0),
        "pctChg": np.zeros(len(dates)),
    })



def test_daily_rankings_only_lists_complete_runs(monkeypatch, tmp_path):
    published_run(tmp_path)
    incomplete = tmp_path / "2026-09-19" / "full_market"
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)

    response = web_app.app.test_client().get("/api/daily-rankings/dates")

    assert response.status_code == 200
    assert response.get_json()["latest"] == "2026-09-18"
    assert len(response.get_json()["dates"]) == 1


def test_daily_rankings_supports_filters_and_detail(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path)
    forbid_live_name_lookups(monkeypatch)
    client = web_app.app.test_client()

    ranking = client.get("/api/daily-rankings?asof=2026-09-18&query=600000&size_group=large")
    detail = client.get("/api/daily-rankings/2026-09-18/600000")

    assert ranking.status_code == 200
    assert ranking.get_json()["total"] == 1
    assert ranking.get_json()["rows"][0]["code"] == "sh.600000"
    assert detail.status_code == 200
    payload = detail.get_json()
    assert payload["model"]["sample_count"] == 5
    assert payload["ranking"]["rank_d10"] == 1
    assert payload["history"] == []
    assert payload["history_source"] is None
    assert "缺少截至信号日的历史行情" in payload["history_note"]
    assert payload["predictions"][0]["close_p50"] == 10.1


def test_daily_rankings_re_ranks_selected_pool(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path)
    forbid_live_name_lookups(monkeypatch)
    client = web_app.app.test_client()

    response = client.get(
        "/api/daily-rankings?asof=2026-09-18&pool=sh.600000,sz.000001"
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["pool_count"] == 2
    assert payload["total"] == 2
    assert [row["rank_d10"] for row in payload["rows"]] == [1, 2]
    assert payload["rows"][0]["code"] == "sh.600000"


def test_home_uses_daily_ranking_experience():
    response = web_app.app.test_client().get("/")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "每日排名" in page
    assert "OOS表现" not in page
    assert "手动预测" not in page


def test_daily_rankings_query_ignores_top_n(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    monkeypatch.setattr(web_app, "query_remote_stock_name", lambda symbol, **kwargs: None)
    client = web_app.app.test_client()

    limited = client.get("/api/daily-rankings?asof=2026-09-18&top=10")
    by_code = client.get("/api/daily-rankings?asof=2026-09-18&query=000063&top=10")
    by_prefixed = client.get(
        "/api/daily-rankings?asof=2026-09-18&query=sz.000063&top=10"
    )

    assert limited.status_code == 200
    assert limited.get_json()["total"] == 2
    assert [row["code"] for row in limited.get_json()["rows"]] == [
        "sh.600000",
        "sz.000001",
    ]

    assert by_code.status_code == 200
    assert by_code.get_json()["total"] == 1
    assert by_code.get_json()["rows"][0]["code"] == "sz.000063"
    assert by_code.get_json()["rows"][0]["rank_d10"] == 11

    assert by_prefixed.status_code == 200
    assert by_prefixed.get_json()["total"] == 1
    assert by_prefixed.get_json()["rows"][0]["code"] == "sz.000063"


def test_daily_rankings_empty_query_still_applies_top_n(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    monkeypatch.setattr(web_app, "query_remote_stock_name", lambda symbol, **kwargs: None)

    response = web_app.app.test_client().get(
        "/api/daily-rankings?asof=2026-09-18&query=%20&top=10"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 2
    assert "sz.000063" not in [row["code"] for row in payload["rows"]]


def test_daily_rankings_page_search_skips_top_and_uses_mobile_cards():
    page = web_app.app.test_client().get("/").get_data(as_text=True)

    assert "top: query ? '' : $('top-select').value" in page
    assert "syncTopForSearch" in page
    assert "card-meta" in page
    assert "grid-template-areas:" in page
    assert "@media (max-width: 430px)" in page
    assert "历史行情 + 未来10日预测" in page
    assert "未来10日预测路径" not in page
    assert "drawChart(result)" in page
    assert "result.history" in page
    assert "当时收盘 ${price(closePrice)}" not in page
    assert "closeLabel" not in page
    assert 'stroke="#916515" stroke-width="1.3" stroke-dasharray="5 4"' in page
    assert 'x1="${left}"' in page
    assert "当时收盘价参考线" in page or "当时收盘" in page
    assert 'x1="${(hasHistory ? histRight : left)' not in page
    assert "Hermes 分析" not in page
    assert "hermes-analysis" not in page
    assert "analyzeWithHermes" not in page
    assert "resetHermesPanel" not in page
    assert "hermes-analyze" not in page
    assert "DeepSeek" not in page


def test_daily_rankings_cold_start_uses_disk_cache_without_baostock(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path, names={
        "sh.600000": "浦发银行",
        "sz.000001": "平安银行",
    })
    forbid_live_name_lookups(monkeypatch)
    web_app.preload_stock_name_cache()
    web_app.stock_name_cache.clear()
    web_app.stock_name_cache_loaded = False
    web_app.stock_name_cache_loaded_path = None
    web_app.stock_name_cache_mtime = None
    web_app.stock_name_cache_updated_at = None

    started = time.perf_counter()
    response = web_app.app.test_client().get(
        "/api/daily-rankings?asof=2026-09-18&top=10"
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 2
    rows = {row["code"]: row["name"] for row in response.get_json()["rows"]}
    assert rows["sh.600000"] == "浦发银行"
    assert rows["sz.000001"] == "平安银行"
    assert [row["code"] for row in response.get_json()["rows"]] == [
        "sh.600000",
        "sz.000001",
    ]


def test_daily_rankings_without_cache_still_returns_codes(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path)
    forbid_live_name_lookups(monkeypatch)

    started = time.perf_counter()
    response = web_app.app.test_client().get(
        "/api/daily-rankings?asof=2026-09-18&top=10"
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 2
    payload = response.get_json()
    assert payload["total"] == 2
    assert payload["rows"][0]["code"] == "sh.600000"
    assert payload["rows"][0]["name"] is None


def test_daily_rankings_uses_csv_name_column_without_remote(monkeypatch, tmp_path):
    published_run(tmp_path, names={"sh.600000": "浦发银行", "sz.000001": "平安银行"})
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path)
    forbid_live_name_lookups(monkeypatch)

    ranking = web_app.app.test_client().get(
        "/api/daily-rankings?asof=2026-09-18&top=10"
    )
    detail = web_app.app.test_client().get("/api/daily-rankings/2026-09-18/600000")

    assert ranking.status_code == 200
    assert ranking.get_json()["rows"][0]["name"] == "浦发银行"
    assert detail.status_code == 200
    assert detail.get_json()["name"] == "浦发银行"


def test_stock_name_refresh_writes_disk_and_survives_cold_memory(monkeypatch, tmp_path):
    cache_path = reset_stock_name_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(
        web_app,
        "fetch_baostock_stock_names",
        lambda: {"sh.600000": "浦发银行", "sz.000001": "平安银行"},
    )
    monkeypatch.setattr(web_app, "query_baostock_stock_name", lambda symbol: None)
    monkeypatch.setattr(web_app, "query_eastmoney_stock_name", lambda symbol: None)

    written = web_app.refresh_stock_name_cache()

    assert written["sh.600000"] == "浦发银行"
    assert cache_path.is_file()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["names"]["sz.000001"] == "平安银行"

    web_app.stock_name_cache.clear()
    web_app.stock_name_cache_loaded = False
    web_app.stock_name_cache_loaded_path = None
    web_app.stock_name_cache_mtime = None
    web_app.stock_name_cache_updated_at = None
    monkeypatch.setattr(
        web_app,
        "fetch_baostock_stock_names",
        lambda: (_ for _ in ()).throw(AssertionError("BaoStock should not rerun")),
    )

    assert web_app.query_remote_stock_name("sh.600000", allow_remote=False) == "浦发银行"
    web_app.load_stock_name_reference()
    assert web_app.stock_name_cache["sz.000001"] == "平安银行"


def test_fresh_disk_cache_skips_baostock_refresh(monkeypatch, tmp_path):
    reset_stock_name_cache(
        monkeypatch,
        tmp_path,
        names={"sh.600000": "浦发银行"},
        updated_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    forbid_live_name_lookups(monkeypatch)

    names = web_app.refresh_stock_name_cache()

    assert names["sh.600000"] == "浦发银行"


def test_daily_rankings_search_still_bypasses_top_n_with_name_cache(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path, names={"sz.000063": "中兴通讯"})
    forbid_live_name_lookups(monkeypatch)

    response = web_app.app.test_client().get(
        "/api/daily-rankings?asof=2026-09-18&query=000063&top=10"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1
    assert payload["rows"][0]["code"] == "sz.000063"
    assert payload["rows"][0]["name"] == "中兴通讯"
    assert payload["rows"][0]["rank_d10"] == 11


def test_daily_ranking_detail_uses_cached_history_through_asof(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    monkeypatch.setattr(web_app, "MARKET_DATA_CACHE_DIR", str(tmp_path / "market_data_cache"))
    reset_stock_name_cache(monkeypatch, tmp_path)
    forbid_live_name_lookups(monkeypatch)

    def explode(*args, **kwargs):
        raise AssertionError("live ranking history lookup should not run")

    monkeypatch.setattr(web_app, "_fetch_ranking_history_remote", explode)
    web_app._merge_market_data_cache("sz.000063", history_frame(close=30.0, extra_after=5))

    by_code = web_app.app.test_client().get("/api/daily-rankings/2026-09-18/000063")
    by_prefixed = web_app.app.test_client().get("/api/daily-rankings/2026-09-18/sz.000063")

    assert by_code.status_code == 200
    payload = by_code.get_json()
    assert payload["code"] == "sz.000063"
    assert payload["history_source"] == "market_data_cache"
    assert payload["history_note"] is None
    assert len(payload["history"]) == 90
    assert payload["history"][0]["timestamp"] < "2026-09-18"
    assert payload["history"][-1]["timestamp"] == "2026-09-18"
    assert payload["history"][-1]["close"] == pytest.approx(30.0)
    assert payload["history"][-1]["high"] >= payload["history"][-1]["close"]
    assert payload["predictions"][-1]["close_p50"] == pytest.approx(31.0)
    assert by_prefixed.get_json()["history"][-1]["timestamp"] == "2026-09-18"
    assert "2026-09-21" not in [row["timestamp"] for row in payload["history"]]


def test_daily_ranking_detail_uses_remote_history_when_cache_is_empty(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    monkeypatch.setattr(web_app, "MARKET_DATA_CACHE_DIR", str(tmp_path / "market_data_cache"))
    reset_stock_name_cache(monkeypatch, tmp_path)
    forbid_live_name_lookups(monkeypatch)
    remote = history_frame(close=10.0, rows=80)

    def fake_remote(symbol, asof, lookback=90):
        assert symbol == "sh.600000"
        return remote, "eastmoney"

    monkeypatch.setattr(web_app, "_fetch_ranking_history_remote", fake_remote)

    payload = web_app.app.test_client().get("/api/daily-rankings/2026-09-18/600000").get_json()

    assert payload["history_source"] == "eastmoney"
    assert payload["history_note"] is None
    assert len(payload["history"]) == 80
    assert payload["history"][-1]["timestamp"] == "2026-09-18"


def test_ranking_chart_history_clips_to_asof_and_skips_modal(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, "MARKET_DATA_CACHE_DIR", str(tmp_path / "market_data_cache"))
    called = {"remote": False}

    def fake_remote(*args, **kwargs):
        called["remote"] = True
        raise AssertionError("cache already covers asof")

    monkeypatch.setattr(web_app, "_fetch_ranking_history_remote", fake_remote)
    web_app._merge_market_data_cache("sh.600000", history_frame(rows=120, extra_after=8))

    payload = web_app.ranking_chart_history("sh.600000", "2026-09-18")

    assert called["remote"] is False
    assert payload["history_source"] == "market_data_cache"
    assert payload["history"][-1]["timestamp"] == "2026-09-18"
    assert len(payload["history"]) == 90
    assert all(row["timestamp"] <= "2026-09-18" for row in payload["history"])


def test_hermes_analysis_route_is_removed(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
    reset_stock_name_cache(monkeypatch, tmp_path)
    forbid_live_name_lookups(monkeypatch)
    client = web_app.app.test_client()

    missing = client.post("/api/daily-rankings/2026-09-18/000063/hermes-analysis")
    detail = client.get("/api/daily-rankings/2026-09-18/000063")

    assert missing.status_code == 404
    assert detail.status_code == 200
    payload = detail.get_json()
    assert payload["code"] == "sz.000063"
    assert "predictions" in payload
    assert "history" in payload
    assert "analysis" not in payload


def test_deploy_no_longer_wires_hermes_analysis():
    root = Path(web_app.PROJECT_ROOT)
    deploy = (root / "deploy" / "oracle-kronos" / "deploy.sh").read_text(encoding="utf-8")
    service = (root / "deploy" / "oracle-kronos" / "kronos-web.service").read_text(encoding="utf-8")
    readme = (root / "deploy" / "oracle-kronos" / "README.md").read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")

    assert 'cp "$PROJECT_DIR/webui/hermes_analysis.py"' not in deploy
    assert 'sudo install -o opc -g opc -m 0644 "$stage/webui/hermes_analysis.py"' not in deploy
    assert 'sudo rm -f "$root/webui/hermes_analysis.py"' in deploy
    assert "KRONOS_HERMES" not in deploy
    assert "KRONOS_HERMES" not in service
    assert "miniconda3/bin/hermes" not in service
    assert "Hermes 分析" not in readme
    assert "KRONOS_HERMES" not in readme
    assert "hermes-analysis" not in root_readme
    assert "Hermes 分析" not in root_readme

