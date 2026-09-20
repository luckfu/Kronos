import json

import pandas as pd

from webui import app as web_app


def published_run(tmp_path):
    run = tmp_path / "2026-09-18" / "full_market"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "status": "complete",
        "prediction_count": 2,
        "batch_count": 1,
        "model": {"sample_count": 5, "model_release": "small-0.1-cosine-c2"},
    }))
    pd.DataFrame([
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
    ]).to_csv(run / "ranking.csv", index=False)
    detail = {
        "asof": "2026-09-18", "code": "sh.600000", "sector_id": 63,
        "sector_label": "J66货币金融服务", "size_percentile": 0.9,
        "close_asof": 10.0,
        "predictions": [{"timestamp": "2026-09-21", "close_p10": 9.8, "close_p50": 10.1, "close_p90": 10.4}],
        "samples": {"close": [[10.1], [10.2], [10.0], [9.9], [10.3]]},
    }
    (run / "predictions.jsonl").write_text(json.dumps(detail) + "\n")
    return run


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
    client = web_app.app.test_client()

    ranking = client.get("/api/daily-rankings?asof=2026-09-18&query=600000&size_group=large")
    detail = client.get("/api/daily-rankings/2026-09-18/600000")

    assert ranking.status_code == 200
    assert ranking.get_json()["total"] == 1
    assert ranking.get_json()["rows"][0]["code"] == "sh.600000"
    assert detail.status_code == 200
    assert detail.get_json()["model"]["sample_count"] == 5
    assert detail.get_json()["ranking"]["rank_d10"] == 1


def test_daily_rankings_re_ranks_selected_pool(monkeypatch, tmp_path):
    published_run(tmp_path)
    monkeypatch.setattr(web_app, "DAILY_PREDICTION_ROOT", tmp_path)
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
