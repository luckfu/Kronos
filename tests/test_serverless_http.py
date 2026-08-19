from api.index import app


def test_health_does_not_load_model():
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "service": "kronos-inference",
    }


def test_predict_rejects_symbol_only_request():
    response = app.test_client().post("/predict", json={"symbol": "600519"})

    assert response.status_code == 400
    assert "data must be" in response.get_json()["error"]
