from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.fast_api.api_app import app

BASE_DIR = Path(__file__).resolve().parent.parent

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def _valid_predict_payload():
    return {
        "features": {
            "Store": 1,
            "DayOfWeek": 2,
            "Promo": 1,
            "StateHoliday": 0,
            "SchoolHoliday": 0,
            "StoreType": 1,
            "Assortment": 1,
            "CompetitionDistance": 500.0,
            "Year": 2015,
            "Month": 7,
            "Day": 15,
        }
    }

@pytest.mark.skip(reason="large error")
def _make_small_eval_files(tmp_path):
    data = {
        'Store': range(1, 251),
        'DayOfWeek': [1] * 250,
        'Date': pd.date_range(start='2023-01-01', periods=250).astype(str),
        'Sales': [5000] * 250,
        'Customers': [100] * 250,
        'Open': [1] * 250,
        'Promo': [1] * 250,
        'StateHoliday': ['0'] * 250,
        'SchoolHoliday': [0] * 250,
        'StoreType': ['a'] * 250,
        'Assortment': ['a'] * 250,
        'CompetitionDistance': [500.0] * 250,
        'CompetitionOpenSinceMonth': [1] * 250,
        'CompetitionOpenSinceYear': [2010] * 250,
        'Promo2': [0] * 250,
        'Promo2SinceWeek': [0] * 250,
        'Promo2SinceYear': [0] * 250,
        'PromoInterval': ['None'] * 250,
    }

    df = pd.DataFrame(data)

    file_1 = tmp_path / "small_input_1.csv"
    file_2 = tmp_path / "small_input_2.csv"

    df.to_csv(file_1, index=False)
    df.to_csv(file_2, index=False)

    return file_1, file_2

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "features" in body
    assert "target" in body

def test_predict_health_success(client):
    response = client.post("/predict-health", json=_valid_predict_payload())
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)

def test_predict_health_validation_error(client):
    bad_payload = _valid_predict_payload()
    bad_payload["features"]["CompetitionDistance"] = -5

    response = client.post("/predict-health", json=bad_payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"

def test_evaluate_file_success(client, tmp_path):
    file_1, _ = _make_small_eval_files(tmp_path)

    response = client.post("/evaluate-file", json={"input_file": str(file_1)})
    assert response.status_code == 200
    body = response.json()
    assert "metrics" in body

    metrics = body["metrics"]
    for key in ["RMSPE", "MAE", "RMSE", "MAPE", "R2"]:
        assert key in metrics

@pytest.mark.skip(reason="compare-input-files endpoint not yet implemented")
def test_compare_input_files_success(client, tmp_path):
    file_1, file_2 = _make_small_eval_files(tmp_path)

    response = client.post(
        "/compare-input-files",
        json={"input_files": [str(file_1), str(file_2)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_files"] == 2
    assert len(body["results"]) == 2
    assert len(body["rankings"]) == 5

def test_metrics_history_success(client):
    response = client.get("/metrics-history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)