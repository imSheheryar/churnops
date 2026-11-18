"""Serving tests: contracts, happy path, validation errors and observability."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churnops.config import ARTIFACTS_DIR
from churnops.monitoring import log_store
from churnops.serving.app import app

MODEL_READY = (ARTIFACTS_DIR / "model.joblib").exists()
pytestmark = pytest.mark.skipif(
    not MODEL_READY, reason="run `churnops pipeline` first to create artifacts/model.joblib"
)

CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.15,
    "TotalCharges": 425.75,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(log_store, "PREDICTIONS_DB_PATH", tmp_path / "predictions.db")
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["model_loaded"] is True


def test_model_endpoint_exposes_metadata(client):
    payload = client.get("/model").json()

    assert payload["model_name"] == "churnops-churn-classifier"
    assert payload["model_version"]
    assert 0 < payload["threshold"] < 1
    assert len(payload["features"]) == 19


def test_predict_returns_calibrated_probability(client):
    response = client.post("/predict", json=CUSTOMER)

    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["churn_probability"] <= 1.0
    assert payload["decision"] is True or payload["decision"] is False
    assert payload["request_id"]
    assert payload["model_version"]
    assert payload["latency_ms"] >= 0


def test_predict_rejects_out_of_range_values(client):
    bad = {**CUSTOMER, "tenure": -3}

    response = client.post("/predict", json=bad)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_rejects_missing_fields_without_nan(client):
    response = client.post("/predict", json={"gender": "Female"})

    assert response.status_code == 422
    body = response.text
    assert "NaN" not in body
    assert "tenure" in body


def test_batch_prediction_matches_request_size(client):
    response = client.post("/predict/batch", json={"customers": [CUSTOMER, CUSTOMER]})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 2
    assert payload["count"] == 2


def test_feedback_requires_known_request_id(client):
    created = client.post("/predict", json=CUSTOMER).json()

    ok = client.post("/feedback", json={"request_id": created["request_id"], "actual": 1})
    missing = client.post("/feedback", json={"request_id": "does-not-exist", "actual": 0})

    assert ok.status_code == 200
    assert ok.json()["status"] == "recorded"
    assert missing.status_code == 404


def test_metrics_expose_prometheus_series(client):
    client.post("/predict", json=CUSTOMER)

    body = client.get("/metrics").text

    assert "churnops_predictions_total" in body
    assert "churnops_prediction_latency_seconds" in body


def test_overview_feeds_the_dashboard(client):
    client.post("/predict", json=CUSTOMER)

    payload = client.get("/api/overview").json()

    assert payload["serving"]["requests"] >= 1
    assert payload["model"]["name"] == "churnops-churn-classifier"
    assert payload["drift"]["status"] in {"stable", "warning", "significant", "unknown"}
