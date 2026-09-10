from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "senior_citizen": False,
    "tenure": 12,
    "monthly_charges": 70.35,
    "total_charges": 845.50,
    "gender": "Female",
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}


def test_health_check():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] == 30


def test_predict_valid_input_returns_prediction():
    r = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in ("Yes", "No")
    assert len(body["top_drivers"]) == 5


def test_predict_rejects_invalid_category():
    bad_payload = dict(VALID_PAYLOAD, contract="Three year")
    r = client.post("/api/v1/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_rejects_missing_field():
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["tenure"]
    r = client.post("/api/v1/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_rejects_negative_tenure():
    bad_payload = dict(VALID_PAYLOAD, tenure=-5)
    r = client.post("/api/v1/predict", json=bad_payload)
    assert r.status_code == 422


def test_high_risk_profile_scores_higher_than_low_risk():
    """Sanity check that the model's directionality makes sense end-to-end,
    not just that the endpoint returns a 200."""
    high_risk = VALID_PAYLOAD
    low_risk = dict(
        VALID_PAYLOAD,
        contract="Two year",
        tenure=60,
        internet_service="DSL",
    )
    r_high = client.post("/api/v1/predict", json=high_risk).json()
    r_low = client.post("/api/v1/predict", json=low_risk).json()
    assert r_high["churn_probability"] > r_low["churn_probability"]
