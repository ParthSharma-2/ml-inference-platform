def valid_customer_payload():
    return {
        "senior_citizen": False,
        "tenure": 12,
        "monthly_charges": 70.35,
        "total_charges": 845.5,
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


def test_health_check(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_type"] == "GradientBoostingClassifier"
    assert isinstance(data["model_roc_auc"], float)
    assert data["model_roc_auc"] > 0
    assert data["n_features"] == 30


def test_prediction_validation_invalid_enum(client):
    payload = valid_customer_payload()
    payload["gender"] = "InvalidGender"

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_prediction_missing_required_field(client):
    payload = valid_customer_payload()
    del payload["total_charges"]

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_prediction_response_structure(client):
    response = client.post(
        "/api/v1/predict",
        json=valid_customer_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    required_fields = {
        "churn_probability",
        "churn_prediction",
        "top_drivers",
        "model_version",
    }

    assert required_fields.issubset(data.keys())


def test_prediction_response_types(client):
    response = client.post(
        "/api/v1/predict",
        json=valid_customer_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["churn_probability"], (int, float))
    assert isinstance(data["churn_prediction"], str)
    assert isinstance(data["top_drivers"], list)
    assert isinstance(data["model_version"], str)


def test_prediction_probability_range(client):
    response = client.post(
        "/api/v1/predict",
        json=valid_customer_payload(),
    )

    assert response.status_code == 200

    probability = response.json()["churn_probability"]

    assert 0 <= probability <= 1


def test_prediction_value(client):
    response = client.post(
        "/api/v1/predict",
        json=valid_customer_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["churn_prediction"] in {"Yes", "No"}
    assert data["model_version"] == "1.0.0"


def test_top_drivers_contract(client):
    response = client.post(
        "/api/v1/predict",
        json=valid_customer_payload(),
    )

    assert response.status_code == 200

    drivers = response.json()["top_drivers"]

    assert len(drivers) <= 5

    for driver in drivers:
        assert "feature" in driver
        assert "shap_value" in driver
        assert "direction" in driver

        assert isinstance(driver["feature"], str)
        assert isinstance(driver["shap_value"], (int, float))

        assert driver["direction"] in {
            "increases_churn_risk",
            "decreases_churn_risk",
        }