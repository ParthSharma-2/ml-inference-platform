from .test_prediction import valid_customer_payload


def test_prediction_history(client):
    response = client.get("/api/v1/predictions")

    assert response.status_code == 200

    data = response.json()

    assert "items" in data
    assert "total" in data

    assert isinstance(data["items"], list)
    assert isinstance(data["total"], int)

    for item in data["items"]:
        assert "id" in item
        assert "churn_probability" in item
        assert "churn_prediction" in item
        assert "model_version" in item
        assert "created_at" in item


def test_prediction_history_item_types(client):
    response = client.get("/api/v1/predictions?skip=0&limit=5")

    assert response.status_code == 200

    data = response.json()

    for item in data["items"]:
        assert isinstance(item["id"], int)
        assert isinstance(item["churn_probability"], (int, float))
        assert item["churn_prediction"] in {"Yes", "No"}
        assert isinstance(item["model_version"], str)
        assert isinstance(item["created_at"], str)


def test_prediction_history_pagination(client):
    response = client.get("/api/v1/predictions?skip=0&limit=1")

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) <= 1


def test_prediction_history_second_page(client):
    response = client.get("/api/v1/predictions?skip=1&limit=1")

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) <= 1


def test_prediction_history_invalid_limit(client):
    response = client.get("/api/v1/predictions?limit=101")

    assert response.status_code == 422


def test_prediction_history_invalid_skip(client):
    response = client.get("/api/v1/predictions?skip=-1")

    assert response.status_code == 422


def test_prediction_not_found(client):
    response = client.get("/api/v1/predictions/999999")

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Prediction not found"


def test_prediction_persistence_and_detail(client):
    # Create a new prediction.
    create_response = client.post(
        "/api/v1/predict",
        json=valid_customer_payload(),
    )

    assert create_response.status_code == 200

    prediction = create_response.json()

    # The API currently returns the prediction result,
    # while the history endpoint provides the database ID.
    history_response = client.get(
        "/api/v1/predictions?skip=0&limit=1"
    )

    assert history_response.status_code == 200

    history = history_response.json()

    assert history["total"] >= 1
    assert len(history["items"]) >= 1

    latest = history["items"][0]

    assert latest["churn_probability"] == prediction["churn_probability"]
    assert latest["churn_prediction"] == prediction["churn_prediction"]
    assert latest["model_version"] == prediction["model_version"]

    # Verify that the persisted record can be retrieved individually.
    detail_response = client.get(
        f"/api/v1/predictions/{latest['id']}"
    )

    assert detail_response.status_code == 200

    detail = detail_response.json()

    assert detail["id"] == latest["id"]
    assert detail["churn_probability"] == latest["churn_probability"]
    assert detail["churn_prediction"] == latest["churn_prediction"]
    assert detail["model_version"] == latest["model_version"]