from types import SimpleNamespace
from unittest.mock import patch

from .test_prediction import valid_customer_payload


def test_batch_prediction_validation(client):
    payload = {
        "customers": [
            valid_customer_payload(),
            valid_customer_payload(),
        ]
    }

    response = client.post(
        "/api/v1/predictions/batch",
        json=payload,
    )

    assert response.status_code == 202

    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert isinstance(data["job_id"], str)


def test_batch_prediction_empty_customers(client):
    response = client.post(
        "/api/v1/predictions/batch",
        json={"customers": []},
    )

    assert response.status_code == 422


def test_batch_prediction_missing_customers(client):
    response = client.post(
        "/api/v1/predictions/batch",
        json={},
    )

    assert response.status_code == 422


def test_batch_prediction_invalid_customer(client):
    payload = {
        "customers": [
            {
                **valid_customer_payload(),
                "gender": "InvalidGender",
            }
        ]
    }

    response = client.post(
        "/api/v1/predictions/batch",
        json=payload,
    )

    assert response.status_code == 422


def test_batch_prediction_maximum_size(client):
    payload = {
        "customers": [
            valid_customer_payload()
            for _ in range(101)
        ]
    }

    response = client.post(
        "/api/v1/predictions/batch",
        json=payload,
    )

    assert response.status_code == 422


def test_batch_prediction_queues_celery_task(client):
    fake_task = SimpleNamespace(
        id="test-job-id-123"
    )

    payload = {
        "customers": [
            valid_customer_payload(),
        ]
    }

    with patch(
        "app.api.v1.endpoints.predict.process_batch_predictions.delay",
        return_value=fake_task,
    ) as mock_delay:

        response = client.post(
            "/api/v1/predictions/batch",
            json=payload,
        )

    assert response.status_code == 202

    data = response.json()

    assert data["job_id"] == "test-job-id-123"
    assert data["status"] == "queued"

    mock_delay.assert_called_once()

    queued_customers = mock_delay.call_args.args[0]

    assert len(queued_customers) == 1
    assert queued_customers[0]["tenure"] == 12
    assert queued_customers[0]["gender"] == "Female"


def test_batch_prediction_status_success(client):
    fake_result = SimpleNamespace(
        state="SUCCESS",
        result={
            "job_id": "test-job-id-123",
            "status": "completed",
            "total": 2,
            "completed": 2,
            "failed": 0,
            "prediction_ids": [100, 101],
        },
        ready=lambda: True,
    )

    with patch(
        "app.api.v1.endpoints.predict.AsyncResult",
        return_value=fake_result,
    ) as mock_async_result:

        response = client.get(
            "/api/v1/predictions/batch/test-job-id-123"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["job_id"] == "test-job-id-123"
    assert data["status"] == "SUCCESS"

    assert data["result"]["status"] == "completed"
    assert data["result"]["total"] == 2
    assert data["result"]["completed"] == 2
    assert data["result"]["failed"] == 0
    assert data["result"]["prediction_ids"] == [100, 101]

    mock_async_result.assert_called_once_with(
        "test-job-id-123",
        app=mock_async_result.call_args.kwargs["app"],
    )


def test_batch_prediction_status_pending(client):
    fake_result = SimpleNamespace(
        state="PENDING",
        result=None,
        ready=lambda: False,
    )

    with patch(
        "app.api.v1.endpoints.predict.AsyncResult",
        return_value=fake_result,
    ):

        response = client.get(
            "/api/v1/predictions/batch/unknown-job-id"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["job_id"] == "unknown-job-id"
    assert data["status"] == "PENDING"
    assert data["result"] is None