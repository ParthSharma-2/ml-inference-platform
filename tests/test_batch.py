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

def test_batch_customer_failure_does_not_retry():
    from unittest.mock import Mock, patch

    from app.workers.tasks import process_batch_predictions

    fake_db = Mock()

    with patch(
        "app.workers.tasks.get_artifacts",
        return_value=Mock(),
    ), patch(
        "app.workers.tasks.SessionLocal",
        return_value=fake_db,
    ), patch.object(
        process_batch_predictions,
        "retry",
    ) as mock_retry:

        result = process_batch_predictions.run(
            [
                {
                    "senior_citizen": False,
                    "tenure": 12,
                    "monthly_charges": 70.35,
                    "total_charges": 845.50,
                    "gender": "InvalidGender",
                }
            ]
        )

    assert result["status"] == "completed"
    assert result["total"] == 1
    assert result["completed"] == 0
    assert result["failed"] == 1

    mock_retry.assert_not_called()
    fake_db.commit.assert_called_once()


def test_batch_database_failure_triggers_retry():
    from unittest.mock import Mock, patch

    from sqlalchemy.exc import OperationalError

    from app.workers.tasks import process_batch_predictions

    fake_db = Mock()
    fake_db.flush.side_effect = OperationalError(
        "INSERT INTO prediction_logs",
        {},
        Exception("temporary database connection failure"),
    )

    fake_features = Mock()
    fake_features.senior_citizen = False
    fake_features.tenure = 12
    fake_features.monthly_charges = 70.35
    fake_features.total_charges = 845.50
    fake_features.gender = "Female"
    fake_features.partner = "Yes"
    fake_features.dependents = "No"
    fake_features.phone_service = "Yes"
    fake_features.multiple_lines = "No"
    fake_features.internet_service = "Fiber optic"
    fake_features.online_security = "No"
    fake_features.online_backup = "Yes"
    fake_features.device_protection = "No"
    fake_features.tech_support = "No"
    fake_features.streaming_tv = "Yes"
    fake_features.streaming_movies = "No"
    fake_features.contract = "Month-to-month"
    fake_features.paperless_billing = "Yes"
    fake_features.payment_method = "Electronic check"

    fake_result = Mock()
    fake_result.churn_probability = 0.25
    fake_result.churn_prediction = "No"
    fake_result.top_drivers = []
    fake_result.model_version = "1.0.0"

    retry_exception = RuntimeError("celery retry")

    with patch(
        "app.workers.tasks.get_artifacts",
        return_value=Mock(),
    ), patch(
        "app.workers.tasks.SessionLocal",
        return_value=fake_db,
    ), patch(
        "app.workers.tasks.CustomerFeatures.model_validate",
        return_value=fake_features,
    ), patch(
        "app.workers.tasks.run_prediction",
        return_value=fake_result,
    ), patch(
        "app.workers.tasks.PredictionLog",
        return_value=Mock(),
    ), patch.object(
        process_batch_predictions,
        "retry",
        side_effect=retry_exception,
    ) as mock_retry:

        try:
            process_batch_predictions.run([{}])
        except RuntimeError as exc:
            assert exc is retry_exception
        else:
            raise AssertionError("Expected Celery retry to be triggered")

    fake_db.rollback.assert_called()
    mock_retry.assert_called_once()

    retry_kwargs = mock_retry.call_args.kwargs

    assert retry_kwargs["countdown"] == 1
    assert isinstance(
        retry_kwargs["exc"],
        OperationalError,
    )


def test_batch_retry_backoff_increases():
    from unittest.mock import Mock, patch

    from sqlalchemy.exc import OperationalError

    from app.workers.tasks import process_batch_predictions

    fake_db = Mock()
    fake_db.flush.side_effect = OperationalError(
        "INSERT INTO prediction_logs",
        {},
        Exception("temporary database failure"),
    )

    fake_features = Mock()
    fake_features.senior_citizen = False
    fake_features.tenure = 12
    fake_features.monthly_charges = 70.35
    fake_features.total_charges = 845.50
    fake_features.gender = "Female"
    fake_features.partner = "Yes"
    fake_features.dependents = "No"
    fake_features.phone_service = "Yes"
    fake_features.multiple_lines = "No"
    fake_features.internet_service = "Fiber optic"
    fake_features.online_security = "No"
    fake_features.online_backup = "Yes"
    fake_features.device_protection = "No"
    fake_features.tech_support = "No"
    fake_features.streaming_tv = "Yes"
    fake_features.streaming_movies = "No"
    fake_features.contract = "Month-to-month"
    fake_features.paperless_billing = "Yes"
    fake_features.payment_method = "Electronic check"

    fake_result = Mock()
    fake_result.churn_probability = 0.25
    fake_result.churn_prediction = "No"
    fake_result.top_drivers = []
    fake_result.model_version = "1.0.0"

    retry_exception = RuntimeError("celery retry")

    with patch(
        "app.workers.tasks.get_artifacts",
        return_value=Mock(),
    ), patch(
        "app.workers.tasks.SessionLocal",
        return_value=fake_db,
    ), patch(
        "app.workers.tasks.CustomerFeatures.model_validate",
        return_value=fake_features,
    ), patch(
        "app.workers.tasks.run_prediction",
        return_value=fake_result,
    ), patch(
        "app.workers.tasks.PredictionLog",
        return_value=Mock(),
    ), patch.object(
        process_batch_predictions,
        "retry",
        side_effect=retry_exception,
    ) as mock_retry:

        process_batch_predictions.request.retries = 2

        try:
            process_batch_predictions.run([{}])
        except RuntimeError:
            pass

    retry_kwargs = mock_retry.call_args.kwargs

    assert retry_kwargs["countdown"] == 4