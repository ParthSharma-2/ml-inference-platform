import logging

from sqlalchemy.exc import DBAPIError, OperationalError

from app.db.database import SessionLocal
from app.db.models import PredictionLog
from app.ml.inference import predict as run_prediction
from app.ml.model_loader import get_artifacts
from app.ml.schema import CustomerFeatures
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task
def test_task(value: int) -> int:
    return value * 2


@celery_app.task(
    bind=True,
    max_retries=3,
    soft_time_limit=60,
    time_limit=90,
)
def process_batch_predictions(self, customers: list[dict]) -> dict:
    """
    Process a batch of customer predictions asynchronously.

    Customer-level validation/inference failures are recorded and
    do not retry the entire batch.

    Database connectivity failures are treated as transient and
    retried with exponential backoff, up to 3 retries.
    """

    artifacts = get_artifacts()
    db = SessionLocal()

    completed = 0
    failed = 0
    prediction_ids = []

    try:
        for customer_data in customers:
            try:
                features = CustomerFeatures.model_validate(customer_data)
                result = run_prediction(features, artifacts)

            except Exception:
                failed += 1
                logger.exception(
                    "Batch prediction failed for one customer"
                )
                continue

            try:
                prediction_log = PredictionLog(
                    senior_citizen=features.senior_citizen,
                    tenure=features.tenure,
                    monthly_charges=features.monthly_charges,
                    total_charges=features.total_charges,
                    gender=features.gender,
                    partner=features.partner,
                    dependents=features.dependents,
                    phone_service=features.phone_service,
                    multiple_lines=features.multiple_lines,
                    internet_service=features.internet_service,
                    online_security=features.online_security,
                    online_backup=features.online_backup,
                    device_protection=features.device_protection,
                    tech_support=features.tech_support,
                    streaming_tv=features.streaming_tv,
                    streaming_movies=features.streaming_movies,
                    contract=features.contract,
                    paperless_billing=features.paperless_billing,
                    payment_method=features.payment_method,
                    churn_probability=result.churn_probability,
                    churn_prediction=result.churn_prediction,
                    top_drivers=[
                        driver.model_dump()
                        for driver in result.top_drivers
                    ],
                    model_version=result.model_version,
                )

                db.add(prediction_log)
                db.flush()

                prediction_ids.append(prediction_log.id)
                completed += 1

            except (OperationalError, DBAPIError) as exc:
                db.rollback()

                retry_countdown = 2 ** self.request.retries

                logger.warning(
                    "Transient database failure for batch job %s. "
                    "Retry %s/%s in %s seconds.",
                    self.request.id,
                    self.request.retries + 1,
                    self.max_retries,
                    retry_countdown,
                )

                raise self.retry(
                    exc=exc,
                    countdown=retry_countdown,
                )

            except Exception:
                db.rollback()

                logger.exception(
                    "Database persistence failed for one customer"
                )

                raise

        db.commit()

        return {
            "job_id": self.request.id,
            "status": "completed",
            "total": len(customers),
            "completed": completed,
            "failed": failed,
            "prediction_ids": prediction_ids,
        }

    except (OperationalError, DBAPIError) as exc:
        db.rollback()

        retry_countdown = 2 ** self.request.retries

        logger.warning(
            "Transient database failure for batch job %s. "
            "Retry %s/%s in %s seconds.",
            self.request.id,
            self.request.retries + 1,
            self.max_retries,
            retry_countdown,
        )

        raise self.retry(
            exc=exc,
            countdown=retry_countdown,
        )

    except Exception:
        db.rollback()

        logger.exception(
            "Batch prediction job failed: %s",
            self.request.id,
        )

        raise

    finally:
        db.close()