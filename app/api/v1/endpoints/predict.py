import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import PredictionLog
from app.ml.model_loader import get_artifacts, ModelArtifacts
from app.ml.schema import CustomerFeatures, PredictionResponse
from app.ml.inference import predict as run_prediction

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    features: CustomerFeatures,
    artifacts: ModelArtifacts = Depends(get_artifacts),
    db: Session = Depends(get_db),
):
    """
    Single-customer churn prediction.

    Takes raw customer attributes, runs the ML model,
    generates SHAP explanations, stores the prediction
    in PostgreSQL, and returns the prediction response.
    """

    try:
        # 1. Run ML prediction
        result = run_prediction(features, artifacts)

        # 2. Create database record
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
            top_drivers=[driver.model_dump() for driver in result.top_drivers],
            model_version=result.model_version,
        )

        # 3. Save to PostgreSQL
        db.add(prediction_log)
        db.commit()
        db.refresh(prediction_log)

        # 4. Return the original API response
        return result

    except Exception:
        db.rollback()
        logger.exception(
            "Prediction failed for input: %s",
            features,
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        )