"""
Turns a raw CustomerFeatures request into the exact 30-column encoded
feature vector the model was trained on, runs prediction, and computes
SHAP-based top drivers for the response.

This mirrors the training-time encoding in the retrained ChurnIQ notebook:
  1. Build a single-row DataFrame with the ORIGINAL dataset column names
  2. pd.get_dummies() on the categorical columns (drop_first=True)
  3. Reindex to the exact training feature_names, filling any missing
     dummy column with 0 (handles categories that don't appear in a
     single-row frame, e.g. only one InternetService value present)
  4. Scale the numeric columns with the fitted StandardScaler
  5. predict_proba + SHAP TreeExplainer for top feature drivers
"""

import logging

import numpy as np
import pandas as pd
import shap

from app.ml.model_loader import ModelArtifacts
from app.ml.schema import CustomerFeatures, TopDriver, PredictionResponse

logger = logging.getLogger(__name__)

# Maps CustomerFeatures (snake_case, API-facing) -> original dataset column names
_FIELD_TO_COLUMN = {
    "senior_citizen": "SeniorCitizen",
    "tenure": "tenure",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "gender": "gender",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "contract": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
}


def _raw_input_to_dataframe(features: CustomerFeatures) -> pd.DataFrame:
    """Builds a single-row DataFrame with original dataset column names/dtypes."""
    data = features.model_dump()
    row = {}
    for field, column in _FIELD_TO_COLUMN.items():
        value = data[field]
        if field == "senior_citizen":
            value = int(value)  # dataset encodes this as 0/1, not bool
        row[column] = value
    return pd.DataFrame([row])


def encode_features(features: CustomerFeatures, artifacts: ModelArtifacts) -> pd.DataFrame:
    """Raw customer input -> exact model-ready feature vector (1 row x 30 cols)."""
    raw_df = _raw_input_to_dataframe(features)

    encoded = pd.get_dummies(
        raw_df, columns=artifacts.categorical_columns, drop_first=True
    )

    # Reindex to the exact training columns; any dummy column not produced
    # by this single row (because that category wasn't the one present) is 0
    encoded = encoded.reindex(columns=artifacts.feature_names, fill_value=0)

    # Scale numeric columns using the fitted scaler (never re-fit at inference)
    encoded[artifacts.numeric_columns] = artifacts.scaler.transform(
        encoded[artifacts.numeric_columns]
    )

    # get_dummies() yields bool dtype for dummy columns; cast to float64 so the
    # frame is homogeneous (required by both the model and SHAP's C extension)
    encoded = encoded.astype("float64")

    return encoded


def predict(features: CustomerFeatures, artifacts: ModelArtifacts) -> PredictionResponse:
    encoded = encode_features(features, artifacts)

    proba = artifacts.model.predict_proba(encoded)[0]
    churn_probability = float(proba[1])
    threshold = float(artifacts.metadata.get("prediction_threshold", 0.4))
    churn_prediction = "Yes" if churn_probability >= threshold else "No"

    top_drivers = _compute_top_drivers(encoded, artifacts)

    return PredictionResponse(
        churn_probability=round(churn_probability, 4),
        churn_prediction=churn_prediction,
        top_drivers=top_drivers,
        model_version=str(artifacts.metadata.get("model_version", "1.0.0")),
    )


def _compute_top_drivers(
    encoded_row: pd.DataFrame, artifacts: ModelArtifacts, top_n: int = 5
) -> list[TopDriver]:
    explainer = shap.TreeExplainer(artifacts.model, artifacts.shap_background)
    shap_values = explainer.shap_values(encoded_row)

    # GradientBoostingClassifier binary case: shap_values is a single 2D array
    # (n_samples, n_features) representing the positive class
    values = shap_values[0] if shap_values.ndim == 2 else shap_values[0][:, 1]

    feature_impact = list(zip(artifacts.feature_names, values))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    drivers = []
    for feature, value in feature_impact[:top_n]:
        drivers.append(
            TopDriver(
                feature=feature,
                shap_value=round(float(value), 4),
                direction="increases_churn_risk" if value > 0 else "decreases_churn_risk",
            )
        )
    return drivers
