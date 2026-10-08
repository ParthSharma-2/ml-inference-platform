"""Reproducible ChurnIQ training/export script.

Requires the versions pinned in requirements.txt. By default it reads:
    data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Run from the project root:
    python scripts/retrain_model.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ART = ROOT / "app" / "ml" / "artifacts"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = "customer-churn-training"
MLFLOW_MODEL_NAME = "customer-churn-model"


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")


print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("Joblib:", joblib.__version__)
print("MLflow tracking URI:", MLFLOW_TRACKING_URI)


# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ---------------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"], errors="ignore")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce",
).fillna(0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

numeric_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_cols = [
    c for c in df.columns
    if c not in numeric_cols + ["Churn"]
]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[numeric_cols] = scaler.fit_transform(
    X_train[numeric_cols]
)

X_test[numeric_cols] = scaler.transform(
    X_test[numeric_cols]
)


# ---------------------------------------------------------------------------
# MLflow run
# ---------------------------------------------------------------------------

with mlflow.start_run() as run:

    # -----------------------------------------------------------------------
    # Train model
    # -----------------------------------------------------------------------

    model = GradientBoostingClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    threshold = 0.4

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(f"ROC-AUC: {roc_auc:.4f}")

    report = classification_report(
        y_test,
        predictions,
        digits=4,
    )

    print(report)


    # -----------------------------------------------------------------------
    # Log parameters
    # -----------------------------------------------------------------------

    mlflow.log_params({
        "algorithm": "GradientBoostingClassifier",
        "random_state": 42,
        "test_size": 0.2,
        "prediction_threshold": threshold,
        "n_features": X_train.shape[1],
        "n_training_samples": X_train.shape[0],
    })


    # -----------------------------------------------------------------------
    # Log metrics
    # -----------------------------------------------------------------------

    mlflow.log_metric(
        "roc_auc",
        float(roc_auc),
    )


    # -----------------------------------------------------------------------
    # Log useful environment information as tags
    # -----------------------------------------------------------------------

    mlflow.set_tags({
        "model_type": "GradientBoostingClassifier",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "serving_model_version": "1.0.0",
    })


    # -----------------------------------------------------------------------
    # Export existing serving artifacts
    # -----------------------------------------------------------------------

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        ART / "churn_model.joblib",
    )

    joblib.dump(
        scaler,
        ART / "preprocessor.joblib",
    )

    feature_names = list(X_train.columns)

    (
        ART / "feature_names.json"
    ).write_text(
        json.dumps(
            feature_names,
            indent=2,
        )
    )

    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42,
    )

    joblib.dump(
        background,
        ART / "shap_background.joblib",
    )


    metadata = {
        "model_type": "GradientBoostingClassifier",
        "roc_auc": float(roc_auc),
        "n_features": len(feature_names),
        "n_training_samples": len(X_train),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "prediction_threshold": threshold,
        "training_random_state": 42,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }

    (
        ART / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


    # -----------------------------------------------------------------------
    # Log supporting artifacts to MLflow
    # -----------------------------------------------------------------------

    mlflow.log_artifact(
        str(ART / "churn_model.joblib"),
        artifact_path="serving_artifacts",
    )

    mlflow.log_artifact(
        str(ART / "preprocessor.joblib"),
        artifact_path="serving_artifacts",
    )

    mlflow.log_artifact(
        str(ART / "feature_names.json"),
        artifact_path="serving_artifacts",
    )

    mlflow.log_artifact(
        str(ART / "metadata.json"),
        artifact_path="serving_artifacts",
    )

    mlflow.log_artifact(
        str(ART / "shap_background.joblib"),
        artifact_path="serving_artifacts",
    )


    # -----------------------------------------------------------------------
    # Log and register model with MLflow
    # -----------------------------------------------------------------------

    signature = infer_signature(
        X_test,
        model.predict_proba(X_test)[:, 1],
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        name="churn_model",
        signature=signature,
        registered_model_name=MLFLOW_MODEL_NAME,
        skops_trusted_types=[
        "sklearn.tree._tree.Tree",
        ],
    )


    # -----------------------------------------------------------------------
    # Immediate round-trip check
    # -----------------------------------------------------------------------

    loaded = joblib.load(
        ART / "churn_model.joblib"
    )

    assert loaded.n_features_in_ == len(
        feature_names
    )

    print("Model round-trip load: SUCCESS")

    print(
        f"Exported {len(feature_names)} features "
        f"to {ART}"
    )

    print("MLflow run ID:", run.info.run_id)

    print(
        "MLflow registered model:",
        MLFLOW_MODEL_NAME,
    )