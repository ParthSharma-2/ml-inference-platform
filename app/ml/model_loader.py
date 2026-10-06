"""
Loads the trained model, scaler, feature schema, and SHAP background
exactly once at application startup.

The model is loaded from MLflow Model Registry.
Supporting inference artifacts remain in app/ml/artifacts.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/customer-churn-model/1",
)


class ModelArtifacts:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.prediction_threshold: float = 0.4
        self.shap_background = None
        self.metadata: dict = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        logger.info(
            "Loading model from MLflow: %s",
            MLFLOW_MODEL_URI,
        )

        logger.info(
            "MLflow tracking URI: %s",
            MLFLOW_TRACKING_URI,
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # ------------------------------------------------------------------
        # Supporting inference artifacts
        # ------------------------------------------------------------------

        scaler_path = ARTIFACTS_DIR / "preprocessor.joblib"
        features_path = ARTIFACTS_DIR / "feature_names.json"
        shap_bg_path = ARTIFACTS_DIR / "shap_background.joblib"
        metadata_path = ARTIFACTS_DIR / "metadata.json"

        for p in [
            scaler_path,
            features_path,
            shap_bg_path,
            metadata_path,
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required inference artifact missing: {p}"
                )

        # ------------------------------------------------------------------
        # Load model from MLflow Model Registry
        # ------------------------------------------------------------------

        self.model = mlflow.sklearn.load_model(
            MLFLOW_MODEL_URI
        )

        # ------------------------------------------------------------------
        # Load supporting artifacts locally
        # ------------------------------------------------------------------

        self.scaler = joblib.load(scaler_path)

        self.shap_background = joblib.load(
            shap_bg_path
        )

        # pandas get_dummies() produces bool dtype columns;
        # SHAP's TreeExplainer requires a homogeneous numeric dtype.
        self.shap_background = (
            self.shap_background.astype("float64")
        )

        with open(features_path) as f:
            self.feature_names = json.load(f)

        with open(metadata_path) as f:
            self.metadata = json.load(f)

            self.numeric_columns = self.metadata.get(
                "numeric_columns",
                [],
            )

            self.categorical_columns = self.metadata.get(
                "categorical_columns",
                [],
            )

            self.prediction_threshold = float(
                self.metadata.get(
                    "prediction_threshold",
                    0.4,
                )
            )

        # ------------------------------------------------------------------
        # Sanity checks
        # ------------------------------------------------------------------

        if self.model.n_features_in_ != len(
            self.feature_names
        ):
            raise ValueError(
                f"Model expects {self.model.n_features_in_} "
                f"features but feature_names.json has "
                f"{len(self.feature_names)}. "
                "Artifacts are out of sync."
            )

        expected_scaler_cols = set(
            self.scaler.feature_names_in_
        )

        if expected_scaler_cols != set(
            self.numeric_columns
        ):
            raise ValueError(
                f"Scaler was fit on {expected_scaler_cols} "
                f"but metadata.json numeric_columns is "
                f"{self.numeric_columns}. "
                "Artifacts are out of sync."
            )

        self._loaded = True

        logger.info(
            "MLflow model loaded successfully: %s",
            MLFLOW_MODEL_URI,
        )

        logger.info(
            "Model: %s | Features: %d | ROC-AUC: %.4f",
            type(self.model).__name__,
            len(self.feature_names),
            self.metadata.get(
                "roc_auc",
                float("nan"),
            ),
        )


@lru_cache
def get_artifacts() -> ModelArtifacts:
    """FastAPI dependency — returns the singleton."""
    artifacts = ModelArtifacts()
    artifacts.load()
    return artifacts