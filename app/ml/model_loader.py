"""
Loads the trained model, scaler, feature schema, and SHAP background sample
exactly once at application startup, and exposes them as module-level
singletons. Avoids re-reading joblib files from disk on every request.
"""

import json
import logging
from pathlib import Path
from functools import lru_cache

import joblib

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


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

        logger.info("Loading model artifacts from %s", ARTIFACTS_DIR)

        model_path = ARTIFACTS_DIR / "churn_model.joblib"
        scaler_path = ARTIFACTS_DIR / "preprocessor.joblib"
        features_path = ARTIFACTS_DIR / "feature_names.json"
        shap_bg_path = ARTIFACTS_DIR / "shap_background.joblib"
        metadata_path = ARTIFACTS_DIR / "metadata.json"

        for p in [model_path, scaler_path, features_path, shap_bg_path, metadata_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required model artifact missing: {p}. "
                    "Ensure the exported ChurnIQ artifacts are placed in app/ml/artifacts/."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.shap_background = joblib.load(shap_bg_path)
        # pandas get_dummies() produces bool dtype columns; SHAP's TreeExplainer
        # C extension requires a single homogeneous numeric dtype
        self.shap_background = self.shap_background.astype("float64")

        with open(features_path) as f:
            self.feature_names = json.load(f)

        with open(metadata_path) as f:
            self.metadata = json.load(f)
            self.numeric_columns = self.metadata.get("numeric_columns", [])
            self.categorical_columns = self.metadata.get("categorical_columns", [])
            self.prediction_threshold = float(self.metadata.get("prediction_threshold", 0.4))

        # Sanity checks — fail loudly at startup rather than silently at inference time
        if self.model.n_features_in_ != len(self.feature_names):
            raise ValueError(
                f"Model expects {self.model.n_features_in_} features but "
                f"feature_names.json has {len(self.feature_names)}. Artifacts are out of sync."
            )

        expected_scaler_cols = set(self.scaler.feature_names_in_)
        if expected_scaler_cols != set(self.numeric_columns):
            raise ValueError(
                f"Scaler was fit on {expected_scaler_cols} but metadata.json numeric_columns "
                f"is {self.numeric_columns}. Artifacts are out of sync."
            )

        self._loaded = True
        logger.info(
            "Model artifacts loaded: %d features, ROC-AUC %.4f (from training)",
            len(self.feature_names),
            self.metadata.get("roc_auc", float("nan")),
        )


@lru_cache
def get_artifacts() -> ModelArtifacts:
    """FastAPI dependency — returns the singleton, loading it on first call."""
    artifacts = ModelArtifacts()
    artifacts.load()
    return artifacts
