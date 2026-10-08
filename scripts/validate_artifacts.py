"""Validate ML artifacts and run a representative prediction.

Run from the project root:
    python scripts/validate_artifacts.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "app" / "ml" / "artifacts"

required = [
    "churn_model.joblib",
    "preprocessor.joblib",
    "feature_names.json",
    "metadata.json",
    "shap_background.joblib",
]

for name in required:
    path = ART / name
    if not path.exists():
        raise FileNotFoundError(path)

model = joblib.load(ART / "churn_model.joblib")
scaler = joblib.load(ART / "preprocessor.joblib")
background = joblib.load(ART / "shap_background.joblib")
features = json.loads((ART / "feature_names.json").read_text())
metadata = json.loads((ART / "metadata.json").read_text())

assert model.n_features_in_ == len(features), (
    f"model expects {model.n_features_in_}, schema has {len(features)}"
)
assert set(scaler.feature_names_in_) == set(metadata["numeric_columns"])
assert background.shape[1] == len(features)

# Representative feature vector using the saved SHAP background.
sample = background.iloc[[0]].copy().astype("float64")
probability = float(model.predict_proba(sample)[0, 1])
threshold = float(metadata.get("prediction_threshold", 0.4))
prediction = "Yes" if probability >= threshold else "No"

print("Artifact validation: SUCCESS")
print(f"Model: {type(model).__name__}")
print(f"Features: {model.n_features_in_}")
print(f"ROC-AUC: {metadata.get('roc_auc')}")
print(f"Threshold: {threshold}")
print(f"Sample probability: {probability:.4f}")
print(f"Sample prediction: {prediction}")
