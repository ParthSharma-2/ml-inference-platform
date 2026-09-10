# ML Model Serving Platform — Phase 1

FastAPI service serving the ChurnIQ Gradient Boosting model (ROC-AUC 0.8415) trained on the Telco Customer Churn dataset.

## Run locally

Use the existing project virtual environment and install the pinned dependencies:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for Swagger UI.

## Validate before starting the API

```powershell
python scripts/validate_artifacts.py
```

The validator checks all five artifacts, feature counts, scaler columns, and a sample prediction.

## Run tests

```powershell
pytest app/tests/ -v
```

## Endpoints (Phase 1)

- `GET /api/v1/health` — model readiness check
- `POST /api/v1/predict` — single-customer churn prediction with SHAP top drivers

## Model artifacts

`app/ml/artifacts/` contains:

- `churn_model.joblib`
- `preprocessor.joblib`
- `feature_names.json`
- `metadata.json`
- `shap_background.joblib`

The training convention is `pd.get_dummies(..., drop_first=True)`, followed by scaling only the four numeric columns. Inference reproduces this convention and reindexes to the saved 30-feature schema.

The prediction threshold is stored in `metadata.json` (`0.4`) instead of being hard-coded in the API.

### Why the model artifact was replaced

The original Colab model was serialized with NumPy 2.x-style references to fitted `RandomState`/BitGenerator and NumPy core internals. The serving environment is pinned to NumPy 1.26.4. The original model therefore failed during `joblib.load()` before FastAPI could start.

The corrected artifact preserves the trained Gradient Boosting estimator and removes fitted random-generator state that is only required during training, while keeping the configured training seed (`42`). NumPy-compatible module references are also normalized for the pinned serving environment.

For future retraining, use:

```powershell
python scripts/retrain_model.py
```

The training script prints the environment versions, evaluates ROC-AUC and threshold-0.4 classification metrics, exports the same artifact contract, and performs a model round-trip load check.

## Coming in later phases

- Phase 2: PostgreSQL + SQLAlchemy prediction logging
- Phase 3: JWT auth protecting `/predict` and `/batch`
- Phase 4: Celery + Redis async batch scoring
- Phase 5: expanded pytest coverage + GitHub Actions CI
- Phase 6: Docker + docker-compose + AWS EC2 deployment
