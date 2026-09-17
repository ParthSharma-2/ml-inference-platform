# ML Inference Platform

**A production-style serving platform for a customer churn prediction model — built with FastAPI, Celery, Redis, and PostgreSQL.**

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Celery](https://img.shields.io/badge/Celery-5.5.3-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen)](#testing--verification)

---

## Overview

This project takes an already-trained customer-churn ML model and wraps it in the engineering layer a real deployment needs: request validation, consistent feature transformation, synchronous and asynchronous inference paths, explainability, durable persistence, and containerized orchestration.

It supports two inference paths:

- **Synchronous, single-customer inference** via FastAPI — returns a prediction, probability, and SHAP-based explanation in one request/response cycle.
- **Asynchronous batch inference** via Redis + Celery — queues up to 100 customer records, processes them individually with retry and failure-isolation logic, and exposes job status through a polling endpoint.

| | |
|---|---|
| **Primary model** | `GradientBoostingClassifier` |
| **Recorded ROC-AUC** | `0.841501976284585` |
| **Feature space** | 30 encoded features |
| **Prediction threshold** | 0.4 |
| **Model version** | `1.0.0` |
| **Current scope** | Minimum Strong Version, through Phase 6.4.2 |
| **Next phase** | MLflow / MLOps |

> Source model repository: [Customer-Churn-Prediction-using-ML](https://github.com/ParthSharma-2/Customer-Churn-Prediction-using-ML)

---

## Architecture

```
Client
 ├─ POST /api/v1/predict
 │    → FastAPI → Pydantic → feature transformation
 │    → GradientBoostingClassifier → probability + SHAP
 │    → PostgreSQL → response
 │
 └─ POST /api/v1/predictions/batch
      → Redis broker → Celery worker
      → validate → transform → predict → SHAP
      → PostgreSQL
      → Redis result backend
      → GET /api/v1/predictions/batch/{job_id}
```

Everything runs as a four-service Docker Compose stack: **PostgreSQL**, **Redis**, the **FastAPI API**, and a **Celery worker**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.11.9 |
| API framework | FastAPI 0.115.0 |
| Validation | Pydantic 2.9.2 / pydantic-settings 2.6.0 |
| ML | scikit-learn 1.6.1 (`GradientBoostingClassifier`) |
| Data handling | pandas 2.2.3, NumPy 1.26.4 |
| Explainability | SHAP 0.46.0 |
| Serialization | joblib 1.4.2 |
| Database | PostgreSQL 16 |
| DB access | SQLAlchemy 2.0.52, psycopg 3.3.5 |
| Migrations | Alembic 1.19.2 |
| Async jobs | Celery 5.5.3 |
| Broker / result backend | Redis 7 (container), redis-py 6.4.0 |
| Testing | pytest 8.3.3, httpx 0.27.2 |
| Containers | Docker + Docker Compose |

---

## ML Model & Feature Pipeline

- `GradientBoostingClassifier` trained with `random_state=42`.
- Recorded ROC-AUC of **0.841501976284585** over **5,634** recorded training samples.
- 30 model features, prediction threshold **0.4**, `model_version=1.0.0`.
- Serialized artifacts: `churn_model.joblib`, `preprocessor.joblib`, `feature_names.json`, `metadata.json`, `shap_background.joblib`.
- The API accepts **raw Telco Customer Churn–style fields** — clients never need to send one-hot/dummy columns.

Feature transformation pipeline:
1. Maps snake_case API fields to the original dataset column names.
2. Converts `senior_citizen` to the model's expected representation.
3. Applies one-hot encoding with `drop_first=True`.
4. Reindexes to the saved 30-feature space.
5. Scales numeric fields with the saved preprocessing artifact.
6. Casts model input to `float64`.

Explanations are generated with a SHAP `TreeExplainer`, ranking features by absolute SHAP magnitude and returning the top 5 drivers with feature name, SHAP value, and direction.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Single synchronous prediction |
| `POST` | `/api/v1/predictions/batch` | Queue an asynchronous batch prediction (1–100 customers) |
| `GET` | `/api/v1/predictions/batch/{job_id}` | Batch job status / result |
| `GET` | `/api/v1/predictions` | Prediction history |
| `GET` | `/api/v1/predictions/{prediction_id}` | Individual persisted prediction |
| `GET` | `/api/v1/health` | Health check and model metadata |
| `GET` | `/` | Service root / status message |

### Example: `POST /api/v1/predict` response

```json
{
  "churn_probability": 0.1953,
  "churn_prediction": "No",
  "top_drivers": [
    { "feature": "InternetService_Fiber optic", "shap_value": -0.4699, "direction": "decreases_churn_risk" },
    { "feature": "PhoneService_Yes", "shap_value": 0.2538, "direction": "increases_churn_risk" },
    { "feature": "tenure", "shap_value": 0.2479, "direction": "increases_churn_risk" },
    { "feature": "Contract_Two year", "shap_value": 0.2369, "direction": "increases_churn_risk" },
    { "feature": "PaperlessBilling_Yes", "shap_value": -0.2201, "direction": "decreases_churn_risk" }
  ],
  "model_version": "1.0.0"
}
```

### Request validation

`CustomerFeatures` covers `senior_citizen`, `tenure`, `monthly_charges`, `total_charges`, `gender`, `partner`, `dependents`, `phone_service`, `multiple_lines`, `internet_service`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`, `contract`, `paperless_billing`, and `payment_method`.

- `tenure`: 0–100
- `monthly_charges`, `total_charges`: ≥ 0
- Categorical fields use `Literal` constraints matching the supported Telco dataset values
- `BatchPredictionRequest` accepts **1–100** customer records

---

## Persistence

- `PredictionLog` stores raw customer fields, `churn_probability`, `churn_prediction`, `top_drivers` (as JSONB), `model_version`, and a timezone-aware `created_at`.
- SQLAlchemy connects via `postgresql+psycopg` with `pool_pre_ping=True`.
- Sessions are created and closed through a session dependency; connectivity is checked with `SELECT 1`.
- Alembic manages schema migrations.
- Async job state/results live in Celery/Redis rather than an extra batch-job database table.

---

## Asynchronous Batch Processing

- Celery uses Redis as both broker and result backend, with JSON serialization, UTC, and `task_track_started` enabled.
- Each customer in a batch is validated **individually** — a bad record fails on its own and does not retry (or fail) the whole batch.
- `OperationalError` / `DBAPIError` are treated as **transient** and retried:
  - `max_retries=3`
  - exponential backoff: `2 ** current_retry_count`
- `soft_time_limit=60s`, hard `time_limit=90s`.
- A successful batch result includes `job_id`, `status`, `total`, `completed`, `failed`, and `prediction_ids`.
- Unhandled batch/database failures are logged and propagated.

---

## Production Hardening

- The Celery worker runs as a dedicated **non-root** `appuser` (verified UID/GID 999 in Docker).
- Transient DB failures are retried separately from customer-level (validation/inference) failures.
- Database rollback happens before any retry or failure propagation.
- Task time limits prevent batch jobs from running indefinitely.
- Prediction and persistence failures are wrapped with exception logging.
- **Deliberately deferred** to avoid over-engineering the current workload: advanced idempotency, `acks_late`, distributed locks, custom queue routing, Kubernetes, Kafka, and Spark.

---

## Deployment (Docker Compose)

| Service | Notes |
|---|---|
| `postgres` | `postgres:16`, persistent named volume, `pg_isready` healthcheck |
| `redis` | `redis:7-alpine`, `redis-cli ping` healthcheck |
| `api` | Built from `Dockerfile`; waits for healthy Postgres/Redis; exposes port `8000`; healthchecks `/api/v1/health` |
| `worker` | Same image as `api`, runs Celery directly as `appuser`, waits for healthy Postgres/Redis |

The `Dockerfile` is based on `python:3.11-slim` with pinned requirements, a dedicated non-root `appuser`, application/Alembic files, and a `docker-entrypoint.sh`. The API entrypoint runs `alembic upgrade head` before starting Uvicorn; the worker service overrides the entrypoint (`entrypoint: []`) to start Celery directly.

```bash
docker compose up --build
```

### Configuration

- `.env` is git-ignored; `.env.example` documents `APP_NAME`, `ENVIRONMENT`, PostgreSQL settings, and placeholders.
- Inside Compose, `POSTGRES_HOST=postgres` and `REDIS_URL=redis://redis:6379/0`.
- Model artifacts are tracked in the repo because Docker builds require them.

> **Note:** the FastAPI description string references "authenticated REST APIs," but authentication/authorization is **not** currently implemented — see [Known Limitations](#known-limitations).

---

## Testing & Verification

- **33 tests passing** (30 original + 3 reliability tests added), 1 unrelated deprecation warning, in ~1.48s.
- The warning comes from Starlette's `TestClient`/`anyio` dependency — not a test failure.
- Reliability tests cover: customer failure without retry, database failure triggering retry, and increasing exponential backoff.
- Manually verified end-to-end: root endpoint, health check, single prediction, persistence/history, individual retrieval, repeated-prediction consistency, and asynchronous batch processing.

**Live evidence (from a verified run):**

```
GET /                          → "ML Model Serving Platform is running. See /docs for API documentation."
GET /api/v1/health              → status=ok, model_type=GradientBoostingClassifier,
                                   model_roc_auc=0.841501976284585, n_features=30
GET /api/v1/predictions?limit=5 → newest record ID 159, probability 0.1953, prediction "No"
GET /api/v1/predictions/96      → persisted prediction with matching probability, label, drivers, version, timestamp
Async job a5ebf5b3-...          → Celery SUCCESS, status=completed, total=2, completed=2, failed=0
```

---

## Model Artifact Portability Fix

The original `joblib` model had cross-environment portability issues — references to newer NumPy internals and fitted `RandomState` objects that didn't survive across NumPy/scikit-learn versions.

Fix summary:
- Artifact inspected under NumPy 2.3.5 / scikit-learn 1.8.0.
- Fitted random-state objects sanitized; tree-estimator `random_state` values normalized to `42`.
- NumPy module references monkeypatched before re-serialization.
- sklearn version metadata aligned to the serving version (1.6.1).
- Round-trip loading verified.
- `validate_artifacts.py` checks all five artifacts, feature count, scaler numeric features, SHAP background shape, and a sample prediction.

This meant diagnosing and correcting a real model-serialization portability problem rather than treating artifact loading as a black box.

---

## Reproducibility Utilities

- `scripts/retrain_model.py` — reproducibly trains `GradientBoostingClassifier(random_state=42)` on an 80/20 stratified split, applies preprocessing, uses threshold 0.4, writes artifacts/metadata, and round-trip loads the output.
- `scripts/validate_artifacts.py` — validates the artifact set and runs a sample inference.

*(Currently untracked in the repository; deliberately excluded from the latest reliability commit.)*

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | ML model + artifacts | ✅ Complete |
| 2 | FastAPI inference API | ✅ Complete |
| 3 | PostgreSQL persistence | ✅ Complete |
| 4 | Docker + Alembic migrations | ✅ Complete |
| 5 | Testing / API contract | ✅ Complete |
| 6.1 | API contract testing | ✅ Complete |
| 6.2 | Redis + Celery infrastructure | ✅ Complete |
| 6.3 | Async batch prediction | ✅ Complete |
| 6.4.1 | Non-root worker | ✅ Complete |
| 6.4.2 | Reliability hardening | ✅ Complete *(final commit/push pending)* |
| 7 | MLflow / MLOps | 🔜 Next |
| 8 | Prometheus / Grafana observability | ⏳ Later |
| 9 | CI/CD | ⏳ Later |
| 10 | Kubernetes | 🧭 Optional / Later |
| 11 | Kafka / Spark | 🧭 Optional / Later |

---

## Design Decisions

- **FastAPI** — typed contracts, validation, auto-generated OpenAPI docs, and a strong fit for Python/ML services.
- **PostgreSQL** — durable, structured, queryable prediction history.
- **Redis + Celery** — straightforward async workers with state, retries, and time limits, without the overhead of a streaming platform.
- **Docker Compose** — reproducible local multi-service environment; Kubernetes deferred until scale justifies it.
- **SHAP** — feature-level explanations alongside every prediction.
- **Alembic** — versioned, reproducible schema evolution.
- **Pinned dependencies** — reduce runtime/model compatibility drift, especially important after the artifact portability incident.

---

## Known Limitations

These are explicitly **not** implemented yet, and shouldn't be assumed:

- ❌ Authentication / authorization
- ❌ MLflow experiment/model tracking
- ❌ Prometheus / Grafana observability
- ❌ CI/CD pipeline
- ❌ Kubernetes deployment
- ❌ Kafka / Spark integration
- ❌ Advanced idempotency, distributed locking, `acks_late`, custom queue routing
- ❌ No production traffic, latency, throughput, SLA, cloud-cost, or user-count metrics
- ❌ No real-world churn-reduction / business-impact claims

---

## Roadmap

1. Run the final `pytest` pass and commit the Phase 6.4.2 reliability changes.
2. Push to `origin/main`.
3. Begin **Phase 7 — MLflow**: experiment tracking for parameters, metrics, artifacts, and model versions, integrated without unnecessarily rewriting the existing serving path.
4. Evaluate observability (Prometheus/Grafana) and CI/CD before any scale-driven infrastructure work.

---

## What This Project Demonstrates

ML model deployment beyond notebooks · API engineering and contracts · Preprocessing consistency · Model artifact debugging · Database-backed inference logging · Asynchronous distributed task execution · Retry/failure semantics · Task time limits · Explainable ML · Containerization · Least-privilege worker execution · Automated testing · End-to-end system verification · Pragmatic technology selection

---

## Links

- **Serving repository:** [ml-inference-platform](https://github.com/ParthSharma-2/ml-inference-platform.git)
- **Original ML repository:** [Customer-Churn-Prediction-using-ML](https://github.com/ParthSharma-2/Customer-Churn-Prediction-using-ML)
