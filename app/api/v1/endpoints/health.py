from fastapi import APIRouter, Depends

from app.ml.model_loader import get_artifacts, ModelArtifacts

router = APIRouter()


@router.get("/health")
def health_check(artifacts: ModelArtifacts = Depends(get_artifacts)):
    """
    Basic liveness + model-readiness check. Returns 200 only if the model
    artifacts loaded successfully at startup (see model_loader sanity checks).
    """
    return {
        "status": "ok",
        "model_type": artifacts.metadata.get("model_type"),
        "model_roc_auc": artifacts.metadata.get("roc_auc"),
        "n_features": len(artifacts.feature_names),
    }
