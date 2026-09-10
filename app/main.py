import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.api.v1.router import api_router
from app.ml.model_loader import get_artifacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast at startup if artifacts are missing/inconsistent, rather than
    # on the first incoming request
    get_artifacts()
    logger.info("Model artifacts loaded successfully.")
    yield


app = FastAPI(
    title=settings.app_name,
    description="Production-style backend serving a churn prediction model "
    "via authenticated REST APIs, with async batch processing and persistent logging.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running. See /docs for API documentation."}
