from fastapi import APIRouter
from app.api.v1.endpoints import health, predict, history
from app.api.v1.endpoints import predict, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predict.router, tags=["prediction"])

api_router.include_router(
    history.router,
    tags=["prediction history"],
)