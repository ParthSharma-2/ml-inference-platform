from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import PredictionLog

router = APIRouter()


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    churn_probability: float
    churn_prediction: str
    top_drivers: list
    model_version: str
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    items: list[PredictionHistoryItem]
    total: int


@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
)
def get_prediction_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Returns previously stored churn predictions.

    Results are ordered from newest to oldest.
    """

    total = db.scalar(
        select(func.count()).select_from(PredictionLog)
    )

    records = db.scalars(
        select(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .offset(skip)
        .limit(limit)
    ).all()

    items = [
        PredictionHistoryItem(
            id=record.id,
            churn_probability=record.churn_probability,
            churn_prediction=record.churn_prediction,
            top_drivers=record.top_drivers,
            model_version=record.model_version,
            created_at=record.created_at,
        )
        for record in records
    ]

    return PredictionHistoryResponse(
        items=items,
        total=total or 0,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionHistoryItem,
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    """
    Returns a single stored prediction by ID.
    """

    record = db.get(PredictionLog, prediction_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return PredictionHistoryItem(
        id=record.id,
        churn_probability=record.churn_probability,
        churn_prediction=record.churn_prediction,
        top_drivers=record.top_drivers,
        model_version=record.model_version,
        created_at=record.created_at,
    )