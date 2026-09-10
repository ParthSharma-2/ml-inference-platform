from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Customer features
    senior_citizen: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    tenure: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    monthly_charges: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    total_charges: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    gender: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    partner: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    dependents: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    phone_service: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    multiple_lines: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    internet_service: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    online_security: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    online_backup: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    device_protection: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    tech_support: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    streaming_tv: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    streaming_movies: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    contract: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    paperless_billing: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    payment_method: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    # Prediction output
    churn_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    churn_prediction: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    # SHAP explanation
    top_drivers: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
    )

    # Model information
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0.0",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )