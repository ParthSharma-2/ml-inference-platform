"""
Pydantic request/response schemas for the churn prediction endpoint.

The RAW input here mirrors the original Telco Customer Churn dataset columns
(before one-hot encoding). Encoding into the 30-column model feature space
happens in inference.py — the API consumer should never need to know about
dummy columns.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Raw customer attributes, as a human/consumer would provide them."""

    senior_citizen: bool = Field(..., description="Whether the customer is a senior citizen")
    tenure: int = Field(..., ge=0, le=100, description="Number of months the customer has stayed")
    monthly_charges: float = Field(..., ge=0, description="Current monthly charge amount")
    total_charges: float = Field(..., ge=0, description="Total amount charged to the customer")

    gender: Literal["Female", "Male"]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "senior_citizen": False,
                "tenure": 12,
                "monthly_charges": 70.35,
                "total_charges": 845.50,
                "gender": "Female",
                "partner": "Yes",
                "dependents": "No",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
            }
        }
    )


class TopDriver(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_churn_risk", "decreases_churn_risk"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    churn_probability: float
    churn_prediction: Literal["Yes", "No"]
    top_drivers: list[TopDriver]
    model_version: str = "1.0.0"
