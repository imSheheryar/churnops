"""API contracts.

Validation at the edge means the model never sees a malformed payload; the
same Literal types double as documentation in the generated OpenAPI schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.15,
                "TotalCharges": 425.75,
            }
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1, description="1 if the customer is a senior citizen")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, le=1000)
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    request_id: str
    churn_probability: float
    decision: bool = Field(description="True when the score crosses the operating threshold")
    threshold: float
    risk_band: Literal["low", "medium", "high"]
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    count: int
    model_version: str
    mean_probability: float
    flagged: int
    predictions: list[PredictionResponse]


class FeedbackRequest(BaseModel):
    request_id: str
    actual: int = Field(ge=0, le=1, description="Ground-truth label: 1 = churned")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    stage: str
    threshold: float
    trained_at: str | None
    features: list[str]
