"""Shared fixtures: a small, schema-valid churn frame built in memory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def make_frame(rows: int = 600, seed: int = 7, churn_rate: float = 0.26) -> pd.DataFrame:
    """Build a frame that satisfies the data contract without touching the network."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"customerID": [f"cust-{i:05d}" for i in range(rows)]})
    for column, values in CATEGORY_VALUES.items():
        frame[column] = rng.choice(values, size=rows)
    frame["SeniorCitizen"] = rng.integers(0, 2, size=rows)
    frame["tenure"] = rng.integers(1, 73, size=rows)
    frame["MonthlyCharges"] = np.round(rng.uniform(18, 120, size=rows), 2)
    frame["TotalCharges"] = np.round(frame["tenure"] * frame["MonthlyCharges"], 2)
    # Make churn depend on tenure so models have something to learn.
    probability = np.clip(0.75 - frame["tenure"] / 100.0, 0.05, 0.9)
    frame["Churn"] = np.where(rng.random(rows) < probability * churn_rate / 0.26, "Yes", "No")
    if frame["Churn"].nunique() == 1:  # pragma: no cover - defensive
        frame.loc[frame.index[: rows // 4], "Churn"] = "Yes"
    return frame


@pytest.fixture()
def churn_frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture()
def reference_sample(churn_frame: pd.DataFrame) -> pd.DataFrame:
    return churn_frame.drop(columns=["Churn"]).head(400)
