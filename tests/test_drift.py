"""Drift tests: PSI maths, KS wiring and the batch verdict."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnops.data.features import compute_reference_stats
from churnops.monitoring.drift import (
    detect_drift,
    psi_categorical,
    psi_numeric,
    simulate_drift,
)


def _wide_frame(rows: int = 2000, seed: int = 3) -> pd.DataFrame:
    """A frame with every monitored column so reference stats cover all features."""
    rng = np.random.default_rng(seed)
    charges = rng.normal(50, 10, size=rows)
    return pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, rows),
            "MonthlyCharges": charges,
            "TotalCharges": charges * 10,
            "SeniorCitizen": rng.integers(0, 2, rows),
            "gender": rng.choice(["Female", "Male"], rows),
            "Partner": rng.choice(["Yes", "No"], rows),
            "Dependents": rng.choice(["Yes", "No"], rows),
            "PhoneService": rng.choice(["Yes", "No"], rows),
            "MultipleLines": rng.choice(["Yes", "No"], rows),
            "InternetService": rng.choice(["DSL", "Fiber optic"], rows),
            "OnlineSecurity": rng.choice(["Yes", "No"], rows),
            "OnlineBackup": rng.choice(["Yes", "No"], rows),
            "DeviceProtection": rng.choice(["Yes", "No"], rows),
            "TechSupport": rng.choice(["Yes", "No"], rows),
            "StreamingTV": rng.choice(["Yes", "No"], rows),
            "StreamingMovies": rng.choice(["Yes", "No"], rows),
            "Contract": rng.choice(["Month-to-month", "Two year"], rows),
            "PaperlessBilling": rng.choice(["Yes", "No"], rows),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], rows),
        }
    )


def test_psi_is_near_zero_for_identical_distributions():
    frame = _wide_frame()
    stats = compute_reference_stats(frame)
    entry = stats["MonthlyCharges"]

    same = psi_numeric(frame["MonthlyCharges"], entry["bin_edges"], entry["bin_share"])
    shifted = psi_numeric(frame["MonthlyCharges"] + 40, entry["bin_edges"], entry["bin_share"])

    assert same < 0.02
    assert shifted > 0.2


def test_psi_categorical_detects_new_mix():
    expected = {"Yes": 0.8, "No": 0.2}

    same = psi_categorical(pd.Series(["Yes"] * 80 + ["No"] * 20), expected)
    flipped = psi_categorical(pd.Series(["Yes"] * 20 + ["No"] * 80), expected)

    assert same < 0.02
    assert flipped > 0.2


def test_stable_batch_is_reported_stable(churn_frame, reference_sample):
    reference_stats = compute_reference_stats(churn_frame.drop(columns=["Churn"]))

    report = detect_drift(
        reference_sample.head(500),
        reference_stats=reference_stats,
        reference_sample=reference_sample,
    )

    assert report["status"] == "stable"
    assert report["drift_share"] < 0.3
    assert report["rows"] == len(reference_sample)
    assert set(report["features"]) == set(reference_stats)


def test_simulated_drift_triggers_alert(churn_frame, reference_sample):
    reference_stats = compute_reference_stats(churn_frame.drop(columns=["Churn"]))
    drifted = simulate_drift(reference_sample, severity="high", seed=11)

    report = detect_drift(
        drifted, reference_stats=reference_stats, reference_sample=reference_sample
    )

    assert report["status"] == "significant"
    assert report["alerting_features"]
    assert report["drift_share"] >= 0.3
    assert all(info["psi"] >= 0 for info in report["features"].values())
