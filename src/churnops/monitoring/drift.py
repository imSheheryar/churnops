"""Distribution drift detection.

Population Stability Index (PSI) is the primary signal because it is
interpretable to non-ML stakeholders and cheap to compute on every batch.
A two-sample Kolmogorov-Smirnov test is added for numeric features when a
reference sample is available, to separate real shifts from binning artefacts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from churnops.config import ARTIFACTS_DIR, REFERENCE_STATS_PATH, settings
from churnops.data.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churnops.data.validate import clean_types
from churnops.logging_utils import get_logger

logger = get_logger(__name__)

REFERENCE_SAMPLE_PATH = ARTIFACTS_DIR / "reference_sample.parquet"


def _severity(psi: float) -> str:
    if psi >= settings.psi_alert:
        return "alert"
    if psi >= settings.psi_warn:
        return "warn"
    return "stable"


def psi_numeric(values: pd.Series, bin_edges: list[float], expected_share: list[float]) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return 0.0
    edges = np.array(bin_edges, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    actual = counts / max(counts.sum(), 1)
    expected = np.clip(np.array(expected_share, dtype=float), 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_categorical(values: pd.Series, expected_share: dict[str, float]) -> float:
    observed = values.astype(str).value_counts(normalize=True)
    keys = set(expected_share) | set(observed.index)
    total = 0.0
    for key in keys:
        expected = max(float(expected_share.get(key, 0.0)), 1e-6)
        actual = max(float(observed.get(key, 0.0)), 1e-6)
        total += (actual - expected) * np.log(actual / expected)
    return float(total)


def ks_test(values: pd.Series, reference: pd.Series) -> tuple[float, float]:
    current = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    if current.size < 20 or ref.size < 20:
        return float("nan"), float("nan")
    result = stats.ks_2samp(current, ref)
    return float(result.statistic), float(result.pvalue)


def detect_drift(
    current: pd.DataFrame,
    reference_stats: dict | None = None,
    reference_sample: pd.DataFrame | None = None,
) -> dict:
    """Compare a batch against the training distribution."""
    reference_stats = reference_stats or json.loads(Path(REFERENCE_STATS_PATH).read_text())
    reference_sample = (
        reference_sample if reference_sample is not None else _load_reference_sample()
    )
    current = clean_types(current)

    features: dict[str, dict] = {}

    for column in NUMERIC_FEATURES:
        if column not in current.columns or column not in reference_stats:
            continue
        stats_entry = reference_stats[column]
        psi = psi_numeric(current[column], stats_entry["bin_edges"], stats_entry["bin_share"])
        detail = {"type": "numeric", "psi": round(psi, 4), "severity": _severity(psi)}
        if reference_sample is not None and column in reference_sample.columns:
            statistic, pvalue = ks_test(current[column], reference_sample[column])
            detail["ks_statistic"] = None if np.isnan(statistic) else round(statistic, 4)
            detail["ks_pvalue"] = None if np.isnan(pvalue) else float(f"{pvalue:.3g}")
        detail["current_mean"] = float(pd.to_numeric(current[column], errors="coerce").mean())
        detail["reference_mean"] = float(stats_entry["mean"])
        features[column] = detail

    for column in CATEGORICAL_FEATURES:
        if column not in current.columns or column not in reference_stats:
            continue
        psi = psi_categorical(current[column], reference_stats[column]["share"])
        features[column] = {
            "type": "categorical",
            "psi": round(psi, 4),
            "severity": _severity(psi),
        }

    drifted = [name for name, info in features.items() if info["severity"] == "alert"]
    warned = [name for name, info in features.items() if info["severity"] == "warn"]
    drift_share = len(drifted) / max(len(features), 1)

    if drift_share >= settings.drift_share_alert:
        status = "significant"
    elif drifted or warned:
        status = "warning"
    else:
        status = "stable"

    report = {
        "status": status,
        "drift_share": round(drift_share, 4),
        "checked_at": datetime.now(UTC).isoformat(),
        "rows": int(len(current)),
        "features": features,
        "alerting_features": drifted,
        "warning_features": warned,
    }
    logger.info(
        "drift computed",
        extra={"status": status, "drift_share": drift_share, "features": len(features)},
    )
    return report


def _load_reference_sample() -> pd.DataFrame | None:
    if REFERENCE_SAMPLE_PATH.exists():
        return pd.read_parquet(REFERENCE_SAMPLE_PATH)
    return None


def simulate_drift(frame: pd.DataFrame, severity: str = "high", seed: int = 7) -> pd.DataFrame:
    """Create a shifted batch for demos and tests.

    Shifts price and tenure distributions and pushes contract mix towards
    month-to-month - the pattern we would expect from a pricing change.
    """
    rng = np.random.default_rng(seed)
    drifted = frame.copy()
    factor = {"low": 1.08, "medium": 1.18, "high": 1.35}[severity]

    drifted["MonthlyCharges"] = (drifted["MonthlyCharges"] * factor).round(2)
    drifted["tenure"] = (drifted["tenure"] * (1 / factor)).clip(lower=0).round(0)

    mask = rng.random(len(drifted)) < 0.35
    drifted.loc[mask, "Contract"] = "Month-to-month"

    # A pricing change also moves add-on uptake and billing behaviour, so the
    # simulated batch touches a realistic spread of features rather than one.
    breadth = {"low": 0.10, "medium": 0.22, "high": 0.38}[severity]
    addon_mask = rng.random(len(drifted)) < breadth
    for column in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"):
        if column in drifted.columns:
            drifted.loc[addon_mask, column] = "No"
    billing_mask = rng.random(len(drifted)) < breadth
    if "PaperlessBilling" in drifted.columns:
        drifted.loc[billing_mask, "PaperlessBilling"] = "Yes"
    if "PaymentMethod" in drifted.columns:
        drifted.loc[billing_mask, "PaymentMethod"] = "Electronic check"
    if "InternetService" in drifted.columns:
        fiber_mask = rng.random(len(drifted)) < breadth * 0.6
        drifted.loc[fiber_mask, "InternetService"] = "Fiber optic"

    if "TotalCharges" in drifted.columns:
        drifted["TotalCharges"] = (
            pd.to_numeric(drifted["TotalCharges"], errors="coerce").fillna(0) * factor
        ).round(2)
    return drifted
