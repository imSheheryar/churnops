"""Dataset ingestion.

Downloads the IBM Telco Customer Churn dataset and records a reproducibility
manifest (row count, checksum, source, timestamp). If the download is not
available (air-gapped CI, upstream outage) a synthetic dataset with the same
schema and a learnable signal is generated instead, so the pipeline never
silently produces an empty artefact.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from churnops.config import MANIFEST_PATH, RAW_DATA_PATH, RAW_DIR, settings
from churnops.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(url: str | None = None, dest: Path | None = None, retries: int = 3) -> Path:
    """Download the CSV with retries. Raises on repeated failure."""
    url = url or settings.data_url
    dest = dest or RAW_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("downloading dataset", extra={"url": url, "attempt": attempt})
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                payload = response.read()
            if len(payload) < 10_000:
                raise ValueError(f"payload too small: {len(payload)} bytes")
            dest.write_bytes(payload)
            logger.info("dataset downloaded", extra={"bytes": len(payload), "path": str(dest)})
            return dest
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            logger.warning("download failed", extra={"attempt": attempt, "error": str(exc)})
            time.sleep(2 * attempt)
    raise RuntimeError(f"unable to download dataset after {retries} attempts: {last_error}")


def generate_synthetic(n_rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Generate a schema-compatible synthetic churn dataset with real signal."""
    rng = np.random.default_rng(seed)

    def pick(values: list[str], probabilities: list[float], size: int) -> np.ndarray:
        return rng.choice(values, size=size, p=probabilities)

    tenure = rng.integers(0, 73, n_rows)
    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24], n_rows)
    internet = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], n_rows)
    monthly = np.clip(rng.normal(64, 30, n_rows), 18, 120).round(2)
    monthly = np.where(internet == "Fiber optic", monthly + 15, monthly).round(2)
    total = (tenure * monthly * rng.normal(1.0, 0.05, n_rows)).clip(min=0).round(2)

    # Logistic signal so the model has something real to learn.
    logit = (
        -0.9
        + 1.15 * (contract == "Month-to-month")
        + 0.55 * (internet == "Fiber optic")
        + 0.035 * (monthly - 64)
        - 0.055 * tenure
        + 0.4 * rng.normal(0, 1, n_rows)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    frame = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:06d}" for i in range(n_rows)],
            "gender": pick(["Female", "Male"], [0.5, 0.5], n_rows),
            "SeniorCitizen": rng.integers(0, 2, n_rows),
            "Partner": pick(["Yes", "No"], [0.48, 0.52], n_rows),
            "Dependents": pick(["Yes", "No"], [0.3, 0.7], n_rows),
            "tenure": tenure,
            "PhoneService": pick(["Yes", "No"], [0.9, 0.1], n_rows),
            "MultipleLines": pick(["Yes", "No", "No phone service"], [0.42, 0.48, 0.1], n_rows),
            "InternetService": internet,
            "OnlineSecurity": pick(
                ["Yes", "No", "No internet service"], [0.29, 0.49, 0.22], n_rows
            ),
            "OnlineBackup": pick(["Yes", "No", "No internet service"], [0.34, 0.44, 0.22], n_rows),
            "DeviceProtection": pick(
                ["Yes", "No", "No internet service"], [0.34, 0.44, 0.22], n_rows
            ),
            "TechSupport": pick(["Yes", "No", "No internet service"], [0.29, 0.49, 0.22], n_rows),
            "StreamingTV": pick(["Yes", "No", "No internet service"], [0.38, 0.4, 0.22], n_rows),
            "StreamingMovies": pick(
                ["Yes", "No", "No internet service"], [0.39, 0.39, 0.22], n_rows
            ),
            "Contract": contract,
            "PaperlessBilling": pick(["Yes", "No"], [0.59, 0.41], n_rows),
            "PaymentMethod": pick(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                [0.34, 0.23, 0.22, 0.21],
                n_rows,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        }
    )
    # Blank TotalCharges for brand-new customers, mirroring the real dataset quirk.
    frame.loc[frame["tenure"] == 0, "TotalCharges"] = " "
    return frame


def write_manifest(path: Path, source: str, rows: int) -> dict:
    manifest = {
        "path": str(path.relative_to(path.parents[2])),
        "sha256": _sha256(path),
        "rows": rows,
        "source": source,
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest written", extra={"sha256": manifest["sha256"], "rows": rows})
    return manifest


def load_dataset(
    force: bool = False,
    sample: int | None = None,
    allow_synthetic: bool = True,
    synthetic_rows: int = 7043,
) -> pd.DataFrame:
    """Load the raw dataset, downloading it if needed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source = "remote"

    if force or not RAW_DATA_PATH.exists():
        try:
            download_dataset()
        except RuntimeError as exc:
            if not allow_synthetic:
                raise
            logger.warning("falling back to synthetic data", extra={"error": str(exc)})
            generate_synthetic(synthetic_rows).to_csv(RAW_DATA_PATH, index=False)
            source = "synthetic"

    frame = pd.read_csv(RAW_DATA_PATH)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    if sample:
        frame = frame.sample(n=min(sample, len(frame)), random_state=settings.random_state)
        frame = frame.reset_index(drop=True)
        source = f"{source}+sampled({len(frame)})"

    write_manifest(RAW_DATA_PATH, source, len(frame))
    logger.info("dataset loaded", extra={"rows": len(frame), "columns": len(frame.columns)})
    return frame


if __name__ == "__main__":  # pragma: no cover
    df = load_dataset(force=True)
    print(df.shape)
    print(df["Churn"].value_counts(normalize=True))
