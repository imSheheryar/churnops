"""Feature engineering and the training/serving contract.

The same ColumnTransformer is fitted at training time and persisted with the
model, which is what guarantees training-serving consistency: the API never
re-implements preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnops.config import REFERENCE_STATS_PATH, settings
from churnops.data.validate import clean_types
from churnops.logging_utils import get_logger

logger = get_logger(__name__)

TARGET = settings.target_column
ID_COLUMN = settings.id_column

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into model input and binary target."""
    cleaned = clean_types(frame)
    missing = set(FEATURE_COLUMNS) - set(cleaned.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")

    X = cleaned[FEATURE_COLUMNS].copy()
    if TARGET not in cleaned.columns:
        raise ValueError(f"target column '{TARGET}' not found")
    y = (cleaned[TARGET] == settings.positive_label).astype(int)
    return X, y


def split_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split."""
    train, rest = train_test_split(
        frame,
        test_size=settings.valid_size + settings.test_size,
        random_state=settings.random_state,
        stratify=frame[TARGET],
    )
    relative_test = settings.test_size / (settings.valid_size + settings.test_size)
    valid, test = train_test_split(
        rest,
        test_size=relative_test,
        random_state=settings.random_state,
        stratify=rest[TARGET],
    )
    logger.info(
        "data split",
        extra={"train": len(train), "valid": len(valid), "test": len(test)},
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def compute_reference_stats(frame: pd.DataFrame, bins: int = 10) -> dict:
    """Summarise the training distribution used as the drift baseline."""
    cleaned = clean_types(frame)
    stats: dict[str, dict] = {}

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(cleaned[column], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if edges.size < 3:
            edges = np.array([values.min(), values.max() + 1e-6])
        stats[column] = {
            "type": "numeric",
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "bin_edges": [float(edge) for edge in edges],
            "bin_share": _bin_share(values, edges),
        }

    for column in CATEGORICAL_FEATURES:
        counts = cleaned[column].astype(str).value_counts(normalize=True)
        stats[column] = {
            "type": "categorical",
            "share": {str(key): float(value) for key, value in counts.items()},
        }

    return stats


def _bin_share(values: np.ndarray, edges: np.ndarray) -> list[float]:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return [float(count / total) if total else 0.0 for count in counts]


def save_reference_stats(stats: dict, path: Path | None = None) -> Path:
    path = path or REFERENCE_STATS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    logger.info("reference stats saved", extra={"path": str(path), "features": len(stats)})
    return path


def load_reference_stats(path: Path | None = None) -> dict:
    path = path or REFERENCE_STATS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"reference stats not found at {path}; run the training pipeline first"
        )
    return json.loads(path.read_text())
