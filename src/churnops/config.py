"""Central configuration for the ChurnOps platform.

All paths are resolved relative to the repository root so that the same code
works locally, in CI, and inside the container.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    env_home = os.environ.get("CHURNOPS_HOME")
    if env_home:
        return Path(env_home).resolve()
    # src/churnops/config.py -> repo root
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

RAW_DATA_PATH = RAW_DIR / "telco_churn.csv"
MANIFEST_PATH = RAW_DIR / "manifest.json"
REFERENCE_STATS_PATH = ARTIFACTS_DIR / "reference_stats.json"
MLFLOW_TRACKING_DIR = ARTIFACTS_DIR / "mlruns"
MLFLOW_DB_PATH = ARTIFACTS_DIR / "mlflow.db"
PREDICTIONS_DB_PATH = ARTIFACTS_DIR / "predictions.db"
AUDIT_LOG_PATH = ARTIFACTS_DIR / "audit_log.jsonl"
PIPELINE_RUN_PATH = ARTIFACTS_DIR / "pipeline_run.json"
MODEL_CARD_PATH = DOCS_DIR / "model_card.md"


class Settings(BaseSettings):
    """Runtime settings, overridable with CHURNOPS_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="CHURNOPS_", env_file=".env", extra="ignore")

    # Data
    data_url: str = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    target_column: str = "Churn"
    positive_label: str = "Yes"
    id_column: str = "customerID"
    random_state: int = 42
    valid_size: float = 0.15
    test_size: float = 0.15

    # Promotion gates
    min_pr_auc: float = 0.60
    min_recall: float = 0.65
    max_brier_score: float = 0.20
    cost_false_negative: float = 1200.0  # lost customer lifetime value
    cost_false_positive: float = 50.0  # retention offer cost

    # Drift thresholds
    psi_warn: float = 0.10
    psi_alert: float = 0.20
    drift_share_alert: float = 0.30

    # Registry / serving
    model_name: str = "churnops-churn-classifier"
    production_stage: str = "Production"
    mlflow_tracking_uri: str = ""
    slack_webhook_url: str = ""
    log_level: str = "INFO"

    @property
    def tracking_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{MLFLOW_DB_PATH}"


settings = Settings()


def ensure_directories() -> None:
    """Create the directories the pipeline writes to."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        SAMPLE_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
        MLFLOW_TRACKING_DIR,
        DOCS_DIR / "images",
    ):
        path.mkdir(parents=True, exist_ok=True)
