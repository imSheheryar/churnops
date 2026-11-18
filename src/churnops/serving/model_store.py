"""Model loading strategy.

Order of preference:
1. Explicit local artefact (artifacts/model.joblib) - used by CI and the container.
2. MLflow registry Production version - used when the tracking server is reachable.

Loading is lazy and thread-safe so that a cold container can answer /health
before the model is ready, and the platform can distinguish liveness from
readiness.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib

from churnops.config import ARTIFACTS_DIR, REPORTS_DIR, settings
from churnops.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_META_PATH = ARTIFACTS_DIR / "model_meta.json"
LOCAL_MODEL_PATH = ARTIFACTS_DIR / "model.joblib"


@dataclass
class ModelBundle:
    model: object
    version: str
    threshold: float
    stage: str = settings.production_stage
    trained_at: str | None = None
    feature_names: list[str] | None = None


class ModelStore:
    def __init__(self) -> None:
        self._bundle: ModelBundle | None = None
        self._lock = threading.Lock()
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def load(self, force: bool = False) -> ModelBundle | None:
        with self._lock:
            if self._bundle is not None and not force:
                return self._bundle
            try:
                self._bundle = self._load_local() or self._load_registry()
                if self._bundle:
                    logger.info(
                        "model loaded",
                        extra={
                            "version": self._bundle.version,
                            "threshold": self._bundle.threshold,
                        },
                    )
                self._error = None if self._bundle else "no model artefact found"
            except Exception as exc:  # pragma: no cover - depends on environment
                self._error = str(exc)
                logger.error("model load failed", extra={"error": str(exc)})
            return self._bundle

    @property
    def bundle(self) -> ModelBundle | None:
        return self._bundle or self.load()

    def _load_local(self) -> ModelBundle | None:
        if not LOCAL_MODEL_PATH.exists():
            return None
        model = joblib.load(LOCAL_MODEL_PATH)
        meta = {}
        if MODEL_META_PATH.exists():
            meta = json.loads(MODEL_META_PATH.read_text())
        threshold = meta.get("threshold") or _threshold_from_metrics()
        return ModelBundle(
            model=model,
            version=str(meta.get("version", "local")),
            threshold=float(threshold),
            trained_at=meta.get("trained_at"),
            feature_names=meta.get("features"),
        )

    def _load_registry(self) -> ModelBundle | None:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(settings.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            settings.model_name, stages=[settings.production_stage]
        )
        if not versions:
            return None
        version = versions[0]
        model = mlflow.sklearn.load_model(f"models:/{settings.model_name}/{version.version}")
        threshold = _threshold_from_metrics()
        return ModelBundle(
            model=model,
            version=str(version.version),
            threshold=float(threshold),
            trained_at=str(version.creation_timestamp),
        )


def _threshold_from_metrics() -> float:
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        for key in ("test", "valid"):
            if key in metrics and "threshold" in metrics[key]:
                return float(metrics[key]["threshold"])
    return 0.5


def write_model_meta(version: str, threshold: float, trained_at: str, features: list[str]) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_META_PATH.write_text(
        json.dumps(
            {
                "version": version,
                "threshold": threshold,
                "trained_at": trained_at,
                "features": features,
            },
            indent=2,
        )
    )
    return MODEL_META_PATH


model_store = ModelStore()
