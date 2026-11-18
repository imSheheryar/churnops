"""Scheduled batch scoring job.

Scores a landed CSV with the production model, writes predictions, computes
drift against the training reference, renders the HTML report and optionally
notifies Slack. This is the unit that a cron job (or GitHub Actions schedule)
calls every day.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from churnops.config import ARTIFACTS_DIR, REPORTS_DIR
from churnops.data.features import prepare_xy
from churnops.logging_utils import get_logger
from churnops.monitoring import log_store
from churnops.monitoring.report import generate_drift_report, notify
from churnops.serving.model_store import model_store

logger = get_logger(__name__)


def score_batch(csv_path: str | Path, threshold: float | None = None) -> dict:
    frame = pd.read_csv(csv_path)
    bundle = model_store.bundle
    if bundle is None:
        raise RuntimeError("no model available; run the training pipeline first")

    X, _ = prepare_xy(frame.assign(**{"Churn": "No"}) if "Churn" not in frame.columns else frame)
    scores = bundle.model.predict_proba(X)[:, 1]
    active_threshold = float(threshold if threshold is not None else bundle.threshold)

    output = frame.copy()
    output["churn_probability"] = scores
    output["decision"] = (scores >= active_threshold).astype(int)
    output["model_version"] = bundle.version

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "batch_predictions.csv"
    output.to_csv(output_path, index=False)

    for index, score in enumerate(scores):
        log_store.log_prediction(
            score=float(score),
            features=frame.iloc[index].drop(labels=["Churn"], errors="ignore").to_dict(),
            model_version=bundle.version,
            threshold=active_threshold,
            decision=int(score >= active_threshold),
            latency_ms=0.0,
        )

    report_path, report = generate_drift_report(
        frame, scores=pd.Series(scores), output_path=REPORTS_DIR / "drift_report.html"
    )
    notify(report)

    summary = {
        "rows": int(len(frame)),
        "mean_score": float(scores.mean()),
        "flagged": int((scores >= active_threshold).sum()),
        "threshold": active_threshold,
        "model_version": bundle.version,
        "predictions_path": str(output_path),
        "report_path": str(report_path),
        "drift": {"status": report["status"], "alerting_features": report["alerting_features"]},
    }
    (ARTIFACTS_DIR / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("batch scoring complete", extra=summary["drift"])
    return summary
