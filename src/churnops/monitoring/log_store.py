"""Prediction logging and lightweight serving analytics.

SQLite keeps the free-tier story honest: no managed database required, yet we
still capture every scored request for drift, latency and outcome analysis.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from churnops.config import PREDICTIONS_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version TEXT,
    score REAL NOT NULL,
    threshold REAL,
    decision INTEGER,
    latency_ms REAL,
    features TEXT,
    actual INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
"""


@contextmanager
def _connect():
    PREDICTIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(PREDICTIONS_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(SCHEMA)


def log_prediction(
    score: float,
    features: dict,
    model_version: str,
    threshold: float,
    decision: int,
    latency_ms: float,
    request_id: str | None = None,
) -> str:
    init_db()
    request_id = request_id or str(uuid.uuid4())
    with _connect() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO predictions "
            "(request_id, created_at, model_version, score, threshold, decision, latency_ms, features) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                datetime.now(UTC).isoformat(),
                model_version,
                float(score),
                float(threshold),
                int(decision),
                float(latency_ms),
                json.dumps(features, default=str),
            ),
        )
    return request_id


def set_actual(request_id: str, actual: int) -> bool:
    init_db()
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE predictions SET actual = ? WHERE request_id = ?", (int(actual), request_id)
        )
    return cursor.rowcount > 0


def recent(limit: int = 50) -> list[dict]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT request_id, created_at, model_version, score, decision, latency_ms "
            "FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def stats(window_minutes: int = 60) -> dict:
    """Aggregate metrics for the dashboard."""
    init_db()
    since = (datetime.now(UTC) - timedelta(minutes=window_minutes)).isoformat()
    with _connect() as connection:
        summary = connection.execute(
            "SELECT COUNT(*) AS n, AVG(score) AS avg_score, AVG(latency_ms) AS avg_latency_ms, "
            "SUM(decision) AS flagged, MAX(created_at) AS last_seen "
            "FROM predictions WHERE created_at >= ?",
            (since,),
        ).fetchone()
        histogram = connection.execute(
            "SELECT CAST(score * 10 AS INT) AS bucket, COUNT(*) AS n FROM predictions "
            "WHERE created_at >= ? GROUP BY bucket ORDER BY bucket",
            (since,),
        ).fetchall()
        labelled = connection.execute(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN actual = decision THEN 1 ELSE 0 END) AS hits, "
            "SUM(CASE WHEN decision = 1 AND actual = 1 THEN 1 ELSE 0 END) AS true_positive, "
            "SUM(CASE WHEN decision = 1 AND actual = 0 THEN 1 ELSE 0 END) AS false_positive, "
            "SUM(CASE WHEN decision = 0 AND actual = 1 THEN 1 ELSE 0 END) AS false_negative "
            "FROM predictions WHERE actual IS NOT NULL"
        ).fetchone()

    total = summary["n"] or 0
    accuracy = None
    precision = None
    recall = None
    if labelled["n"]:
        accuracy = float(labelled["hits"] or 0) / float(labelled["n"])
        true_positive = float(labelled["true_positive"] or 0)
        false_positive = float(labelled["false_positive"] or 0)
        false_negative = float(labelled["false_negative"] or 0)
        if true_positive + false_positive:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative:
            recall = true_positive / (true_positive + false_negative)

    return {
        "window_minutes": window_minutes,
        "requests": total,
        "avg_score": float(summary["avg_score"]) if summary["avg_score"] is not None else None,
        "avg_latency_ms": float(summary["avg_latency_ms"])
        if summary["avg_latency_ms"] is not None
        else None,
        "flagged": int(summary["flagged"] or 0),
        "last_seen": summary["last_seen"],
        "score_histogram": [{"bucket": row["bucket"], "count": row["n"]} for row in histogram],
        "labelled_samples": int(labelled["n"] or 0),
        "decision_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
