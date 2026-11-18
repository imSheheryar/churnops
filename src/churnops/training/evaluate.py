"""Model evaluation.

Beyond vanilla metrics we translate the confusion matrix into money: a missed
churner costs the customer's lifetime value, a wasted retention offer costs the
discount. The operating threshold is chosen to minimise that expected cost,
which is the number a business stakeholder actually cares about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnops.config import REPORTS_DIR, settings


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "support": int(len(y_true)),
    }


def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_false_negative: float | None = None,
    cost_false_positive: float | None = None,
) -> float:
    """Expected cost for 1,000 scored customers at a given threshold."""
    cost_fn = cost_false_negative or settings.cost_false_negative
    cost_fp = cost_false_positive or settings.cost_false_positive
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)
    if total == 0:
        return 0.0
    per_customer = (fn * cost_fn + fp * cost_fp) / total
    return float(per_customer * 1000)


def no_model_cost(y_true: np.ndarray, cost_false_negative: float | None = None) -> float:
    """Cost of doing nothing: every churner is lost (1,000-customer basis)."""
    cost_fn = cost_false_negative or settings.cost_false_negative
    positives = int(np.sum(y_true))
    return float(positives / max(len(y_true), 1) * cost_fn * 1000)


def choose_threshold(
    y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray | None = None
) -> dict[str, float]:
    """Pick the threshold minimising expected cost, tie-breaking on recall."""
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.01)
    best: dict[str, float] | None = None
    for threshold in grid:
        cost = expected_cost(y_true, y_score, float(threshold))
        y_pred = (y_score >= threshold).astype(int)
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        candidate = {"threshold": float(threshold), "expected_cost_per_1k": cost, "recall": recall}
        if best is None or (cost, -recall) < (best["expected_cost_per_1k"], -best["recall"]):
            best = candidate
    assert best is not None
    baseline = no_model_cost(y_true)
    best["baseline_cost_per_1k"] = baseline
    best["savings_per_1k"] = float(baseline - best["expected_cost_per_1k"])
    best["savings_pct"] = float((baseline - best["expected_cost_per_1k"]) / baseline * 100)
    return best


def evaluate_model(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    """Full evaluation bundle: metrics at the operating point + economics."""
    if threshold is None:
        economics = choose_threshold(y_true, y_score)
        threshold = economics["threshold"]
    else:
        economics = {
            "threshold": float(threshold),
            "expected_cost_per_1k": expected_cost(y_true, y_score, threshold),
            "baseline_cost_per_1k": no_model_cost(y_true),
        }
        economics["savings_per_1k"] = (
            economics["baseline_cost_per_1k"] - economics["expected_cost_per_1k"]
        )
        economics["savings_pct"] = (
            economics["savings_per_1k"] / economics["baseline_cost_per_1k"] * 100
        )
        economics["recall"] = classification_metrics(y_true, y_score, threshold)["recall"]

    metrics = classification_metrics(y_true, y_score, threshold)
    metrics.update({f"economics_{key}": value for key, value in economics.items()})
    return metrics


def save_metrics(metrics: dict, path: Path | None = None) -> Path:
    path = path or REPORTS_DIR / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path
