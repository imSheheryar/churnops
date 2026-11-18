"""Model training with champion/challenger selection and MLflow tracking.

Two candidates are trained on every run - a regularised logistic regression
(interpretable, cheap, hard to beat on tabular churn data) and a gradient
boosting model (non-linear interactions). The winner on validation PR-AUC is
registered; the loser is kept in MLflow for the record. Selection is data
driven, not aspirational.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from churnops.config import ARTIFACTS_DIR, REPORTS_DIR, settings  # noqa: E402
from churnops.data.features import (  # noqa: E402
    build_preprocessor,
    compute_reference_stats,
    prepare_xy,
    save_reference_stats,
)
from churnops.logging_utils import get_logger  # noqa: E402
from churnops.monitoring.drift import REFERENCE_SAMPLE_PATH  # noqa: E402
from churnops.serving.model_store import write_model_meta  # noqa: E402
from churnops.training.evaluate import evaluate_model, save_metrics  # noqa: E402

logger = get_logger(__name__)

MODEL_ARTIFACT = ARTIFACTS_DIR / "model.joblib"
EXPERIMENT_NAME = "churnops-training"


@dataclass
class TrainingResult:
    run_id: str
    model_version: str
    metrics: dict
    threshold: float
    model_path: Path
    algorithm: str = "hgb"
    challengers: dict = field(default_factory=dict)


def _positive_weight(y: pd.Series) -> np.ndarray:
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - positives), 1)
    return np.where(y.to_numpy() == 1, negatives / positives, 1.0)


def _candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.12,
                        random_state=settings.random_state,
                    ),
                ),
            ]
        ),
    }


def _plot_curves(y_valid: pd.Series, y_score: np.ndarray, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_valid, y_score, ax=ax)
    ax.set_title("ROC curve (validation)")
    fig.tight_layout()
    path = out_dir / "roc_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_valid, y_score, ax=ax)
    ax.set_title("Precision-recall curve (validation)")
    fig.tight_layout()
    path = out_dir / "precision_recall_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_importance(
    model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series, out_dir: Path
) -> tuple[Path, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = X_valid.sample(n=min(800, len(X_valid)), random_state=settings.random_state)
    result = permutation_importance(
        model,
        sample,
        y_valid.loc[sample.index],
        scoring="roc_auc",
        n_repeats=3,
        random_state=settings.random_state,
        n_jobs=1,
    )
    ranking = (
        pd.DataFrame({"feature": X_valid.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(ranking["feature"][::-1], ranking["importance"][::-1], color="#2f6f4e")
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    ax.set_title("Top features")
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, ranking.to_dict(orient="records")


def _plot_confusion(
    y_true: pd.Series, y_score: np.ndarray, threshold: float, out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (y_score >= threshold).astype(int), ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    register: bool = True,
) -> TrainingResult:
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train = prepare_xy(train_df)
    X_valid, y_valid = prepare_xy(valid_df)
    X_test, y_test = prepare_xy(test_df)

    save_reference_stats(compute_reference_stats(train_df))

    fitted: dict[str, Pipeline] = {}
    scores: dict[str, dict[str, np.ndarray]] = {}
    evaluations: dict[str, dict] = {}

    for name, pipeline in _candidates().items():
        pipeline.fit(X_train, y_train, classifier__sample_weight=_positive_weight(y_train))
        valid_score = pipeline.predict_proba(X_valid)[:, 1]
        valid_metrics = evaluate_model(y_valid.to_numpy(), valid_score)
        threshold = valid_metrics["economics_threshold"]
        test_score = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_model(y_test.to_numpy(), test_score, threshold=threshold)
        fitted[name] = pipeline
        scores[name] = {"valid": valid_score, "test": test_score}
        evaluations[name] = {"valid": valid_metrics, "test": test_metrics}
        logger.info(
            "candidate evaluated",
            extra={
                "candidate": name,
                "pr_auc": test_metrics["pr_auc"],
                "roc_auc": test_metrics["roc_auc"],
            },
        )

    winner = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["valid"]["pr_auc"],
            -evaluations[name]["valid"]["brier"],
        ),
    )
    logger.info("champion selected", extra={"champion": winner})
    best = fitted[winner]
    best_metrics = evaluations[winner]["test"]
    threshold = evaluations[winner]["valid"]["economics_threshold"]
    valid_score = scores[winner]["valid"]
    test_score = scores[winner]["test"]

    with mlflow.start_run(run_name=f"churn-{winner}") as run:
        mlflow.log_params(
            {
                "champion": winner,
                "threshold": threshold,
                "train_rows": len(X_train),
                "positive_rate": float(y_train.mean()),
                "candidates": ",".join(sorted(fitted)),
            }
        )
        for name, evaluation in evaluations.items():
            prefix = "champion" if name == winner else "challenger"
            mlflow.log_metrics(
                {
                    f"{prefix}_{name}_valid_{key}": value
                    for key, value in evaluation["valid"].items()
                    if isinstance(value, (int, float))
                }
            )
            mlflow.log_metrics(
                {
                    f"{prefix}_{name}_test_{key}": value
                    for key, value in evaluation["test"].items()
                    if isinstance(value, (int, float))
                }
            )

        curve_paths = _plot_curves(y_valid, valid_score, ARTIFACTS_DIR / "plots")
        importance_path, importance = _plot_importance(
            best, X_valid, y_valid, ARTIFACTS_DIR / "plots"
        )
        confusion_path = _plot_confusion(y_test, test_score, threshold, ARTIFACTS_DIR / "plots")
        for path in [*curve_paths, importance_path, confusion_path]:
            mlflow.log_artifact(str(path), artifact_path="plots")

        summary = {
            "champion": winner,
            "threshold": threshold,
            "candidates": {
                name: {
                    "valid": evaluations[name]["valid"],
                    "test": evaluations[name]["test"],
                }
                for name in evaluations
            },
            "champion_test_metrics": best_metrics,
            "feature_importance": importance,
        }
        summary_path = REPORTS_DIR / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=float))
        mlflow.log_artifact(str(summary_path))

        signature = mlflow.models.infer_signature(
            X_valid.head(5), best.predict_proba(X_valid.head(5))[:, 1]
        )
        mlflow.sklearn.log_model(
            sk_model=best,
            artifact_path="model",
            signature=signature,
            input_example=X_valid.head(3),
            registered_model_name=settings.model_name if register else None,
        )
        mlflow.set_tags(
            {
                "git_sha": _git_sha(),
                "owner": "ml-platform",
                "data_manifest": "data/raw/manifest.json",
                "champion": winner,
            }
        )
        run_id = run.info.run_id

    model_version = ""
    if register:
        versions = mlflow.MlflowClient().search_model_versions(f"name='{settings.model_name}'")
        matching = [v for v in versions if v.run_id == run_id]
        model_version = matching[0].version if matching else ""

    MODEL_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_ARTIFACT)
    reference_sample = train_df.sample(
        n=min(2000, len(train_df)), random_state=settings.random_state
    )
    reference_sample.to_parquet(REFERENCE_SAMPLE_PATH, index=False)
    write_model_meta(
        version=str(model_version or "local"),
        threshold=threshold,
        trained_at=datetime.now(UTC).isoformat(),
        features=list(X_train.columns),
    )
    save_metrics(
        {
            "champion": winner,
            "test": best_metrics,
            "valid": evaluations[winner]["valid"],
            "challengers": {name: ev["test"] for name, ev in evaluations.items() if name != winner},
        }
    )

    logger.info(
        "training complete",
        extra={
            "run_id": run_id,
            "model_version": model_version,
            "champion": winner,
            "pr_auc": best_metrics["pr_auc"],
            "roc_auc": best_metrics["roc_auc"],
            "threshold": threshold,
        },
    )
    return TrainingResult(
        run_id=run_id,
        model_version=model_version,
        metrics=best_metrics,
        threshold=threshold,
        model_path=MODEL_ARTIFACT,
        algorithm=winner,
        challengers={name: ev["test"] for name, ev in evaluations.items() if name != winner},
    )


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git not always available
        return "unknown"
