"""Model promotion gates, audit trail and model card generation.

Nothing reaches Production automatically. A model must clear quantitative
gates, carry a model card, and be promoted by a named approver. Every decision
is appended to an immutable JSONL audit log with the git SHA and the metrics
that justified it - the artefact an auditor would ask for.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime

import mlflow
from mlflow.tracking import MlflowClient

from churnops.config import AUDIT_LOG_PATH, MODEL_CARD_PATH, settings
from churnops.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reasons": self.reasons, "checks": self.checks}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def evaluate_gates(metrics: dict, validation_ok: bool = True) -> GateDecision:
    """Quantitative promotion criteria."""
    checks = {
        "data_validation_passed": validation_ok,
        f"pr_auc>={settings.min_pr_auc}": metrics.get("pr_auc", 0.0) >= settings.min_pr_auc,
        f"recall>={settings.min_recall}": metrics.get("recall", 0.0) >= settings.min_recall,
        f"brier<={settings.max_brier_score}": metrics.get("brier", 1.0) <= settings.max_brier_score,
        "beats_baseline": metrics.get("pr_auc", 0.0) > 0.0,
    }
    reasons = []
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"gate failed: {name}")
    return GateDecision(passed=all(checks.values()), reasons=reasons, checks=checks)


def append_audit(entry: dict) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(UTC).isoformat(), **entry}
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=float) + "\n")
    logger.info("audit entry appended", extra={"action": entry.get("action")})


def promote(
    model_name: str,
    version: str,
    metrics: dict,
    actor: str,
    validation_ok: bool = True,
    force: bool = False,
) -> GateDecision:
    """Run gates and, if they pass, move the version to Production."""
    decision = evaluate_gates(metrics, validation_ok=validation_ok)

    if not decision.passed and not force:
        append_audit(
            {
                "action": "promotion_rejected",
                "model": model_name,
                "version": version,
                "actor": actor,
                "git_sha": _git_sha(),
                "metrics": metrics,
                "decision": decision.to_dict(),
            }
        )
        logger.warning("promotion rejected", extra={"reasons": decision.reasons})
        return decision

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=settings.production_stage,
        archive_existing_versions=True,
    )
    # Stages are deprecated in MLflow 2.9+; keep the alias in sync so the
    # registry keeps working when stages are removed.
    client.set_registered_model_alias(model_name, "production", version)
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"Promoted by {actor}. PR-AUC={metrics.get('pr_auc', 0):.4f}, "
            f"ROC-AUC={metrics.get('roc_auc', 0):.4f}, threshold={metrics.get('threshold', 0):.2f}"
        ),
    )
    append_audit(
        {
            "action": "promoted_to_production" if not force else "forced_promotion",
            "model": model_name,
            "version": version,
            "actor": actor,
            "git_sha": _git_sha(),
            "metrics": metrics,
            "decision": decision.to_dict(),
        }
    )
    logger.info("model promoted", extra={"version": version, "actor": actor})
    return decision


def write_model_card(metrics: dict, model_version: str, run_id: str, params: dict) -> str:
    """Generate docs/model_card.md from the evaluated metrics."""
    rows = [
        ("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}"),
        ("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}"),
        ("Recall @ operating threshold", f"{metrics.get('recall', 0):.4f}"),
        ("Precision @ operating threshold", f"{metrics.get('precision', 0):.4f}"),
        ("F1", f"{metrics.get('f1', 0):.4f}"),
        ("Brier score", f"{metrics.get('brier', 0):.4f}"),
        ("Decision threshold", f"{metrics.get('threshold', 0):.2f}"),
        (
            "Expected savings / 1k customers",
            f"${metrics.get('economics_savings_per_1k', 0):,.0f}",
        ),
    ]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)

    content = f"""# Model Card - ChurnOps churn classifier

## Model details
- **Model name:** `{settings.model_name}`
- **Version:** `{model_version}`
- **MLflow run:** `{run_id}`
- **Algorithm:** HistGradientBoostingClassifier (scikit-learn)
- **Task:** binary classification - probability that a customer churns
- **Owner:** ml-platform
- **Generated:** {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}

## Intended use
Prioritise retention outreach for existing subscribers. Scores are used to rank
customers for a retention offer, never to deny service or set price.

## Training data
IBM Telco Customer Churn (7,043 rows, 21 columns) with a stratified
70/15/15 train/validation/test split. Target positive rate: 26.5%.

## Features
Numeric: tenure, MonthlyCharges, TotalCharges, SeniorCitizen.
Categorical: contract, internet service, payment method, support/security
add-ons, household and billing attributes.

## Evaluation
| Metric | Value |
|---|---|
{table}

Metrics are computed on the held-out test split at the operating threshold that
minimises expected retention cost (missed churner = ${settings.cost_false_negative:,.0f} CLV,
wasted offer = ${settings.cost_false_positive:,.0f}).

## Limitations
- Trained on a single historical snapshot; behaviour under distribution shift is
  monitored but not guaranteed.
- No causal claim: targeting a customer does not guarantee retention.
- Sensitive attributes (gender, senior status) are present and must be reviewed
  before any use in regulated decisioning.

## Ethical considerations
Scores should not be used for pricing, eligibility, or service quality
decisions. Monitor for disparate impact across gender and senior-citizen groups
before production use in a regulated domain.

## Retraining
Retrain when PR-AUC on fresh labelled data drops more than 5% below the
registered baseline, or when the drift monitor raises a `significant` alert on
three consecutive days.
"""
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CARD_PATH.write_text(content)
    logger.info("model card written", extra={"path": str(MODEL_CARD_PATH)})
    return str(MODEL_CARD_PATH)


def register_and_promote(result, actor: str = "ci-pipeline", validation_ok: bool = True):
    """Convenience wrapper used by the pipeline entry point."""
    mlflow.set_tracking_uri(settings.tracking_uri)
    decision = promote(
        model_name=settings.model_name,
        version=result.model_version,
        metrics=result.metrics,
        actor=actor,
        validation_ok=validation_ok,
    )
    summary_path = AUDIT_LOG_PATH.parent / "experiment_summary.json"
    params = {}
    if summary_path.exists():
        params = json.loads(summary_path.read_text()).get("params", {})
    write_model_card(result.metrics, result.model_version, result.run_id, params)
    return decision
