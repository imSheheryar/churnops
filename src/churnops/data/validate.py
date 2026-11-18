"""Data contracts for the churn dataset.

Validation runs before training and before batch scoring. Critical failures
stop the pipeline; warnings are recorded in the report and surfaced on the
dashboard so that silent data corruption cannot reach a model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from churnops.config import REPORTS_DIR, settings
from churnops.data.ingest import REQUIRED_COLUMNS
from churnops.logging_utils import get_logger

logger = get_logger(__name__)

CATEGORICAL_DOMAINS: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


@dataclass
class Check:
    name: str
    passed: bool
    severity: str
    detail: str


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    rows: int = 0
    columns: int = 0
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks if check.severity == "critical")

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def add(self, name: str, passed: bool, detail: str, severity: str = "critical") -> None:
        self.checks.append(Check(name=name, passed=passed, severity=severity, detail=detail))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "columns": self.columns,
            "generated_at": self.generated_at,
            "checks": [asdict(check) for check in self.checks],
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or REPORTS_DIR / "validation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def clean_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges to numeric (the raw file contains blank strings)."""
    cleaned = frame.copy()
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    return cleaned


def validate_dataframe(frame: pd.DataFrame, min_rows: int = 500) -> ValidationReport:
    report = ValidationReport(rows=len(frame), columns=len(frame.columns))

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    report.add(
        "required_columns",
        not missing,
        "all required columns present" if not missing else f"missing: {missing}",
    )

    report.add(
        "row_count",
        len(frame) >= min_rows,
        f"{len(frame)} rows (minimum {min_rows})",
    )

    if settings.id_column in frame.columns:
        duplicates = int(frame[settings.id_column].duplicated().sum())
        report.add(
            "unique_customer_id",
            duplicates == 0,
            f"{duplicates} duplicate customer ids",
        )

    if settings.target_column in frame.columns:
        target = frame[settings.target_column]
        unexpected = sorted(set(target.dropna().unique()) - {settings.positive_label, "No"})
        report.add(
            "target_domain",
            not unexpected,
            f"unexpected target values: {unexpected}" if unexpected else "target values valid",
        )
        rate = float((target == settings.positive_label).mean())
        report.add(
            "target_balance",
            0.03 <= rate <= 0.60,
            f"positive rate {rate:.3f}",
            severity="warning",
        )

    if {"tenure", "MonthlyCharges"} <= set(frame.columns):
        tenure_bad = int(((frame["tenure"] < 0) | (frame["tenure"] > 100)).sum())
        charges_bad = int((frame["MonthlyCharges"] <= 0).sum())
        report.add(
            "numeric_ranges",
            tenure_bad == 0 and charges_bad == 0,
            f"out-of-range tenure={tenure_bad}, monthly_charges={charges_bad}",
        )

    if "TotalCharges" in frame.columns:
        coerced = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        bad_share = float(coerced.isna().mean())
        report.add(
            "total_charges_numeric",
            bad_share <= 0.05,
            f"{bad_share:.2%} of TotalCharges values are not numeric",
            severity="warning",
        )

    null_share = frame.isna().mean().sort_values(ascending=False)
    worst_column = str(null_share.index[0]) if len(null_share) else "n/a"
    worst_share = float(null_share.iloc[0]) if len(null_share) else 0.0
    report.add(
        "null_share",
        worst_share <= 0.20,
        f"highest null share {worst_share:.2%} in '{worst_column}'",
    )

    domain_violations: list[str] = []
    for column, allowed in CATEGORICAL_DOMAINS.items():
        if column not in frame.columns:
            continue
        observed = set(frame[column].dropna().unique())
        unexpected = observed - allowed
        if unexpected:
            domain_violations.append(f"{column}: {sorted(unexpected)}")
    report.add(
        "categorical_domains",
        not domain_violations,
        "; ".join(domain_violations) if domain_violations else "all categorical values known",
        severity="warning",
    )

    duplicate_rows = int(frame.duplicated().sum())
    report.add(
        "duplicate_rows",
        duplicate_rows / max(len(frame), 1) <= 0.01,
        f"{duplicate_rows} duplicate rows",
        severity="warning",
    )

    return report


def validate_or_raise(frame: pd.DataFrame) -> ValidationReport:
    report = validate_dataframe(frame)
    path = report.save()
    if not report.ok:
        failed = [check.name for check in report.checks if not check.passed]
        raise ValueError(f"data validation failed: {failed} (report: {path})")
    logger.info(
        "data validation passed",
        extra={"checks": len(report.checks), "warnings": len(report.failures)},
    )
    return report
