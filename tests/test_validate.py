"""Data contract tests: the gate that keeps bad data out of training."""

from __future__ import annotations

import pandas as pd

from churnops.data.validate import clean_types, validate_dataframe


def _check(report, name: str):
    return next(check for check in report.checks if check.name == name)


def test_valid_frame_passes_critical_checks(churn_frame):
    report = validate_dataframe(churn_frame, min_rows=100)

    assert report.ok, report.failures
    assert _check(report, "required_columns").passed
    assert _check(report, "row_count").passed
    assert _check(report, "unique_customer_id").passed
    assert _check(report, "target_domain").passed
    assert report.rows == len(churn_frame)


def test_missing_column_is_critical(churn_frame):
    broken = churn_frame.drop(columns=["MonthlyCharges"])

    report = validate_dataframe(broken, min_rows=100)

    assert not report.ok
    assert not _check(report, "required_columns").passed
    assert "MonthlyCharges" in _check(report, "required_columns").detail


def test_duplicate_customer_ids_are_rejected(churn_frame):
    broken = churn_frame.copy()
    broken.loc[broken.index[1], "customerID"] = broken.loc[broken.index[0], "customerID"]

    report = validate_dataframe(broken, min_rows=100)

    assert not report.ok
    assert not _check(report, "unique_customer_id").passed


def test_unexpected_target_value_is_rejected(churn_frame):
    broken = churn_frame.copy()
    broken.loc[broken.index[0], "Churn"] = "Maybe"

    report = validate_dataframe(broken, min_rows=100)

    assert not report.ok
    assert "Maybe" in _check(report, "target_domain").detail


def test_small_frame_fails_row_count(churn_frame):
    report = validate_dataframe(churn_frame.head(20), min_rows=100)

    assert not report.ok
    assert not _check(report, "row_count").passed


def test_clean_types_coerces_total_charges():
    frame = pd.DataFrame({"TotalCharges": ["29.85", " ", "1889.5"]})

    cleaned = clean_types(frame)

    assert cleaned["TotalCharges"].isna().sum() == 1
    assert cleaned["TotalCharges"].iloc[0] == 29.85


def test_report_serialises_to_dict(churn_frame):
    payload = validate_dataframe(churn_frame, min_rows=100).to_dict()

    assert payload["ok"] is True
    assert isinstance(payload["checks"], list)
    assert {"name", "passed", "severity", "detail"} == set(payload["checks"][0])
