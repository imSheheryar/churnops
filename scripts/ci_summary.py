"""Render pipeline metrics as a GitHub Actions job summary.

The script is deliberately dependency-free (stdlib only) so that it can run
even when the training environment failed to install.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = os.environ.get("GITHUB_STEP_SUMMARY")


def _load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> int:
    metrics = _load(ROOT / "reports" / "metrics.json")
    validation = _load(ROOT / "reports" / "validation_report.json")
    drift = _load(ROOT / "reports" / "drift_report.json")

    test = metrics.get("test") or {}
    challengers = metrics.get("challengers") or {}
    challenger_name, challenger = next(iter(challengers.items()), ("n/a", {}))
    checks = validation.get("checks") or []
    passed = sum(1 for check in checks if check.get("passed"))

    rows = [
        ("Champion", str(metrics.get("champion", "n/a"))),
        ("PR-AUC", _fmt(test.get("pr_auc"))),
        ("ROC-AUC", _fmt(test.get("roc_auc"))),
        ("Recall", _fmt(test.get("recall"), 3)),
        ("Precision", _fmt(test.get("precision"), 3)),
        ("Decision threshold", _fmt(test.get("threshold"), 2)),
        ("Expected savings / 1k", f"${float(test.get('economics_savings_per_1k') or 0):,.0f}"),
        ("Validation checks passed", f"{passed}/{len(checks)}"),
        ("Latest drift report", str(drift.get("status", "not run"))),
    ]

    lines = ["## ChurnOps pipeline summary", "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    if challenger:
        lines += [
            "",
            f"Challenger `{challenger_name}`: PR-AUC {_fmt(challenger.get('pr_auc'))}, "
            f"ROC-AUC {_fmt(challenger.get('roc_auc'))}, recall {_fmt(challenger.get('recall'), 3)}.",
        ]
    body = "\n".join(lines) + "\n"

    print(body)
    if SUMMARY_PATH:
        with open(SUMMARY_PATH, "a", encoding="utf-8") as handle:
            handle.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
