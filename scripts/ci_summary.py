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
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main() -> int:
    metrics = _load(ROOT / "reports" / "metrics.json")
    validation = _load(ROOT / "reports" / "validation_report.json")
    drift = _load(ROOT / "reports" / "drift_report.json")

    champion = metrics.get("champion", {})
    challenger = metrics.get("challenger", {})
    rows = [
        ("Champion", champion.get("model", "n/a")),
        ("PR-AUC", f"{champion.get('pr_auc', 0):.4f}"),
        ("ROC-AUC", f"{champion.get('roc_auc', 0):.4f}"),
        ("Recall", f"{champion.get('recall', 0):.3f}"),
        ("Precision", f"{champion.get('precision', 0):.3f}"),
        ("Decision threshold", f"{metrics.get('threshold', 0):.3f}"),
        ("Expected savings / 1k", f"${metrics.get('expected_savings_per_1000', 0):,.0f}"),
        (
            "Validation checks passed",
            f"{sum(c['passed'] for c in validation.get('checks', []))}/{len(validation.get('checks', []))}",
        ),
        ("Drift status", drift.get("status", "not run")),
    ]

    lines = ["## ChurnOps pipeline summary", "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    if challenger:
        lines += [
            "",
            f"Challenger `{challenger.get('model', 'n/a')}`: "
            f"PR-AUC {challenger.get('pr_auc', 0):.4f}, ROC-AUC {challenger.get('roc_auc', 0):.4f}.",
        ]
    body = "\n".join(lines) + "\n"

    print(body)
    if SUMMARY_PATH:
        with open(SUMMARY_PATH, "a", encoding="utf-8") as handle:
            handle.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
