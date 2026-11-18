#!/usr/bin/env python3
"""End-to-end training pipeline: ingest -> validate -> split -> train -> promote.

Usage
-----
    python pipelines/run_pipeline.py                 # full run on the real dataset
    python pipelines/run_pipeline.py --sample 1500   # fast smoke run for CI
    python pipelines/run_pipeline.py --no-promote    # train + evaluate only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnops.config import ARTIFACTS_DIR, PIPELINE_RUN_PATH, ensure_directories  # noqa: E402
from churnops.data.features import split_data  # noqa: E402
from churnops.data.ingest import load_dataset  # noqa: E402
from churnops.data.validate import validate_or_raise  # noqa: E402
from churnops.logging_utils import get_logger  # noqa: E402
from churnops.training.registry import register_and_promote  # noqa: E402
from churnops.training.train import train_model  # noqa: E402

logger = get_logger("pipeline")


def run(
    sample: int | None = None,
    force_download: bool = False,
    promote: bool = True,
    actor: str = "ci-pipeline",
) -> dict:
    ensure_directories()
    started = time.time()
    timings: dict[str, float] = {}

    step = time.time()
    frame = load_dataset(force=force_download, sample=sample)
    timings["ingest_s"] = round(time.time() - step, 2)

    step = time.time()
    validation = validate_or_raise(frame)
    timings["validate_s"] = round(time.time() - step, 2)

    step = time.time()
    train_df, valid_df, test_df = split_data(frame)
    timings["split_s"] = round(time.time() - step, 2)

    step = time.time()
    result = train_model(train_df, valid_df, test_df, register=promote)
    timings["train_s"] = round(time.time() - step, 2)

    decision = None
    if promote:
        step = time.time()
        decision = register_and_promote(result, actor=actor, validation_ok=validation.ok)
        timings["promote_s"] = round(time.time() - step, 2)

    summary = {
        "status": "success" if (decision is None or decision.passed) else "gate_failed",
        "started_at": datetime.now(UTC).isoformat(),
        "duration_s": round(time.time() - started, 2),
        "timings": timings,
        "rows": int(len(frame)),
        "validation_ok": validation.ok,
        "validation_failures": [c.name for c in validation.failures],
        "run_id": result.run_id,
        "model_version": result.model_version,
        "threshold": result.threshold,
        "metrics": result.metrics,
        "gate": decision.to_dict() if decision else None,
    }
    PIPELINE_RUN_PATH.write_text(json.dumps(summary, indent=2, default=float))
    (ARTIFACTS_DIR / "pipeline_run.json").write_text(json.dumps(summary, indent=2, default=float))

    logger.info(
        "pipeline finished",
        extra={
            "status": summary["status"],
            "duration_s": summary["duration_s"],
            "pr_auc": result.metrics.get("pr_auc"),
        },
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("status", "duration_s", "run_id", "model_version")}, indent=2
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ChurnOps training pipeline")
    parser.add_argument(
        "--sample", type=int, default=None, help="train on a random sample (CI smoke)"
    )
    parser.add_argument("--force-download", action="store_true", help="re-download the dataset")
    parser.add_argument("--no-promote", action="store_true", help="skip registry promotion")
    parser.add_argument("--actor", default="ci-pipeline", help="who approved the promotion")
    args = parser.parse_args()

    summary = run(
        sample=args.sample,
        force_download=args.force_download,
        promote=not args.no_promote,
        actor=args.actor,
    )
    if summary["status"] != "success":
        logger.error("promotion gate failed", extra={"gate": summary["gate"]})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
