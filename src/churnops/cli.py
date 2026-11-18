"""ChurnOps command line interface.

churnops pipeline --sample 1500     # train + evaluate + promote
churnops serve --port 8080          # run the inference API
churnops drift --input data/sample/new_batch.csv
churnops score --input data/sample/new_batch.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from churnops.config import ROOT


def _pipeline(args: argparse.Namespace) -> int:
    command = [sys.executable, str(ROOT / "pipelines" / "run_pipeline.py")]
    if args.sample:
        command += ["--sample", str(args.sample)]
    if args.no_promote:
        command += ["--no-promote"]
    if args.force_download:
        command += ["--force-download"]
    return subprocess.call(command)


def _serve(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "churnops.serving.app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        command.append("--reload")
    return subprocess.call(command)


def _drift(args: argparse.Namespace) -> int:
    import pandas as pd

    from churnops.monitoring.report import generate_drift_report, notify

    frame = pd.read_csv(args.input)
    path, report = generate_drift_report(frame)
    notify(report)
    print(json.dumps({"report": str(path), "status": report["status"]}, indent=2))
    return 0 if report["status"] != "significant" else 2


def _score(args: argparse.Namespace) -> int:
    from churnops.monitoring.batch import score_batch

    print(json.dumps(score_batch(args.input), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churnops", description="ChurnOps MLOps platform")
    sub = parser.add_subparsers(dest="command", required=True)

    pipeline = sub.add_parser("pipeline", help="run the training pipeline")
    pipeline.add_argument("--sample", type=int, default=None)
    pipeline.add_argument("--no-promote", action="store_true")
    pipeline.add_argument("--force-download", action="store_true")
    pipeline.set_defaults(func=_pipeline)

    serve = sub.add_parser("serve", help="start the inference API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    drift = sub.add_parser("drift", help="generate a drift report for a CSV batch")
    drift.add_argument("--input", required=True, type=Path)
    drift.set_defaults(func=_drift)

    score = sub.add_parser("score", help="score a CSV batch and log predictions")
    score.add_argument("--input", required=True, type=Path)
    score.set_defaults(func=_score)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
