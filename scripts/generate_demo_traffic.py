#!/usr/bin/env python3
"""Send a realistic mix of customers to the running API.

Used to populate the live dashboard for screenshots and demos.
    python scripts/generate_demo_traffic.py --url http://127.0.0.1:8080 --count 120
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnops.config import RAW_DATA_PATH  # noqa: E402
from churnops.data.features import FEATURE_COLUMNS  # noqa: E402


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    frame = pd.read_csv(RAW_DATA_PATH)
    sample = frame.sample(n=args.count, random_state=args.seed)
    random.seed(args.seed)

    flagged = 0
    for _, row in sample.iterrows():
        payload = {}
        for column in FEATURE_COLUMNS:
            value = row[column]
            if column == "TotalCharges":
                numeric = pd.to_numeric(value, errors="coerce")
                value = float(numeric) if pd.notna(numeric) else None
            elif column in {"tenure", "SeniorCitizen"}:
                value = int(value)
            elif column == "MonthlyCharges":
                value = float(value)
            payload[column] = value

        result = post(f"{args.url}/predict", payload)
        flagged += int(result["decision"])

        # Attach ground truth for a third of the traffic, mimicking a delayed
        # label feed from the CRM.
        if random.random() < 0.33:
            post(
                f"{args.url}/feedback",
                {"request_id": result["request_id"], "actual": int(row["Churn"] == "Yes")},
            )

    print(f"scored {args.count} customers, {flagged} flagged for outreach")


if __name__ == "__main__":
    main()
