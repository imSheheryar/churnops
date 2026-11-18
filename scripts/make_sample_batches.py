#!/usr/bin/env python3
"""Create demo batches: one representative, one deliberately drifted.

Used by the README screenshots and by `make demo-drift`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnops.config import RAW_DATA_PATH, SAMPLE_DIR  # noqa: E402
from churnops.monitoring.drift import simulate_drift  # noqa: E402


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RAW_DATA_PATH)
    batch = frame.sample(n=800, random_state=11).reset_index(drop=True)

    stable_path = SAMPLE_DIR / "baseline_batch.csv"
    batch.to_csv(stable_path, index=False)

    drifted_path = SAMPLE_DIR / "drifted_batch.csv"
    simulate_drift(batch, severity="high").to_csv(drifted_path, index=False)

    print(f"stable batch:  {stable_path} ({len(batch)} rows)")
    print(f"drifted batch: {drifted_path} ({len(batch)} rows)")


if __name__ == "__main__":
    main()
