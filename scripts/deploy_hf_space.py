"""Publish the ChurnOps serving stack to a free Hugging Face Space.

Usage
-----
    export HF_TOKEN=hf_xxx
    python scripts/deploy_hf_space.py --space-id your-username/churnops

The script assembles a minimal Space repository (code, serving requirements,
model artefacts, drift reports and the Dockerfile) in a temporary directory and
uploads it. Re-running it updates the Space in place.

Note: since July 2026 Hugging Face requires a PRO plan for Docker Spaces on
``cpu-basic``; without one, ``create_repo`` fails with 402 Payment Required. The
free deployment path is the Render blueprint at the repository root.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "pyproject.toml",
    "requirements-serve.txt",
    "src",
    "artifacts/model.joblib",
    "artifacts/model_meta.json",
    "artifacts/reference_stats.json",
    "artifacts/reference_sample.parquet",
    "reports/drift_report.html",
    "reports/drift_report.json",
    "reports/metrics.json",
]


def stage(target: Path) -> Path:
    """Copy the serving subset of the repository into ``target``."""
    for relative in INCLUDE:
        source = ROOT / relative
        if not source.exists():
            print(f"skip (missing): {relative}")
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
            )
        else:
            shutil.copy2(source, destination)
    shutil.copy2(ROOT / "deploy" / "hf-spaces" / "Dockerfile", target / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "hf-spaces" / "README.md", target / "README.md")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default="", help="Hugging Face repo id, e.g. user/churnops")
    parser.add_argument("--dry-run", action="store_true", help="Stage locally without uploading")
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging = stage(Path(tmp) / "space")
        files = sorted(path.relative_to(staging) for path in staging.rglob("*") if path.is_file())
        print(f"staged {len(files)} files in {staging}")
        for path in files:
            print(f"  {path}")

        if args.dry_run:
            return 0
        if not args.space_id:
            parser.error("--space-id is required unless --dry-run is used")

        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub is not installed: pip install huggingface_hub")
            return 1

        api = HfApi()
        url = api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        print(f"repository ready: {url}")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.space_id,
            repo_type="space",
            commit_message="Deploy ChurnOps serving stack",
        )
        print(f"deployed: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
