# syntax=docker/dockerfile:1
# ChurnOps inference image.
# Build from the repository root:  docker build -f docker/Dockerfile -t churnops:local .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHURNOPS_HOME=/app \
    PORT=8000

WORKDIR /app

# Dependencies first so that code edits do not invalidate the wheel cache.
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# The package itself (no training extras).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

# Model artefacts promoted by the pipeline. Small enough to bake into the image,
# which keeps the container self-contained and makes cold starts fast.
COPY artifacts/model.joblib artifacts/model_meta.json artifacts/reference_stats.json \
     artifacts/reference_sample.parquet ./artifacts/
COPY reports ./reports
COPY docs ./docs

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health').read()"

CMD ["sh", "-c", "uvicorn churnops.serving.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
