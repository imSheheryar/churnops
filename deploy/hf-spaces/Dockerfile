# Hugging Face Spaces (Docker SDK) - free CPU tier.
# The Space repository is assembled by scripts/deploy_hf_space.py, which copies
# this file to the repository root together with the code and model artefacts.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURNOPS_HOME=/app \
    PORT=7860 \
    HF_HOME=/tmp/huggingface

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

COPY artifacts/model.joblib artifacts/model_meta.json artifacts/reference_stats.json \
     artifacts/reference_sample.parquet ./artifacts/
COPY reports ./reports

# Spaces runs containers as uid 1000; match it so the SQLite store is writable.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "churnops.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
