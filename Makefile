# ChurnOps developer shortcuts.
# Every target is a thin wrapper around the CLI so that CI and humans run
# exactly the same commands.
PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help install smoke pipeline serve test lint format drift score demo docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install the package with dev extras
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@echo "Activate with: source $(BIN)/activate"

smoke: ## Fast pipeline run on 2k rows without promotion (used by CI)
	$(BIN)/churnops pipeline --sample 2000 --no-promote

pipeline: ## Full training pipeline: validate, train, log, gate, promote
	$(BIN)/churnops pipeline

serve: ## Run the inference API on :8080
	$(BIN)/churnops serve --port 8080

test: ## Run the test suite
	$(BIN)/python -m pytest

lint: ## Lint and check formatting
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Auto-fix lint issues and format code
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

drift: ## Score a batch and write the drift report (default: drifted demo batch)
	$(BIN)/churnops drift --input $${INPUT:-data/sample/drifted_batch.csv}

score: ## Batch-score a CSV without a drift report
	$(BIN)/churnops score --input $${INPUT:-data/sample/baseline_batch.csv}

demo: ## Generate synthetic demo traffic against a running API
	$(BIN)/python scripts/generate_demo_traffic.py --url http://127.0.0.1:8080 --count 300

docker-build: ## Build the inference image
	docker build -f docker/Dockerfile -t churnops:local .

docker-up: ## Start API + Prometheus with docker compose
	docker compose up --build

clean: ## Remove caches and generated artefacts (keeps the model)
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f artifacts/predictions.db artifacts/api.log reports/batch_predictions.csv
