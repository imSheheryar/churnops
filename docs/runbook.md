# Runbook

On-call notes for the ChurnOps demo. Each scenario lists the signal you will
see, the first diagnostic command, and the decision to make.

## Drift alert

**Signal:** the scheduled workflow opens an issue labelled `drift`, or the
dashboard pill turns `significant`, or Slack posts a drift message.

1. Open the attached `drift_report.html`. The PSI chart ranks features; the
   verdict table shows PSI, KS statistic and severity per feature.
2. Decide whether the change is real or a data bug:
   - A single feature with PSI > 0.5 and a suspicious mean usually means an
     upstream unit change or a broken join, not genuine behavioural drift.
   - Several behavioural features (tenure, contract, payment method) moving
     together usually means a real population shift.
3. If it is a data bug, fix ingestion and re-run the batch. Do not retrain on
   corrupt data - the contract will likely block you anyway.
4. If it is real drift, retrain on fresh labelled data and let the gates decide:

   ```bash
   churnops pipeline            # trains, evaluates, promotes only if gates pass
   ```

5. If the new model fails the gates, keep the current Production version and
   escalate. A failed gate is a valid outcome, not an incident.

## Model promotion failed

**Signal:** `reports/metrics.json` shows `gate.passed = false`, or CI logs
`promotion blocked`.

1. Read the gate reasons in `artifacts/audit_log.jsonl` (last line).
2. Compare champion and challenger metrics in `reports/experiment_summary.json`.
3. Typical causes: PR-AUC below 0.60 (label shift or too little data), recall
   below 0.65 at the cost-optimal threshold, Brier above 0.20 (poorly calibrated
   probabilities).
4. Fix the cause and re-run; never edit the thresholds to force a promotion
   without recording the decision in an ADR.

## API is unhealthy

```bash
curl -s localhost:8000/health | jq
curl -s localhost:8000/model | jq
docker logs churnops
```

- `model_loaded: false` means the artefact is missing or incompatible. Check that
  `artifacts/model.joblib` exists and that the scikit-learn version in the image
  matches the one used for training (`requirements-serve.txt` pins it).
- 503 on `/predict` with a healthy container usually means a cold start that has
  not finished loading; the readiness probe retries for 60 seconds.
- To recover a bad deployment, redeploy the previous image tag from GHCR.

## Rollback a model

The registry keeps every version, so rollback is a stage transition, not a
retrain:

```bash
mlflow models update-model-version --name churnops-churn-classifier --version 1 --stage Production
curl -X POST localhost:8000/admin/reload
```

For the container deployment, redeploy the previous image tag (`sha-<commit>` or
`vX.Y.Z`) and confirm `/model` reports the expected version.

## Feedback loop looks wrong

**Signal:** dashboard precision collapses or recall stays at 1.0 with precision
below 0.5.

1. Confirm labels are arriving: `/api/overview` -> `serving.labelled_samples`.
2. Check the label distribution in SQLite:

   ```bash
   sqlite3 artifacts/predictions.db \
     "select actual, count(*) from predictions where actual is not null group by actual;"
   ```

3. A recall of 1.0 is expected at the current operating point: the threshold is
   tuned so that a missed churner ($1,200 CLV) costs far more than a wasted offer
   ($50). If the business changes those costs, update
   `CHURNOPS_COST_FALSE_NEGATIVE` / `CHURNOPS_COST_FALSE_POSITIVE` and retrain so
   the threshold is re-derived.

## Prediction store growth

SQLite is fine for a demo but grows with traffic. Check the size and prune:

```bash
sqlite3 artifacts/predictions.db "select count(*) from predictions;"
sqlite3 artifacts/predictions.db "delete from predictions where created_at < date('now','-90 day');"
```

In a real deployment this table belongs in BigQuery or Postgres; the interface in
`monitoring/log_store.py` is intentionally narrow so it can be swapped.

## Scheduled jobs

| Workflow | Schedule | Purpose |
| --- | --- | --- |
| `ci.yml` | every push / PR | lint, tests, smoke pipeline, image build |
| `publish-image.yml` | push to `main`, tags | build and push the image to GHCR |
| `drift-monitor.yml` | Mondays 06:00 UTC | score a drifted batch, open an issue on alert |
| `deploy-hf.yml` | manual | publish to a Hugging Face Space (PRO plan only since Jul 2026) |

## Deploying the live demo

Render builds the service from `render.yaml`, so there is nothing to run by hand:

1. Dashboard -> *New -> Blueprint*, connect the repository, apply the plan.
2. Wait for the first build; Render gates it on `GET /health` returning 200.
3. Note the `https://<service>.onrender.com` URL and set it as the repository
   homepage.

Free instances sleep after fifteen minutes without traffic and take about a
minute to wake, so the first click from a recruiter is slower than the rest. If a
deploy fails, read the build log first: the usual causes are a missing artefact
in the image (`artifacts/model.joblib`) or an import that only exists in the
training extras, which `requirements-serve.txt` deliberately omits.

The Hugging Face path is no longer free: `create_repo` returns `402 Payment
Required` for Docker Spaces on `cpu-basic`. Do not debug it as an application
fault.
