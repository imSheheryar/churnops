"""ChurnOps inference API.

Endpoints
---------
GET  /health           liveness + model readiness
GET  /model            deployed model metadata
POST /predict          single customer score
POST /predict/batch    up to 1,000 customers
POST /feedback         attach the ground-truth label to a scored request
GET  /metrics          Prometheus exposition
GET  /api/overview     JSON used by the dashboard
GET  /dashboard        live HTML dashboard
POST /admin/reload     hot-reload the model after a promotion
"""

from __future__ import annotations

import json
import math
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from churnops.config import REPORTS_DIR, settings
from churnops.logging_utils import get_logger
from churnops.monitoring import log_store
from churnops.serving.model_store import model_store
from churnops.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    FeedbackRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = get_logger(__name__)
START_TIME = time.time()

PREDICTION_COUNTER = Counter(
    "churnops_predictions_total", "Number of scored customers", ["decision"]
)
PREDICTION_LATENCY = Histogram(
    "churnops_prediction_latency_seconds",
    "Time spent scoring a request",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
SCORE_HISTOGRAM = Histogram(
    "churnops_churn_probability",
    "Distribution of predicted churn probability",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
REQUEST_COUNTER = Counter(
    "churnops_http_requests_total", "HTTP requests", ["method", "path", "status"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store.load()
    log_store.init_db()
    logger.info("service started", extra={"model_loaded": model_store.bundle is not None})
    yield
    logger.info("service stopped")


app = FastAPI(
    title="ChurnOps Inference API",
    description=(
        "Production-style churn scoring service with validation, prediction logging, "
        "Prometheus metrics and drift reporting."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _sanitize(value):
    """Make validation details JSON-safe (NaN/Infinity are not valid JSON)."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _sanitize(exc.errors())})


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{elapsed_ms:.2f}"
    REQUEST_COUNTER.labels(request.method, request.url.path, str(response.status_code)).inc()
    return response


def _risk_band(score: float) -> str:
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"


def _score_one(customer: CustomerFeatures) -> PredictionResponse:
    bundle = model_store.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    payload = customer.model_dump()
    started = time.perf_counter()
    frame = pd.DataFrame([payload])
    probability = float(bundle.model.predict_proba(frame)[0][1])
    latency_ms = (time.perf_counter() - started) * 1000
    decision = probability >= bundle.threshold

    request_id = log_store.log_prediction(
        score=probability,
        features=payload,
        model_version=bundle.version,
        threshold=bundle.threshold,
        decision=int(decision),
        latency_ms=latency_ms,
    )

    PREDICTION_COUNTER.labels("flagged" if decision else "clear").inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    SCORE_HISTOGRAM.observe(probability)

    return PredictionResponse(
        request_id=request_id,
        churn_probability=round(probability, 4),
        decision=decision,
        threshold=round(bundle.threshold, 4),
        risk_band=_risk_band(probability),
        model_version=bundle.version,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    bundle = model_store.bundle
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/ready", tags=["ops"])
def ready() -> Response:
    if model_store.bundle is None:
        return Response(
            content=json.dumps({"ready": False}), status_code=503, media_type="application/json"
        )
    return Response(content=json.dumps({"ready": True}), media_type="application/json")


@app.get("/model", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    bundle = model_store.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return ModelInfoResponse(
        model_name=settings.model_name,
        model_version=bundle.version,
        stage=bundle.stage,
        threshold=bundle.threshold,
        trained_at=bundle.trained_at,
        features=bundle.feature_names or [],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    return _score_one(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    bundle = model_store.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()
    frame = pd.DataFrame([customer.model_dump() for customer in request.customers])
    probabilities = bundle.model.predict_proba(frame)[:, 1]
    latency_ms = (time.perf_counter() - started) * 1000

    predictions: list[PredictionResponse] = []
    for customer, probability in zip(request.customers, probabilities, strict=True):
        probability = float(probability)
        decision = probability >= bundle.threshold
        request_id = log_store.log_prediction(
            score=probability,
            features=customer.model_dump(),
            model_version=bundle.version,
            threshold=bundle.threshold,
            decision=int(decision),
            latency_ms=latency_ms / max(len(request.customers), 1),
        )
        PREDICTION_COUNTER.labels("flagged" if decision else "clear").inc()
        SCORE_HISTOGRAM.observe(probability)
        predictions.append(
            PredictionResponse(
                request_id=request_id,
                churn_probability=round(probability, 4),
                decision=decision,
                threshold=round(bundle.threshold, 4),
                risk_band=_risk_band(probability),
                model_version=bundle.version,
                latency_ms=round(latency_ms, 2),
            )
        )

    return BatchPredictionResponse(
        count=len(predictions),
        model_version=bundle.version,
        mean_probability=round(
            float(sum(p.churn_probability for p in predictions) / len(predictions)), 4
        ),
        flagged=sum(1 for p in predictions if p.decision),
        predictions=predictions,
    )


@app.post("/feedback", tags=["monitoring"])
def feedback(payload: FeedbackRequest) -> dict:
    updated = log_store.set_actual(payload.request_id, payload.actual)
    if not updated:
        raise HTTPException(status_code=404, detail="request_id not found")
    return {"status": "recorded", "request_id": payload.request_id}


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/reload", tags=["ops"])
def reload_model() -> dict:
    bundle = model_store.load(force=True)
    if bundle is None:
        raise HTTPException(status_code=503, detail="reload failed: no model artefact")
    return {"status": "reloaded", "version": bundle.version, "threshold": bundle.threshold}


def _drift_summary() -> dict:
    path = REPORTS_DIR / "drift_report.json"
    if not path.exists():
        return {"status": "unknown", "alerting_features": []}
    report = json.loads(path.read_text())
    return {
        "status": report.get("status", "unknown"),
        "drift_share": report.get("drift_share"),
        "alerting_features": report.get("alerting_features", []),
        "checked_at": report.get("checked_at"),
    }


def _economics_summary() -> dict:
    """Cost assumptions and the savings measured on the last evaluated model."""
    path = REPORTS_DIR / "metrics.json"
    test = json.loads(path.read_text()).get("test", {}) if path.exists() else {}
    return {
        "cost_false_negative": settings.cost_false_negative,
        "cost_false_positive": settings.cost_false_positive,
        "savings_per_1000": test.get("economics_savings_per_1k"),
        "baseline_cost_per_1000": test.get("economics_baseline_cost_per_1k"),
    }


@app.get("/api/overview", tags=["dashboard"])
def overview(window_minutes: int = 60) -> dict:
    bundle = model_store.bundle
    return {
        "model": {
            "name": settings.model_name,
            "version": bundle.version if bundle else None,
            "stage": bundle.stage if bundle else None,
            "threshold": bundle.threshold if bundle else None,
        },
        "serving": log_store.stats(window_minutes=window_minutes),
        "drift": _drift_summary(),
        "economics": _economics_summary(),
        "recent": log_store.recent(limit=10),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>ChurnOps - live dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 :root { --ink:#1f2933; --muted:#6b7280; --line:#e5e7eb; --green:#2f6f4e; --amber:#b7791f; --red:#b42318; }
 * { box-sizing: border-box; }
 body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f6f7f9; color:var(--ink); }
 header { background:#12263a; color:#fff; padding:18px 28px; display:flex; justify-content:space-between; align-items:center; }
 header h1 { font-size:18px; margin:0; letter-spacing:.02em; }
 header .meta { font-size:12px; color:#c9d6e3; }
 .wrap { max-width:1180px; margin:0 auto; padding:24px; }
 .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:16px; }
 .card { background:#fff; border-radius:12px; padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,.07); }
 .card .label { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
 .card .value { font-size:26px; font-weight:700; margin-top:6px; }
 .card .hint { font-size:12px; color:var(--muted); margin-top:4px; }
 .grid2 { display:grid; grid-template-columns:1.3fr 1fr; gap:16px; margin-top:16px; }
 @media (max-width:900px){ .grid2 { grid-template-columns:1fr; } }
 .panel { background:#fff; border-radius:12px; padding:18px 20px; box-shadow:0 1px 3px rgba(0,0,0,.07); }
 .panel h2 { font-size:14px; margin:0 0 14px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
 .pill { display:inline-block; padding:4px 12px; border-radius:999px; color:#fff; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
 .pill.stable { background:var(--green); } .pill.warning { background:var(--amber); } .pill.significant { background:var(--red); } .pill.unknown { background:#6b7280; }
 table { width:100%; border-collapse:collapse; font-size:13px; }
 th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); padding:8px 6px; border-bottom:1px solid var(--line); }
 td { padding:8px 6px; border-bottom:1px solid #f3f4f6; }
 code { background:#eef2f7; padding:2px 6px; border-radius:6px; font-size:12px; }
 .flag { color:var(--red); font-weight:600; }
 .clear { color:var(--green); }
 footer { text-align:center; color:var(--muted); font-size:12px; padding:24px 0 40px; }
</style></head>
<body>
<header>
  <h1>ChurnOps &middot; live inference dashboard</h1>
  <div class="meta" id="header-meta">connecting…</div>
</header>
<div class="wrap">
  <div class="cards">
    <div class="card"><div class="label">Scored (60 min)</div><div class="value" id="kpi-requests">–</div><div class="hint">requests through /predict</div></div>
    <div class="card"><div class="label">Mean churn score</div><div class="value" id="kpi-score">–</div><div class="hint">probability, 0–1</div></div>
    <div class="card"><div class="label">Mean latency</div><div class="value" id="kpi-latency">–</div><div class="hint">milliseconds</div></div>
    <div class="card"><div class="label">Flagged for outreach</div><div class="value" id="kpi-flagged">–</div><div class="hint">above decision threshold</div></div>
    <div class="card"><div class="label">Drift status</div><div class="value"><span id="kpi-drift" class="pill unknown">unknown</span></div><div class="hint" id="drift-hint">awaiting report</div></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Score distribution</h2>
      <canvas id="scoreChart" height="260"></canvas>
    </div>
    <div class="panel">
      <h2>Model</h2>
      <table>
        <tr><th>Name</th><td><code id="m-name">–</code></td></tr>
        <tr><th>Version</th><td id="m-version">–</td></tr>
        <tr><th>Stage</th><td id="m-stage">–</td></tr>
        <tr><th>Threshold</th><td id="m-threshold">–</td></tr>
        <tr><th>Labelled feedback</th><td id="m-labelled">–</td></tr>
        <tr><th>Precision (labelled)</th><td id="m-precision">–</td></tr>
        <tr><th>Recall (labelled)</th><td id="m-recall">–</td></tr>
        <tr><th>Cost / missed churner</th><td id="m-cost-fn">–</td></tr>
        <tr><th>Cost / wasted offer</th><td id="m-cost-fp">–</td></tr>
        <tr><th>Expected savings / 1k</th><td id="m-savings">–</td></tr>
      </table>
      <div class="hint" style="font-size:11px;color:#6b7280;margin-top:10px;line-height:1.5">
        Labelled outcomes arrive through <code>/feedback</code>. Precision and recall are measured
        at the recall-optimised threshold, where accuracy is not the target metric.
      </div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px">
    <h2>Recent predictions</h2>
    <table id="recent"><thead><tr><th>Time (UTC)</th><th>Request</th><th>Score</th><th>Decision</th><th>Latency</th><th>Model</th></tr></thead><tbody></tbody></table>
  </div>
</div>
<footer>ChurnOps &middot; FastAPI + MLflow + Prometheus &middot; <a href="/docs">API docs</a> &middot; <a href="/metrics">Prometheus metrics</a></footer>
<script>
let chart;
async function refresh() {
  const res = await fetch('/api/overview');
  const data = await res.json();
  const s = data.serving, m = data.model, d = data.drift;

  document.getElementById('kpi-requests').textContent = s.requests ?? 0;
  document.getElementById('kpi-score').textContent = s.avg_score != null ? s.avg_score.toFixed(3) : '–';
  document.getElementById('kpi-latency').textContent = s.avg_latency_ms != null ? s.avg_latency_ms.toFixed(1) + ' ms' : '–';
  document.getElementById('kpi-flagged').textContent = s.flagged ?? 0;

  const pill = document.getElementById('kpi-drift');
  pill.textContent = d.status;
  pill.className = 'pill ' + d.status;
  document.getElementById('drift-hint').textContent = d.alerting_features && d.alerting_features.length
    ? d.alerting_features.length + ' feature(s) in alert' : 'all features within tolerance';

  document.getElementById('m-name').textContent = m.name;
  document.getElementById('m-version').textContent = m.version ?? 'not loaded';
  document.getElementById('m-stage').textContent = m.stage ?? '–';
  document.getElementById('m-threshold').textContent = m.threshold != null ? m.threshold.toFixed(2) : '–';
  document.getElementById('m-labelled').textContent = s.labelled_samples ?? 0;
  document.getElementById('m-precision').textContent = s.precision != null ? (s.precision * 100).toFixed(1) + '%' : 'n/a';
  document.getElementById('m-recall').textContent = s.recall != null ? (s.recall * 100).toFixed(1) + '%' : 'n/a';
  const e = data.economics || {};
  const money = v => v == null ? 'n/a' : '$' + Math.round(v).toLocaleString('en-US');
  document.getElementById('m-cost-fn').textContent = money(e.cost_false_negative);
  document.getElementById('m-cost-fp').textContent = money(e.cost_false_positive);
  document.getElementById('m-savings').textContent = money(e.savings_per_1000);
  document.getElementById('header-meta').textContent = 'model v' + (m.version ?? '?') + ' · updated ' + data.generated_at;

  const labels = Array.from({length: 10}, (_, i) => (i/10).toFixed(1));
  const counts = new Array(10).fill(0);
  (s.score_histogram || []).forEach(b => { if (b.bucket >= 0 && b.bucket < 10) counts[b.bucket] = b.count; });
  if (chart) { chart.data.datasets[0].data = counts; chart.update(); }
  else {
    chart = new Chart(document.getElementById('scoreChart'), {
      type: 'bar',
      data: { labels, datasets: [{ label: 'customers', data: counts, backgroundColor: '#355d9c', borderRadius: 4 }] },
      options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'churn probability' } }, y: { beginAtZero: true } } }
    });
  }

  const tbody = document.querySelector('#recent tbody');
  tbody.innerHTML = (data.recent || []).map(r => `<tr>
     <td>${(r.created_at || '').slice(11, 19)}</td>
     <td><code>${(r.request_id || '').slice(0, 8)}</code></td>
     <td>${Number(r.score).toFixed(3)}</td>
     <td class="${r.decision ? 'flag' : 'clear'}">${r.decision ? 'outreach' : 'no action'}</td>
     <td>${Number(r.latency_ms).toFixed(1)} ms</td>
     <td>v${r.model_version}</td></tr>`).join('');
}
refresh(); setInterval(refresh, 5000);
</script>
</body></html>"""


@app.get("/dashboard", response_class=HTMLResponse, tags=["dashboard"])
def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(
        "<h1>ChurnOps</h1><p>Inference API is running.</p>"
        "<ul><li><a href='/docs'>API docs</a></li><li><a href='/dashboard'>Live dashboard</a></li>"
        "<li><a href='/metrics'>Prometheus metrics</a></li></ul>"
    )


def _report_path() -> Path:
    return REPORTS_DIR / "drift_report.html"
