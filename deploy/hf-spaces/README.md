---
title: ChurnOps
emoji: "\U0001F504"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Churn prediction API with drift monitoring and a live dashboard
---

# ChurnOps - live demo

Customer-churn inference API built with FastAPI, scikit-learn and MLflow. This
Space runs the serving container only; the full pipeline (validation, training,
model registry gates, drift reports) lives in the GitHub repository.

- `/dashboard` - live inference dashboard (latency, score mix, drift status)
- `/docs` - OpenAPI documentation
- `/health`, `/model`, `/predict`, `/predict/batch`, `/feedback`, `/metrics`

The model is a histogram gradient boosting classifier trained on the IBM Telco
churn dataset. The decision threshold is cost-sensitive (a missed churner is
assumed to cost 24x a wasted retention offer), so the service deliberately
trades precision for recall.
