# Model Card - ChurnOps churn classifier

## Model details
- **Model name:** `churnops-churn-classifier`
- **Version:** `2`
- **MLflow run:** `75234014a9274cf5891478656cc338e5`
- **Algorithm:** HistGradientBoostingClassifier (scikit-learn)
- **Task:** binary classification - probability that a customer churns
- **Owner:** ml-platform
- **Generated:** 2026-11-18 17:32 UTC

## Intended use
Prioritise retention outreach for existing subscribers. Scores are used to rank
customers for a retention offer, never to deny service or set price.

## Training data
IBM Telco Customer Churn (7,043 rows, 21 columns) with a stratified
70/15/15 train/validation/test split. Target positive rate: 26.5%.

## Features
Numeric: tenure, MonthlyCharges, TotalCharges, SeniorCitizen.
Categorical: contract, internet service, payment method, support/security
add-ons, household and billing attributes.

## Evaluation
| Metric | Value |
|---|---|
| ROC-AUC | 0.8394 |
| PR-AUC | 0.6616 |
| Recall @ operating threshold | 0.9573 |
| Precision @ operating threshold | 0.3690 |
| F1 | 0.5327 |
| Brier score | 0.1584 |
| Decision threshold | 0.14 |
| Expected savings / 1k customers | $283,633 |

Metrics are computed on the held-out test split at the operating threshold that
minimises expected retention cost (missed churner = $1,200 CLV,
wasted offer = $50).

## Limitations
- Trained on a single historical snapshot; behaviour under distribution shift is
  monitored but not guaranteed.
- No causal claim: targeting a customer does not guarantee retention.
- Sensitive attributes (gender, senior status) are present and must be reviewed
  before any use in regulated decisioning.

## Ethical considerations
Scores should not be used for pricing, eligibility, or service quality
decisions. Monitor for disparate impact across gender and senior-citizen groups
before production use in a regulated domain.

## Retraining
Retrain when PR-AUC on fresh labelled data drops more than 5% below the
registered baseline, or when the drift monitor raises a `significant` alert on
three consecutive days.
