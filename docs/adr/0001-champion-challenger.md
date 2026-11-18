# ADR 0001 - Champion/challenger training with validation-based selection

- Status: accepted
- Date: 2026-11-18

## Context

The platform needs a model that is good enough to demo, cheap to retrain in CI,
and explainable to a reviewer. Two families are plausible for tabular churn:
gradient-boosted trees, which usually win on accuracy, and logistic regression,
which is linear, fast and easy to defend to a risk committee.

Picking a single algorithm hides the trade-off and gives reviewers no evidence
that alternatives were considered.

## Decision

Train both models on identical folds in every pipeline run:

- **Champion:** `HistGradientBoostingClassifier` (handles non-linearity and
  interactions, native categorical support not required because the shared
  `ColumnTransformer` one-hot encodes).
- **Challenger:** `LogisticRegression` inside a pipeline with median imputation,
  standard scaling and one-hot encoding.

The champion is selected on **validation PR-AUC**, never on test metrics. The
test split is evaluated once, for the record. Both models are logged to MLflow
with their metrics and curves, so the comparison is reproducible.

## Consequences

On the Telco dataset the champion reaches a test PR-AUC of 0.6616 and ROC-AUC of
0.8394; the challenger reaches 0.6579 and 0.8446. The challenger is competitive
and better on ROC-AUC, but the champion wins on the metric that matters for an
imbalanced retention problem (PR-AUC) and is therefore promoted. The gap is
small, which is exactly the point: the decision is documented, not asserted.

Cost: roughly 9 seconds of extra training per run. Acceptable in CI and locally.
