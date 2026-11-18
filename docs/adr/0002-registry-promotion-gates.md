# ADR 0002 - Promotion gates and an append-only audit log

- Status: accepted
- Date: 2026-11-18

## Context

A registry that promotes whatever was trained last is a deployment risk. A model
that regresses on recall, or whose probabilities are badly calibrated, should not
reach production just because the pipeline succeeded.

Gates also need evidence. "The model passed" is not auditable; a record of the
metrics, thresholds and decisions is.

## Decision

Promotion to the `Production` stage requires all of:

| Gate | Threshold | Rationale |
| --- | --- | --- |
| Validation PR-AUC | >= 0.60 | Primary ranking quality on an imbalanced target |
| Recall at operating threshold | >= 0.65 | Retention misses are the expensive error |
| Brier score | <= 0.20 | Probabilities must be usable as probabilities |
| Critical data-contract checks | all pass | No promotion on corrupt data |

Gates are evaluated on validation metrics. Every run - promoted or not - writes:

1. A model card (`docs/model_card.md`) with metrics, intended use, limitations.
2. An append-only JSONL entry (`artifacts/audit_log.jsonl`) containing run id,
   model version, metrics, threshold, gate decisions and timestamp.

The pipeline exits non-zero when a gate fails, so CI surfaces the failure instead
of silently shipping a worse model.

## Consequences

Promotion can fail, and that is a feature: the previous Production version stays
live. Thresholds are configuration (`CHURNOPS_MIN_PR_AUC`, ...), so they can be
changed - but a change requires an ADR, which keeps the decision trail honest.

The audit log is a file, not a database. For a demo that is a deliberate
simplification; in a regulated setting it would be shipped to immutable storage.
