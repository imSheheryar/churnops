# ADR 0003 - Cost-sensitive decision threshold

- Status: accepted
- Date: 2026-11-18

## Context

The default 0.5 threshold assumes a false positive and a false negative cost the
same. In retention they do not: a missed churner loses the customer's remaining
lifetime value, while a wasted retention offer costs a discount or a call.

With a 26.5% positive rate, accuracy is also misleading - predicting "nobody
churns" scores 73.5%.

## Decision

Choose the operating threshold by minimising expected cost on the validation
split, with costs supplied as configuration:

- False negative (missed churner): **$1,200** expected CLV.
- False positive (wasted offer): **$50** cost of the offer.

The threshold search is exhaustive over the observed probability grid, so the
result is reproducible and auditable. The chosen threshold is persisted with the
model (`model_meta.json`) and used by the API; it is not hard-coded in the
serving layer.

## Consequences

The selected threshold is **0.14**, which yields recall 0.957 and precision 0.369
on the test split. The model flags aggressively by design. Expected cost falls
from $319,016 to $35,383 per 1,000 customers, a saving of roughly **$283,633
(88.9%)** against the "contact nobody" baseline.

Reviewers sometimes read precision 0.37 as a weak model. The dashboard and model
card therefore show precision and recall together, with the cost assumption
next to them, and the runbook explains when a recall of ~1.0 is expected rather
than suspicious.

If the business changes the cost ratio, the threshold must be re-derived by
retraining, not by editing a constant in the API.
