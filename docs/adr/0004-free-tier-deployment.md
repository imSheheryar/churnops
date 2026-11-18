# ADR 0004 - Free-tier deployment: Hugging Face Spaces first, Cloud Run optional

- Status: accepted
- Date: 2026-11-18

## Context

The project has to be publicly demonstrable without a paid cloud subscription.
The constraint is a hard requirement: no credit card, no trial credit that
expires, no "free for 90 days" trap.

Candidates considered:

| Option | Cost | Always-on | Container | Notes |
| --- | --- | --- | --- | --- |
| Hugging Face Spaces (Docker) | free CPU tier | sleeps when idle, wakes on request | yes | 16 GB RAM, public URL, no card |
| Render / Railway free tiers | free with limits | sleeps | yes | card often required, monthly hour caps |
| Fly.io | pay-as-you-go, small free allowance | yes | yes | card required |
| GCP Cloud Run | free tier | scale to zero | yes | billing account required |
| Streamlit Community Cloud | free | sleeps | no | not a container, different runtime |
| GitHub Codespaces | free monthly hours | no | dev only | not a public URL |

## Decision

Ship three deployment paths, in order of cost:

1. **Hugging Face Spaces (default).** The Space runs the same `docker/Dockerfile`
   contract with `app_port: 7860`. `scripts/deploy_hf_space.py` stages the code,
   model artefacts and reports, then uploads them; the GitHub workflow
   `deploy-hf.yml` wraps it behind a manual dispatch.
2. **GHCR image.** `publish-image.yml` builds and pushes the image on every push
   to `main`, tagged with branch, semver and commit SHA. Anyone can run it:
   `docker run -p 8000:8000 ghcr.io/<owner>/churnops:latest`.
3. **GCP Cloud Run (optional).** Terraform in `infra/terraform` provisions
   Artifact Registry, a runtime service account and a scale-to-zero Cloud Run
   service with `max_instances = 2`. Documented as optional because it requires a
   billing account, even though the workload fits in the free tier.

## Consequences

The live demo is free forever, at the cost of cold starts (the Space sleeps) and
ephemeral disk: the SQLite prediction store resets when the Space restarts. The
dashboard therefore treats traffic as illustrative, and the committed drift
report supplies the monitoring story.

Keeping one image for all three targets avoids "works on my laptop" drift. The
only environment-specific knob is `PORT` (8000 locally, 7860 on Spaces).

Cloud Run remains the migration path for anyone with a billing account; the
Terraform is written to be applied and destroyed without leaving paid resources
behind.
