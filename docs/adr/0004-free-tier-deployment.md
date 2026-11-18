# ADR 0004 - Free-tier deployment: Render first, Hugging Face and Cloud Run optional

- Status: accepted
- Date: 2026-11-18
- Supersedes: the Hugging Face-first draft of this ADR (same date)

## Context

The project has to be publicly demonstrable without a paid cloud subscription.
The constraint is a hard requirement: no credit card, no trial credit that
expires, no "free for 90 days" trap.

Hugging Face Spaces was the original choice: a Docker Space on `cpu-basic`
needed no card. That changed in July 2026 - the API now answers
`402 Payment Required` with "hosting Gradio and Docker Spaces on free cpu-basic
requires a PRO subscription". Static Spaces remain free, but a static page
cannot serve `/predict`. The decision was therefore revisited before the first
deploy, not after.

Candidates considered:

| Option | Cost | Always-on | Container | Notes |
| --- | --- | --- | --- | --- |
| Render free web service | $0 | sleeps after 15 min idle | yes | 512 MB RAM, 0.1 CPU, 750 instance-hours/month, no card |
| Hugging Face Spaces (Docker) | PRO plan required since Jul 2026 | sleeps when idle | yes | was free, now `402` on free hardware |
| Koyeb free instance | $0 | scales to zero | yes | card with a $29 pre-authorisation required since Feb 2026 |
| Fly.io | pay-as-you-go | yes | yes | card required |
| GCP Cloud Run | free tier | scale to zero | yes | billing account required |
| Streamlit Community Cloud | free | sleeps | no | not a container, different runtime |
| GitHub Codespaces | free monthly hours | no | dev only | not a public URL |

## Decision

Ship four deployment paths, in order of cost:

1. **Render free web service (default).** `render.yaml` describes a Docker
   service built from `docker/Dockerfile`, health-checked on `/health`, with
   `plan: free`. Render injects `PORT`; the image already binds `${PORT}`, so the
   same container runs locally, in CI and on Render. No secrets and no card.
2. **GHCR image.** `publish-image.yml` builds and pushes the image on every push
   to `main`, tagged with branch, semver and commit SHA. Anyone can run it:
   `docker run -p 8000:8000 ghcr.io/<owner>/churnops:latest`.
3. **Hugging Face Spaces (optional, PRO).** `scripts/deploy_hf_space.py` and
   `deploy-hf.yml` remain for anyone with a subscription; they are documented as
   paid rather than silently failing.
4. **GCP Cloud Run (optional).** Terraform in `infra/terraform` provisions
   Artifact Registry, a runtime service account and a scale-to-zero Cloud Run
   service with `max_instances = 2`. Documented as optional because it requires a
   billing account, even though the workload fits in the free tier.

## Consequences

The live demo is free indefinitely, at the cost of cold starts (about a minute
after fifteen idle minutes) and ephemeral disk: the SQLite prediction store
resets when the instance restarts. The dashboard therefore treats traffic as
illustrative, and the committed drift report supplies the monitoring story.
Measured resident memory with the model loaded is roughly 215 MB, comfortably
inside the 512 MB free instance; the app is deliberately single-worker because
0.1 CPU cannot usefully run more.

Keeping one image for every target avoids "works on my laptop" drift. The only
environment-specific knob is `PORT` (8000 locally and in GHCR, injected by
Render, 7860 on Spaces).

Cloud Run remains the migration path for anyone with a billing account; the
Terraform is written to be applied and destroyed without leaving paid resources
behind.
