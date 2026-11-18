# Optional GCP deployment (Cloud Run)

This stack is **optional**. The default, zero-cost deployment for ChurnOps is a
Hugging Face Space (`deploy/hf-spaces`), which needs no billing account at all.

Cloud Run itself fits comfortably inside the free tier for a demo: it scales to
zero, and the free allowance covers roughly 180k vCPU-seconds and 360k GiB-seconds
per month. Google still requires a billing account to be attached to the project,
so treat this path as "free-tier, but billing-enabled".

## What gets created

| Resource | Purpose |
| --- | --- |
| Artifact Registry repository `churnops` | Stores the API image |
| Service account `churnops-runtime` | Least-privilege runtime identity |
| Cloud Run v2 service `churnops-api` | Scales 0 to `max_instances`, public by default |
| IAM binding `allUsers` -> `roles/run.invoker` | Public dashboard and API (disable with `allow_unauthenticated = false`) |

## Usage

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in your project id
terraform init
terraform apply -var="image=us-central1-docker.pkg.dev/PROJECT/churnops/api:latest"
```

Build and push the image first:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -f docker/Dockerfile -t us-central1-docker.pkg.dev/PROJECT/churnops/api:latest .
docker push us-central1-docker.pkg.dev/PROJECT/churnops/api:latest
```

## Cost guardrails

- `max_instances = 2` caps the blast radius of a traffic spike.
- `min_instance_count = 0` means no idle cost.
- The container is CPU-only and 512 MiB, well inside the free tier.

Destroy everything with `terraform destroy` when you are done demoing.
