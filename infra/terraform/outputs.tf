output "service_url" {
  description = "Public URL of the Cloud Run service."
  value       = google_cloud_run_v2_service.api.uri
}

output "dashboard_url" {
  description = "Live inference dashboard."
  value       = "${google_cloud_run_v2_service.api.uri}/dashboard"
}

output "artifact_registry" {
  description = "Docker repository to push images to."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.churnops.repository_id}"
}

output "runtime_service_account" {
  description = "Service account used by Cloud Run."
  value       = google_service_account.runtime.email
}
