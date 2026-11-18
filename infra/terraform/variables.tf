variable "project_id" {
  description = "GCP project that hosts the demo (billing must be enabled for Cloud Run)."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run and Artifact Registry."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "churnops-api"
}

variable "image" {
  description = "Container image to deploy, e.g. us-central1-docker.pkg.dev/PROJECT/churnops/api:latest"
  type        = string
}

variable "allow_unauthenticated" {
  description = "Expose the API publicly (needed for the live dashboard demo)."
  type        = bool
  default     = true
}

variable "cpu" {
  description = "CPU limit for the container."
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory limit for the container."
  type        = string
  default     = "512Mi"
}

variable "max_instances" {
  description = "Upper bound on instances, the main guardrail against surprise cost."
  type        = number
  default     = 2
}
