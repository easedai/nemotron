# ── Secrets (required — no defaults) ─────────────────────────────────────────

variable "vastai_api_key" {
  description = "vast.ai API key used by the orchestrator to search offers, place bids, and manage worker instances"
  type        = string
  sensitive   = true
}

variable "discord_webhook_url" {
  description = "Discord webhook URL for orchestrator lifecycle notifications (bids, worker up/down, errors)"
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "HuggingFace API token — stored in Secrets Manager as a reference; model weights are baked into the worker image at build time"
  type        = string
  sensitive   = true
}

variable "authorizer_token" {
  description = "Bearer token callers must supply in the Authorization header"
  type        = string
  sensitive   = true
}

variable "ghcr_pat" {
  description = "GitHub Personal Access Token (read:packages scope) — used by ECS to pull the private orchestrator image from GHCR"
  type        = string
  sensitive   = true
}

# ── Images ────────────────────────────────────────────────────────────────────

variable "orchestrator_image" {
  description = "GHCR image URI for the eased orchestrator container. Updated automatically by CI on each deploy."
  type        = string
  default     = "ghcr.io/easedai/eased:latest"
}

variable "lb_image" {
  description = "Docker image for the load-balancer service. Updated automatically by CI on each deploy."
  type        = string
  default     = "ghcr.io/easedai/eased-lb:latest"
}

variable "worker_image" {
  description = "Docker Hub image for the vLLM worker container (built by github.com/easedai/nemotron)"
  type        = string
  default     = "easedai/nemotron-vastai:latest"
}

# ── DynamoDB ──────────────────────────────────────────────────────────────────

variable "dynamodb_table_name" {
  description = "DynamoDB table name for orchestrator worker state"
  type        = string
  default     = "eased-workers"
}

variable "lb_workers_table_name" {
  description = "DynamoDB table name for the load-balancer worker pool (healthy workers only)"
  type        = string
  default     = "eased-lb-workers"
}

variable "events_table_name" {
  description = "DynamoDB table name for instance lifecycle events (7-day TTL)"
  type        = string
  default     = "eased-instance-events"
}

variable "history_table_name" {
  description = "DynamoDB table name for permanent worker history (never deleted)"
  type        = string
  default     = "eased-workers-history"
}

# ── vLLM / model ──────────────────────────────────────────────────────────────

variable "model_id" {
  description = "HuggingFace model ID passed as MODEL_ID to the orchestrator (and on to vLLM --model)"
  type        = string
  default     = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
}

variable "max_model_len" {
  description = "vLLM context length passed as VLLM_MAX_MODEL_LEN to the orchestrator"
  type        = number
  default     = 32768
}
