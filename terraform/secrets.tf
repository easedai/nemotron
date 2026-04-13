# -----------------------------------------------------------------------
# vast.ai API key — used by the orchestrator to bid on and manage GPU workers
# -----------------------------------------------------------------------
resource "aws_secretsmanager_secret" "vastai_api_key" {
  name                    = "${local.name_prefix}/vastai-api-key"
  description             = "vast.ai API key for the eased orchestrator"
  recovery_window_in_days = 0

  tags = { Name = "${local.name_prefix}-vastai-api-key" }
}

resource "aws_secretsmanager_secret_version" "vastai_api_key" {
  secret_id     = aws_secretsmanager_secret.vastai_api_key.id
  secret_string = var.vastai_api_key
}

# -----------------------------------------------------------------------
# Discord webhook URL — orchestrator lifecycle notifications
# -----------------------------------------------------------------------
resource "aws_secretsmanager_secret" "discord_webhook_url" {
  name                    = "${local.name_prefix}/discord-webhook-url"
  description             = "Discord webhook URL for eased orchestrator alerts"
  recovery_window_in_days = 0

  tags = { Name = "${local.name_prefix}-discord-webhook-url" }
}

resource "aws_secretsmanager_secret_version" "discord_webhook_url" {
  secret_id     = aws_secretsmanager_secret.discord_webhook_url.id
  secret_string = var.discord_webhook_url
}

# -----------------------------------------------------------------------
# GHCR PAT — allows ECS task execution role to pull the private
# orchestrator image from GitHub Container Registry
# -----------------------------------------------------------------------
resource "aws_secretsmanager_secret" "ghcr_pat" {
  name                    = "${local.name_prefix}/ghcr-pat"
  description             = "GitHub PAT (read:packages) for pulling the private eased orchestrator image"
  recovery_window_in_days = 0

  tags = { Name = "${local.name_prefix}-ghcr-pat" }
}

resource "aws_secretsmanager_secret_version" "ghcr_pat" {
  secret_id = aws_secretsmanager_secret.ghcr_pat.id
  # ECS repositoryCredentials expects {"username":"<user>","password":"<pat>"}
  secret_string = jsonencode({
    username = "easedai"
    password = var.ghcr_pat
  })
}

# -----------------------------------------------------------------------
# HuggingFace token — kept for reference; model weights are baked into
# the worker image so this is not mounted at runtime
# -----------------------------------------------------------------------
resource "aws_secretsmanager_secret" "hf_token" {
  name                    = "${local.name_prefix}/hf-token"
  description             = "HuggingFace API token for Nemotron model access"
  recovery_window_in_days = 0

  tags = { Name = "${local.name_prefix}-hf-token" }
}

resource "aws_secretsmanager_secret_version" "hf_token" {
  secret_id     = aws_secretsmanager_secret.hf_token.id
  secret_string = var.hf_token
}

# -----------------------------------------------------------------------
# API authorizer token — validated by the Lambda authorizer
# -----------------------------------------------------------------------
resource "aws_secretsmanager_secret" "authorizer_token" {
  name                    = "${local.name_prefix}/authorizer-token"
  description             = "Bearer token for API Gateway Lambda authorizer"
  recovery_window_in_days = 0

  tags = { Name = "${local.name_prefix}-authorizer-token" }
}

resource "aws_secretsmanager_secret_version" "authorizer_token" {
  secret_id     = aws_secretsmanager_secret.authorizer_token.id
  secret_string = var.authorizer_token
}
