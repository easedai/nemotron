output "api_url" {
  description = "Public API endpoint (custom domain)"
  value       = "https://${local.api_domain}"
}

output "api_gateway_default_url" {
  description = "API Gateway default invoke URL (pre-custom-domain fallback)"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name for the orchestrator"
  value       = aws_ecs_service.orchestrator.name
}

output "orchestrator_log_group" {
  description = "CloudWatch log group for orchestrator container logs"
  value       = aws_cloudwatch_log_group.orchestrator.name
}

output "orchestrator_ssh_public_key" {
  description = "OpenSSH public key for the orchestrator's vast.ai SSH access (registered at task start)"
  value       = tls_private_key.orchestrator_ssh.public_key_openssh
}

output "orchestrator_ssh_key_secret_arn" {
  description = "Secrets Manager ARN for the orchestrator Ed25519 private key"
  value       = aws_secretsmanager_secret.orchestrator_ssh_key.arn
}

output "dynamodb_table_name" {
  description = "DynamoDB table used for orchestrator worker state"
  value       = aws_dynamodb_table.eased_workers.name
}

output "events_table_name" {
  description = "DynamoDB table used for instance lifecycle events (7-day TTL)"
  value       = aws_dynamodb_table.eased_instance_events.name
}

output "vastai_secret_arn" {
  description = "ARN of the vast.ai API key secret"
  value       = aws_secretsmanager_secret.vastai_api_key.arn
}

output "discord_secret_arn" {
  description = "ARN of the Discord webhook URL secret"
  value       = aws_secretsmanager_secret.discord_webhook_url.arn
}

output "hf_token_secret_arn" {
  description = "ARN of the HuggingFace token secret"
  value       = aws_secretsmanager_secret.hf_token.arn
}

output "authorizer_token_secret_arn" {
  description = "ARN of the API authorizer token secret"
  value       = aws_secretsmanager_secret.authorizer_token.arn
}

output "dynamodb_user_access_key_id" {
  description = "Access key ID for the DynamoDB IAM user"
  value       = aws_iam_access_key.dynamodb.id
}

output "dynamodb_user_secret_access_key" {
  description = "Secret access key for the DynamoDB IAM user"
  value       = aws_iam_access_key.dynamodb.secret
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "models_s3_bucket" {
  description = "S3 bucket holding the model weights"
  value       = aws_s3_bucket.models.id
}

output "models_s3_uri" {
  description = "S3 URI for the Nemotron model weights"
  value       = "s3://${aws_s3_bucket.models.id}/${local.model_s3_prefix}/"
}
