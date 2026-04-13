# -----------------------------------------------------------------------
# CloudWatch log group — orchestrator container logs
# -----------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "orchestrator" {
  name              = "/ecs/${local.name_prefix}-orchestrator"
  retention_in_days = 30

  tags = { Name = "${local.name_prefix}-orchestrator-logs" }
}

# -----------------------------------------------------------------------
# ECS Task Definition — orchestrator
#
# The orchestrator is a FastAPI app that:
#   • Bids on / manages GPU workers on vast.ai
#   • Proxies OpenAI-compatible requests to the winning worker
#   • Registers its SSH public key with vast.ai on startup so it can
#     SSH into worker containers to tail /tmp/vllm.log during startup
#
# Secrets injected at container start by ECS (not env-file):
#   VASTAI_API_KEY                — vast.ai REST API credentials
#   DISCORD_WEBHOOK_URL           — lifecycle alert notifications
#   ADMIN_TOKEN                   — bearer token for /admin/* endpoints
#   ORCHESTRATOR_SSH_PRIVATE_KEY  — Ed25519 key from ssh_keys.tf
#                                   asyncssh.import_private_key() accepts
#                                   the OpenSSH-format string directly
# -----------------------------------------------------------------------
resource "aws_ecs_task_definition" "orchestrator" {
  family                   = "${local.name_prefix}-orchestrator"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"

  # 0.5 vCPU / 1 GB — the orchestrator is I/O-bound (HTTP + DynamoDB)
  cpu    = "512"
  memory = "1024"

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "orchestrator"
      image = var.orchestrator_image

      # GHCR credentials so the ECS agent can pull the private image
      repositoryCredentials = {
        credentialsParameter = aws_secretsmanager_secret.ghcr_pat.arn
      }

      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "AWS_REGION",        value = local.region },
        { name = "DYNAMODB_TABLE",    value = var.dynamodb_table_name },
        { name = "HISTORY_TABLE",     value = var.history_table_name },
        { name = "EVENTS_TABLE",      value = var.events_table_name },
        { name = "LOG_LEVEL",         value = "INFO" },
        { name = "LB_WORKERS_TABLE",  value = var.lb_workers_table_name },
        { name = "WORKER_IMAGE",      value = var.worker_image },
        { name = "MODEL_ID",          value = var.model_id },
        { name = "VLLM_MAX_MODEL_LEN", value = tostring(var.max_model_len) },
        # Identifies this replica in Discord notifications — distinguishes
        # prod from local dev or any other environment running concurrently.
        { name = "ORCHESTRATOR_ID",   value = "prod" },
      ]

      # Secrets are fetched at task start by the ECS agent (using the
      # execution role) and injected as plain env vars into the container.
      secrets = [
        {
          name      = "VASTAI_API_KEY"
          valueFrom = aws_secretsmanager_secret.vastai_api_key.arn
        },
        {
          name      = "DISCORD_WEBHOOK_URL"
          valueFrom = aws_secretsmanager_secret.discord_webhook_url.arn
        },
        {
          name      = "ADMIN_TOKEN"
          valueFrom = aws_secretsmanager_secret.authorizer_token.arn
        },
        {
          # OpenSSH private key — read by asyncssh.import_private_key()
          # in worker_manager.start() instead of generating an ephemeral key.
          name      = "ORCHESTRATOR_SSH_PRIVATE_KEY"
          valueFrom = aws_secretsmanager_secret.orchestrator_ssh_key.arn
        },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.orchestrator.name
          awslogs-region        = local.region
          awslogs-stream-prefix = "orchestrator"
        }
      }

      essential = true
    }
  ])

  tags = { Name = "${local.name_prefix}-orchestrator-task" }
}

# -----------------------------------------------------------------------
# ECS Service — orchestrator
#
# Traffic path: API Gateway HTTP API → VPC Link → Cloud Map SRV record
# → this service's task ENI on port 8000.  No ALB needed.
# -----------------------------------------------------------------------
resource "aws_ecs_service" "orchestrator" {
  name            = "${local.name_prefix}-orchestrator"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.orchestrator.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  # Register each task with Cloud Map so API Gateway can discover the task
  # IP:port via SRV records without an ALB.
  service_registries {
    registry_arn   = aws_service_discovery_service.vllm.arn
    container_name = "orchestrator"
    container_port = 8000
  }

  # Stop-then-start deployment: shut down the old task before starting the new
  # one.  Prevents two orchestrator replicas from overlapping during a rolling
  # deploy, which would cause duplicate Discord notifications and a race where
  # both replicas independently launch vast.ai bid campaigns.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  lifecycle {
    # CI updates the task definition on every deploy via `aws ecs update-service`.
    # Ignore here to prevent Terraform from rolling back to the tfstate version.
    ignore_changes = [task_definition, desired_count]
  }

  tags = { Name = "${local.name_prefix}-orchestrator-service" }
}
