# -----------------------------------------------------------------------
# CloudWatch log group — load-balancer container logs
# -----------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "lb" {
  name              = "/ecs/${local.name_prefix}-lb"
  retention_in_days = 14

  tags = { Name = "${local.name_prefix}-lb-logs" }
}

# -----------------------------------------------------------------------
# ECS Task Definition — load balancer
#
# Stateless HTTP proxy: round-robins /v1/* requests to healthy vLLM
# workers registered in the lb-workers DynamoDB table.
# The orchestrator writes to that table; this service only reads it.
# -----------------------------------------------------------------------
resource "aws_ecs_task_definition" "lb" {
  family                   = "${local.name_prefix}-lb"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"

  # 256 vCPU / 512 MB — pure I/O: proxy + periodic DynamoDB scan
  cpu    = "256"
  memory = "512"

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_lb_task.arn

  container_definitions = jsonencode([
    {
      name  = "lb"
      image = var.lb_image

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
        { name = "AWS_REGION",       value = local.region },
        { name = "LB_WORKERS_TABLE", value = var.lb_workers_table_name },
        { name = "LOG_LEVEL",        value = "INFO" },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.lb.name
          awslogs-region        = local.region
          awslogs-stream-prefix = "lb"
        }
      }

      essential = true
    }
  ])

  tags = { Name = "${local.name_prefix}-lb-task" }
}

# -----------------------------------------------------------------------
# ECS Service — load balancer
#
# Traffic path: API Gateway /v1/* → VPC Link → Cloud Map SRV (lb) → here
# -----------------------------------------------------------------------
resource "aws_ecs_service" "lb" {
  name            = "${local.name_prefix}-lb"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.lb.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn   = aws_service_discovery_service.lb.arn
    container_name = "lb"
    container_port = 8000
  }

  # Rolling deploy: allow a second task to start before stopping the old one.
  # Unlike the orchestrator, multiple LB replicas can safely coexist.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 200

  lifecycle {
    ignore_changes = [task_definition, desired_count]
  }

  tags = { Name = "${local.name_prefix}-lb-service" }
}
