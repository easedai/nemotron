# -----------------------------------------------------------------------
# ECS Cluster
# Hosts the orchestrator Fargate service.
# GPU workers run on vast.ai — no EC2 capacity providers needed.
# -----------------------------------------------------------------------
resource "aws_ecs_cluster" "main" {
  name = local.name_prefix

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${local.name_prefix}-cluster" }
}
