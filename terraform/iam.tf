# -----------------------------------------------------------------------
# Shared assume-role policy documents
# -----------------------------------------------------------------------
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# -----------------------------------------------------------------------
# ECS Task Execution Role
# Used by the ECS agent to pull the orchestrator image and inject secrets.
# -----------------------------------------------------------------------
resource "aws_iam_role" "ecs_task_execution" {
  name               = "${local.name_prefix}-ecs-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_managed" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Secrets the orchestrator task execution role must be able to fetch.
# ECS injects these as env vars at task start; the task role is NOT used here.
data "aws_iam_policy_document" "task_execution_secrets" {
  statement {
    effect  = "Allow"
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.vastai_api_key.arn,
      aws_secretsmanager_secret.discord_webhook_url.arn,
      aws_secretsmanager_secret.ghcr_pat.arn,
      # Bearer token for /admin/* endpoints (injected as ADMIN_TOKEN)
      aws_secretsmanager_secret.authorizer_token.arn,
      # Ed25519 private key for orchestrator → vast.ai worker SSH
      aws_secretsmanager_secret.orchestrator_ssh_key.arn,
    ]
  }
}

resource "aws_iam_role_policy" "task_execution_secrets" {
  name   = "orchestrator-secrets"
  role   = aws_iam_role.ecs_task_execution.id
  policy = data.aws_iam_policy_document.task_execution_secrets.json
}

# -----------------------------------------------------------------------
# ECS Task Role
# Attached to the running orchestrator container.
# -----------------------------------------------------------------------
resource "aws_iam_role" "ecs_task" {
  name               = "${local.name_prefix}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# DynamoDB access — orchestrator reads/writes worker state, event log,
# and the load-balancer pool (register/deregister only)
data "aws_iam_policy_document" "ecs_task_dynamodb" {
  statement {
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:DeleteItem",
      "dynamodb:Scan",
      "dynamodb:Query",
    ]
    resources = [
      aws_dynamodb_table.eased_workers.arn,
      aws_dynamodb_table.eased_workers_history.arn,
      aws_dynamodb_table.eased_instance_events.arn,
      # Allow GSI queries on the events table
      "${aws_dynamodb_table.eased_instance_events.arn}/index/*",
      # LB table — orchestrator registers/deregisters workers here
      aws_dynamodb_table.eased_lb_workers.arn,
    ]
  }
}

resource "aws_iam_role_policy" "ecs_task_dynamodb" {
  name   = "worker-state-dynamodb"
  role   = aws_iam_role.ecs_task.id
  policy = data.aws_iam_policy_document.ecs_task_dynamodb.json
}

# -----------------------------------------------------------------------
# Load-balancer ECS Task Role
# Scoped to the LB worker pool table only.
# -----------------------------------------------------------------------
resource "aws_iam_role" "ecs_lb_task" {
  name               = "${local.name_prefix}-ecs-lb-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "ecs_lb_task_dynamodb" {
  statement {
    effect = "Allow"
    actions = [
      "dynamodb:Scan",
      "dynamodb:GetItem",
      "dynamodb:UpdateItem",
    ]
    resources = [aws_dynamodb_table.eased_lb_workers.arn]
  }
}

resource "aws_iam_role_policy" "ecs_lb_task_dynamodb" {
  name   = "lb-pool-dynamodb"
  role   = aws_iam_role.ecs_lb_task.id
  policy = data.aws_iam_policy_document.ecs_lb_task_dynamodb.json
}

# -----------------------------------------------------------------------
# Lambda Authorizer Execution Role
# -----------------------------------------------------------------------
resource "aws_iam_role" "lambda_authorizer" {
  name               = "${local.name_prefix}-lambda-authorizer"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_authorizer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "lambda_secrets" {
  statement {
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.authorizer_token.arn]
  }
}

resource "aws_iam_role_policy" "lambda_secrets" {
  name   = "authorizer-token-secret"
  role   = aws_iam_role.lambda_authorizer.id
  policy = data.aws_iam_policy_document.lambda_secrets.json
}
