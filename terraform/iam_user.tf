# -----------------------------------------------------------------------
# IAM user — programmatic DynamoDB access
#
# Useful for local development, scripts, or any tool that needs direct
# read/write access to the worker-state and event tables without going
# through the ECS task role.
#
# After applying, retrieve credentials:
#   terraform output dynamodb_user_access_key_id
#   terraform output --raw dynamodb_user_secret_access_key
# -----------------------------------------------------------------------

resource "aws_iam_user" "dynamodb" {
  name = "${local.name_prefix}-dynamodb-user"
  tags = { Name = "${local.name_prefix}-dynamodb-user" }
}

data "aws_iam_policy_document" "dynamodb_user" {
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
      "${aws_dynamodb_table.eased_workers.arn}/index/*",
      aws_dynamodb_table.eased_workers_history.arn,
      "${aws_dynamodb_table.eased_workers_history.arn}/index/*",
      aws_dynamodb_table.eased_instance_events.arn,
      "${aws_dynamodb_table.eased_instance_events.arn}/index/*",
      aws_dynamodb_table.eased_lb_workers.arn,
    ]
  }

  # Read access to the Secrets Manager entries managed by this stack —
  # needed so local scripts/tools using these credentials can pull the
  # orchestrator SSH key (and other runtime secrets) without going
  # through the ECS task role.
  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
    ]
    resources = [
      aws_secretsmanager_secret.orchestrator_ssh_key.arn,
      aws_secretsmanager_secret.vastai_api_key.arn,
      aws_secretsmanager_secret.discord_webhook_url.arn,
      aws_secretsmanager_secret.ghcr_pat.arn,
      aws_secretsmanager_secret.hf_token.arn,
      aws_secretsmanager_secret.authorizer_token.arn,
    ]
  }
}

resource "aws_iam_user_policy" "dynamodb" {
  name   = "dynamodb-access"
  user   = aws_iam_user.dynamodb.name
  policy = data.aws_iam_policy_document.dynamodb_user.json
}

resource "aws_iam_access_key" "dynamodb" {
  user = aws_iam_user.dynamodb.name
}
