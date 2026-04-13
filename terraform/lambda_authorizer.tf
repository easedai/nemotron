# -----------------------------------------------------------------------
# Package the Lambda source
# -----------------------------------------------------------------------
data "archive_file" "authorizer" {
  type        = "zip"
  source_file = "${path.module}/lambda/authorizer.py"
  output_path = "${path.module}/.build/authorizer.zip"
}

# -----------------------------------------------------------------------
# CloudWatch Log Group (pre-create so retention is applied immediately)
# -----------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "lambda_authorizer" {
  name              = "/aws/lambda/${local.name_prefix}-authorizer"
  retention_in_days = 14
  tags              = { Name = "${local.name_prefix}-authorizer-logs" }
}

# -----------------------------------------------------------------------
# Lambda Function
# Not VPC-attached — accesses Secrets Manager via public endpoint,
# which avoids VPC cold-start penalty and NAT dependency.
# -----------------------------------------------------------------------
resource "aws_lambda_function" "authorizer" {
  function_name    = "${local.name_prefix}-authorizer"
  filename         = data.archive_file.authorizer.output_path
  source_code_hash = data.archive_file.authorizer.output_base64sha256
  role             = aws_iam_role.lambda_authorizer.arn
  handler          = "authorizer.handler"
  runtime          = "python3.13"
  timeout          = 5
  memory_size      = 128

  environment {
    variables = {
      AUTHORIZER_SECRET_ARN = aws_secretsmanager_secret.authorizer_token.arn
      SECRET_CACHE_TTL_SEC  = "300"
      AWS_REGION_NAME       = local.region
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda_authorizer,
    aws_iam_role_policy_attachment.lambda_basic,
  ]

  tags = { Name = "${local.name_prefix}-authorizer" }
}

# -----------------------------------------------------------------------
# Allow API Gateway to invoke the Lambda
# -----------------------------------------------------------------------
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.authorizer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.vllm.execution_arn}/*"
}
