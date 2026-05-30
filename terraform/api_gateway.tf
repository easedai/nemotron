# -----------------------------------------------------------------------
# VPC Link — tunnels API Gateway traffic into the private VPC subnets
# -----------------------------------------------------------------------
resource "aws_apigatewayv2_vpc_link" "main" {
  name               = "${local.name_prefix}-vpc-link"
  security_group_ids = [aws_security_group.vpc_link.id]
  subnet_ids         = aws_subnet.private[*].id

  tags = { Name = "${local.name_prefix}-vpc-link" }
}

# -----------------------------------------------------------------------
# HTTP API
# -----------------------------------------------------------------------
resource "aws_apigatewayv2_api" "vllm" {
  name          = "${local.name_prefix}-api"
  protocol_type = "HTTP"
  description   = "vLLM Nemotron inference — OpenAI-compatible"

  tags = { Name = "${local.name_prefix}-api" }
}

# -----------------------------------------------------------------------
# Lambda Authorizer (REQUEST type, simple-response, 5-min result cache)
# -----------------------------------------------------------------------
resource "aws_apigatewayv2_authorizer" "lambda" {
  api_id           = aws_apigatewayv2_api.vllm.id
  authorizer_type  = "REQUEST"
  authorizer_uri   = aws_lambda_function.authorizer.invoke_arn
  identity_sources = ["$request.header.Authorization"]
  name             = "bearer-token-authorizer"

  # Format 2.0 + simple response: Lambda returns {"isAuthorized": bool}
  authorizer_payload_format_version = "2.0"
  enable_simple_responses           = true

  # API GW caches the authorisation result per unique Authorization header
  # value for 5 minutes, matching the Lambda's own in-process secret cache.
  authorizer_result_ttl_in_seconds = 300
}

# -----------------------------------------------------------------------
# Private integrations — API GW → VPC Link → Cloud Map → ECS tasks
# -----------------------------------------------------------------------

# Orchestrator: handles /admin/* and all other non-/v1 routes
resource "aws_apigatewayv2_integration" "vllm" {
  api_id             = aws_apigatewayv2_api.vllm.id
  integration_type   = "HTTP_PROXY"
  integration_method = "ANY"

  # Cloud Map service ARN: API GW uses SRV records to discover task IP:port
  integration_uri = aws_service_discovery_service.vllm.arn

  connection_type        = "VPC_LINK"
  connection_id          = aws_apigatewayv2_vpc_link.main.id
  payload_format_version = "1.0"
}

# Load balancer: handles all /v1/* (OpenAI-compatible inference) routes
resource "aws_apigatewayv2_integration" "lb" {
  api_id             = aws_apigatewayv2_api.vllm.id
  integration_type   = "HTTP_PROXY"
  integration_method = "ANY"

  integration_uri = aws_service_discovery_service.lb.arn

  connection_type        = "VPC_LINK"
  connection_id          = aws_apigatewayv2_vpc_link.main.id
  payload_format_version = "1.0"
}

# -----------------------------------------------------------------------
# Routes — proxy everything through the authorizer
#
# API Gateway HTTP API matches the most-specific route first:
#   /v1/{proxy+}  →  load balancer  (OpenAI /v1/chat/completions, etc.)
#   /{proxy+}     →  orchestrator   (/admin/*, /health, etc.)
# -----------------------------------------------------------------------
resource "aws_apigatewayv2_route" "v1_proxy" {
  api_id             = aws_apigatewayv2_api.vllm.id
  route_key          = "ANY /v1/{proxy+}"
  target             = "integrations/${aws_apigatewayv2_integration.lb.id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.lambda.id
}

resource "aws_apigatewayv2_route" "proxy" {
  api_id             = aws_apigatewayv2_api.vllm.id
  route_key          = "ANY /{proxy+}"
  target             = "integrations/${aws_apigatewayv2_integration.vllm.id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.lambda.id
}

resource "aws_apigatewayv2_route" "root" {
  api_id             = aws_apigatewayv2_api.vllm.id
  route_key          = "ANY /"
  target             = "integrations/${aws_apigatewayv2_integration.vllm.id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.lambda.id
}

# -----------------------------------------------------------------------
# Stage with access logging
# -----------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${local.name_prefix}"
  retention_in_days = 14
  tags              = { Name = "${local.name_prefix}-apigw-logs" }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.vllm.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId        = "$context.requestId"
      sourceIp         = "$context.identity.sourceIp"
      requestTime      = "$context.requestTime"
      httpMethod       = "$context.httpMethod"
      routeKey         = "$context.routeKey"
      status           = "$context.status"
      responseLength   = "$context.responseLength"
      integrationError = "$context.integrationErrorMessage"
      authorizerError  = "$context.authorizer.error"
    })
  }

  tags = { Name = "${local.name_prefix}-stage" }
}

# -----------------------------------------------------------------------
# Custom domain mapping
# -----------------------------------------------------------------------
resource "aws_apigatewayv2_domain_name" "api" {
  domain_name = local.api_domain

  domain_name_configuration {
    certificate_arn = aws_acm_certificate_validation.api.certificate_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }

  tags = { Name = "${local.name_prefix}-custom-domain" }
}

resource "aws_apigatewayv2_api_mapping" "api" {
  api_id      = aws_apigatewayv2_api.vllm.id
  domain_name = aws_apigatewayv2_domain_name.api.id
  stage       = aws_apigatewayv2_stage.default.id
}
