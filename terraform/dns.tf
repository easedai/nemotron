# -----------------------------------------------------------------------
# Route 53 — look up the existing eased.ai hosted zone
# -----------------------------------------------------------------------
data "aws_route53_zone" "eased_ai" {
  name         = "eased.ai."
  private_zone = false
}

# -----------------------------------------------------------------------
# ACM certificate for dev.api.eased.ai (same region as HTTP API)
# -----------------------------------------------------------------------
resource "aws_acm_certificate" "api" {
  domain_name       = local.api_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = { Name = "${local.name_prefix}-cert" }
}

# Write the ACM DNS validation CNAME records into the hosted zone
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.api.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  zone_id         = data.aws_route53_zone.eased_ai.zone_id
  name            = each.value.name
  type            = each.value.type
  records         = [each.value.record]
  ttl             = 60
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "api" {
  certificate_arn         = aws_acm_certificate.api.arn
  validation_record_fqdns = [for r in aws_route53_record.cert_validation : r.fqdn]
}

# -----------------------------------------------------------------------
# Route 53 alias A record — dev.api.eased.ai → API Gateway regional domain
# -----------------------------------------------------------------------
resource "aws_route53_record" "api" {
  zone_id = data.aws_route53_zone.eased_ai.zone_id
  name    = local.api_domain
  type    = "A"

  alias {
    name                   = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].target_domain_name
    zone_id                = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].hosted_zone_id
    evaluate_target_health = false
  }
}
