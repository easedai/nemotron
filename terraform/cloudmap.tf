# -----------------------------------------------------------------------
# Cloud Map — private DNS namespace + service
# API Gateway HTTP API integrates with the service ARN via the VPC Link,
# using SRV records to discover the ECS task IP:port.
# -----------------------------------------------------------------------
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "${local.name_prefix}.local"
  vpc         = aws_vpc.main.id
  description = "Private DNS namespace for ${local.name_prefix}"

  tags = { Name = "${local.name_prefix}-namespace" }
}

resource "aws_service_discovery_service" "vllm" {
  name        = "vllm"
  description = "Orchestrator — admin/ops endpoints"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    # SRV carries both IP and port — required for API GW Cloud Map integration
    dns_records {
      ttl  = 10
      type = "SRV"
    }

    # A record for direct VPC DNS resolution
    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }


  tags = { Name = "${local.name_prefix}-sd-service" }
}

resource "aws_service_discovery_service" "lb" {
  name        = "lb"
  description = "Load balancer — round-robin /v1/* proxy to vLLM workers"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "SRV"
    }

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  tags = { Name = "${local.name_prefix}-sd-lb" }
}
