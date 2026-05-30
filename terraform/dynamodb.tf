# -----------------------------------------------------------------------
# DynamoDB — orchestrator worker state
#
# Stores the state of all vast.ai worker instances so the orchestrator
# can survive Fargate task restarts without losing track of running workers.
# -----------------------------------------------------------------------
resource "aws_dynamodb_table" "eased_workers" {
  name         = var.dynamodb_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "worker_id"

  attribute {
    name = "worker_id"
    type = "S"
  }

  attribute {
    name = "instance_id"
    type = "S"
  }

  global_secondary_index {
    name            = "instance_id-index"
    key_schema {
      attribute_name = "instance_id"
      key_type       = "HASH"
    }
    projection_type = "ALL"
  }

  # Retain the table (and all worker records) if Terraform is destroyed.
  # Set to false only when intentionally tearing down the full stack.
  lifecycle {
    prevent_destroy = true
  }

  tags = { Name = "${local.name_prefix}-workers" }
}

# -----------------------------------------------------------------------
# DynamoDB — permanent worker history
#
# A permanent record of every worker that has ever existed.  Written in
# parallel with eased-workers on every save/update; never deleted.
# The terminated_reason field is populated on failure so post-mortems can
# query "why did each instance die?" without consulting the event log.
# -----------------------------------------------------------------------
resource "aws_dynamodb_table" "eased_workers_history" {
  name         = var.history_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "worker_id"

  attribute {
    name = "worker_id"
    type = "S"
  }

  attribute {
    name = "instance_id"
    type = "S"
  }

  global_secondary_index {
    name            = "instance_id-index"
    key_schema {
      attribute_name = "instance_id"
      key_type       = "HASH"
    }
    projection_type = "ALL"
  }

  lifecycle {
    prevent_destroy = true
  }

  tags = { Name = "${local.name_prefix}-workers-history" }
}

# -----------------------------------------------------------------------
# DynamoDB — load-balancer worker pool
#
# Written by the orchestrator (register/deregister).
# Read by the load-balancer service to pick the next upstream.
# Only contains workers that are healthy and accepting requests.
# -----------------------------------------------------------------------
resource "aws_dynamodb_table" "eased_lb_workers" {
  name         = var.lb_workers_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "worker_id"

  attribute {
    name = "worker_id"
    type = "S"
  }

  lifecycle {
    prevent_destroy = false
  }

  tags = { Name = "${local.name_prefix}-lb-workers" }
}

# -----------------------------------------------------------------------
# DynamoDB — instance event log
#
# Rolling 7-day log of instance lifecycle events, status changes, and
# container log snapshots. Useful for debugging and post-mortem analysis.
#
# Schema:
#   PK  worker_id  (S)  — identifies the logical worker
#   SK  ts         (S)  — ISO-8601 timestamp, enables time-ordered queries
#   GSI instance_id-index on instance_id (S) — query events by vast.ai ID
# -----------------------------------------------------------------------
resource "aws_dynamodb_table" "eased_instance_events" {
  name         = var.events_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "worker_id"
  range_key    = "ts"

  # TTL attribute — set by the application to epoch + 7 days
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  attribute {
    name = "worker_id"
    type = "S"
  }

  attribute {
    name = "ts"
    type = "S"
  }

  attribute {
    name = "instance_id"
    type = "S"
  }

  global_secondary_index {
    name            = "instance_id-index"
    key_schema {
      attribute_name = "instance_id"
      key_type       = "HASH"
    }
    key_schema {
      attribute_name = "ts"
      key_type       = "RANGE"
    }
    projection_type = "ALL"
  }

  # Events are ephemeral — fine to destroy with the stack.
  lifecycle {
    prevent_destroy = false
  }

  tags = { Name = "${local.name_prefix}-instance-events" }
}
