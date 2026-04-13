# -----------------------------------------------------------------------
# Locals
# -----------------------------------------------------------------------
locals {
  model_s3_prefix = "nemotron-nano-12b-vl-bf16"
}

# -----------------------------------------------------------------------
# Account ID — used to give the bucket a globally unique name
# -----------------------------------------------------------------------
data "aws_caller_identity" "current" {}

# -----------------------------------------------------------------------
# S3 Bucket — model weight store (backup / source of truth)
#
# Model weights are baked into the worker Docker image at build time
# (see github.com/easedai/nemotron). This bucket acts as the canonical
# store for the weights and is not mounted at runtime.
# -----------------------------------------------------------------------
resource "aws_s3_bucket" "models" {
  bucket = "${local.name_prefix}-models-${data.aws_caller_identity.current.account_id}"

  tags = { Name = "${local.name_prefix}-models" }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    id     = "abort-incomplete-mpu"
    status = "Enabled"
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }
}
