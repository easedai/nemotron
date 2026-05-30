# NOTE: Before running `terraform init`, ensure the S3 bucket exists and has
# versioning enabled — required for use_lockfile = true (S3 native locking,
# Terraform >= 1.10, no DynamoDB needed).
terraform {
  required_version = ">= 1.12"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket       = "terrafrom-471112713419-us-east-1-an"
    key          = "nemotron/vllm/terraform.tfstate"
    region       = "us-east-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-east-1"

  default_tags {
    tags = {
      Project   = "nemotron-vllm"
      ManagedBy = "terraform"
    }
  }
}
