locals {
  name_prefix = "nemotron-vllm"
  region      = "us-east-1"

  azs           = ["us-east-1a", "us-east-1b"]
  vpc_cidr      = "10.0.0.0/16"
  public_cidrs  = ["10.0.0.0/24", "10.0.1.0/24"]
  private_cidrs = ["10.0.10.0/24", "10.0.11.0/24"]

  api_domain = "dev.api.eased.ai"
}
