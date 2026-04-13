from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── vast.ai ──────────────────────────────────────────────────────────────
    # Required for the orchestrator; also used by the LB's /admin/terminate.
    vastai_api_key: Optional[str] = None

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id:     Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    dynamodb_endpoint_url: Optional[str] = None

    # ── DynamoDB tables ───────────────────────────────────────────────────────
    dynamodb_table:  str = "eased-workers"
    history_table:   str = "eased-workers-history"
    events_table:    str = "eased-instance-events"

    # The orchestrator registers/deregisters workers here;
    # the LB reads this table to pick upstreams.
    lb_workers_table: str = "eased-lb-workers"

    # ── Admin auth ────────────────────────────────────────────────────────────
    admin_token: str

    # ── Discord ───────────────────────────────────────────────────────────────
    # Required for the orchestrator; unused by the LB.
    discord_webhook_url: Optional[str] = None

    # ── Load-balancer round-robin ─────────────────────────────────────────────
    # Seconds between DynamoDB scans to refresh the worker pool.
    worker_cache_ttl: float = 5.0

    # ── Worker image ──────────────────────────────────────────────────────────
    worker_image: str = "easedai/nemotron-vastai:latest"
    worker_disk_gb: float = 100.0
    ghcr_username: Optional[str] = None
    ghcr_pat:      Optional[str] = None

    # ── vLLM / model ──────────────────────────────────────────────────────────
    vllm_port: int = 8080
    model_id: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    hf_home: str = "/hf"
    vllm_cache_root: str = "/vllm-cache"
    vllm_max_model_len: int = 32768
    vllm_gpu_memory_utilization: float = 0.95
    vllm_video_loader_backend: str = "opencv"

    # ── Bidding ───────────────────────────────────────────────────────────────
    bid_start_pct: float = 0.50
    bid_step_pct: float = 0.05
    bid_retry_interval_sec: int = 300
    bid_max_multiplier: float = 1.10

    # ── GPU requirements ──────────────────────────────────────────────────────
    min_gpu_ram_gb: int = 40
    min_disk_gb: float = 100.0
    min_inet_down_mbps: float = 300.0
    min_reliability: float = 0.90

    # ── Health checking ───────────────────────────────────────────────────────
    health_check_interval_sec: int = 30
    health_check_timeout_sec: int = 10
    health_check_fail_threshold: int = 3
    instance_running_timeout_sec: int = 2400
    worker_startup_timeout_sec: int = 900
    vast_check_interval_sec: int = 60
    status_report_interval_sec: int = 1800

    # ── Instance limits ───────────────────────────────────────────────────────
    max_instances: int = 1
    keep_debug_instance: bool = False

    # ── SSH ───────────────────────────────────────────────────────────────────
    orchestrator_ssh_private_key: Optional[str] = None

    # ── CloudWatch metrics ────────────────────────────────────────────────────
    cloudwatch_enabled: bool = True
    latency_warn_threshold_ms: float = 30_000

    # ── Identity ──────────────────────────────────────────────────────────────
    orchestrator_id: str = "prod"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
