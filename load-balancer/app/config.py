from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── AWS ───────────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id:     Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    dynamodb_endpoint_url: Optional[str] = None

    # ── Load-balancer worker table ────────────────────────────────────────
    # Written by the orchestrator; read here to pick the next upstream.
    lb_workers_table: str = "eased-lb-workers"

    # ── Round-robin cache ─────────────────────────────────────────────────
    # Seconds between DynamoDB scans to refresh the worker pool.
    worker_cache_ttl: float = 5.0

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
