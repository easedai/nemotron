from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


# ── Orchestrator worker models ────────────────────────────────────────────────

class WorkerStatus(str, Enum):
    BIDDING    = "bidding"
    PENDING    = "pending"
    STARTING   = "starting"
    RUNNING    = "running"
    UNHEALTHY  = "unhealthy"
    DRAINING   = "draining"
    TERMINATED = "terminated"


class WorkerType(str, Enum):
    INTERRUPTIBLE = "interruptible"
    ON_DEMAND     = "on_demand"


class Worker(BaseModel):
    worker_id:            str
    instance_id:          Optional[str]      = None
    label:                Optional[str]      = None
    status:               WorkerStatus
    worker_type:          WorkerType
    provider:             str                = "vastai"
    api_key:              str
    host:                 Optional[str]      = None
    port:                 int                = 8000
    gpu_name:             Optional[str]      = None
    gpu_ram_gb:           Optional[float]    = None
    num_gpus:             int                = 1
    bid_price:            Optional[float]    = None
    market_price:         Optional[float]    = None
    bid_attempts:         int                = 0
    consecutive_failures: int                = 0
    running_since:            Optional[datetime] = None
    image_pull_started_at:    Optional[datetime] = None
    image_pull_duration_sec:  Optional[float]   = None
    specs:                    Optional[dict]    = None
    created_at:               datetime
    updated_at:               datetime

    @property
    def base_url(self) -> Optional[str]:
        if self.host and self.port:
            if self.port == 443:
                return f"https://{self.host}"
            return f"http://{self.host}:{self.port}"
        return None

    @property
    def is_available(self) -> bool:
        return self.status == WorkerStatus.RUNNING and self.host is not None


# ── Load-balancer worker model ────────────────────────────────────────────────

class LBWorker(BaseModel):
    """A worker that is healthy and accepting requests."""

    worker_id:   str
    source_type: str
    host:        str
    port:        int
    api_key:     str
    added_at:    datetime

    ssh_port:            Optional[int]      = None
    successful_requests: int = 0
    failed_requests:     int = 0
    last_request_at:     Optional[datetime] = None

    @property
    def base_url(self) -> str:
        if self.port == 443:
            return f"https://{self.host}"
        return f"http://{self.host}:{self.port}"
