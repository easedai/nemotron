from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class WorkerStatus(str, Enum):
    BIDDING  = "bidding"
    PENDING  = "pending"   # instance created, waiting for it to come up
    STARTING = "starting"  # vast.ai running, vLLM health check in progress
    RUNNING      = "running"
    UNHEALTHY    = "unhealthy"
    DRAINING     = "draining"
    TERMINATED   = "terminated"


class WorkerType(str, Enum):
    INTERRUPTIBLE = "interruptible"
    ON_DEMAND     = "on_demand"


class Worker(BaseModel):
    worker_id:            str
    instance_id:          Optional[str]      = None
    label:                Optional[str]      = None   # vast.ai instance label (eased-{worker_id})
    status:               WorkerStatus
    worker_type:          WorkerType
    api_key:              str
    host:                 Optional[str]      = None
    port:                 int                = 8000
    gpu_name:             Optional[str]      = None
    gpu_ram_gb:           Optional[float]    = None
    bid_price:            Optional[float]    = None
    market_price:         Optional[float]    = None
    bid_attempts:         int                = 0
    consecutive_failures: int                = 0
    running_since:          Optional[datetime] = None   # when vLLM first became healthy
    image_pull_started_at:  Optional[datetime] = None   # when create_instance() completed
    image_pull_duration_sec: Optional[float]  = None   # seconds from create → vast running
    specs:                  Optional[dict]    = None   # hardware/perf snapshot from vast.ai
    created_at:             datetime
    updated_at:             datetime

    @property
    def base_url(self) -> Optional[str]:
        if self.host and self.port:
            return f"http://{self.host}:{self.port}"
        return None

    @property
    def is_available(self) -> bool:
        return self.status == WorkerStatus.RUNNING and self.host is not None
