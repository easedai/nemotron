from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class LBWorker(BaseModel):
    """A worker that is healthy and accepting requests."""

    worker_id:   str
    source_type: str            # "vastai", etc.
    host:        str
    port:        int
    api_key:     str
    added_at:    datetime       # when the orchestrator first registered this worker

    # Cumulative counters — incremented atomically via DynamoDB ADD
    successful_requests: int = 0
    failed_requests:     int = 0
    last_request_at:     Optional[datetime] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
