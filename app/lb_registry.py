from __future__ import annotations

from typing import Optional

import structlog

from .lb_queue import WorkerQueue

log = structlog.get_logger(__name__)


class LBRegistry:
    """
    Thin async shim over WorkerQueue for the orchestrator.

    The orchestrator calls register/deregister here; WorkerQueue writes to Redis
    so the load balancer picks up changes immediately without a DynamoDB scan.
    """

    def __init__(self, queue: WorkerQueue) -> None:
        self._q = queue

    async def register(
        self,
        worker_id:   str,
        source_type: str,
        host:        str,
        port:        int,
        api_key:     str,
        ssh_port:    Optional[int] = None,
    ) -> None:
        await self._q.register(
            worker_id,
            host=host,
            port=port,
            api_key=api_key,
            source_type=source_type,
            ssh_port=ssh_port,
        )

    async def deregister(self, worker_id: str) -> None:
        await self._q.deregister(worker_id)
