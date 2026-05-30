from __future__ import annotations

from typing import Optional

from ..lb_queue import WorkerQueue
from ..models import LBWorker


class WorkerRouter:
    """Thin wrapper over WorkerQueue for use in the LB request path."""

    def __init__(self, queue: WorkerQueue, lb_id: str) -> None:
        self._queue = queue
        self._lb_id = lb_id

    async def checkout(self) -> Optional[LBWorker]:
        return await self._queue.checkout(self._lb_id)

    async def checkin(self, worker_id: str) -> str:
        return await self._queue.checkin(worker_id, self._lb_id)
