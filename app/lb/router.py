from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog

from ..config import settings
from ..lb_db import LBDB
from ..models import LBWorker

log = structlog.get_logger(__name__)


class RoundRobin:
    """
    Thread-safe round-robin worker selector backed by a DynamoDB scan cache.

    The worker list is refreshed from DynamoDB at most every
    ``settings.worker_cache_ttl`` seconds so individual requests do not
    each incur a DynamoDB round-trip.  A lock prevents thundering-herd
    on simultaneous refreshes.
    """

    def __init__(self, db: LBDB) -> None:
        self.db               = db
        self._workers:        list[LBWorker] = []
        self._index:          int            = 0
        self._last_refresh:   float          = 0.0
        self._lock                           = asyncio.Lock()

    async def get_next(self) -> Optional[LBWorker]:
        await self._maybe_refresh()
        if not self._workers:
            return None
        worker       = self._workers[self._index % len(self._workers)]
        self._index  = (self._index + 1) % len(self._workers)
        return worker

    async def force_refresh(self) -> None:
        """Bypass the TTL and immediately re-scan DynamoDB."""
        async with self._lock:
            await self._do_refresh()

    async def _maybe_refresh(self) -> None:
        if time.monotonic() - self._last_refresh < settings.worker_cache_ttl:
            return
        async with self._lock:
            if time.monotonic() - self._last_refresh < settings.worker_cache_ttl:
                return  # another coroutine refreshed while we waited for the lock
            await self._do_refresh()

    async def _do_refresh(self) -> None:
        workers = await asyncio.to_thread(self.db.list_workers)

        old_ids = {w.worker_id for w in self._workers}
        new_ids = {w.worker_id for w in workers}

        if old_ids != new_ids:
            self._index = 0  # reset position when the pool membership changes
            log.info(
                "round_robin.pool_changed",
                added=sorted(new_ids - old_ids),
                removed=sorted(old_ids - new_ids),
                total=len(workers),
            )

        self._workers     = workers
        self._last_refresh = time.monotonic()
