from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
import structlog

from .models import LBWorker, Worker

log = structlog.get_logger(__name__)

# ── Key constants ─────────────────────────────────────────────────────────────
_KEY_AVAILABLE   = "workers:available"    # LIST    – round-robin queue
_KEY_DRAINING    = "workers:draining"     # SET     – tombstones for in-flight removal
_KEY_503_COUNT   = "workers:stats:503"    # STRING  – running count of 503s since last reset
_PFX_DATA        = "workers:data:"        # HASH    – per-worker metadata
_PFX_LEASED      = "workers:leased:"      # HASH    – active LB leases (with TTL)
_PFX_LAST_ACTIVE = "workers:last_active:" # STRING  – ISO timestamp of last completed request
_KEY_CTRL_BID    = "control:scale_bid"    # STRING  – admin UI → orchestrator: start bid campaign
_KEY_CTRL_OD     = "control:scale_od"     # STRING  – admin UI → orchestrator: start on-demand

# ── Lua scripts ───────────────────────────────────────────────────────────────
# KEYS[1]=workers:available  KEYS[2]=workers:draining
# ARGV[1]=lb_id  ARGV[2]=leased_at  ARGV[3]=lease_ttl_sec
#
# Atomically pop the next healthy worker and create a lease.
# Iterates the full list length so stale entries (draining/data-gone)
# are skipped and effectively consumed (cleaned up without re-enqueue).
_CHECKOUT_LUA = """\
local list_len = redis.call('LLEN', KEYS[1])
for i = 1, list_len do
    local wid = redis.call('LPOP', KEYS[1])
    if not wid then break end
    local draining = redis.call('SISMEMBER', KEYS[2], wid)
    local has_data = redis.call('EXISTS', 'workers:data:' .. wid)
    if draining == 0 and has_data == 1 then
        redis.call('HSET', 'workers:leased:' .. wid,
                   'lb_id', ARGV[1], 'leased_at', ARGV[2])
        redis.call('EXPIRE', 'workers:leased:' .. wid, tonumber(ARGV[3]))
        return wid
    end
end
return false
"""

# KEYS[1]=workers:available  KEYS[2]=workers:draining
# ARGV[1]=worker_id  ARGV[2]=lb_id
#
# Return a leased worker to available, or drop if tombstoned/data-gone.
# Returns a status string consumed by the LB for logging.
_CHECKIN_LUA = """\
local leased_key = 'workers:leased:' .. ARGV[1]
local current_lb = redis.call('HGET', leased_key, 'lb_id')
if current_lb and current_lb ~= ARGV[2] then
    return 'NOT_OWNER'
end
redis.call('DEL', leased_key)
local is_draining = redis.call('SISMEMBER', KEYS[2], ARGV[1])
if is_draining == 1 then
    redis.call('SREM', KEYS[2], ARGV[1])
    return 'DRAINED'
end
local has_data = redis.call('EXISTS', 'workers:data:' .. ARGV[1])
if has_data == 0 then
    return 'DATA_GONE'
end
redis.call('RPUSH', KEYS[1], ARGV[1])
return 'RETURNED'
"""

# KEYS[1]=workers:available  ARGV[1]=worker_id
#
# Add worker_id to the tail of the list only if not already present.
# O(N) scan — acceptable for expected worker counts (< 100).
_ENSURE_QUEUED_LUA = """\
local members = redis.call('LRANGE', KEYS[1], 0, -1)
for _, v in ipairs(members) do
    if v == ARGV[1] then return 0 end
end
redis.call('RPUSH', KEYS[1], ARGV[1])
return 1
"""


class WorkerQueue:
    """
    Redis-backed round-robin queue for vLLM worker routing.

    Data model
    ----------
    workers:available           LIST    LPOP = checkout, RPUSH = return (round-robin)
    workers:data:{id}           HASH    host, port, api_key, ssh_port, source_type, added_at
    workers:leased:{id}         HASH    lb_id, leased_at — TTL = lease_ttl_sec (stale-lease guard)
    workers:draining            SET     tombstones: workers removed while an LB holds the lease
    workers:last_active:{id}    STRING  ISO timestamp of the last completed (RETURNED) checkin
                                        Used by the orchestrator for idle-based scale-down.

    Orchestrator writes: register, deregister, sync, reclaim_orphaned, get_idle_workers.
    LB writes:           checkout, checkin (also updates last_active on RETURNED).
    Both sides share the same Redis namespace.
    """

    def __init__(self, r: aioredis.Redis, lease_ttl: int = 300) -> None:
        self._r                  = r
        self._lease_ttl          = lease_ttl
        self._checkout_sha:      str = ""
        self._checkin_sha:       str = ""
        self._ensure_queued_sha: str = ""

    @classmethod
    async def create(cls, redis_url: str, lease_ttl: int = 300) -> "WorkerQueue":
        r = aioredis.from_url(redis_url, decode_responses=True)
        await r.ping()
        q = cls(r, lease_ttl)
        q._checkout_sha      = await r.script_load(_CHECKOUT_LUA)
        q._checkin_sha       = await r.script_load(_CHECKIN_LUA)
        q._ensure_queued_sha = await r.script_load(_ENSURE_QUEUED_LUA)
        log.info("worker_queue.connected", redis_url=redis_url, lease_ttl_sec=lease_ttl)
        return q

    async def close(self) -> None:
        await self._r.aclose()

    # ── Orchestrator operations ───────────────────────────────────────────────

    async def register(
        self,
        worker_id:   str,
        host:        str,
        port:        int,
        api_key:     str,
        source_type: str,
        *,
        ssh_port:    Optional[int] = None,
        added_at:    Optional[datetime] = None,
    ) -> None:
        """Register a worker into the queue. Idempotent: safe to call repeatedly."""
        now_str = (added_at or datetime.now(timezone.utc)).isoformat()
        data: dict[str, str] = {
            "host":        host,
            "port":        str(port),
            "api_key":     api_key,
            "source_type": source_type,
            "added_at":    now_str,
            "ssh_port":    str(ssh_port) if ssh_port else "",
        }
        pipe = self._r.pipeline(transaction=True)
        pipe.hset(_PFX_DATA + worker_id, mapping=data)
        pipe.srem(_KEY_DRAINING, worker_id)   # clear any stale tombstone
        await pipe.execute()

        added = await self._r.evalsha(
            self._ensure_queued_sha, 1, _KEY_AVAILABLE, worker_id
        )
        if added:
            log.info(
                "worker_queue.registered",
                worker_id=worker_id, host=host, port=port, ssh_port=ssh_port,
            )
        else:
            log.debug("worker_queue.register.already_present", worker_id=worker_id)

    async def deregister(self, worker_id: str) -> None:
        """
        Remove a worker from the pool.

        If an LB currently holds the lease, the worker is tombstoned in
        ``workers:draining`` so the LB drops it after the in-flight request
        instead of re-enqueueing it.  The lease key's TTL is left intact
        as a stale-lease guard.
        """
        pipe = self._r.pipeline(transaction=True)
        pipe.lrem(_KEY_AVAILABLE, 0, worker_id)
        pipe.sadd(_KEY_DRAINING, worker_id)
        pipe.delete(_PFX_DATA + worker_id)
        pipe.delete(_PFX_LAST_ACTIVE + worker_id)
        results = await pipe.execute()

        removed_from_queue = int(results[0]) > 0
        is_leased          = bool(await self._r.exists(_PFX_LEASED + worker_id))

        log.info(
            "worker_queue.deregistered",
            worker_id=worker_id,
            was_in_queue=removed_from_queue,
            is_leased_by_lb=is_leased,
            note=(
                "tombstoned — LB will drop worker after current request completes"
                if is_leased
                else "removed immediately"
            ),
        )

    async def sync(self, workers: list[Worker]) -> dict[str, list[str]]:
        """
        Reconcile the Redis queue to match the orchestrator's ground truth.

        Called on orchestrator startup after DynamoDB state reconciliation.
        Repairs any desync caused by a Redis restart, network partition, or
        orchestrator crash.

        Logs a warning for every discrepancy found.
        Returns ``{"added": [...], "removed": [...]}`` with repaired worker IDs.
        """
        expected: dict[str, Worker] = {
            w.worker_id: w for w in workers if w.is_available
        }
        expected_ids = set(expected)

        available_raw = await self._r.lrange(_KEY_AVAILABLE, 0, -1)
        available_ids = set(available_raw)

        leased_keys = await self._r.keys(_PFX_LEASED + "*")
        leased_ids  = {k[len(_PFX_LEASED):] for k in leased_keys}

        data_keys = await self._r.keys(_PFX_DATA + "*")
        data_ids  = {k[len(_PFX_DATA):] for k in data_keys}

        redis_ids = available_ids | leased_ids | data_ids
        to_add    = expected_ids - redis_ids
        to_remove = redis_ids - expected_ids

        result: dict[str, list[str]] = {"added": [], "removed": []}

        for wid in to_add:
            w = expected[wid]
            log.warning(
                "worker_queue.sync.missing_from_redis",
                worker_id=wid, host=w.host, port=w.port,
                note="RUNNING in DynamoDB but absent from Redis — re-registering",
            )
            await self.register(
                wid,
                host=w.host or "",
                port=w.port,
                api_key=w.api_key,
                source_type=w.provider,
            )
            result["added"].append(wid)

        for wid in to_remove:
            already_draining = await self._r.sismember(_KEY_DRAINING, wid)
            if already_draining:
                continue   # a prior deregister already tombstoned it
            log.warning(
                "worker_queue.sync.extra_in_redis",
                worker_id=wid,
                note="in Redis but absent from DynamoDB RUNNING workers — deregistering",
            )
            await self.deregister(wid)
            result["removed"].append(wid)

        if result["added"] or result["removed"]:
            log.warning(
                "worker_queue.sync.desync_repaired",
                added=result["added"],
                removed=result["removed"],
                expected=sorted(expected_ids),
                redis_state_before_sync={
                    "available": sorted(available_ids),
                    "leased":    sorted(leased_ids),
                    "data":      sorted(data_ids),
                },
            )
        else:
            log.info(
                "worker_queue.sync.in_sync",
                worker_count=len(expected_ids),
                available=sorted(available_ids),
                leased=sorted(leased_ids),
            )

        return result

    async def reclaim_orphaned(self) -> int:
        """
        Return workers with expired leases to the available queue.

        A lease TTL expiry means the LB that checked the worker out crashed
        or was killed before calling checkin.  The orchestrator health monitor
        calls this periodically to return those workers to rotation.

        Returns the number of workers reclaimed.
        """
        data_keys = await self._r.keys(_PFX_DATA + "*")
        if not data_keys:
            return 0

        data_ids      = {k[len(_PFX_DATA):] for k in data_keys}
        available_raw = await self._r.lrange(_KEY_AVAILABLE, 0, -1)
        available_ids = set(available_raw)
        leased_keys   = await self._r.keys(_PFX_LEASED + "*")
        leased_ids    = {k[len(_PFX_LEASED):] for k in leased_keys}
        draining_raw  = await self._r.smembers(_KEY_DRAINING)
        draining_ids  = set(draining_raw)

        # Orphaned = has metadata but is neither available, leased, nor draining.
        # These are workers whose lease TTL expired without a checkin.
        orphaned = data_ids - available_ids - leased_ids - draining_ids
        for wid in orphaned:
            await self._r.rpush(_KEY_AVAILABLE, wid)
            log.warning(
                "worker_queue.reclaim_orphaned",
                worker_id=wid,
                note="lease TTL expired without checkin — LB likely crashed; returning to pool",
            )

        if orphaned:
            log.warning(
                "worker_queue.reclaim_orphaned.summary",
                count=len(orphaned),
                worker_ids=sorted(orphaned),
            )

        return len(orphaned)

    # ── LB operations ────────────────────────────────────────────────────────

    async def checkout(self, lb_id: str) -> Optional[LBWorker]:
        """
        Atomically pop the next available worker and lease it to *lb_id*.

        Stale queue entries (draining or data-gone) are skipped and consumed
        in the same Lua call so the queue self-cleans without extra round-trips.

        Returns ``None`` when no healthy worker is currently available.
        """
        now = datetime.now(timezone.utc).isoformat()
        wid: Optional[str] = await self._r.evalsha(
            self._checkout_sha,
            2,
            _KEY_AVAILABLE, _KEY_DRAINING,
            lb_id, now, str(self._lease_ttl),
        )
        if not wid:
            log.debug("worker_queue.checkout.none_available", lb_id=lb_id)
            return None

        data = await self._r.hgetall(_PFX_DATA + wid)
        if not data:
            # Lua confirmed data exists; this is a very narrow TOCTOU edge case.
            log.error(
                "worker_queue.checkout.data_vanished",
                worker_id=wid, lb_id=lb_id,
                note="data key gone between Lua EXISTS check and HGETALL — possible race",
            )
            return None

        worker = _deserialize(wid, data)
        log.debug(
            "worker_queue.checkout",
            worker_id=wid, host=worker.host, lb_id=lb_id,
        )
        return worker

    async def checkin(self, worker_id: str, lb_id: str) -> str:
        """
        Return a leased worker to the available queue, or drop if tombstoned.

        Returns one of:
          ``'RETURNED'``   — re-enqueued; worker is available for the next request
          ``'DRAINED'``    — tombstoned by orchestrator; dropped after in-flight request
          ``'DATA_GONE'``  — metadata missing without a tombstone; dropped
          ``'NOT_OWNER'``  — lease belongs to a different LB instance (TTL reclaim race)
        """
        result: str = await self._r.evalsha(
            self._checkin_sha,
            2,
            _KEY_AVAILABLE, _KEY_DRAINING,
            worker_id, lb_id,
        )
        if result == "RETURNED":
            log.debug("worker_queue.checkin.returned", worker_id=worker_id, lb_id=lb_id)
            await self._r.set(
                _PFX_LAST_ACTIVE + worker_id,
                datetime.now(timezone.utc).isoformat(),
            )
        elif result == "DRAINED":
            log.info(
                "worker_queue.checkin.drained",
                worker_id=worker_id, lb_id=lb_id,
                note="orchestrator tombstoned this worker; dropped after in-flight request",
            )
        elif result == "DATA_GONE":
            log.warning(
                "worker_queue.checkin.data_gone",
                worker_id=worker_id, lb_id=lb_id,
            )
        elif result == "NOT_OWNER":
            log.warning(
                "worker_queue.checkin.not_owner",
                worker_id=worker_id, lb_id=lb_id,
                note="lease owned by another LB — our lease likely expired (slow/crashed LB?)",
            )
        return result

    # ── 503 pressure tracking ────────────────────────────────────────────────

    async def record_503(self) -> None:
        """Increment the 503 counter. Called by the LB on every no-worker response."""
        await self._r.incr(_KEY_503_COUNT)

    async def pop_503_count(self) -> int:
        """
        Atomically read and reset the 503 counter.

        Called by the orchestrator on each health-check tick.  Returns the
        number of 503s that occurred since the last call (0 if none).
        """
        count = await self._r.getdel(_KEY_503_COUNT)
        return int(count) if count else 0

    # ── Auto-scaling support ─────────────────────────────────────────────────

    async def get_utilization(self) -> tuple[int, int, float]:
        """
        Return (available, leased, utilization_ratio) as a single snapshot.

        utilization_ratio = leased / (available + leased), or 0.0 when no workers exist.
        Used by the orchestrator to drive scale-up decisions.
        """
        available = await self._r.llen(_KEY_AVAILABLE)
        leased_keys = await self._r.keys(_PFX_LEASED + "*")
        leased = len(leased_keys)
        total  = available + leased
        ratio  = leased / total if total else 0.0
        return available, leased, ratio

    async def get_idle_workers(self, min_idle_sec: float) -> list[tuple[str, float]]:
        """
        Return workers idle longer than *min_idle_sec*, sorted most-idle first.

        Idle time is measured from ``workers:last_active:{id}`` (set on every
        successful checkin).  Workers that have never served a request use their
        ``added_at`` as the baseline so brand-new workers aren't prematurely
        considered idle.

        Returns a list of ``(worker_id, idle_seconds)`` tuples.
        """
        data_keys = await self._r.keys(_PFX_DATA + "*")
        if not data_keys:
            return []

        now      = datetime.now(timezone.utc)
        result: list[tuple[str, float]] = []

        for k in data_keys:
            wid = k[len(_PFX_DATA):]

            last_raw = await self._r.get(_PFX_LAST_ACTIVE + wid)
            if last_raw:
                last_dt  = datetime.fromisoformat(last_raw)
                idle_sec = (now - last_dt).total_seconds()
            else:
                # Worker has never completed a request — use added_at so a new
                # worker that hasn't received traffic yet doesn't look like it's
                # been idle since epoch.
                data = await self._r.hgetall(k)
                if not data:
                    continue
                added_dt = datetime.fromisoformat(data["added_at"])
                idle_sec = (now - added_dt).total_seconds()

            if idle_sec >= min_idle_sec:
                result.append((wid, idle_sec))
                log.debug(
                    "worker_queue.idle_worker",
                    worker_id=wid,
                    idle_sec=round(idle_sec),
                )

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # ── Admin UI control signals ─────────────────────────────────────────────

    async def signal_scale_bid(self, config: dict | None = None) -> None:
        """Signal orchestrator to start an interruptible bid campaign.
        Config may include: provider (str), image (str)."""
        await self._r.set(_KEY_CTRL_BID, json.dumps(config or {}), ex=300)

    async def signal_scale_on_demand(self, config: dict | None = None) -> None:
        """Signal orchestrator to launch an on-demand instance."""
        await self._r.set(_KEY_CTRL_OD, json.dumps(config or {}), ex=300)

    async def pop_control_signal(self) -> tuple[str | None, dict]:
        """
        Consume and return (kind, config) if a manual scale signal is pending.
        kind is 'bid', 'on_demand', or None.  Consumes only one signal per call.
        """
        for key, kind in [(_KEY_CTRL_BID, "bid"), (_KEY_CTRL_OD, "on_demand")]:
            raw = await self._r.getdel(key)
            if raw is not None:
                try:
                    cfg = json.loads(raw)
                except Exception:
                    cfg = {}
                return kind, cfg
        return None, {}

    # ── Observability ────────────────────────────────────────────────────────

    async def list_state(self) -> dict:
        """Return a snapshot of current queue state for health/debug endpoints."""
        available_raw = await self._r.lrange(_KEY_AVAILABLE, 0, -1)
        leased_keys   = await self._r.keys(_PFX_LEASED + "*")
        draining_raw  = await self._r.smembers(_KEY_DRAINING)
        now           = datetime.now(timezone.utc)

        available_detail = []
        for wid in available_raw:
            last_raw = await self._r.get(_PFX_LAST_ACTIVE + wid)
            idle_sec = (
                round((now - datetime.fromisoformat(last_raw)).total_seconds())
                if last_raw else None
            )
            available_detail.append({"worker_id": wid, "idle_sec": idle_sec})

        leased = []
        for k in leased_keys:
            wid  = k[len(_PFX_LEASED):]
            info = await self._r.hgetall(k)
            ttl  = await self._r.ttl(k)
            leased.append({
                "worker_id":           wid,
                "lb_id":               info.get("lb_id"),
                "leased_at":           info.get("leased_at"),
                "lease_ttl_remaining": ttl,
            })

        return {
            "available": available_detail,
            "leased":    leased,
            "draining":  list(draining_raw),
        }


# ── Deserialization ───────────────────────────────────────────────────────────

def _deserialize(worker_id: str, data: dict) -> LBWorker:
    ssh_raw = data.get("ssh_port", "")
    return LBWorker(
        worker_id=worker_id,
        source_type=data.get("source_type", ""),
        host=data["host"],
        port=int(data["port"]),
        api_key=data["api_key"],
        added_at=datetime.fromisoformat(data["added_at"]),
        ssh_port=int(ssh_raw) if ssh_raw else None,
    )
