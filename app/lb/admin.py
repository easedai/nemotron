from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..config import settings
from ..event_store import EventStore
from ..models import WorkerStatus
from ..worker_db import DynamoDB

log = structlog.get_logger(__name__)

router  = APIRouter(prefix="/admin", tags=["admin"])
_bearer = HTTPBearer(auto_error=False)

# Module-level singletons that don't touch Redis (created once on import)
_wdb    = DynamoDB()
_events = EventStore()


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if credentials is None or credentials.credentials != settings.admin_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Health & fleet status ─────────────────────────────────────────────────────

@router.get("/health", dependencies=[Depends(require_admin)])
async def admin_health():
    """Detailed fleet health — requires admin token."""
    now     = datetime.now(timezone.utc)
    workers = _wdb.list_workers()

    by_status: dict[str, list] = {}
    for w in workers:
        by_status.setdefault(w.status.value, []).append(w)

    def _instance_cost(w) -> float:
        if w.bid_price and w.running_since:
            hrs = (now - w.running_since).total_seconds() / 3600
            return w.bid_price * hrs
        return 0.0

    running    = by_status.get("running",    [])
    unhealthy  = by_status.get("unhealthy",  [])
    starting   = by_status.get("starting",   [])
    pending    = by_status.get("pending",    [])
    terminated = by_status.get("terminated", [])

    active_rate = sum(
        w.bid_price for w in workers
        if w.bid_price and w.status.value in ("running", "unhealthy", "starting", "pending")
    )
    total_spent = sum(_instance_cost(w) for w in workers)

    worker_detail = [
        {
            "worker_id":     w.worker_id,
            "instance_id":   w.instance_id,
            "label":         w.label,
            "status":        w.status.value,
            "worker_type":   w.worker_type.value,
            "gpu_name":      w.gpu_name,
            "gpu_ram_gb":    w.gpu_ram_gb,
            "host":          w.host,
            "port":          w.port,
            "bid_price":     w.bid_price,
            "market_price":  w.market_price,
            "running_since": w.running_since.isoformat() if w.running_since else None,
            "uptime_hr":     round((now - w.running_since).total_seconds() / 3600, 2)
                             if w.running_since else None,
            "cost_usd":      round(_instance_cost(w), 6),
            "specs":         w.specs,
        }
        for w in workers
        if w.status.value != "terminated"
    ]

    return {
        "status":    "ok",
        "counts": {
            "running":    len(running),
            "unhealthy":  len(unhealthy),
            "starting":   len(starting),
            "pending":    len(pending),
            "terminated": len(terminated),
        },
        "spend_rate_per_hr": round(active_rate, 6),
        "total_spent_usd":   round(total_spent, 6),
        "workers":           worker_detail,
    }


@router.get("/workers", dependencies=[Depends(require_admin)])
async def list_workers():
    """List all workers and their current state from DynamoDB."""
    workers = _wdb.list_workers()
    return {
        "workers": [w.model_dump(mode="json") for w in workers],
        "total":   len(workers),
    }


@router.get("/queue", dependencies=[Depends(require_admin)])
async def queue_state(request: Request):
    """Live Redis queue state — available, leased, and draining workers."""
    queue = request.app.state.queue
    return await queue.list_state()


# ── Terminate ─────────────────────────────────────────────────────────────────

@router.post("/workers/{worker_id}/terminate", dependencies=[Depends(require_admin)])
async def terminate_worker(worker_id: str, request: Request):
    """
    Destroy a specific worker and remove it from the pool.

    Calls the vast.ai API to immediately destroy the instance, marks the
    worker TERMINATED in DynamoDB, and deregisters it from the Redis queue.
    If an LB currently holds the lease, the worker is tombstoned so it is
    dropped after the in-flight request completes.
    """
    worker = _wdb.get_worker(worker_id)
    if not worker:
        raise HTTPException(404, f"Worker {worker_id!r} not found")

    log.info("admin.terminate_worker", worker_id=worker_id, instance_id=worker.instance_id)

    if worker.instance_id and settings.vastai_api_key:
        from ..orchestrator.vast_client import VastAIClient
        try:
            await VastAIClient().destroy_instance(worker.instance_id)
        except Exception as exc:
            log.warning("admin.terminate_worker.destroy_failed", error=str(exc))
    elif worker.instance_id and not settings.vastai_api_key:
        log.warning(
            "admin.terminate_worker.no_vastai_key",
            worker_id=worker_id,
            note="VASTAI_API_KEY not set — instance not destroyed on vast.ai",
        )

    _wdb.update_worker_status(worker_id, WorkerStatus.TERMINATED)

    queue = request.app.state.queue
    await queue.deregister(worker_id)

    return {"status": "terminated", "worker_id": worker_id}


# ── Events ────────────────────────────────────────────────────────────────────

@router.get("/events/worker/{worker_id}", dependencies=[Depends(require_admin)])
async def get_worker_events(worker_id: str, limit: int = 100):
    """Lifecycle events + log snapshots for a specific worker_id."""
    return {"events": _events.query_by_worker(worker_id, limit=limit)}


@router.get("/events/instance/{instance_id}", dependencies=[Depends(require_admin)])
async def get_instance_events(instance_id: str, limit: int = 100):
    """Lifecycle events + log snapshots for a specific vast.ai instance ID."""
    return {"events": _events.query_by_instance(instance_id, limit=limit)}


@router.get("/events/label/{label}", dependencies=[Depends(require_admin)])
async def get_label_events(label: str, limit: int = 100):
    """Lifecycle events + log snapshots for an instance label (e.g. eased-abc123)."""
    return {"events": _events.query_by_label(label, limit=limit)}


# ── Config (read-only) ────────────────────────────────────────────────────────

@router.get("/config", dependencies=[Depends(require_admin)])
async def get_config():
    """Return current orchestrator settings (read-only, safe to expose to the UI)."""
    return {
        "providers":               settings.provider_list,
        "worker_image":            settings.worker_image,
        "model_id":                settings.model_id,
        "max_instances":           settings.max_instances,
        "min_instances":           settings.min_instances,
        "bid_start_pct":           settings.bid_start_pct,
        "bid_max_multiplier":      settings.bid_max_multiplier,
        "min_gpu_ram_gb":          settings.min_gpu_ram_gb,
        "scale_up_threshold":      settings.scale_up_threshold,
        "scale_up_cooldown_sec":   settings.scale_up_cooldown_sec,
        "health_check_interval_sec": settings.health_check_interval_sec,
    }


# ── Scale controls ────────────────────────────────────────────────────────────

class ScaleRequest(BaseModel):
    provider:   Optional[str] = None   # pin to a specific provider (e.g. "vastai")
    image:      Optional[str] = None   # override worker image for this launch


@router.post("/scale/bid", dependencies=[Depends(require_admin)])
async def trigger_bid(body: ScaleRequest, request: Request):
    """Signal the orchestrator to start an interruptible (spot) bid campaign."""
    queue = request.app.state.queue
    cfg: dict = {}
    if body.provider:
        cfg["provider"] = body.provider
    if body.image:
        cfg["image"] = body.image
    await queue.signal_scale_bid(cfg or None)
    return {
        "status": "signal_sent",
        "kind":   "bid",
        "config": cfg,
        "note":   "Orchestrator will start bidding on its next health-check tick (~30 s).",
    }


@router.post("/scale/on-demand", dependencies=[Depends(require_admin)])
async def trigger_on_demand(body: ScaleRequest, request: Request):
    """Signal the orchestrator to launch an on-demand (non-spot) instance."""
    queue = request.app.state.queue
    cfg: dict = {}
    if body.provider:
        cfg["provider"] = body.provider
    if body.image:
        cfg["image"] = body.image
    await queue.signal_scale_on_demand(cfg or None)
    return {
        "status": "signal_sent",
        "kind":   "on_demand",
        "config": cfg,
        "note":   "Orchestrator will launch on-demand on its next health-check tick (~30 s).",
    }
