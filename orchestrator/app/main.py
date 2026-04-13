from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .models import WorkerStatus
from .worker_manager import WorkerManager

# ── Structured logging ────────────────────────────────────────────────────────

logging.basicConfig(format="%(message)s", level=settings.log_level.upper())

# httpx and httpcore emit very verbose transport-level events (connect_tcp,
# send_request_body, receive_response_headers, etc.) at DEBUG level.
# Keep them at WARNING regardless of the app log level.
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Human-readable in DEBUG; JSON everywhere else
        (
            structlog.dev.ConsoleRenderer()
            if settings.log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer()
        ),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper())
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)

# ── Admin authentication ──────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if credentials is None or credentials.credentials != settings.admin_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── App lifecycle ─────────────────────────────────────────────────────────────

manager = WorkerManager()
_start_time = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "orchestrator.startup",
        orchestrator_id=settings.orchestrator_id,
        log_level=settings.log_level,
        dynamodb_table=settings.dynamodb_table,
        worker_image=settings.worker_image,
        bid_start_pct=settings.bid_start_pct,
        bid_max_multiplier=settings.bid_max_multiplier,
    )
    await manager.start()
    yield
    log.info("orchestrator.shutdown")
    await manager.stop()


app = FastAPI(
    title="eased orchestrator",
    description="Agentic GPU worker orchestrator — bids on vast.ai, proxies vLLM requests.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Health & status ───────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Basic liveness probe — no auth required."""
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()
    return {"status": "ok", "uptime_sec": round(uptime)}


@app.get("/admin/health", tags=["admin"], dependencies=[Depends(require_admin)])
async def admin_health():
    """Detailed fleet health — requires admin token."""
    now     = datetime.now(timezone.utc)
    uptime  = (now - _start_time).total_seconds()
    workers = manager.db.list_workers()

    by_status: dict[str, list] = {}
    for w in workers:
        by_status.setdefault(w.status.value, []).append(w)

    def _instance_cost(w) -> float:
        if w.bid_price and w.running_since:
            hrs = (now - w.running_since).total_seconds() / 3600
            return w.bid_price * hrs
        return 0.0

    running   = by_status.get("running", [])
    unhealthy = by_status.get("unhealthy", [])
    starting  = by_status.get("starting", [])
    pending   = by_status.get("pending", [])
    terminated = by_status.get("terminated", [])

    active_rate = sum(
        w.bid_price for w in workers
        if w.bid_price and w.status.value in ("running", "unhealthy", "starting", "pending")
    )
    total_spent = sum(_instance_cost(w) for w in workers)

    worker_detail = [
        {
            "worker_id":    w.worker_id,
            "instance_id":  w.instance_id,
            "label":        w.label,
            "status":       w.status.value,
            "worker_type":  w.worker_type.value,
            "gpu_name":     w.gpu_name,
            "gpu_ram_gb":   w.gpu_ram_gb,
            "host":         w.host,
            "port":         w.port,
            "bid_price":    w.bid_price,
            "market_price": w.market_price,
            "running_since": w.running_since.isoformat() if w.running_since else None,
            "uptime_hr":    round((now - w.running_since).total_seconds() / 3600, 2)
                            if w.running_since else None,
            "cost_usd":     round(_instance_cost(w), 6),
            "specs":        w.specs,
        }
        for w in workers
        if w.status.value not in ("terminated",)
    ]

    return {
        "status":          "ok",
        "uptime_sec":      round(uptime),
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


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.get("/admin/workers", tags=["admin"], dependencies=[Depends(require_admin)])
async def list_workers():
    """List all workers and their current state."""
    workers = manager.db.list_workers()
    return {
        "workers": [w.model_dump(mode="json") for w in workers],
        "total":   len(workers),
    }


@app.post("/admin/workers/refresh", tags=["admin"], dependencies=[Depends(require_admin)])
async def refresh_workers():
    """
    Manually trigger state reconciliation and ensure-worker logic.
    Useful after a vast.ai interruption or orchestrator restart.
    """
    log.info("admin.refresh_workers")
    await manager._reconcile_state()
    await manager._ensure_worker()
    return {"status": "triggered"}


@app.post("/admin/workers/{worker_id}/terminate", tags=["admin"], dependencies=[Depends(require_admin)])
async def terminate_worker(worker_id: str):
    """Destroy a specific worker and remove it from the pool."""
    worker = manager.db.get_worker(worker_id)
    if not worker:
        raise HTTPException(404, f"Worker {worker_id!r} not found")

    log.info("admin.terminate_worker", worker_id=worker_id, instance_id=worker.instance_id)
    if worker.instance_id:
        try:
            await manager.vast.destroy_instance(worker.instance_id)
        except Exception as exc:
            log.warning("admin.terminate_worker.destroy_failed", error=str(exc))

    manager.db.update_worker_status(worker_id, WorkerStatus.TERMINATED)
    manager.lb.deregister(worker_id)
    return {"status": "terminated", "worker_id": worker_id}


@app.get("/admin/events/worker/{worker_id}", tags=["admin"], dependencies=[Depends(require_admin)])
async def get_worker_events(worker_id: str, limit: int = 100):
    """Lifecycle events + log snapshots for a specific worker_id."""
    return {"events": manager.events.query_by_worker(worker_id, limit=limit)}


@app.get("/admin/events/instance/{instance_id}", tags=["admin"], dependencies=[Depends(require_admin)])
async def get_instance_events(instance_id: str, limit: int = 100):
    """Lifecycle events + log snapshots for a specific vast.ai instance ID."""
    return {"events": manager.events.query_by_instance(instance_id, limit=limit)}


@app.get("/admin/events/label/{label}", tags=["admin"], dependencies=[Depends(require_admin)])
async def get_label_events(label: str, limit: int = 100):
    """Lifecycle events + log snapshots for an instance label (e.g. eased-abc123)."""
    return {"events": manager.events.query_by_label(label, limit=limit)}


@app.post("/admin/bid", tags=["admin"], dependencies=[Depends(require_admin)])
async def trigger_bid():
    """Manually kick off a new bid campaign (e.g. to scale up)."""
    import asyncio
    log.info("admin.trigger_bid")
    asyncio.create_task(manager._bidding_campaign(), name="admin-bid-campaign")
    return {"status": "bid_campaign_started"}


# /v1/* requests are handled by the load-balancer service, not here.
