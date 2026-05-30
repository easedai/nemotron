from __future__ import annotations

import asyncio
import logging
import socket
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import settings
from ..lb_db import LBDB
from ..lb_queue import WorkerQueue
from .admin import router as admin_router
from .proxy import proxy_request
from .router import WorkerRouter
from .ui import router as ui_router

# ── Structured logging ────────────────────────────────────────────────────────

logging.basicConfig(format="%(message)s", level=settings.log_level.upper())
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
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

# ── App lifecycle ─────────────────────────────────────────────────────────────

db     = LBDB()
_start = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    lb_id  = settings.lb_instance_id or socket.gethostname()
    queue  = await WorkerQueue.create(
        settings.redis_url,
        lease_ttl=settings.redis_lease_ttl_sec,
    )
    router = WorkerRouter(queue, lb_id)

    app.state.queue  = queue
    app.state.router = router
    app.state.lb_id  = lb_id

    state = await queue.list_state()
    log.info(
        "lb.startup",
        lb_id=lb_id,
        redis_url=settings.redis_url,
        lease_ttl_sec=settings.redis_lease_ttl_sec,
        queue_available=[e["worker_id"] for e in state["available"]],
        queue_leased=[e["worker_id"] for e in state["leased"]],
    )
    yield
    await queue.close()
    log.info("lb.shutdown", lb_id=lb_id)


app = FastAPI(
    title="eased load balancer",
    description="Round-robin load balancer + admin API for eased vLLM workers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(admin_router)
app.include_router(ui_router)

# ── Ops ───────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health(request: Request):
    """Liveness probe — no auth required."""
    queue  = request.app.state.queue
    state  = await queue.list_state()
    uptime = (datetime.now(timezone.utc) - _start).total_seconds()
    return {
        "status":          "ok",
        "lb_id":           request.app.state.lb_id,
        "uptime_sec":      round(uptime),
        "available_count": len(state["available"]),
        "leased_count":    len(state["leased"]),
    }


@app.get("/workers", tags=["ops"])
async def list_workers(request: Request):
    """Return current queue state from Redis."""
    queue = request.app.state.queue
    state = await queue.list_state()
    return {
        "available": state["available"],
        "leased":    state["leased"],
        "draining":  state["draining"],
        "total":     len(state["available"]) + len(state["leased"]),
        "utilization": round(
            len(state["leased"]) / max(1, len(state["available"]) + len(state["leased"])), 2
        ),
    }


# ── Proxy ─────────────────────────────────────────────────────────────────────

@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    tags=["proxy"],
)
async def proxy(path: str, request: Request):
    """Round-robin proxy to the next available vLLM worker."""
    router: WorkerRouter = request.app.state.router
    lb_id:  str          = request.app.state.lb_id

    worker = await router.checkout()
    if not worker:
        log.warning("lb.no_workers_available", path=path)
        # Record pressure so the orchestrator can react even before any worker is registered.
        asyncio.create_task(request.app.state.queue.record_503())
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "The service is temporarily unavailable — please retry shortly.",
                    "type":    "service_unavailable",
                    "code":    503,
                }
            },
        )

    structlog.contextvars.bind_contextvars(worker_id=worker.worker_id)
    return await proxy_request(request, worker, db, router)
