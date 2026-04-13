from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, HTTPException, Request

from .config import settings
from .db import LBDB
from .proxy import proxy_request
from .router import RoundRobin

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
router = RoundRobin(db)
_start = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "lb.startup",
        lb_workers_table=settings.lb_workers_table,
        worker_cache_ttl=settings.worker_cache_ttl,
        log_level=settings.log_level,
    )
    await router.force_refresh()
    yield
    log.info("lb.shutdown")


app = FastAPI(
    title="eased load balancer",
    description="Round-robin load balancer for vLLM workers.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Ops ───────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe — no auth required."""
    workers = db.list_workers()
    uptime  = (datetime.now(timezone.utc) - _start).total_seconds()
    return {
        "status":      "ok",
        "uptime_sec":  round(uptime),
        "worker_count": len(workers),
    }


@app.get("/workers", tags=["ops"])
async def list_workers():
    """Return all workers currently in the load-balancer pool."""
    workers = db.list_workers()
    return {
        "workers": [w.model_dump(mode="json") for w in workers],
        "total":   len(workers),
    }


# ── Proxy ─────────────────────────────────────────────────────────────────────

@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    tags=["proxy"],
)
async def proxy(path: str, request: Request):
    """Round-robin proxy to the next healthy vLLM worker."""
    worker = await router.get_next()
    if not worker:
        log.warning("lb.no_workers_available", path=path)
        raise HTTPException(503, detail="No workers available")

    structlog.contextvars.bind_contextvars(worker_id=worker.worker_id)
    return await proxy_request(request, worker, db)
