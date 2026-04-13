from __future__ import annotations

import asyncio
import logging
import signal

import structlog

from ..config import settings
from .worker_manager import WorkerManager

# ── Structured logging ────────────────────────────────────────────────────────

logging.basicConfig(format="%(message)s", level=settings.log_level.upper())
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


async def main() -> None:
    if not settings.vastai_api_key:
        raise RuntimeError("VASTAI_API_KEY is required for the orchestrator")
    if not settings.discord_webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL is required for the orchestrator")

    manager = WorkerManager()

    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        log.info("orchestrator.signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    log.info(
        "orchestrator.startup",
        orchestrator_id=settings.orchestrator_id,
        log_level=settings.log_level,
        dynamodb_table=settings.dynamodb_table,
        worker_image=settings.worker_image,
    )

    await manager.start()
    try:
        await stop_event.wait()
    finally:
        log.info("orchestrator.shutdown")
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
