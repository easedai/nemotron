from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional

import httpx
import structlog
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .db import LBDB
from .models import LBWorker

log = structlog.get_logger(__name__)

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "host", "content-length",
})


async def proxy_request(request: Request, worker: LBWorker, db: LBDB) -> Response:
    """
    Forward an incoming request to *worker* and stream the response back.

    Increments ``successful_requests`` or ``failed_requests`` on the worker's
    DynamoDB record after each request completes (fire-and-forget so stats
    never block the response path).
    """
    t_start    = time.monotonic()
    target_url = f"{worker.base_url}{request.url.path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    forwarded_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    # Replace the client's Bearer token with the worker-specific API key
    forwarded_headers["Authorization"] = f"Bearer {worker.api_key}"

    body = await request.body()

    log.info(
        "lb.proxy.forward",
        method=request.method,
        path=request.url.path,
        worker_id=worker.worker_id,
        worker_host=worker.host,
        body_bytes=len(body),
    )

    try:
        client           = httpx.AsyncClient(timeout=None)
        upstream_request = client.build_request(
            method=request.method,
            url=target_url,
            headers=forwarded_headers,
            content=body,
        )
        upstream = await client.send(upstream_request, stream=True)

        response_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }

        log.info(
            "lb.proxy.upstream_response",
            status_code=upstream.status_code,
            worker_id=worker.worker_id,
            headers_ms=round((time.monotonic() - t_start) * 1000, 1),
        )

        succeeded = upstream.status_code < 400

        async def stream_and_close() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes(chunk_size=4096):
                    yield chunk
            finally:
                latency_ms = round((time.monotonic() - t_start) * 1000, 1)
                await upstream.aclose()
                await client.aclose()
                log.info(
                    "lb.proxy.complete",
                    worker_id=worker.worker_id,
                    status_code=upstream.status_code,
                    latency_ms=latency_ms,
                )
                # Fire-and-forget stat update — never blocks the response
                if succeeded:
                    asyncio.create_task(
                        asyncio.to_thread(db.increment_success, worker.worker_id)
                    )
                else:
                    asyncio.create_task(
                        asyncio.to_thread(db.increment_failure, worker.worker_id)
                    )

        return StreamingResponse(
            stream_and_close(),
            status_code=upstream.status_code,
            headers=response_headers,
        )

    except httpx.ConnectError as exc:
        _log_and_count_error(db, worker.worker_id, t_start, "connect_error", str(exc))
        raise HTTPException(502, f"Cannot connect to worker at {worker.host}: {exc}")
    except httpx.TimeoutException as exc:
        _log_and_count_error(db, worker.worker_id, t_start, "timeout", str(exc))
        raise HTTPException(504, "Worker request timed out")
    except Exception as exc:
        _log_and_count_error(db, worker.worker_id, t_start, "error", str(exc))
        raise HTTPException(502, f"Proxy error: {exc}")


def _log_and_count_error(
    db: LBDB, worker_id: str, t_start: float, kind: str, detail: str
) -> None:
    log.error(
        f"lb.proxy.{kind}",
        worker_id=worker_id,
        latency_ms=round((time.monotonic() - t_start) * 1000, 1),
        error=detail,
    )
    asyncio.create_task(
        asyncio.to_thread(db.increment_failure, worker_id)
    )
