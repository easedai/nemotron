from __future__ import annotations

import time
from typing import AsyncIterator, Optional

import httpx
import structlog
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .config import settings
from .metrics import classify_request, metrics
from .models import Worker

log = structlog.get_logger(__name__)

# Headers that must not be forwarded to the upstream worker
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "host", "content-length",
})

# SSE prefix that marks real token data (skip "[DONE]" sentinels and empty lines)
_SSE_DATA_PREFIX = b"data: "


async def proxy_request(request: Request, worker: Worker) -> Response:
    """
    Forward an incoming client request to the active worker and stream the
    response back.  Handles both regular JSON responses and SSE streams
    (used by vLLM when stream=true).

    Timing
    ------
    t_start      — clock at entry (before body read)
    t_first_byte — clock when the first non-empty SSE chunk arrives (TTFT for
                   streaming requests; first response byte for non-streaming)
    t_end        — clock when the last byte leaves the generator

    All three are recorded as CloudWatch metrics and logged via structlog.
    """
    if not worker.base_url:
        raise HTTPException(502, "Worker has no address")

    t_start = time.monotonic()

    target_url = f"{worker.base_url}{request.url.path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Strip hop-by-hop headers; inject the worker-specific Bearer token
    forwarded_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    forwarded_headers["Authorization"] = f"Bearer {worker.api_key}"

    body = await request.body()
    req_type = classify_request(request.url.path, body)

    log.info(
        "proxy.forward",
        method=request.method,
        path=request.url.path,
        request_type=req_type,
        worker_id=worker.worker_id,
        worker_host=worker.host,
        target_url=target_url,
        body_bytes=len(body),
    )

    try:
        client = httpx.AsyncClient(timeout=None)
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

        t_headers = time.monotonic()
        log.info(
            "proxy.upstream_response",
            status_code=upstream.status_code,
            worker_id=worker.worker_id,
            request_type=req_type,
            content_type=upstream.headers.get("content-type"),
            headers_ms=round((t_headers - t_start) * 1000, 1),
        )

        status = "success" if upstream.status_code < 400 else "error"

        # State tracked inside the generator closure
        state: dict = {
            "t_first_chunk": None,  # monotonic time of first non-empty chunk
            "chunks":        0,
        }

        async def stream_and_close() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes(chunk_size=4096):
                    if chunk:
                        if state["t_first_chunk"] is None:
                            state["t_first_chunk"] = time.monotonic()
                        state["chunks"] += 1
                    yield chunk
            finally:
                t_end = time.monotonic()
                await upstream.aclose()
                await client.aclose()

                latency_ms = (t_end - t_start) * 1000

                # TTFT: time from request start to the first chunk that
                # carries actual SSE token data.  For non-streaming requests
                # this equals the time-to-first-byte (same thing).
                ttft_ms: Optional[float] = None
                if state["t_first_chunk"] is not None:
                    ttft_ms = (state["t_first_chunk"] - t_start) * 1000

                log_fn = (
                    log.warning
                    if latency_ms > settings.latency_warn_threshold_ms
                    else log.info
                )
                log_fn(
                    "proxy.complete",
                    worker_id=worker.worker_id,
                    request_type=req_type,
                    status=status,
                    status_code=upstream.status_code,
                    latency_ms=round(latency_ms, 1),
                    ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
                    chunks=state["chunks"],
                )

                metrics.emit(
                    request_type=req_type,
                    worker_id=worker.worker_id,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    status=status,
                )

        return StreamingResponse(
            stream_and_close(),
            status_code=upstream.status_code,
            headers=response_headers,
        )

    except httpx.ConnectError as exc:
        _emit_error(req_type, worker.worker_id, t_start)
        log.error(
            "proxy.connect_error",
            worker_id=worker.worker_id,
            target_url=target_url,
            error=str(exc),
        )
        raise HTTPException(502, f"Cannot connect to worker at {worker.host}: {exc}")
    except httpx.TimeoutException as exc:
        _emit_error(req_type, worker.worker_id, t_start)
        log.error(
            "proxy.timeout",
            worker_id=worker.worker_id,
            target_url=target_url,
            error=str(exc),
        )
        raise HTTPException(504, "Worker request timed out")
    except Exception as exc:
        _emit_error(req_type, worker.worker_id, t_start)
        log.error(
            "proxy.error",
            worker_id=worker.worker_id,
            target_url=target_url,
            error=str(exc),
        )
        raise HTTPException(502, f"Proxy error: {exc}")


def _emit_error(request_type: str, worker_id: str, t_start: float) -> None:
    """Emit a CloudWatch metric for a request that failed before streaming."""
    metrics.emit(
        request_type=request_type,
        worker_id=worker_id,
        latency_ms=(time.monotonic() - t_start) * 1000,
        status="error",
    )
