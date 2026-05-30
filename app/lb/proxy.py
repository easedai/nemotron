from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Optional

import httpx
import structlog
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..config import settings
from ..lb_db import LBDB
from ..models import LBWorker
from .router import WorkerRouter
from .ssh_logs import fetch_vllm_logs

log = structlog.get_logger(__name__)

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "host", "content-length",
    # Always stripped so the LB can inject the correct worker API key below.
    "authorization",
    # vLLM 0.19+ has CSRF protection: it rejects requests where Origin / Referer
    # doesn't match the Host header.  The LB rewrites Host to the worker URL but
    # forwards Origin unchanged from the client → CSRF token mismatch (400).
    # Strip both so vLLM's CSRF check never fires on proxied requests.
    "origin", "referer",
})


def _summarise_body(body: bytes) -> Optional[str]:
    """
    Return a concise one-line summary of a request body for logging.

    For chat completions: "model=X messages=N [system:text, user:text+image_url]"
    For anything else:    first 120 chars of raw UTF-8.
    """
    if not body:
        return None
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError
        parts = []
        if "model" in data:
            parts.append(f"model={data['model']}")
        msgs = data.get("messages")
        if isinstance(msgs, list):
            parts.append(f"messages={len(msgs)}")
            content_summary = []
            for m in msgs:
                role    = m.get("role", "?")
                content = m.get("content", "")
                if isinstance(content, str):
                    content_summary.append(f"{role}:text")
                elif isinstance(content, list):
                    types = "+".join(
                        p.get("type", "?") for p in content if isinstance(p, dict)
                    )
                    content_summary.append(f"{role}:{types}")
            if content_summary:
                parts.append("[" + ", ".join(content_summary) + "]")
        elif "prompt" in data:
            prompt = str(data["prompt"])
            parts.append(f"prompt={prompt[:80]!r}")
        return " ".join(parts) if parts else body[:120].decode("utf-8", errors="replace")
    except Exception:
        return body[:120].decode("utf-8", errors="replace")


async def _log_worker_ssh_logs(worker: LBWorker) -> None:
    """Fire-and-forget: SSH into worker and log the vLLM tail on 5xx."""
    if not worker.ssh_port:
        log.debug("lb.ssh_logs.skipped", worker_id=worker.worker_id, reason="ssh_port not set")
        return
    text = await fetch_vllm_logs(
        host=worker.host,
        ssh_port=worker.ssh_port,
    )
    if text:
        log.error(
            "lb.worker_vllm_logs",
            worker_id=worker.worker_id,
            host=worker.host,
            ssh_port=worker.ssh_port,
            logs=text,
        )


async def proxy_request(
    request: Request,
    worker:  LBWorker,
    db:      LBDB,
    router:  WorkerRouter,
) -> Response:
    """
    Forward an incoming request to *worker* and stream the response back.

    The worker is checked out (leased) before this call.  checkin is
    guaranteed to run at every exit path — streaming finally block,
    buffered error returns, and connection-level exceptions — so the
    worker always re-enters the Redis queue (or is dropped if draining).

    DynamoDB success/failure counters are updated fire-and-forget so stats
    never block the response path.
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
    forwarded_headers["Authorization"] = f"Bearer {worker.api_key}"

    # Salad gateway auth: every request to a Salad worker must carry the
    # organization API key so the HTTPS gateway accepts it.
    if worker.source_type == "salad" and settings.salad_api_key:
        forwarded_headers["Salad-Api-Key"] = settings.salad_api_key

    body = await request.body()

    log.info(
        "lb.proxy.forward",
        method=request.method,
        path=request.url.path,
        worker_id=worker.worker_id,
        worker_host=worker.host,
        body_bytes=len(body),
        body=_summarise_body(body),
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

        succeeded  = upstream.status_code < 400
        headers_ms = round((time.monotonic() - t_start) * 1000, 1)

        log.info(
            "lb.proxy.upstream_response",
            status_code=upstream.status_code,
            worker_id=worker.worker_id,
            headers_ms=headers_ms,
        )

        # ── 5xx: buffer body, log it, SSH in for vLLM tail ────────────────
        if upstream.status_code >= 500:
            error_body = await upstream.aread()
            latency_ms = round((time.monotonic() - t_start) * 1000, 1)
            await upstream.aclose()
            await client.aclose()
            log.error(
                "lb.proxy.upstream_error",
                status_code=upstream.status_code,
                worker_id=worker.worker_id,
                worker_host=worker.host,
                path=request.url.path,
                latency_ms=latency_ms,
                request=_summarise_body(body),
                response=error_body[:500].decode("utf-8", errors="replace"),
            )
            asyncio.create_task(asyncio.to_thread(db.increment_failure, worker.worker_id))
            asyncio.create_task(router.checkin(worker.worker_id))
            asyncio.create_task(_log_worker_ssh_logs(worker))
            return Response(
                content=error_body,
                status_code=upstream.status_code,
                headers=response_headers,
            )

        # ── 4xx: buffer and log without SSH (client error, not worker fault)
        if not succeeded:
            error_body = await upstream.aread()
            latency_ms = round((time.monotonic() - t_start) * 1000, 1)
            await upstream.aclose()
            await client.aclose()
            log.warning(
                "lb.proxy.client_error",
                status_code=upstream.status_code,
                worker_id=worker.worker_id,
                path=request.url.path,
                latency_ms=latency_ms,
                response=error_body[:300].decode("utf-8", errors="replace"),
            )
            asyncio.create_task(asyncio.to_thread(db.increment_failure, worker.worker_id))
            asyncio.create_task(router.checkin(worker.worker_id))
            return Response(
                content=error_body,
                status_code=upstream.status_code,
                headers=response_headers,
            )

        # ── 2xx/3xx: stream and checkin when stream is exhausted ──────────
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
                asyncio.create_task(
                    asyncio.to_thread(db.increment_success, worker.worker_id)
                )
                asyncio.create_task(router.checkin(worker.worker_id))

        return StreamingResponse(
            stream_and_close(),
            status_code=upstream.status_code,
            headers=response_headers,
        )

    except httpx.ConnectError as exc:
        _handle_connection_error(db, router, worker, t_start, "connect_error", str(exc))
        raise HTTPException(502, f"Cannot connect to worker at {worker.host}: {exc}")
    except httpx.TimeoutException as exc:
        _handle_connection_error(db, router, worker, t_start, "timeout", str(exc))
        raise HTTPException(504, "Worker request timed out")
    except Exception as exc:
        _handle_connection_error(db, router, worker, t_start, "error", str(exc))
        raise HTTPException(502, f"Proxy error: {exc}")


def _handle_connection_error(
    db:      LBDB,
    router:  WorkerRouter,
    worker:  LBWorker,
    t_start: float,
    kind:    str,
    detail:  str,
) -> None:
    """Log, record failure stat, and checkin the worker so it re-enters the queue."""
    log.error(
        f"lb.proxy.{kind}",
        worker_id=worker.worker_id,
        latency_ms=round((time.monotonic() - t_start) * 1000, 1),
        error=detail,
    )
    asyncio.create_task(asyncio.to_thread(db.increment_failure, worker.worker_id))
    asyncio.create_task(router.checkin(worker.worker_id))
