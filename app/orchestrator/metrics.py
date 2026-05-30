from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

import boto3
import structlog

from ..config import settings

log = structlog.get_logger(__name__)

NAMESPACE = "Eased/vLLM"


def classify_request(path: str, body: bytes) -> str:
    """
    Return a request-type label used as a CloudWatch dimension.

    chat          POST /v1/chat/completions — text only, non-streaming
    chat_stream   POST /v1/chat/completions — text only, stream=true
    vision        POST /v1/chat/completions — contains image_url, non-streaming
    vision_stream POST /v1/chat/completions — contains image_url, stream=true
    other         anything else (/v1/models, /v1/embeddings, …)
    """
    if "/v1/chat/completions" not in path:
        return "other"

    try:
        data = json.loads(body) if body else {}
    except Exception:
        return "chat"

    is_streaming = bool(data.get("stream"))

    has_image = any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for msg in data.get("messages", [])
        for part in (
            msg.get("content")
            if isinstance(msg.get("content"), list)
            else []
        )
    )

    if has_image and is_streaming:
        return "vision_stream"
    if has_image:
        return "vision"
    if is_streaming:
        return "chat_stream"
    return "chat"


class CloudWatchMetrics:
    """
    Thin wrapper around boto3 CloudWatch.put_metric_data.

    All calls run in a thread pool executor (boto3 is synchronous) and are
    fire-and-forget — a failure never propagates back to the proxy caller.
    """

    def __init__(self) -> None:
        kwargs: dict = {"region_name": settings.aws_region}
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        self._cw = boto3.client("cloudwatch", **kwargs)

    # ── internal sync put (runs in executor) ─────────────────────────────

    def _put_sync(self, metric_data: list[dict]) -> None:
        try:
            self._cw.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
        except Exception as exc:
            log.warning("metrics.cloudwatch.put_failed", error=str(exc))

    # ── public async interface ────────────────────────────────────────────

    def emit(
        self,
        *,
        request_type: str,
        worker_id: str,
        latency_ms: float,
        ttft_ms: Optional[float] = None,
        status: str = "success",
    ) -> None:
        """
        Schedule a CloudWatch put in the background.  Non-blocking — safe to
        call from hot-path async code.
        """
        if not settings.cloudwatch_enabled:
            return

        now = datetime.now(timezone.utc)
        dims = [
            {"Name": "RequestType", "Value": request_type},
            {"Name": "WorkerId",    "Value": worker_id},
        ]

        metric_data: list[dict] = [
            {
                "MetricName": "RequestLatencyMs",
                "Dimensions": dims,
                "Value":      latency_ms,
                "Unit":       "Milliseconds",
                "Timestamp":  now,
            },
            {
                "MetricName": "RequestCount",
                "Dimensions": [*dims, {"Name": "Status", "Value": status}],
                "Value":      1,
                "Unit":       "Count",
                "Timestamp":  now,
            },
        ]

        if ttft_ms is not None:
            metric_data.append(
                {
                    "MetricName": "TimeToFirstTokenMs",
                    "Dimensions": dims,
                    "Value":      ttft_ms,
                    "Unit":       "Milliseconds",
                    "Timestamp":  now,
                }
            )

        # Fire-and-forget: run the blocking boto3 call in the thread pool.
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._put_sync, metric_data)


# Module-level singleton — created once when the module is first imported.
metrics = CloudWatchMetrics()
