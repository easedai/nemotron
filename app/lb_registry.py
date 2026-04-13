from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import boto3
import structlog
from botocore.exceptions import ClientError

from .config import settings

log = structlog.get_logger(__name__)


class LBRegistry:
    """
    Write-side of the load-balancer DynamoDB table.

    The orchestrator is the sole writer.  Workers are registered when
    vLLM becomes healthy and deregistered when the instance is terminated
    for any reason.  The load-balancer service reads this table to select
    upstreams for each incoming request.
    """

    def __init__(self) -> None:
        kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.dynamodb_endpoint_url:
            kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"]     = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        resource   = boto3.resource("dynamodb", **kwargs)
        self.table = resource.Table(settings.lb_workers_table)
        log.info("lb_registry.init", table=settings.lb_workers_table)

    def register(
        self,
        worker_id:   str,
        source_type: str,
        host:        str,
        port:        int,
        api_key:     str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self.table.update_item(
                Key={"worker_id": worker_id},
                UpdateExpression=(
                    "SET source_type = :st, "
                    "    #h = :h, "
                    "    #p = :p, "
                    "    api_key = :k, "
                    "    added_at = if_not_exists(added_at, :ts), "
                    "    successful_requests = if_not_exists(successful_requests, :zero), "
                    "    failed_requests     = if_not_exists(failed_requests, :zero)"
                ),
                ExpressionAttributeNames={"#h": "host", "#p": "port"},
                ExpressionAttributeValues={
                    ":st":   source_type,
                    ":h":    host,
                    ":p":    port,
                    ":k":    api_key,
                    ":ts":   now,
                    ":zero": 0,
                },
            )
            log.info("lb_registry.registered", worker_id=worker_id, source_type=source_type, host=host, port=port)
        except ClientError as exc:
            log.error("lb_registry.register_failed", worker_id=worker_id, error=repr(exc))

    def deregister(self, worker_id: str) -> None:
        try:
            self.table.delete_item(Key={"worker_id": worker_id})
            log.info("lb_registry.deregistered", worker_id=worker_id)
        except ClientError as exc:
            log.warning("lb_registry.deregister_failed", worker_id=worker_id, error=repr(exc))
