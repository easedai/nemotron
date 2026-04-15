from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import boto3
import structlog
from botocore.exceptions import ClientError

from .config import settings

log = structlog.get_logger(__name__)


class LBDB:
    """
    DynamoDB stats sink for load-balancer request counters.

    Worker routing is handled entirely by Redis (WorkerQueue).
    This class only updates per-worker success/failure counters in DynamoDB
    for observability and billing attribution.
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
        log.info("lb_db.init", table=settings.lb_workers_table)

    def increment_success(self, worker_id: str) -> None:
        try:
            self.table.update_item(
                Key={"worker_id": worker_id},
                UpdateExpression=(
                    "ADD successful_requests :one "
                    "SET last_request_at = :ts"
                ),
                ExpressionAttributeValues={
                    ":one": 1,
                    ":ts":  datetime.now(timezone.utc).isoformat(),
                },
            )
        except ClientError as exc:
            log.warning("lb_db.increment_success.failed", worker_id=worker_id, error=repr(exc))

    def increment_failure(self, worker_id: str) -> None:
        try:
            self.table.update_item(
                Key={"worker_id": worker_id},
                UpdateExpression="ADD failed_requests :one",
                ExpressionAttributeValues={":one": 1},
            )
        except ClientError as exc:
            log.warning("lb_db.increment_failure.failed", worker_id=worker_id, error=repr(exc))
