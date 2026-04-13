from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import boto3
import structlog
from botocore.exceptions import ClientError

from .config import settings
from .models import LBWorker

log = structlog.get_logger(__name__)


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _deserialize(item: dict) -> LBWorker:
    return LBWorker(
        worker_id=item["worker_id"],
        source_type=item["source_type"],
        host=item["host"],
        port=_to_int(item.get("port", 8080)),
        api_key=item["api_key"],
        added_at=datetime.fromisoformat(item["added_at"]),
        successful_requests=_to_int(item.get("successful_requests", 0)),
        failed_requests=_to_int(item.get("failed_requests", 0)),
        last_request_at=(
            datetime.fromisoformat(item["last_request_at"])
            if item.get("last_request_at")
            else None
        ),
    )


class LBDB:
    """Read / update side of the load-balancer worker table."""

    def __init__(self) -> None:
        kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.dynamodb_endpoint_url:
            kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"]     = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        resource = boto3.resource("dynamodb", **kwargs)
        self.table = resource.Table(settings.lb_workers_table)
        log.info("lb_db.init", table=settings.lb_workers_table)

    def list_workers(self) -> list[LBWorker]:
        resp    = self.table.scan()
        workers = [_deserialize(item) for item in resp.get("Items", [])]
        log.debug("lb_db.list_workers", count=len(workers))
        return workers

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
