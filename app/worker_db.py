from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import boto3
import structlog
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from .config import settings
from .models import Worker, WorkerStatus, WorkerType

log = structlog.get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_decimal(value: Any) -> Any:
    """DynamoDB rejects float; convert to Decimal for numeric fields."""
    if isinstance(value, float):
        return Decimal(str(value))
    return value


class DynamoDB:
    def __init__(self) -> None:
        kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.dynamodb_endpoint_url:
            kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        resource = boto3.resource("dynamodb", **kwargs)
        self.table         = resource.Table(settings.dynamodb_table)
        self.history_table = resource.Table(settings.history_table)
        log.info(
            "db.init",
            table=settings.dynamodb_table,
            history_table=settings.history_table,
            endpoint=settings.dynamodb_endpoint_url or "aws",
        )

    # ── Write ─────────────────────────────────────────────────────────────

    def save_worker(self, worker: Worker) -> None:
        item = worker.model_dump()
        item["status"]      = worker.status.value
        item["worker_type"] = worker.worker_type.value
        item["created_at"]  = worker.created_at.isoformat()
        item["updated_at"]  = worker.updated_at.isoformat()
        for k in ("bid_price", "market_price", "gpu_ram_gb", "image_pull_duration_sec"):
            if item.get(k) is not None:
                item[k] = _to_decimal(item[k])
        if item.get("running_since") is not None:
            item["running_since"] = worker.running_since.isoformat()
        if item.get("image_pull_started_at") is not None:
            item["image_pull_started_at"] = worker.image_pull_started_at.isoformat()
        if item.get("specs") is not None:
            item["specs"] = json.dumps(item["specs"])
        log.info(
            "db.save_worker",
            worker_id=worker.worker_id,
            status=worker.status,
            worker_type=worker.worker_type,
        )
        self.table.put_item(Item=item)
        self.history_table.put_item(Item=item)

    def update_worker_status(
        self,
        worker_id: str,
        status: WorkerStatus,
        **extra_fields: Any,
    ) -> None:
        log.info("db.update_worker_status", worker_id=worker_id, status=status)
        update_expr = "SET #st = :st, updated_at = :ua"
        expr_names: dict[str, str]  = {"#st": "status"}
        expr_values: dict[str, Any] = {
            ":st": status.value,
            ":ua": _utcnow(),
        }
        for k, v in extra_fields.items():
            update_expr += f", #{k} = :{k}"
            expr_names[f"#{k}"] = k
            if isinstance(v, datetime):
                expr_values[f":{k}"] = v.isoformat()
            elif isinstance(v, float):
                expr_values[f":{k}"] = _to_decimal(v)
            elif isinstance(v, dict):
                expr_values[f":{k}"] = json.dumps(v)
            else:
                expr_values[f":{k}"] = v

        update_kwargs = dict(
            Key={"worker_id": worker_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
        self.table.update_item(**update_kwargs)
        self.history_table.update_item(**update_kwargs)

    def delete_worker(self, worker_id: str) -> None:
        log.info("db.delete_worker", worker_id=worker_id)
        self.table.delete_item(Key={"worker_id": worker_id})

    # ── Read ──────────────────────────────────────────────────────────────

    def get_worker(self, worker_id: str) -> Optional[Worker]:
        resp = self.table.get_item(Key={"worker_id": worker_id})
        item = resp.get("Item")
        if not item:
            log.debug("db.get_worker.not_found", worker_id=worker_id)
            return None
        return self._deserialize(item)

    def list_workers(self, status: Optional[WorkerStatus] = None) -> list[Worker]:
        if status:
            resp = self.table.scan(FilterExpression=Attr("status").eq(status.value))
        else:
            resp = self.table.scan()
        workers = [self._deserialize(i) for i in resp.get("Items", [])]
        log.debug("db.list_workers", count=len(workers), status_filter=status)
        return workers

    def get_active_workers(self) -> list[Worker]:
        active = [
            WorkerStatus.BIDDING,
            WorkerStatus.PENDING,
            WorkerStatus.STARTING,
            WorkerStatus.RUNNING,
            WorkerStatus.UNHEALTHY,
        ]
        resp = self.table.scan(
            FilterExpression=Attr("status").is_in([s.value for s in active])
        )
        workers = [self._deserialize(i) for i in resp.get("Items", [])]
        log.debug("db.get_active_workers", count=len(workers))
        return workers

    def get_worker_by_instance_id(self, instance_id: str) -> Optional[Worker]:
        resp = self.table.query(
            IndexName="instance_id-index",
            KeyConditionExpression=Key("instance_id").eq(instance_id),
        )
        items = resp.get("Items", [])
        return self._deserialize(items[0]) if items else None

    def get_known_instance_ids(self) -> set[str]:
        try:
            resp = self.table.scan(
                IndexName="instance_id-index",
                ProjectionExpression="instance_id",
            )
            return {item["instance_id"] for item in resp.get("Items", [])}
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise
            log.warning("db.get_known_instance_ids.gsi_backfilling — falling back to full scan")
            resp = self.table.scan(ProjectionExpression="instance_id")
            return {
                item["instance_id"]
                for item in resp.get("Items", [])
                if item.get("instance_id")
            }

    def get_running_workers(self) -> list[Worker]:
        resp = self.table.scan(
            FilterExpression=Attr("status").eq(WorkerStatus.RUNNING.value)
        )
        workers = [self._deserialize(i) for i in resp.get("Items", [])]
        log.debug("db.get_running_workers", count=len(workers))
        return workers

    # ── Deserialisation ───────────────────────────────────────────────────

    @staticmethod
    def _deserialize(item: dict[str, Any]) -> Worker:
        item["status"]      = WorkerStatus(item["status"])
        item["worker_type"] = WorkerType(item["worker_type"])
        item["created_at"]  = datetime.fromisoformat(item["created_at"])
        item["updated_at"]  = datetime.fromisoformat(item["updated_at"])
        for k in ("port", "bid_attempts", "consecutive_failures"):
            if k in item:
                item[k] = int(item[k])
        for k in ("bid_price", "market_price", "gpu_ram_gb", "image_pull_duration_sec"):
            if item.get(k) is not None:
                item[k] = float(item[k])
        if item.get("running_since") is not None:
            item["running_since"] = datetime.fromisoformat(item["running_since"])
        if item.get("image_pull_started_at") is not None:
            item["image_pull_started_at"] = datetime.fromisoformat(item["image_pull_started_at"])
        if item.get("specs") is not None:
            item["specs"] = json.loads(item["specs"])
        return Worker(**item)
