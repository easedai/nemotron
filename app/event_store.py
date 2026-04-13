from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import structlog
from boto3.dynamodb.conditions import Attr, Key

from .config import settings

log = structlog.get_logger(__name__)

_TTL_SEC = 7 * 24 * 3600
_MAX_LOG_CHARS = 80_000


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventStore:
    """
    Append-only log of worker lifecycle events, status transitions, and
    container log snapshots.  All rows expire after 7 days via DynamoDB TTL.

    Table schema
    ────────────
    PK  worker_id  (string)
    SK  ts         (string)  — ISO timestamp, sorts chronologically
    TTL expires_at (number)  — Unix epoch seconds

    GSI  instance_id-index
         PK  instance_id  (string)
         SK  ts           (string)
    """

    def __init__(self) -> None:
        kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.dynamodb_endpoint_url:
            kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"]     = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        resource   = boto3.resource("dynamodb", **kwargs)
        self.table = resource.Table(settings.events_table)
        log.info("event_store.init", table=settings.events_table)

    # ── Write ─────────────────────────────────────────────────────────────

    def record(
        self,
        worker_id:   str,
        event_type:  str,
        status:      str,
        message:     str,
        instance_id: Optional[str]  = None,
        label:       Optional[str]  = None,
        prev_status: Optional[str]  = None,
        meta:        Optional[dict] = None,
    ) -> None:
        ts  = _utcnow_iso()
        ttl = int(time.time()) + _TTL_SEC
        item: dict[str, Any] = {
            "worker_id":  worker_id,
            "ts":         ts,
            "event_type": event_type,
            "status":     status,
            "message":    message,
            "expires_at": ttl,
        }
        if instance_id:
            item["instance_id"] = instance_id
        if label:
            item["label"] = label
        if prev_status:
            item["prev_status"] = prev_status
        if meta:
            item["meta"] = json.dumps(meta)
        self._put(item)

    def record_logs(
        self,
        worker_id:   str,
        status:      str,
        log_text:    str,
        instance_id: Optional[str] = None,
        label:       Optional[str] = None,
        trigger:     str = "manual",
    ) -> None:
        if len(log_text) > _MAX_LOG_CHARS:
            log_text = "…(truncated — showing last 80 KB)\n" + log_text[-_MAX_LOG_CHARS:]
        ts  = _utcnow_iso()
        ttl = int(time.time()) + _TTL_SEC
        item: dict[str, Any] = {
            "worker_id":  worker_id,
            "ts":         ts,
            "event_type": "log.snapshot",
            "status":     status,
            "message":    f"Container log snapshot ({trigger})",
            "log_text":   log_text,
            "expires_at": ttl,
        }
        if instance_id:
            item["instance_id"] = instance_id
        if label:
            item["label"] = label
        self._put(item)

    def _put(self, item: dict[str, Any]) -> None:
        try:
            self.table.put_item(Item=item)
            log.debug(
                "event_store.put",
                worker_id=item.get("worker_id"),
                event_type=item.get("event_type"),
            )
        except Exception as exc:
            log.warning(
                "event_store.put.failed",
                worker_id=item.get("worker_id"),
                event_type=item.get("event_type"),
                error=str(exc),
            )

    # ── Query ─────────────────────────────────────────────────────────────

    def query_by_worker(self, worker_id: str, limit: int = 100) -> list[dict]:
        try:
            resp = self.table.query(
                KeyConditionExpression=Key("worker_id").eq(worker_id),
                ScanIndexForward=False,
                Limit=limit,
            )
            return self._clean(resp.get("Items", []))
        except Exception as exc:
            log.warning("event_store.query_by_worker.failed", worker_id=worker_id, error=str(exc))
            return []

    def query_by_label(self, label: str, limit: int = 100) -> list[dict]:
        if label.startswith("eased-") and len(label) > 6:
            return self.query_by_worker(label[6:], limit)
        try:
            resp = self.table.scan(
                FilterExpression=Attr("label").eq(label),
                Limit=limit * 10,
            )
            items = sorted(
                resp.get("Items", []),
                key=lambda x: x.get("ts", ""),
                reverse=True,
            )[:limit]
            return self._clean(items)
        except Exception as exc:
            log.warning("event_store.query_by_label.failed", label=label, error=str(exc))
            return []

    def query_by_instance(self, instance_id: str, limit: int = 100) -> list[dict]:
        try:
            resp = self.table.query(
                IndexName="instance_id-index",
                KeyConditionExpression=Key("instance_id").eq(instance_id),
                ScanIndexForward=False,
                Limit=limit,
            )
            return self._clean(resp.get("Items", []))
        except Exception as exc:
            log.warning("event_store.query_by_instance.failed", instance_id=instance_id, error=str(exc))
            return []

    @staticmethod
    def _clean(items: list[dict]) -> list[dict]:
        for item in items:
            if "meta" in item:
                try:
                    item["meta"] = json.loads(item["meta"])
                except Exception:
                    pass
            item.pop("expires_at", None)
        return items
