from __future__ import annotations

import asyncio
from typing import Optional

import asyncssh
import boto3
import structlog
from botocore.exceptions import ClientError

from ..config import settings

log = structlog.get_logger(__name__)

_SSH_SECRET = "nemotron-vllm/orchestrator-ssh-private-key"
_cached_key: Optional[asyncssh.SSHKey] = None
_key_lock = asyncio.Lock()


async def _resolve_ssh_key() -> Optional[asyncssh.SSHKey]:
    """
    Resolve the SSH key using the same priority as the orchestrator:
      1. ORCHESTRATOR_SSH_PRIVATE_KEY env var
      2. AWS Secrets Manager (nemotron-vllm/orchestrator-ssh-private-key)
    Result is cached for the lifetime of the process.
    """
    global _cached_key
    async with _key_lock:
        if _cached_key is not None:
            return _cached_key

        pem: Optional[str] = settings.orchestrator_ssh_private_key
        if not pem:
            try:
                client = boto3.client("secretsmanager", region_name=settings.aws_region)
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.get_secret_value(SecretId=_SSH_SECRET),
                )
                pem = resp.get("SecretString")
                if pem:
                    log.info("lb.ssh_logs.key_from_secrets_manager", secret=_SSH_SECRET)
            except ClientError as exc:
                log.warning("lb.ssh_logs.key_fetch_failed", secret=_SSH_SECRET, error=repr(exc))

        if not pem:
            log.debug("lb.ssh_logs.no_key_available")
            return None

        _cached_key = asyncssh.import_private_key(pem)
        return _cached_key


async def fetch_vllm_logs(
    host: str,
    ssh_port: int,
    lines: int = 150,
    timeout: int = 20,
) -> Optional[str]:
    """
    SSH into a worker and return debug logs on 5xx.

    Fetches in a single connection (matching orchestrator behaviour):
      • /var/log/portal/vllm.log or /tmp/vllm.log  — vLLM stdout/stderr
      • /var/log/onstart.log                        — startup / EXTRA_COMMANDS output

    Returns None on any SSH / timeout error so callers can ignore failures silently.
    """
    cmd = (
        f"echo '=== vllm log (last {lines} lines) ===';"
        f" tail -n {lines} /var/log/portal/vllm.log 2>/dev/null"
        f" || tail -n {lines} /tmp/vllm.log 2>/dev/null"
        " || echo '(no vllm log found yet)';"
        " echo; echo '=== onstart.log (last 50 lines) ===';"
        " tail -n 50 /var/log/onstart.log 2>/dev/null"
        " || echo '(no onstart.log yet)'"
    )
    key = await _resolve_ssh_key()
    if key is None:
        log.debug("lb.ssh_logs.skipped", reason="no SSH key configured", host=host)
        return None

    log.info(
        "lb.ssh_logs.connecting",
        host=host,
        ssh_port=ssh_port,
    )
    try:
        async with asyncssh.connect(
            host,
            port=ssh_port,
            username="root",
            client_keys=[key],
            known_hosts=None,
            connect_timeout=timeout,
        ) as conn:
            result = await conn.run(cmd, timeout=timeout)
            text = (result.stdout or "").strip()
            log.info(
                "lb.ssh_logs.ok",
                host=host,
                ssh_port=ssh_port,
                bytes=len(text),
            )
            return text or None
    except Exception as exc:
        log.warning(
            "lb.ssh_logs.failed",
            host=host,
            ssh_port=ssh_port,
            error=str(exc),
        )
        return None
