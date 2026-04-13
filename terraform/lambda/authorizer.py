"""
API Gateway HTTP API Lambda Authorizer (REQUEST type, payload format 2.0).

Design goals
------------
- Fast:   module-level caching means Secrets Manager is called at most once
          per Lambda execution environment per cache TTL (default 5 min).
- Safe:   constant-time HMAC comparison prevents timing oracle attacks.
- Simple: returns {"isAuthorized": bool} (simple-response mode).

Caching behaviour
-----------------
API Gateway caches the authorizer *result* per unique Authorization header
value for `authorizer_result_ttl_in_seconds` (set to 300 s in Terraform).
The Lambda's in-process cache guards against the case where many distinct
tokens arrive, or the API GW cache is cold after a deployment.

Cold start cost: ~1 Secrets Manager call per new execution environment.
Warm invocation cost: 0 Secrets Manager calls (cache hit).
"""
from __future__ import annotations

import hmac
import logging
import os
import time

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Boto3 client is created once per execution environment — reused on warm calls
_client = boto3.client(
    "secretsmanager",
    region_name=os.environ.get("AWS_REGION_NAME", "us-east-1"),
)
_SECRET_ARN: str = os.environ["AUTHORIZER_SECRET_ARN"]
_CACHE_TTL: int = int(os.environ.get("SECRET_CACHE_TTL_SEC", "300"))

# Module-level cache — lives for the lifetime of the execution environment
_cached_token: str | None = None
_cache_expiry: float = 0.0


def _get_expected_token() -> str:
    """Return the expected bearer token, fetching from Secrets Manager only on
    cache miss or expiry."""
    global _cached_token, _cache_expiry

    now = time.monotonic()
    if _cached_token is not None and now < _cache_expiry:
        return _cached_token

    logger.info("Secret cache miss — fetching from Secrets Manager")
    response = _client.get_secret_value(SecretId=_SECRET_ARN)
    _cached_token = response["SecretString"]
    _cache_expiry = now + _CACHE_TTL
    return _cached_token


def handler(event: dict, context: object) -> dict:  # noqa: ARG001
    """Lambda handler — returns {"isAuthorized": bool}."""
    try:
        headers: dict = event.get("headers") or {}
        # HTTP/2 headers are always lowercase; handle both casings defensively
        auth_header: str = headers.get("authorization") or headers.get("Authorization", "")

        if not auth_header.lower().startswith("bearer "):
            return {"isAuthorized": False}

        provided: str = auth_header[7:]  # strip "Bearer " prefix
        expected: str = _get_expected_token()

        # hmac.compare_digest runs in constant time regardless of token length,
        # preventing timing-based token enumeration.
        return {"isAuthorized": hmac.compare_digest(provided, expected)}

    except Exception:
        logger.exception("Unhandled authorizer error")
        return {"isAuthorized": False}
