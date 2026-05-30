from __future__ import annotations

import base64
import textwrap
from typing import Any, Optional

import httpx
import structlog

from .base import CreateConfig, GPUOffer, GPUProvider, InstanceInfo

log = structlog.get_logger(__name__)

_LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"

# Lambda Labs status → orchestrator actual_status
_STATE_MAP: dict[str, str] = {
    "active":       "running",
    "booting":      "pending",
    "unhealthy":    "failed",
    "terminating":  "offline",
    "terminated":   "deleted",
}

# GPU name substrings → VRAM GB (for types that don't embed GB in the name).
_VRAM_LOOKUP: dict[str, float] = {
    "h100 sxm":  80.0,
    "h100 pcie": 80.0,
    "h100":      80.0,
    "a100 80":   80.0,
    "a100 40":   40.0,
    "a100":      40.0,
    "a6000":     48.0,
    "a40":       48.0,
    "a30":       24.0,
    "a10":       24.0,
    "a5000":     24.0,
    "a4000":     16.0,
    "rtx 4090":  24.0,
    "rtx 3090":  24.0,
    "rtx 3080":  10.0,
    "v100":      16.0,
    "t4":        16.0,
    "l40s":      48.0,
    "l40":       48.0,
    "l4":        24.0,
}


def _gpu_vram(description: str) -> float:
    """Parse VRAM GB from a Lambda Labs instance description string."""
    import re

    # e.g. "1x A100 (80 GB SXM4)" or "1x RTX 3090 (24 GB)"
    m = re.search(r"(\d+)\s*[Gg][Bb]", description)
    if m:
        return float(m.group(1))
    lower = description.lower()
    for key, gb in _VRAM_LOOKUP.items():
        if key in lower:
            return gb
    return 0.0


def _make_user_data(
    worker_api_key: str,
    model_id: str,
    vllm_max_model_len: int,
    vllm_gpu_memory_utilization: float,
    vllm_video_loader_backend: str,
    hf_home: str,
    vllm_cache_root: str,
    worker_image: str,
) -> str:
    """
    Build a cloud-init user_data script that pulls and runs the vLLM worker
    Docker container on first boot.

    Lambda Labs Ubuntu instances ship with Docker pre-installed.  The script
    waits for the Docker socket to be ready before pulling, in case Docker is
    still initialising at boot time.
    """
    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail

        # Wait for Docker daemon
        until docker info >/dev/null 2>&1; do sleep 2; done

        # Write environment variables to a file so values are never exposed in
        # the process list.
        cat > /tmp/vllm.env << 'ENVEOF'
        VLLM_API_KEY={worker_api_key}
        MODEL_ID={model_id}
        VLLM_PORT=8080
        VLLM_MAX_MODEL_LEN={vllm_max_model_len}
        VLLM_GPU_MEMORY_UTILIZATION={vllm_gpu_memory_utilization}
        VLLM_VIDEO_LOADER_BACKEND={vllm_video_loader_backend}
        HF_HOME={hf_home}
        VLLM_CACHE_ROOT={vllm_cache_root}
        HF_HUB_ENABLE_HF_TRANSFER=1
        ENVEOF

        docker run -d \\
            --gpus all \\
            --env-file /tmp/vllm.env \\
            -p 8080:8080 \\
            --name vllm-worker \\
            --restart unless-stopped \\
            {worker_image}
    """)
    return script


class LambdaLabsProvider(GPUProvider):
    """
    Lambda Labs GPU provider.

    Lambda Labs provides on-demand GPU VMs (no spot / bidding).  Instances are
    launched with a cloud-init user_data script that starts the vLLM worker
    Docker container on first boot.

    Networking: instances receive a public IP; vLLM is accessible at
    ``http://<ip>:8080``.  SSH (port 22) is also available.

    ``change_bid`` always returns False (fixed pricing, no spot market).
    """

    name = "lambdalabs"
    supports_ssh = False   # SSH key management via Lambda API uses string IDs;
                           # manual key registration is recommended instead.

    def __init__(self) -> None:
        from ...config import settings

        if not settings.lambdalabs_api_key:
            raise RuntimeError("LAMBDALABS_API_KEY is required for the lambdalabs provider")
        self._api_key = settings.lambdalabs_api_key

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _client(self) -> httpx.AsyncClient:
        # Lambda Labs uses HTTP Basic Auth with the API key as username and empty password.
        credentials = base64.b64encode(f"{self._api_key}:".encode()).decode()
        return httpx.AsyncClient(
            base_url=_LAMBDA_API_BASE,
            headers={"Authorization": f"Basic {credentials}"},
            timeout=30.0,
        )

    # ── Offer discovery ───────────────────────────────────────────────────────

    async def search_offers(self, on_demand: bool = False) -> list[GPUOffer]:
        """
        Returns one offer per instance type / region combination that has
        available capacity.  Lambda Labs has no spot market; on_demand is ignored.
        """
        from ...config import settings

        async with self._client() as c:
            resp = await c.get("/instance-types")
            resp.raise_for_status()
            data = resp.json()

        offers: list[GPUOffer] = []
        for type_name, entry in (data.get("data") or {}).items():
            instance_type = entry.get("instance_type") or {}
            regions = entry.get("regions_with_capacity_available") or []
            if not regions:
                continue

            description = instance_type.get("description", type_name)
            price_cents = instance_type.get("price_cents_per_hour") or 0
            price_hr    = price_cents / 100.0

            vram_gb = _gpu_vram(description)
            if vram_gb == 0.0:
                log.warning(
                    "lambdalabs.search_offers.unknown_vram",
                    type_name=type_name,
                    description=description,
                )
                continue
            if vram_gb < settings.min_gpu_ram_gb:
                continue

            min_per_gpu_gb = settings.min_gpu_overhead_gb / (
                1.0 - settings.vllm_gpu_memory_utilization
            )
            if vram_gb < min_per_gpu_gb:
                continue

            if settings.max_hourly_rate > 0 and price_hr > settings.max_hourly_rate:
                continue

            # Use first available region; encode region into offer_id.
            region = regions[0]["name"]
            offer_id = f"{region}:{type_name}"

            offers.append(GPUOffer(
                offer_id    =offer_id,
                provider    ="lambdalabs",
                price_per_hr=price_hr,
                gpu_name    =description,
                gpu_ram_gb  =vram_gb,
                num_gpus    =1,
                specs       ={"instance_type": instance_type, "region": region},
                raw         =entry,
            ))

        offers.sort(key=lambda o: o.price_per_hr)
        return offers

    def get_market_price(self, offers: list[GPUOffer]) -> float:
        if not offers:
            return 0.0
        prices = sorted(o.price_per_hr for o in offers)
        mid = len(prices) // 2
        return prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2

    # ── Instance lifecycle ────────────────────────────────────────────────────

    async def create_instance(self, offer: GPUOffer, config: CreateConfig) -> str:
        from ...config import settings

        # offer_id encodes "region:instance_type_name"
        region, instance_type_name = offer.offer_id.split(":", 1)

        # Lambda Labs VMs run Docker via cloud-init.  Use the generic image
        # (plain ENTRYPOINT, no vastai supervisor) unless overridden.
        worker_image = (
            settings.lambdalabs_worker_image
            or "ghcr.io/easedai/nemotron:latest"
        )

        user_data = _make_user_data(
            worker_api_key              =config.worker_api_key,
            model_id                    =settings.model_id,
            vllm_max_model_len          =settings.vllm_max_model_len,
            vllm_gpu_memory_utilization =settings.vllm_gpu_memory_utilization,
            vllm_video_loader_backend   =settings.vllm_video_loader_backend,
            hf_home                     ="/hf",   # generic image bakes HF_HOME=/hf
            vllm_cache_root             ="/vllm-cache",
            worker_image                =worker_image,
        )

        body: dict[str, Any] = {
            "region_name":          region,
            "instance_type_name":   instance_type_name,
            "name":                 config.label,
            "user_data":            user_data,
        }

        # Attach the orchestrator's SSH key if one has been pre-registered.
        ssh_key_name = getattr(settings, "lambdalabs_ssh_key_name", None)
        if ssh_key_name:
            body["ssh_key_names"] = [ssh_key_name]

        async with self._client() as c:
            resp = await c.post("/instance-operations/launch", json=body)
            resp.raise_for_status()
            result = resp.json()

        ids: list[str] = (result.get("data") or {}).get("instance_ids") or []
        if not ids:
            raise ValueError(f"No instance ID in Lambda Labs response: {result}")
        instance_id = ids[0]
        log.info(
            "lambdalabs.create_instance",
            instance_id=instance_id,
            instance_type=instance_type_name,
            region=region,
        )
        return instance_id

    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        async with self._client() as c:
            resp = await c.get(f"/instances/{instance_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
        raw = data.get("data") or {}
        return self._to_instance_info(raw) if raw else None

    async def list_instances(self) -> list[InstanceInfo]:
        async with self._client() as c:
            resp = await c.get("/instances")
            resp.raise_for_status()
            data = resp.json()
        return [self._to_instance_info(item) for item in (data.get("data") or [])]

    async def destroy_instance(self, instance_id: str) -> None:
        async with self._client() as c:
            resp = await c.post(
                "/instance-operations/terminate",
                json={"instance_ids": [instance_id]},
            )
            if resp.status_code == 404:
                return
            resp.raise_for_status()

    async def change_bid(self, instance_id: str, new_price: float) -> bool:
        """Lambda Labs does not support spot bidding — always returns False."""
        return False

    async def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        """Lambda Labs does not provide a container logs API."""
        return (
            f"[lambdalabs] Log retrieval is not available via the Lambda Labs API "
            f"for instance {instance_id!r}.  SSH to the instance and run: "
            f"docker logs --tail {tail} vllm-worker"
        )

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _to_instance_info(self, raw: dict[str, Any]) -> InstanceInfo:
        instance_id = str(raw.get("id") or "")
        status      = raw.get("status", "unknown")
        ip          = raw.get("ip")

        actual_status = _STATE_MAP.get(status, "unknown")

        instance_type = raw.get("instance_type") or {}
        description   = instance_type.get("description", "")
        vram_gb       = _gpu_vram(description) or None

        return InstanceInfo(
            instance_id  =instance_id,
            provider     ="lambdalabs",
            actual_status=actual_status,
            cur_state    =status,
            status_msg   ="",
            next_state   ="",
            gpu_name     =description or None,
            gpu_ram_gb   =vram_gb,
            label        =raw.get("name") or "",
            image        ="",   # Lambda runs Docker via cloud-init; no image field
            host         =ip,
            port         =8080 if ip else None,
            ssh_host     =ip,
            ssh_port     =22 if ip else None,
            specs        =raw,
            raw          =raw,
        )
