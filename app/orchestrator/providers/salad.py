from __future__ import annotations

import re
from typing import Any, Optional

import httpx
import structlog

from .base import CreateConfig, GPUOffer, GPUProvider, InstanceInfo

log = structlog.get_logger(__name__)

_SALAD_API_BASE = "https://api.salad.com/api/public"

# Container group name max length on Salad; must be DNS-safe (lowercase, hyphens, digits).
_MAX_NAME_LEN = 63

# Salad state → orchestrator actual_status
# Known VRAM (GB) for GPU names that don't embed the size in their name.
# Keys are lowercase substrings matched against the GPU class name.
_VRAM_LOOKUP: dict[str, float] = {
    "h100 sxm":  80.0,
    "h100 pcie": 80.0,
    "h100":      80.0,
    "a100 80":   80.0,
    "a100 40":   40.0,
    "a100":      40.0,   # default A100 (40 GB PCIe); overridden above if "80" is in name
    "a40":       48.0,
    "a30":       24.0,
    "a10":       24.0,
    "a6000":     48.0,
    "a5000":     24.0,
    "a4500":     20.0,
    "a4000":     16.0,
    "rtx 4090":  24.0,
    "rtx 4080":  16.0,
    "rtx 3090":  24.0,
    "rtx 3080":  10.0,
    "rtx 3070":   8.0,
    "l40s":      48.0,
    "l40":       48.0,
    "l4":        24.0,
    "4090":      24.0,
    "3090":      24.0,
}

_STATE_MAP: dict[str, str] = {
    "allocating":  "pending",
    "downloading": "pending",
    "creating":    "pending",
    "running":     "running",
    "stopped":     "offline",
    "failed":      "failed",
    "deleted":     "deleted",
}


def _safe_name(label: str) -> str:
    """Convert an arbitrary label to a DNS-safe Salad container group name."""
    name = label.lower()
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name[:_MAX_NAME_LEN]


class SaladProvider(GPUProvider):
    """
    Salad.com GPU provider.

    Salad does not use spot bidding — containers run at a fixed priority tier
    (batch / low / medium / high).  ``change_bid`` always returns False.

    Networking: Salad exposes containers via an HTTPS gateway URL
    (``https://<name>-<hash>.salad.cloud``).  InstanceInfo.host is set to the
    bare hostname (no scheme) and port to 443 so that Worker/LBWorker.base_url
    produces the correct ``https://`` URL.
    """

    name = "salad"

    def __init__(self) -> None:
        from ...config import settings

        if not settings.salad_api_key:
            raise RuntimeError("SALAD_API_KEY is required for the salad provider")
        if not settings.salad_org_name:
            raise RuntimeError("SALAD_ORG_NAME is required for the salad provider")
        if not settings.salad_project_name:
            raise RuntimeError("SALAD_PROJECT_NAME is required for the salad provider")

        self._api_key  = settings.salad_api_key
        self._org      = settings.salad_org_name
        self._project  = settings.salad_project_name
        self._priority = settings.salad_priority
        self._cpu      = settings.salad_container_cpu
        self._mem_mb   = settings.salad_container_memory_mb

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_SALAD_API_BASE,
            headers={"Salad-Api-Key": self._api_key},
            timeout=30.0,
        )

    def _containers_url(self, name: Optional[str] = None) -> str:
        base = f"/organizations/{self._org}/projects/{self._project}/containers"
        return f"{base}/{name}" if name else base

    # ── Offer discovery ───────────────────────────────────────────────────────

    async def search_offers(self, on_demand: bool = False) -> list[GPUOffer]:
        """
        Return one GPUOffer per Salad GPU class, filtered to those with enough
        VRAM to run the configured model (min_gpu_ram_gb setting).

        GPU classes whose VRAM cannot be determined are excluded so the
        orchestrator never tries to schedule on an undersized GPU.
        """
        from ...config import settings

        async with self._client() as c:
            resp = await c.get(f"/organizations/{self._org}/gpu-classes")
            resp.raise_for_status()
            data = resp.json()

        offers: list[GPUOffer] = []
        for item in data.get("items", []):
            offer = self._gpu_class_to_offer(item)
            if offer.gpu_ram_gb == 0.0:
                log.warning(
                    "salad.search_offers.unknown_vram",
                    gpu_class_id=offer.offer_id,
                    gpu_name=offer.gpu_name,
                    note="VRAM unknown — excluded from offers; add to _VRAM_LOOKUP if needed",
                )
                continue
            if offer.gpu_ram_gb < settings.min_gpu_ram_gb:
                log.debug(
                    "salad.search_offers.vram_too_small",
                    gpu_name=offer.gpu_name,
                    gpu_ram_gb=offer.gpu_ram_gb,
                    min_gpu_ram_gb=settings.min_gpu_ram_gb,
                )
                continue
            offers.append(offer)

        offers.sort(key=lambda o: o.price_per_hr)
        return offers

    def get_market_price(self, offers: list[GPUOffer]) -> float:
        """
        Salad has fixed prices per GPU class and priority tier, not auction-based.
        Return the median price across available offers as a reference.
        """
        if not offers:
            return 0.0
        prices = sorted(o.price_per_hr for o in offers)
        mid = len(prices) // 2
        return prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2

    # ── Instance lifecycle ────────────────────────────────────────────────────

    async def create_instance(self, offer: GPUOffer, config: CreateConfig) -> str:
        from ...config import settings

        name = _safe_name(config.label)
        env: dict[str, str] = {
            "VLLM_API_KEY":               config.worker_api_key,
            "MODEL_ID":                   settings.model_id,
            "VLLM_PORT":                  "8080",
            "VLLM_MAX_MODEL_LEN":         str(settings.vllm_max_model_len),
            "VLLM_GPU_MEMORY_UTILIZATION": str(settings.vllm_gpu_memory_utilization),
            "VLLM_VIDEO_LOADER_BACKEND":  settings.vllm_video_loader_backend,
            "HF_HOME":                    settings.hf_home,
            "VLLM_CACHE_ROOT":            settings.vllm_cache_root,
            "HF_HUB_ENABLE_HF_TRANSFER":  "1",
        }

        body: dict[str, Any] = {
            "name": name,
            "container": {
                "image": settings.worker_image,
                "resources": {
                    "cpu":         self._cpu,
                    "memory":      self._mem_mb,
                    "gpu_classes": [offer.offer_id],
                },
                "environment_variables": env,
            },
            "autostart_policy": True,
            "restart_policy":   "always",
            "replicas":         1,
            "networking": {
                "protocol": "http",
                "port":     8080,
                # Salad gateway auth: every request must carry
                # Salad-Api-Key: <org api key>.  The LB proxy injects this
                # header automatically for workers with source_type="salad".
                "auth":     True,
            },
        }

        if not config.on_demand and self._priority not in ("high", "medium"):
            body["queue_autoscaler"] = None   # no autoscaler for batch/low

        async with self._client() as c:
            resp = await c.post(self._containers_url(), json=body)
            resp.raise_for_status()
            result = resp.json()

        instance_id = result.get("id") or result.get("name") or name
        log.info(
            "salad.create_instance",
            name=name,
            gpu_class=offer.offer_id,
            instance_id=instance_id,
        )
        return str(instance_id)

    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        async with self._client() as c:
            resp = await c.get(self._containers_url(instance_id))
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._to_instance_info(resp.json())

    async def list_instances(self) -> list[InstanceInfo]:
        async with self._client() as c:
            resp = await c.get(self._containers_url())
            resp.raise_for_status()
            data = resp.json()

        return [
            self._to_instance_info(item)
            for item in data.get("items", [])
        ]

    async def destroy_instance(self, instance_id: str) -> None:
        async with self._client() as c:
            resp = await c.delete(self._containers_url(instance_id))
            if resp.status_code in (404, 204):
                return   # already gone or deleted successfully
            resp.raise_for_status()

    async def change_bid(self, instance_id: str, new_price: float) -> bool:
        """Salad does not support spot bidding — always returns False."""
        return False

    async def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        """Salad does not provide a container logs API."""
        return f"[salad] Log retrieval is not available via the Salad API for instance {instance_id!r}."

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _gpu_class_to_offer(self, raw: dict[str, Any]) -> GPUOffer:
        """
        Map a Salad GPU class object to a GPUOffer.

        Salad pricing is not included in the gpu-classes response — it varies by
        priority tier and is shown in the Salad portal.  price_per_hr is set to 0.0
        as a placeholder; the orchestrator's market-price logic falls back to 0 when
        no real price is available.
        """
        gpu_id   = raw.get("id", "")
        gpu_name = raw.get("name", "unknown")

        # 1. Structured VRAM field from API (Salad returns MB in the `vram` field).
        vram_gb = 0.0
        raw_vram = raw.get("vram")
        if raw_vram:
            mb = float(raw_vram)
            # Values >= 1000 are almost certainly MB; smaller values are likely already GB.
            vram_gb = mb / 1024 if mb >= 1000 else mb

        # 2. Parse GB suffix from the GPU name (e.g. "A100 80GB", "RTX 3090 24 GB").
        if vram_gb == 0.0:
            m = re.search(r"(\d+)\s*[Gg][Bb]", gpu_name)
            if m:
                vram_gb = float(m.group(1))

        # 3. Lookup table for well-known GPUs that don't embed VRAM in their name.
        if vram_gb == 0.0:
            name_lower = gpu_name.lower()
            for key, gb in _VRAM_LOOKUP.items():
                if key in name_lower:
                    vram_gb = gb
                    break

        return GPUOffer(
            offer_id    =gpu_id,
            provider    ="salad",
            price_per_hr=0.0,       # not available from GPU-classes API
            gpu_name    =gpu_name,
            gpu_ram_gb  =vram_gb,
            num_gpus    =1,
            specs       =raw,
            raw         =raw,
        )

    def _to_instance_info(self, raw: dict[str, Any]) -> InstanceInfo:
        state    = raw.get("current_state", {})
        status   = state.get("status") or raw.get("status", "unknown")
        replicas = raw.get("instances", [])

        # Try to get the running replica's GPU info
        gpu_name   = None
        gpu_ram_gb = None
        for replica in replicas:
            machine = replica.get("machine") or {}
            if machine.get("gpu"):
                gpu_name = machine["gpu"].get("name")
                break

        # Networking: Salad exposes containers via an HTTPS gateway hostname.
        # host = bare hostname (no scheme), port = 443 so base_url → https://host
        networking = raw.get("networking") or {}
        dns = networking.get("dns") or ""
        host = dns if dns else None
        port = 443 if host else None

        actual_status = _STATE_MAP.get(status, "unknown")

        return InstanceInfo(
            instance_id  =str(raw.get("id") or raw.get("name", "")),
            provider     ="salad",
            actual_status=actual_status,
            cur_state    =status,
            status_msg   =state.get("description") or "",
            next_state   ="",
            gpu_name     =gpu_name,
            gpu_ram_gb   =gpu_ram_gb,
            label        =raw.get("name", ""),
            image        =(raw.get("container") or {}).get("image", ""),
            host         =host,
            port         =port,
            ssh_host     =None,
            ssh_port     =None,
            specs        =raw,
            raw          =raw,
        )
