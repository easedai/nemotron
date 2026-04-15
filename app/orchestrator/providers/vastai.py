from __future__ import annotations

from typing import Any, Optional

import structlog

from ...models import WorkerType
from ..vast_client import VastAIClient
from .base import CreateConfig, GPUOffer, GPUProvider, InstanceInfo

log = structlog.get_logger(__name__)


class VastAIProvider(GPUProvider):
    """
    vast.ai GPU provider.

    Wraps VastAIClient and converts raw vast.ai API dicts into the normalized
    GPUOffer / InstanceInfo types that the orchestrator works with.
    """
    name = "vastai"
    supports_ssh = True

    def __init__(self) -> None:
        self._client = VastAIClient()

    # ── Offer discovery ───────────────────────────────────────────────────

    async def search_offers(self, on_demand: bool = False) -> list[GPUOffer]:
        from ...config import settings
        raw_offers = await self._client.search_offers(on_demand=on_demand)
        offers = [self._to_gpu_offer(o) for o in raw_offers]
        # Client-side filters.
        # VRAM (total): total across all GPUs must meet the minimum (multi-GPU
        #   offers like 2×24 GB are accepted even when each card falls below).
        # VRAM (per-GPU headroom): reject cards where vLLM's allocation leaves
        #   insufficient room for CUDA graph capture + NCCL overhead.
        #   At util=0.95 a 24 GB card has only 1.2 GB free — causes OOM.
        #   Formula: gpu_ram_gb × (1 - utilization) >= min_gpu_overhead_gb
        #         ↔  gpu_ram_gb >= min_gpu_overhead_gb / (1 - utilization)
        # Rate: drop offers above the configured hourly ceiling (0 = no cap).
        offers = [o for o in offers if o.total_gpu_ram_gb >= settings.min_gpu_ram_gb]
        min_per_gpu_gb = settings.min_gpu_overhead_gb / (
            1.0 - settings.vllm_gpu_memory_utilization
        )
        offers = [o for o in offers if o.gpu_ram_gb >= min_per_gpu_gb]
        if settings.max_hourly_rate > 0:
            offers = [o for o in offers if o.price_per_hr <= settings.max_hourly_rate]
        return offers

    def get_market_price(self, offers: list[GPUOffer]) -> float:
        if not offers:
            raise RuntimeError("No offers returned — cannot determine market price")
        prices = sorted(o.price_per_hr for o in offers)
        mid = len(prices) // 2
        median = prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2
        log.info(
            "vastai.market_price",
            median=f"{median:.6f}",
            sample_size=len(prices),
            min_price=prices[0],
            max_price=prices[-1],
        )
        return median

    # ── Instance lifecycle ────────────────────────────────────────────────

    async def create_instance(self, offer: GPUOffer, config: CreateConfig) -> str:
        worker_type = (
            WorkerType.ON_DEMAND if config.on_demand else WorkerType.INTERRUPTIBLE
        )
        result = await self._client.create_instance(
            offer_id=int(offer.offer_id),
            price=config.price,
            worker_api_key=config.worker_api_key,
            worker_type=worker_type,
            num_gpus=offer.num_gpus,
            label=config.label,
            ssh_public_key=config.ssh_public_key,
        )
        instance_id = str(result.get("new_contract") or result.get("id") or "")
        if not instance_id:
            raise ValueError(f"No instance ID in vast.ai response: {result}")
        return instance_id

    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        raw = await self._client.get_instance(instance_id)
        return self._to_instance_info(raw) if raw else None

    async def list_instances(self) -> list[InstanceInfo]:
        raw_list = await self._client.list_instances()
        return [self._to_instance_info(r) for r in raw_list]

    async def destroy_instance(self, instance_id: str) -> None:
        await self._client.destroy_instance(instance_id)

    async def change_bid(self, instance_id: str, new_price: float) -> bool:
        return await self._client.change_bid(instance_id, new_price)

    async def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        return await self._client.get_instance_logs(instance_id, tail=tail)

    # ── SSH support ───────────────────────────────────────────────────────

    async def list_ssh_keys(self) -> list[dict[str, Any]]:
        return await self._client.list_ssh_keys()

    async def add_ssh_key(self, pubkey_text: str) -> dict[str, Any]:
        return await self._client.add_ssh_key(pubkey_text)

    async def delete_ssh_key(self, key_id: int) -> None:
        await self._client.delete_ssh_key(key_id)

    async def attach_ssh_key(self, instance_id: str, pubkey_text: str) -> dict[str, Any]:
        return await self._client.attach_ssh_key(instance_id, pubkey_text)

    # ── Conversion helpers ────────────────────────────────────────────────

    def _to_gpu_offer(self, raw: dict[str, Any]) -> GPUOffer:
        return GPUOffer(
            offer_id=str(raw["id"]),
            provider="vastai",
            price_per_hr=raw.get("dph_base", 0.0),
            gpu_name=raw.get("gpu_name", "unknown"),
            gpu_ram_gb=round(raw.get("gpu_ram", 0) / 1024, 1),
            num_gpus=int(raw.get("num_gpus") or 1),
            specs=VastAIClient.extract_instance_specs(raw),
            raw=raw,
        )

    def _to_instance_info(self, raw: dict[str, Any]) -> InstanceInfo:
        addr     = self._client.extract_worker_address(raw)
        ssh_addr = self._client.extract_ssh_address(raw)
        gpu_ram  = raw.get("gpu_ram")
        return InstanceInfo(
            instance_id=str(raw.get("id", "")),
            provider="vastai",
            actual_status=raw.get("actual_status", "unknown"),
            cur_state=raw.get("cur_state", "") or "",
            status_msg=raw.get("status_msg", "") or "",
            next_state=raw.get("next_state", "") or "",
            gpu_name=raw.get("gpu_name"),
            gpu_ram_gb=round(gpu_ram / 1024, 1) if gpu_ram else None,
            label=raw.get("label", "") or "",
            image=raw.get("image", "") or "",
            host=addr[0] if addr else None,
            port=addr[1] if addr else None,
            ssh_host=ssh_addr[0] if ssh_addr else None,
            ssh_port=ssh_addr[1] if ssh_addr else None,
            specs=VastAIClient.extract_instance_specs(raw),
            raw=raw,
        )
