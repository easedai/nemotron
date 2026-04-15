from __future__ import annotations

import textwrap
from typing import Any, Optional

import httpx
import structlog

from .base import CreateConfig, GPUOffer, GPUProvider, InstanceInfo

log = structlog.get_logger(__name__)

_TENSORDOCK_API_BASE = "https://marketplace.tensordock.com/api/v0"

# TensorDock VM status → orchestrator actual_status.
# TensorDock uses "running", "stopped", "paused", "error" as primary states.
_STATE_MAP: dict[str, str] = {
    "running":     "running",
    "starting":    "pending",
    "provisioning": "pending",
    "stopped":     "offline",
    "paused":      "offline",
    "error":       "failed",
    "deleted":     "deleted",
}

# GPU model string fragments → VRAM GB lookup.
_VRAM_LOOKUP: dict[str, float] = {
    "h100 sxm":  80.0,
    "h100 pcie": 80.0,
    "h100":      80.0,
    "a100 80gb": 80.0,
    "a100 40gb": 40.0,
    "a100":      40.0,
    "a6000":     48.0,
    "a40":       48.0,
    "a30":       24.0,
    "a10":       24.0,
    "a5000":     24.0,
    "a4000":     16.0,
    "rtx 4090":  24.0,
    "rtx 4080":  16.0,
    "rtx 3090":  24.0,
    "rtx 3080":  10.0,
    "rtx 3070":   8.0,
    "v100 32gb": 32.0,
    "v100":      16.0,
    "t4":        16.0,
    "l40s":      48.0,
    "l40":       48.0,
    "l4":        24.0,
    "4090":      24.0,
    "3090":      24.0,
}


def _gpu_vram(gpu_model: str) -> float:
    """Infer VRAM GB from a TensorDock GPU model name."""
    import re

    m = re.search(r"(\d+)\s*[Gg][Bb]", gpu_model)
    if m:
        return float(m.group(1))
    lower = gpu_model.lower()
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
    Build a cloud-init user_data script that installs Docker (if missing) and
    starts the vLLM worker container on first boot.

    TensorDock deploys Ubuntu VMs; Docker may need to be installed.
    """
    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail

        # Install Docker if not present
        if ! command -v docker >/dev/null 2>&1; then
            curl -fsSL https://get.docker.com | sh
            systemctl enable --now docker
        fi

        # Install NVIDIA Container Toolkit if not present
        if ! dpkg -l nvidia-container-toolkit >/dev/null 2>&1; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\
                gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt-get update -qq
            apt-get install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker
            systemctl restart docker
        fi

        # Wait for Docker daemon
        until docker info >/dev/null 2>&1; do sleep 2; done

        # Write env vars (never expose secrets in process list)
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


class TensorDockProvider(GPUProvider):
    """
    TensorDock GPU marketplace provider.

    TensorDock offers spot-like pricing with hourly billing on bare-metal GPU
    nodes.  Instances are VMs; the vLLM worker Docker container is started via
    a cloud-init user_data script on first boot.

    Networking: the VM receives a public IP; vLLM is accessible at
    ``http://<ip>:8080``.  SSH is available on port 22.

    ``change_bid`` is not supported and always returns False.
    """

    name = "tensordock"
    supports_ssh = False  # no SSH key management API; manual key setup required

    def __init__(self) -> None:
        from ...config import settings

        if not settings.tensordock_api_key:
            raise RuntimeError("TENSORDOCK_API_KEY is required for the tensordock provider")
        if not settings.tensordock_org_id:
            raise RuntimeError("TENSORDOCK_ORG_ID is required for the tensordock provider")

        self._api_key = settings.tensordock_api_key
        self._org_id  = settings.tensordock_org_id

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_TENSORDOCK_API_BASE,
            timeout=60.0,
        )

    def _auth_params(self) -> dict[str, str]:
        return {"api": self._api_key, "org": self._org_id}

    # ── Offer discovery ───────────────────────────────────────────────────────

    async def search_offers(self, on_demand: bool = False) -> list[GPUOffer]:
        """
        Returns one offer per (hostnode, GPU model) combination with available
        stock.  TensorDock has no separate spot vs. on-demand tiers.
        """
        from ...config import settings

        async with self._client() as c:
            resp = await c.post(
                "/client/deploy/hostnodes",
                data=self._auth_params(),
            )
            resp.raise_for_status()
            data = resp.json()

        hostnodes: dict[str, Any] = data.get("hostnodes") or {}
        offers: list[GPUOffer] = []

        for node_id, node in hostnodes.items():
            specs  = node.get("specs") or {}
            gpus   = specs.get("gpu") or {}
            status = node.get("status") or {}

            for gpu_model, gpu_info in gpus.items():
                amount      = int(gpu_info.get("amount") or 0)
                price_cents = float(gpu_info.get("price") or 0)  # $/hr in some versions
                price_hr    = price_cents  # TensorDock reports $/hr directly

                if amount < 1:
                    continue

                vram_gb = _gpu_vram(gpu_model)
                if vram_gb == 0.0:
                    log.warning(
                        "tensordock.search_offers.unknown_vram",
                        node_id=node_id,
                        gpu_model=gpu_model,
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

                offer_id = f"{node_id}:{gpu_model}"
                offers.append(GPUOffer(
                    offer_id    =offer_id,
                    provider    ="tensordock",
                    price_per_hr=price_hr,
                    gpu_name    =gpu_model,
                    gpu_ram_gb  =vram_gb,
                    num_gpus    =1,
                    specs       ={"node_id": node_id, "gpu": gpu_info, "node": node},
                    raw         =node,
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

        node_id, gpu_model = offer.offer_id.split(":", 1)

        user_data = _make_user_data(
            worker_api_key              =config.worker_api_key,
            model_id                    =settings.model_id,
            vllm_max_model_len          =settings.vllm_max_model_len,
            vllm_gpu_memory_utilization =settings.vllm_gpu_memory_utilization,
            vllm_video_loader_backend   =settings.vllm_video_loader_backend,
            hf_home                     =settings.hf_home,
            vllm_cache_root             =settings.vllm_cache_root,
            worker_image                =settings.worker_image,
        )

        params: dict[str, Any] = {
            **self._auth_params(),
            "hostnode":         node_id,
            "gpu_model":        gpu_model,
            "gpu_count":        1,
            "vcpus":            4,
            "ram":              int(settings.tensordock_ram_gb) * 1024,  # MB
            "storage":          int(settings.worker_disk_gb),
            "operating_system": "Ubuntu 22.04 LTS",
            "name":             config.label,
            "user_data":        user_data,
            # Expose vLLM port 8080 externally.
            "external_ports":   '{"8080/tcp": 8080}',
        }

        async with self._client() as c:
            resp = await c.post("/client/deploy/single", data=params)
            resp.raise_for_status()
            result = resp.json()

        instance_id = str(
            result.get("server") or result.get("id") or result.get("uuid") or ""
        )
        if not instance_id:
            raise ValueError(f"No server ID in TensorDock response: {result}")
        log.info(
            "tensordock.create_instance",
            instance_id=instance_id,
            node_id=node_id,
            gpu_model=gpu_model,
        )
        return instance_id

    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        instances = await self.list_instances()
        for inst in instances:
            if inst.instance_id == instance_id:
                return inst
        return None

    async def list_instances(self) -> list[InstanceInfo]:
        async with self._client() as c:
            resp = await c.post("/client/list", data=self._auth_params())
            resp.raise_for_status()
            data = resp.json()

        servers: dict[str, Any] = data.get("servers") or {}
        return [
            self._to_instance_info(server_id, info)
            for server_id, info in servers.items()
        ]

    async def destroy_instance(self, instance_id: str) -> None:
        async with self._client() as c:
            resp = await c.delete(
                f"/client/delete/{instance_id}",
                params=self._auth_params(),
            )
            if resp.status_code == 404:
                return
            resp.raise_for_status()

    async def change_bid(self, instance_id: str, new_price: float) -> bool:
        """TensorDock does not support in-place bid updates — returns False."""
        return False

    async def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        """TensorDock does not provide a container logs API."""
        return (
            f"[tensordock] Log retrieval is not available via the TensorDock API "
            f"for instance {instance_id!r}.  SSH to the instance and run: "
            f"docker logs --tail {tail} vllm-worker"
        )

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _to_instance_info(self, server_id: str, raw: dict[str, Any]) -> InstanceInfo:
        status = (raw.get("status") or "unknown").lower()
        actual_status = _STATE_MAP.get(status, "unknown")

        ip: Optional[str] = raw.get("ip") or raw.get("public_ip")

        # TensorDock port mappings may be under "port_forwards" or "networking".
        vllm_port:  Optional[int] = None
        networking = raw.get("networking") or raw.get("port_forwards") or {}
        if isinstance(networking, dict):
            for ext_port, int_port in networking.items():
                try:
                    if int(int_port) == 8080:
                        vllm_port = int(str(ext_port).split("/")[0])
                except (ValueError, TypeError):
                    pass

        gpu_info  = raw.get("gpu") or {}
        gpu_model = gpu_info.get("model") if isinstance(gpu_info, dict) else None
        vram_gb   = _gpu_vram(gpu_model) if gpu_model else None

        return InstanceInfo(
            instance_id  =server_id,
            provider     ="tensordock",
            actual_status=actual_status,
            cur_state    =status,
            status_msg   =raw.get("message") or "",
            next_state   ="",
            gpu_name     =gpu_model,
            gpu_ram_gb   =vram_gb or None,
            label        =raw.get("name") or "",
            image        ="",
            host         =ip,
            port         =vllm_port if ip else None,
            ssh_host     =ip,
            ssh_port     =22 if ip else None,
            specs        =raw,
            raw          =raw,
        )
