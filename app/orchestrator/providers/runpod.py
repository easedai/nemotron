from __future__ import annotations

from typing import Any, Optional

import httpx
import structlog

from .base import CreateConfig, GPUOffer, GPUProvider, InstanceInfo

log = structlog.get_logger(__name__)

_RUNPOD_GQL_URL = "https://api.runpod.io/graphql"

# RunPod desiredStatus → orchestrator actual_status
_STATE_MAP: dict[str, str] = {
    "RUNNING":     "running",
    "STARTING":    "pending",
    "RESTARTING":  "pending",
    "PAUSED":      "offline",
    "STOPPED":     "offline",
    "TERMINATED":  "deleted",
    "EXITED":      "exited",
    "FAILED":      "failed",
    "DEAD":        "failed",
}


class RunPodProvider(GPUProvider):
    """
    RunPod GPU provider.

    RunPod supports both on-demand (SECURE cloud) and spot/interruptible
    (COMMUNITY cloud) pods.  Docker containers are deployed directly — no
    VM bootstrapping required.

    Networking: RunPod exposes HTTP ports via a proxy URL of the form
    ``https://<pod_id>-8080.proxy.runpod.net``.  SSH is available on TCP
    ports and returns a public IP + random port from the runtime.ports list.

    Bid updates are not supported; ``change_bid`` always returns False.
    """

    name = "runpod"
    supports_ssh = True

    def __init__(self) -> None:
        from ...config import settings

        if not settings.runpod_api_key:
            raise RuntimeError("RUNPOD_API_KEY is required for the runpod provider")
        self._api_key = settings.runpod_api_key

    # ── HTTP / GraphQL helpers ────────────────────────────────────────────────

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=f"{_RUNPOD_GQL_URL}?api_key={self._api_key}",
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

    async def _gql(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        async with self._client() as c:
            resp = await c.post("", json=payload)
            resp.raise_for_status()
            body = resp.json()
        if "errors" in body:
            raise RuntimeError(f"RunPod GraphQL error: {body['errors']}")
        return body.get("data") or {}

    # ── Offer discovery ───────────────────────────────────────────────────────

    async def search_offers(self, on_demand: bool = False) -> list[GPUOffer]:
        from ...config import settings

        data = await self._gql("""
            query GpuTypes {
              gpuTypes {
                id
                displayName
                memoryInGb
                securePrice
                communityPrice
                lowestPrice { minimumBidPrice }
                maxGpuCount
                secureCloud
                communityCloud
              }
            }
        """)
        gpu_types: list[dict[str, Any]] = data.get("gpuTypes") or []

        offers: list[GPUOffer] = []
        for gpu in gpu_types:
            if on_demand and not gpu.get("secureCloud"):
                continue
            if not on_demand and not gpu.get("communityCloud"):
                continue

            if on_demand:
                price = float(gpu.get("securePrice") or 0.0)
            else:
                price = float(gpu.get("communityPrice") or 0.0)
                if price == 0.0:
                    lowest = gpu.get("lowestPrice") or {}
                    price = float(lowest.get("minimumBidPrice") or 0.0)

            vram_gb = float(gpu.get("memoryInGb") or 0)
            if vram_gb < settings.min_gpu_ram_gb:
                continue

            min_per_gpu_gb = settings.min_gpu_overhead_gb / (
                1.0 - settings.vllm_gpu_memory_utilization
            )
            if vram_gb < min_per_gpu_gb:
                continue

            if settings.max_hourly_rate > 0 and price > settings.max_hourly_rate:
                continue

            offers.append(GPUOffer(
                offer_id    =gpu["id"],
                provider    ="runpod",
                price_per_hr=price,
                gpu_name    =gpu.get("displayName") or gpu["id"],
                gpu_ram_gb  =vram_gb,
                num_gpus    =1,
                specs       =gpu,
                raw         =gpu,
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

        env_vars = [
            {"key": "VLLM_API_KEY",                "value": config.worker_api_key},
            {"key": "MODEL_ID",                    "value": settings.model_id},
            {"key": "VLLM_PORT",                   "value": "8080"},
            {"key": "VLLM_MAX_MODEL_LEN",          "value": str(settings.vllm_max_model_len)},
            {"key": "VLLM_GPU_MEMORY_UTILIZATION", "value": str(settings.vllm_gpu_memory_utilization)},
            {"key": "VLLM_VIDEO_LOADER_BACKEND",   "value": settings.vllm_video_loader_backend},
            {"key": "HF_HOME",                     "value": settings.hf_home},
            {"key": "VLLM_CACHE_ROOT",             "value": settings.vllm_cache_root},
            {"key": "HF_HUB_ENABLE_HF_TRANSFER",   "value": "1"},
        ]

        if config.on_demand:
            mutation = """
                mutation DeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
                  podFindAndDeployOnDemand(input: $input) {
                    id imageName desiredStatus
                  }
                }
            """
            pod_input: dict[str, Any] = {
                "cloudType":          "SECURE",
                "gpuCount":           1,
                "volumeInGb":         0,
                "containerDiskInGb":  int(settings.worker_disk_gb),
                "gpuTypeId":          offer.offer_id,
                "name":               config.label,
                "imageName":          settings.worker_image,
                "env":                env_vars,
                "ports":              "8080/http,22/tcp",
            }
            variables = {"input": pod_input}
            data = await self._gql(mutation, variables)
            pod = data.get("podFindAndDeployOnDemand") or {}
        else:
            mutation = """
                mutation DeployInterruptable($input: PodRentInterruptableInput!) {
                  podRentInterruptable(input: $input) {
                    id imageName desiredStatus
                  }
                }
            """
            pod_input = {
                "bidPerGpu":          config.price,
                "cloudType":          "COMMUNITY",
                "gpuCount":           1,
                "volumeInGb":         0,
                "containerDiskInGb":  int(settings.worker_disk_gb),
                "gpuTypeId":          offer.offer_id,
                "name":               config.label,
                "imageName":          settings.worker_image,
                "env":                env_vars,
                "ports":              "8080/http,22/tcp",
            }
            variables = {"input": pod_input}
            data = await self._gql(mutation, variables)
            pod = data.get("podRentInterruptable") or {}

        instance_id = pod.get("id")
        if not instance_id:
            raise ValueError(f"No pod ID in RunPod response: {data}")
        log.info("runpod.create_instance", pod_id=instance_id, gpu=offer.gpu_name)
        return str(instance_id)

    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        data = await self._gql(
            """
            query GetPod($input: PodFilter!) {
              pod(input: $input) {
                id name desiredStatus imageName
                runtime {
                  uptimeInSeconds
                  ports { ip isIpPublic privatePort publicPort type }
                  gpus { id gpuUtilPercent memoryUtilPercent }
                }
                machine { podHostId gpuDisplayName gpuCount }
              }
            }
            """,
            {"input": {"podId": instance_id}},
        )
        raw = data.get("pod")
        return self._to_instance_info(raw) if raw else None

    async def list_instances(self) -> list[InstanceInfo]:
        data = await self._gql("""
            query {
              myself {
                pods {
                  id name desiredStatus imageName
                  runtime {
                    uptimeInSeconds
                    ports { ip isIpPublic privatePort publicPort type }
                    gpus { id }
                  }
                  machine { podHostId gpuDisplayName gpuCount }
                }
              }
            }
        """)
        pods: list[dict[str, Any]] = (data.get("myself") or {}).get("pods") or []
        return [self._to_instance_info(p) for p in pods]

    async def destroy_instance(self, instance_id: str) -> None:
        try:
            await self._gql(
                """
                mutation TerminatePod($input: PodTerminateInput!) {
                  podTerminate(input: $input)
                }
                """,
                {"input": {"podId": instance_id}},
            )
        except Exception as exc:
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                return
            raise

    async def change_bid(self, instance_id: str, new_price: float) -> bool:
        """RunPod does not support in-place bid updates — returns False."""
        return False

    async def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        """RunPod does not expose a container logs API."""
        return (
            f"[runpod] Log retrieval is not available via the RunPod API "
            f"for pod {instance_id!r}. Use SSH or the RunPod web console."
        )

    # ── SSH key management ────────────────────────────────────────────────────

    async def list_ssh_keys(self) -> list[dict[str, Any]]:
        data = await self._gql("""
            query { myself { publicKeys { id keyValue } } }
        """)
        keys = (data.get("myself") or {}).get("publicKeys") or []
        return [{"id": k["id"], "public_key": k.get("keyValue", "")} for k in keys]

    async def add_ssh_key(self, pubkey_text: str) -> dict[str, Any]:
        data = await self._gql(
            """
            mutation AddKey($input: UserPublicKeyInput!) {
              savePublicKey(input: $input) { id keyValue }
            }
            """,
            {"input": {"keyValue": pubkey_text}},
        )
        key = (data.get("savePublicKey") or {})
        return {"id": key.get("id"), "public_key": key.get("keyValue", "")}

    async def delete_ssh_key(self, key_id: int) -> None:
        # RunPod key IDs are strings; the base class uses int for compatibility.
        await self._gql(
            """
            mutation DeleteKey($id: String!) {
              removePublicKey(keyId: $id)
            }
            """,
            {"id": str(key_id)},
        )

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _to_instance_info(self, raw: dict[str, Any]) -> InstanceInfo:
        pod_id   = str(raw.get("id") or "")
        status   = (raw.get("desiredStatus") or "UNKNOWN").upper()
        runtime  = raw.get("runtime") or {}
        machine  = raw.get("machine") or {}

        actual_status = _STATE_MAP.get(status, "unknown")

        host: Optional[str] = None
        port: Optional[int] = None
        ssh_host: Optional[str] = None
        ssh_port: Optional[int] = None

        for p in runtime.get("ports") or []:
            private  = p.get("privatePort")
            pub_port = p.get("publicPort")
            ip       = p.get("ip") or ""
            ptype    = (p.get("type") or "").lower()

            if private == 8080 and "http" in ptype:
                # RunPod HTTP proxy URL is returned in the ip field for http ports.
                host = ip if ip else f"{pod_id}-8080.proxy.runpod.net"
                port = 443
            elif private == 22:
                ssh_host = ip
                ssh_port = pub_port

        # Construct proxy URL if pod is running but ports not yet reported.
        if host is None and actual_status == "running":
            host = f"{pod_id}-8080.proxy.runpod.net"
            port = 443

        return InstanceInfo(
            instance_id  =pod_id,
            provider     ="runpod",
            actual_status=actual_status,
            cur_state    =status,
            status_msg   ="",
            next_state   ="",
            gpu_name     =machine.get("gpuDisplayName"),
            gpu_ram_gb   =None,
            label        =raw.get("name") or "",
            image        =raw.get("imageName") or "",
            host         =host,
            port         =port,
            ssh_host     =ssh_host,
            ssh_port     =ssh_port,
            specs        =raw,
            raw          =raw,
        )
