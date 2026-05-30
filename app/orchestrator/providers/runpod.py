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
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

    async def _gql(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        async with self._client() as c:
            resp = await c.post(
                _RUNPOD_GQL_URL,
                params={"api_key": self._api_key},
                json=payload,
            )
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

        worker_image = settings.runpod_worker_image or settings.worker_image

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
                "imageName":          worker_image,
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
                "imageName":          worker_image,
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
                machine { podHostId gpuDisplayName }
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
                  machine { podHostId gpuDisplayName }
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
        """
        Fetch container logs via SSH (docker logs).

        RunPod has no API log endpoint — SSH is the only option.
        Requires the orchestrator public key to be registered in the RunPod
        console under Settings → SSH Public Key.
        """
        import asyncssh
        from ...config import settings

        info = await self.get_instance(instance_id)
        if not info:
            return f"[runpod] Pod {instance_id!r} was not found on RunPod — it has likely been terminated."
        if not info.ssh_host or not info.ssh_port:
            return (
                f"[runpod] SSH address not yet available for pod {instance_id!r} "
                f"(status: {info.actual_status}). The container may still be pulling its image."
            )

        ssh_key_pem = settings.orchestrator_ssh_private_key
        if not ssh_key_pem:
            return (
                f"[runpod] ORCHESTRATOR_SSH_PRIVATE_KEY is not set. "
                "Cannot fetch logs for pod {instance_id!r} via SSH."
            )

        try:
            ssh_key = asyncssh.import_private_key(ssh_key_pem)
        except Exception as exc:
            return f"[runpod] Failed to load SSH private key: {exc}"

        # Grab the running container's logs; fall back to the most recently
        # exited container if no running container is found.
        cmd = (
            f"cid=$(docker ps -q --filter status=running | head -1); "
            f"[ -z \"$cid\" ] && cid=$(docker ps -aq | head -1); "
            f"[ -n \"$cid\" ] && docker logs --tail {tail} \"$cid\" 2>&1 "
            f"|| echo '[runpod] no container found'"
        )

        try:
            async with asyncssh.connect(
                info.ssh_host,
                port=info.ssh_port,
                username="root",
                client_keys=[ssh_key],
                known_hosts=None,
                connect_timeout=20,
            ) as conn:
                result = await conn.run(cmd, timeout=60)
                output = (result.stdout or "") + (result.stderr or "")
                log.info(
                    "runpod.get_instance_logs.ok",
                    instance_id=instance_id,
                    bytes=len(output),
                )
                return output or "(no log output)"
        except asyncssh.PermissionDenied:
            return (
                f"[runpod] SSH permission denied for pod {instance_id!r}. "
                "Make sure the orchestrator public key is set in the RunPod console "
                "under Settings → SSH Public Key."
            )
        except Exception as exc:
            log.warning("runpod.get_instance_logs.failed", instance_id=instance_id, error=repr(exc))
            return f"[runpod] Log fetch via SSH failed: {exc}"

    # ── SSH key management ────────────────────────────────────────────────────
    # RunPod stores all SSH keys as a single newline-separated string in the
    # account-level `pubKey` field (like an authorized_keys file).
    # We parse and write it as a list so we can append / remove individual keys
    # without clobbering keys set by other tools or in the RunPod console.

    async def _get_pub_keys(self) -> list[str]:
        """Return the current list of SSH public keys stored on the account."""
        data = await self._gql("query { myself { pubKey } }")
        raw = (data.get("myself") or {}).get("pubKey") or ""
        return [line for line in raw.splitlines() if line.strip()]

    async def _set_pub_keys(self, keys: list[str]) -> None:
        """Overwrite the account pubKey field with the given list of keys."""
        # Use $pubKey: String! directly — the UpdateUserInput type name varies
        # across RunPod API versions and causes 400 GRAPHQL_VALIDATION_FAILED.
        await self._gql(
            """
            mutation SetPubKey($pubKey: String!) {
              updateUserSettings(input: { pubKey: $pubKey }) { id }
            }
            """,
            {"pubKey": "\n".join(keys)},
        )

    async def list_ssh_keys(self) -> list[dict[str, Any]]:
        keys = await self._get_pub_keys()
        # Return each key as a separate entry; id is the line index so
        # delete_ssh_key can remove the right one.
        return [{"id": i, "public_key": k} for i, k in enumerate(keys)]

    async def add_ssh_key(self, pubkey_text: str) -> dict[str, Any]:
        new_parts = pubkey_text.split()[:2]  # [type, base64] — ignore comment
        keys = await self._get_pub_keys()

        for k in keys:
            if k.split()[:2] == new_parts:
                log.debug("runpod.ssh_key.already_present", pubkey_prefix=pubkey_text[:40])
                return {"id": keys.index(k), "public_key": k}

        keys.append(pubkey_text.strip())
        await self._set_pub_keys(keys)
        log.info("runpod.ssh_key.appended", pubkey_prefix=pubkey_text[:40], total=len(keys))
        return {"id": len(keys) - 1, "public_key": pubkey_text}

    async def delete_ssh_key(self, key_id: int) -> None:
        keys = await self._get_pub_keys()
        if 0 <= key_id < len(keys):
            keys.pop(key_id)
            await self._set_pub_keys(keys)
            log.info("runpod.ssh_key.removed", key_id=key_id, remaining=len(keys))

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _to_instance_info(self, raw: dict[str, Any]) -> InstanceInfo:
        pod_id  = str(raw.get("id") or "")
        status  = (raw.get("desiredStatus") or "UNKNOWN").upper()
        runtime = raw.get("runtime")   # None until the container actually starts
        machine = raw.get("machine") or {}

        # RunPod sets desiredStatus=RUNNING as soon as the pod is *allocated*,
        # but `runtime` stays None while the Docker image is still being pulled.
        # Only report actual_status="running" once the container is genuinely up.
        base_status = _STATE_MAP.get(status, "unknown")
        if base_status == "running" and not runtime:
            actual_status = "pending"   # image still pulling
        else:
            actual_status = base_status

        host: Optional[str] = None
        port: Optional[int] = None
        ssh_host: Optional[str] = None
        ssh_port: Optional[int] = None

        for p in (runtime or {}).get("ports") or []:
            private  = p.get("privatePort")
            pub_port = p.get("publicPort")
            ip       = p.get("ip") or ""
            ptype    = (p.get("type") or "").lower()

            if private == 8080 and "http" in ptype:
                host = ip if ip else f"{pod_id}-8080.proxy.runpod.net"
                port = 443
            elif private == 22:
                ssh_host = ip
                ssh_port = pub_port

        # Once the container is running and runtime is present, expose the proxy
        # URL even if RunPod hasn't populated the ports list yet.
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
