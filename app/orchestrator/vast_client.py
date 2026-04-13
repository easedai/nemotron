from __future__ import annotations

import asyncio
import base64
from typing import Any, Optional

import httpx
import structlog

from ..config import settings
from ..models import WorkerType

# ── vLLM startup patches ──────────────────────────────────────────────────────
#
# vLLM 0.19.0 (NVIDIA Nemotron fork) has three interrelated startup crashes
# when loading NanoNemotronVLProcessor.  All occur in the dummy-input path
# used to profile the multimodal encoder budget before serving begins.
#
# Patch 1 — encoder_budget.py
#   get_mm_max_toks_per_item calls get_dummy_mm_inputs which crashes.
#   Fix: wrap in try/except, fall back to max_model_len per modality.
#
# Patch 2 — transformers_utils/processor.py
#   call_hf_processor_mm_only calls processor._merge_kwargs() which
#   NanoNemotronVLProcessor does not implement.
#   Fix: hasattr guard; fall back to distributing common kwargs per modality.
#
# Patch 3 — v1/worker/gpu_model_runner.py
#   profile_run calls _get_mm_dummy_batch which also calls get_dummy_mm_inputs.
#   Fix: wrap in try/except; skip encoder profiling on failure (first real
#   request warms up the encoder instead).
#
# This Python script is base64-encoded and run via EXTRA_COMMANDS so it applies
# automatically to every new instance before onstart.sh launches vLLM.
# ─────────────────────────────────────────────────────────────────────────────
_VLLM_PATCH_SCRIPT = """\
import glob

ROOTS = ['/opt', '/usr', '/root']

def _find(name):
    for r in ROOTS:
        for p in glob.glob(r + '/**/' + name, recursive=True):
            return p
    return None

def _patch(path, old, new, label):
    if not path:
        print('SKIP (not found):', label); return
    src = open(path).read()
    if old not in src:
        print('SKIP (already patched?):', label); return
    open(path, 'w').write(src.replace(old, new, 1))
    print('patched:', label)

# ── Patch 1: encoder_budget.py ──────────────────────────────────────────────
_patch(
    _find('encoder_budget.py'),
    (
        '    mm_inputs = mm_registry.get_dummy_mm_inputs(\\n'
        '        model_config,\\n'
        '        mm_counts=mm_counts,\\n'
        '        processor=processor,\\n'
        '    )\\n'
        '\\n'
        '    return {\\n'
        '        modality: sum(item.get_num_embeds() for item in placeholders)\\n'
        '        for modality, placeholders in mm_inputs[\\"mm_placeholders\\"].items()\\n'
        '    }'
    ),
    (
        '    try:\\n'
        '        mm_inputs = mm_registry.get_dummy_mm_inputs(\\n'
        '            model_config,\\n'
        '            mm_counts=mm_counts,\\n'
        '            processor=processor,\\n'
        '        )\\n'
        '    except Exception as _exc:\\n'
        '        import logging as _l\\n'
        '        _l.getLogger(__name__).warning(\\n'
        '            "get_dummy_mm_inputs failed for %s (%s); "\\n'
        '            "falling back to max_model_len=%d per modality",\\n'
        '            type(processor).__name__, _exc, model_config.max_model_len,\\n'
        '        )\\n'
        '        return {m: model_config.max_model_len for m in mm_counts}\\n'
        '\\n'
        '    return {\\n'
        '        modality: sum(item.get_num_embeds() for item in placeholders)\\n'
        '        for modality, placeholders in mm_inputs[\\"mm_placeholders\\"].items()\\n'
        '    }'
    ),
    'encoder_budget.py',
)

# ── Patch 2: transformers_utils/processor.py ────────────────────────────────
_proc = None
for _r in ROOTS:
    for _p in glob.glob(_r + '/**/transformers_utils/processor.py', recursive=True):
        _proc = _p; break
_patch(
    _proc,
    (
        '    output_kwargs = processor._merge_kwargs(\\n'
        '        get_processor_kwargs_type(processor),\\n'
        '        **kwargs,\\n'
        '    )'
    ),
    (
        '    if hasattr(processor, \\"_merge_kwargs\\"):\\n'
        '        output_kwargs = processor._merge_kwargs(\\n'
        '            get_processor_kwargs_type(processor),\\n'
        '            **kwargs,\\n'
        '        )\\n'
        '    else:\\n'
        '        _mk = {\\"text_kwargs\\", \\"audio_kwargs\\", \\"images_kwargs\\",\\n'
        '               \\"videos_kwargs\\", \\"cross_attention_kwargs\\"}\\n'
        '        _c = {k: v for k, v in kwargs.items() if k not in _mk}\\n'
        '        output_kwargs = {\\n'
        '            \\"audio_kwargs\\":  {**_c, **kwargs.get(\\"audio_kwargs\\",  {})},\\n'
        '            \\"images_kwargs\\": {**_c, **kwargs.get(\\"images_kwargs\\", {})},\\n'
        '            \\"videos_kwargs\\": {**_c, **kwargs.get(\\"videos_kwargs\\", {})},\\n'
        '        }'
    ),
    'transformers_utils/processor.py',
)

# ── Patch 3: gpu_model_runner.py ────────────────────────────────────────────
_patch(
    _find('gpu_model_runner.py'),
    (
        '                        # Create dummy batch of multimodal inputs.\\n'
        '                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(\\n'
        '                            dummy_modality,\\n'
        '                            max_mm_items_per_batch,\\n'
        '                        )\\n'
        '\\n'
        '                        # Run multimodal encoder.\\n'
        '                        dummy_encoder_outputs = self.model.embed_multimodal(\\n'
        '                            **batched_dummy_mm_inputs\\n'
        '                        )\\n'
        '\\n'
        '                        sanity_check_mm_encoder_outputs(\\n'
        '                            dummy_encoder_outputs,\\n'
        '                            expected_num_items=max_mm_items_per_batch,\\n'
        '                        )\\n'
        '                        for i, output in enumerate(dummy_encoder_outputs):\\n'
        '                            self.encoder_cache[f\\"tmp_{i}\\"] = output'
    ),
    (
        '                        # Create dummy batch of multimodal inputs.\\n'
        '                        try:\\n'
        '                            batched_dummy_mm_inputs = self._get_mm_dummy_batch(\\n'
        '                                dummy_modality,\\n'
        '                                max_mm_items_per_batch,\\n'
        '                            )\\n'
        '                        except Exception as _gmr_exc:\\n'
        '                            import logging as _l\\n'
        '                            _l.getLogger(__name__).warning(\\n'
        '                                "Skipping encoder profiling for %s - "\\n'
        '                                "_get_mm_dummy_batch failed (%s). "\\n'
        '                                "First real request will warm up the encoder.",\\n'
        '                                dummy_modality, _gmr_exc,\\n'
        '                            )\\n'
        '                        else:\\n'
        '                            dummy_encoder_outputs = self.model.embed_multimodal(\\n'
        '                                **batched_dummy_mm_inputs\\n'
        '                            )\\n'
        '                            sanity_check_mm_encoder_outputs(\\n'
        '                                dummy_encoder_outputs,\\n'
        '                                expected_num_items=max_mm_items_per_batch,\\n'
        '                            )\\n'
        '                            for i, output in enumerate(dummy_encoder_outputs):\\n'
        '                                self.encoder_cache[f\\"tmp_{i}\\"] = output'
    ),
    'gpu_model_runner.py',
)
"""

_VLLM_PATCH_B64 = base64.b64encode(_VLLM_PATCH_SCRIPT.encode()).decode()


def _build_start_cmd(ssh_public_key: Optional[str] = None) -> str:
    """
    Build the EXTRA_COMMANDS value injected into every new vast.ai instance.

    Runs before onstart.sh:
      1. Overwrite onstart.sh — exec entrypoint.sh, tee stdout to /tmp/vllm.log
         so the orchestrator can SSH in and read vLLM output during startup.
      2. Inject orchestrator SSH public key into authorized_keys (if provided).
      3. Apply three vLLM 0.19.0 patches for NanoNemotronVLProcessor compat:
           - encoder_budget.py: wrap get_dummy_mm_inputs in try/except
           - transformers_utils/processor.py: guard _merge_kwargs with hasattr
           - gpu_model_runner.py: skip encoder profiling when dummy batch fails
    """
    parts = [
        # Tee vLLM output to a file readable over SSH
        "printf '#!/bin/bash\\nexec /entrypoint.sh 2>&1 | tee /tmp/vllm.log\\n'"
        " > /root/onstart.sh && chmod +x /root/onstart.sh",
    ]

    if ssh_public_key:
        # Single quotes extremely unlikely in ed25519 keys but sanitise anyway
        safe_key = ssh_public_key.replace("'", r"'\''")
        parts.append(
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh"
            f" && printf '%s\\n' '{safe_key}' >> /root/.ssh/authorized_keys"
            " && chmod 600 /root/.ssh/authorized_keys"
        )

    parts.append(f"echo {_VLLM_PATCH_B64} | base64 -d | python3")

    return " && ".join(parts)

log = structlog.get_logger(__name__)

VAST_API_BASE = "https://console.vast.ai/api/v0"


class VastAIClient:
    def __init__(self) -> None:
        self._headers = {"Authorization": f"Bearer {settings.vastai_api_key}"}

    # ── Offer search ──────────────────────────────────────────────────────

    async def search_offers(self, on_demand: bool = False) -> list[dict[str, Any]]:
        """
        Return GPU offers sorted ascending by price (dph_base).

        on_demand=False  → interruptible ("bid") offers — cheaper, can be reclaimed
        on_demand=True   → non-interruptible offers for the on-demand fallback
        """
        # vast.ai Search Offers API: POST /bundles/ with JSON body
        # type values: "ondemand", "bid", "reserved"
        body: dict[str, Any] = {
            "verified":    {"eq": True},
            "type":        "ondemand" if on_demand else "bid",
            "rentable":    {"eq": True},
            # vast.ai reports GPU RAM in MB
            "gpu_ram":     {"gte": settings.min_gpu_ram_gb * 1024},
            "disk_space":  {"gte": settings.min_disk_gb},
            "inet_down":   {"gte": settings.min_inet_down_mbps},
            "reliability2": {"gte": settings.min_reliability},
            "num_gpus":    {"eq": 1},
            # North America only — US and Canada datacenters
            "geolocation": {"in": ["US", "CA"]},
        }
        log.info(
            "vast.search_offers",
            on_demand=on_demand,
            min_gpu_ram_gb=settings.min_gpu_ram_gb,
            min_disk_gb=settings.min_disk_gb,
        )
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.post(f"{VAST_API_BASE}/bundles/", json=body)
            r.raise_for_status()
            offers: list[dict] = r.json().get("offers", [])

        offers.sort(key=lambda o: o.get("dph_base", float("inf")))
        log.info(
            "vast.search_offers.result",
            on_demand=on_demand,
            count=len(offers),
            cheapest=offers[0].get("dph_base") if offers else None,
        )
        return offers

    def get_market_price(self, offers: list[dict[str, Any]]) -> float:
        """
        Median dph_base across matched offers — used as the reference price
        for calculating bid percentages.
        """
        prices = sorted(o["dph_base"] for o in offers if "dph_base" in o)
        if not prices:
            raise RuntimeError("No offers returned — cannot determine market price")
        mid = len(prices) // 2
        median = prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2
        log.info(
            "vast.market_price",
            median=f"{median:.6f}",
            sample_size=len(prices),
            min_price=prices[0],
            max_price=prices[-1],
        )
        return median

    # ── Instance lifecycle ────────────────────────────────────────────────

    async def create_instance(
        self,
        offer_id: int,
        price: float,
        worker_api_key: str,
        worker_type: WorkerType,
        label: str = "",
        ssh_public_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create an instance on offer_id.

        For interruptible workers the `price` is the bid amount.
        For on-demand workers it should equal the listed dph_base.

        If `ssh_public_key` is provided it is injected into root's authorized_keys
        via EXTRA_COMMANDS so the orchestrator can SSH in during startup.
        """
        # vast.ai expects env vars as {"-e KEY=VALUE": "1"} — Docker flag format.
        #
        # vast.ai's ssh_direc/ssh_proxy runtype bypasses the Docker ENTRYPOINT
        # and instead runs /root/onstart.sh.  EXTRA_COMMANDS runs before onstart.sh
        # and is used to:
        #   • overwrite onstart.sh to exec entrypoint.sh (tee to /tmp/vllm.log)
        #   • inject the orchestrator SSH public key into authorized_keys
        #   • patch encoder_budget.py to fix the vLLM 0.19.0 startup crash
        raw_env: dict[str, str] = {
            "VLLM_API_KEY":                  worker_api_key,
            "MODEL_ID":                      settings.model_id,
            "HF_HOME":                       settings.hf_home,
            "VLLM_CACHE_ROOT":               settings.vllm_cache_root,
            "HF_HUB_ENABLE_HF_TRANSFER":     "1",
            "VLLM_PORT":                     str(settings.vllm_port),
            "VLLM_MAX_MODEL_LEN":            str(settings.vllm_max_model_len),
            "VLLM_GPU_MEMORY_UTILIZATION":   str(settings.vllm_gpu_memory_utilization),
            "VLLM_VIDEO_LOADER_BACKEND":     settings.vllm_video_loader_backend,
            "CUDA_VISIBLE_DEVICES":          "0",
            "EXTRA_COMMANDS":                _build_start_cmd(ssh_public_key),
        }
        env_vars = {f"-e {k}={v}": "1" for k, v in raw_env.items()}
        payload: dict[str, Any] = {
            "client_id": "me",
            "image":     settings.worker_image,
            "disk":      settings.worker_disk_gb,
            "env":       env_vars,
            # "ssh_direc ssh_proxy" gives SSH access; Docker ENTRYPOINT is bypassed
            "runtype":   "ssh_direc ssh_proxy",
            # Expose the vLLM port so vast.ai maps it to a random host port.
            # extract_worker_address() reads instance["ports"]["8080/tcp"] to
            # find this mapping — without it the worker never becomes reachable.
            "ports":     str(settings.vllm_port),
            # Label appears in the vast.ai UI and is used for orphan detection
            "label":     label or "eased",
        }

        # Inject GHCR credentials so vast.ai workers can pull a private image.
        # vast.ai passes this string directly to `docker login` on the host.
        if settings.ghcr_username and settings.ghcr_pat:
            registry = settings.worker_image.split("/")[0]  # e.g. ghcr.io
            payload["login"] = f"-u {settings.ghcr_username} -p {settings.ghcr_pat} {registry}"
            log.debug("vast.create_instance.registry_auth", registry=registry, username=settings.ghcr_username)
        if worker_type == WorkerType.INTERRUPTIBLE:
            payload["price"] = price  # Submit as bid below market

        log.info(
            "vast.create_instance",
            offer_id=offer_id,
            price=price,
            worker_type=worker_type,
            image=settings.worker_image,
        )
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.put(f"{VAST_API_BASE}/asks/{offer_id}/", json=payload)
            log.debug("vast.create_instance.response", status=r.status_code, body=r.text[:500])
            r.raise_for_status()
            return r.json()

    async def get_instance(self, instance_id: str) -> dict[str, Any]:
        log.debug("vast.get_instance", instance_id=instance_id)
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.get(f"{VAST_API_BASE}/instances/{instance_id}/")
            r.raise_for_status()
            result = self._extract_single_instance(r.json(), instance_id)
        if result:
            return result
        # Single-instance endpoint returned ambiguous data (instances lack id field).
        # Fall back to listing all instances and matching by ID.
        log.debug("vast.get_instance.fallback_list", instance_id=instance_id)
        all_instances = await self.list_instances()
        target = str(instance_id)
        for inst in all_instances:
            if str(inst.get("id", "")) == target:
                return inst
        return {}

    async def list_instances(self) -> list[dict[str, Any]]:
        log.debug("vast.list_instances")
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.get(f"{VAST_API_BASE}/instances/", params={"owner": "me"})
            r.raise_for_status()
            instances = self._normalise_instances(r.json())
        log.info("vast.list_instances.result", count=len(instances))
        return instances

    async def change_bid(self, instance_id: str, new_price: float) -> bool:
        """
        Attempt to raise the bid on an existing vast.ai instance.

        Uses PUT /instances/{id}/ with {"price": new_price}.
        Returns True if the bid was accepted, False if the instance is already
        gone (404) — caller should then fall back to a fresh bid campaign.
        """
        log.info("vast.change_bid", instance_id=instance_id, new_price=f"{new_price:.6f}")
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.put(
                f"{VAST_API_BASE}/instances/{instance_id}/",
                json={"price": new_price},
            )
            if r.status_code == 404:
                log.warning("vast.change_bid.instance_gone", instance_id=instance_id)
                return False
            r.raise_for_status()
        log.info("vast.change_bid.accepted", instance_id=instance_id, new_price=f"{new_price:.6f}")
        return True

    async def destroy_instance(self, instance_id: str) -> None:
        log.info("vast.destroy_instance", instance_id=instance_id)
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.delete(f"{VAST_API_BASE}/instances/{instance_id}/")
            if r.status_code == 404:
                log.warning("vast.destroy_instance.already_gone", instance_id=instance_id)
                return
            r.raise_for_status()
        log.info("vast.destroy_instance.done", instance_id=instance_id)

    async def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        """
        Request recent container logs via vast.ai's log endpoint.

        vast.ai generates the log file asynchronously and returns a presigned
        S3 URL.  We poll the URL up to 5 times (3 s apart) until content appears.
        Returns the raw log text, or a descriptive error string on failure.
        """
        log.info("vast.get_instance_logs", instance_id=instance_id, tail=tail)
        async with httpx.AsyncClient(headers=self._headers, timeout=60) as client:
            try:
                r = await client.put(
                    f"{VAST_API_BASE}/instances/request_logs/{instance_id}/",
                    json={"tail": tail},
                )
                r.raise_for_status()
                result_url: Optional[str] = r.json().get("result_url")
                log.debug(
                    "vast.get_instance_logs.url",
                    instance_id=instance_id,
                    result_url=result_url,
                )
            except Exception as exc:
                log.warning("vast.get_instance_logs.request_failed", error=str(exc))
                return f"(log request failed: {exc})"

            if not result_url:
                return "(no log URL returned by vast.ai)"

            for attempt in range(5):
                await asyncio.sleep(3)
                try:
                    log_r = await client.get(result_url)
                    if log_r.status_code == 200 and log_r.text.strip():
                        log.info(
                            "vast.get_instance_logs.ok",
                            instance_id=instance_id,
                            bytes=len(log_r.text),
                            attempt=attempt + 1,
                        )
                        return log_r.text
                except Exception as exc:
                    log.debug(
                        "vast.get_instance_logs.poll_failed",
                        attempt=attempt + 1,
                        error=str(exc),
                    )

        return "(logs not yet available — check the vast.ai dashboard)"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_instances(data: Any) -> list[dict[str, Any]]:
        """
        vast.ai returns instances in two formats depending on the endpoint
        and API version:

          List format (expected):  {"instances": [{...}, ...]}
          Dict format (observed):  {"instances": {"0": {...}, "1": {...}}}

        Normalise both into a plain Python list of instance dicts.
        """
        if isinstance(data, dict):
            raw = data.get("instances", [])
        elif isinstance(data, list):
            raw = data
        else:
            log.warning("vast.normalise_instances.unexpected_type", got=type(data).__name__)
            return []

        if isinstance(raw, dict):
            # Single-instance format — the dict IS the instance (has "id" key).
            # Seen on GET /instances/{id}/ responses.
            if "id" in raw:
                return [raw]
            # Dict-of-instances format — values are the actual instance objects.
            return [v for v in raw.values() if isinstance(v, dict)]
        if isinstance(raw, list):
            return [v for v in raw if isinstance(v, dict)]

        log.warning("vast.normalise_instances.unexpected_instances_type", got=type(raw).__name__)
        return []

    @classmethod
    def _extract_single_instance(
        cls, data: Any, instance_id: str
    ) -> dict[str, Any]:
        """
        Pull a specific instance out of any response format.
        Returns {} if the instance is not found.
        """
        instances = cls._normalise_instances(data)
        if not instances:
            return {}

        # Prefer an exact ID match
        target = str(instance_id)
        for inst in instances:
            if str(inst.get("id", "")) == target:
                return inst

        # Single-instance endpoint — trust the only result
        if len(instances) == 1:
            return instances[0]

        log.debug(
            "vast.extract_single_instance.not_found",
            instance_id=instance_id,
            candidates=[str(i.get("id")) for i in instances],
        )
        return {}

    def extract_ssh_address(
        self, instance: dict[str, Any]
    ) -> Optional[tuple[str, int]]:
        """
        Extract (host, ssh_port) for SSH access to a running vast.ai instance.

        vast.ai maps container port 22/tcp to a random host port when the
        instance runtype is ssh_direc or ssh_proxy.
        """
        host = instance.get("public_ipaddr")
        ports: dict = instance.get("ports", {})
        mapped = ports.get("22/tcp", [])
        if host and mapped:
            port = int(mapped[0]["HostPort"])
            log.debug(
                "vast.extract_ssh_address.ok",
                host=host,
                port=port,
                instance_id=instance.get("id"),
            )
            return host, port
        log.debug(
            "vast.extract_ssh_address.missing",
            host=host,
            ports=list(ports.keys()),
            instance_id=instance.get("id"),
        )
        return None

    # ── SSH key management ────────────────────────────────────────────────

    async def list_ssh_keys(self) -> list[dict[str, Any]]:
        """Return all SSH keys registered on the vast.ai account."""
        log.debug("vast.list_ssh_keys")
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.get(f"{VAST_API_BASE}/keys/")
            r.raise_for_status()
            data = r.json()
        keys = data.get("keys", [])
        log.debug("vast.list_ssh_keys.result", count=len(keys))
        return keys if isinstance(keys, list) else []

    async def add_ssh_key(self, pubkey_text: str) -> dict[str, Any]:
        """Register a new SSH public key on the vast.ai account."""
        log.info("vast.add_ssh_key", key_prefix=pubkey_text[:40])
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.post(
                f"{VAST_API_BASE}/keys/",
                json={"public_key": pubkey_text},
            )
            log.debug("vast.add_ssh_key.response", status=r.status_code, body=r.text[:300])
            r.raise_for_status()
            return r.json()

    async def delete_ssh_key(self, key_id: int) -> None:
        """Remove an SSH key from the vast.ai account by its numeric ID."""
        log.info("vast.delete_ssh_key", key_id=key_id)
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            r = await client.delete(f"{VAST_API_BASE}/keys/{key_id}/")
            if r.status_code == 404:
                log.debug("vast.delete_ssh_key.already_gone", key_id=key_id)
                return
            r.raise_for_status()
        log.debug("vast.delete_ssh_key.done", key_id=key_id)

    @staticmethod
    def extract_instance_specs(instance: dict[str, Any]) -> dict[str, Any]:
        """
        Pull hardware / performance details from a vast.ai instance or offer dict.

        Field mapping (vast.ai → specs key):
          id               → instance_id
          host_id          → host_id
          machine_id       → machine_id
          cuda_max_good    → cuda_max
          total_flops      → tflops          (TFLOPS, GPU compute)
          gpu_mem_bw       → mem_bw_gbps     (GPU memory bandwidth)
          dlperf           → dlperf          (DL performance score)
          dlperf_usd       → dlperf_per_hr   (DL perf per $/hr)
          direct_port_count→ num_ports
          inet_down        → inet_down_mbps
          inet_up          → inet_up_mbps
          cpu_name         → cpu_name
          cpu_cores_effective → cpu_cores
          cpu_ram          → cpu_ram_gb
          disk_bw          → disk_bw_mbps
          disk_space       → disk_gb
          mobo_name        → mobo
          pcie_bw          → pcie_bw_gbps
          pcie_gen         → pcie_gen
          pcie_lanes       → pcie_lanes
          num_gpus         → num_gpus
          vram_costperhour → vram_costperhour
        """
        def _f(key: str) -> Optional[float]:
            v = instance.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _i(key: str) -> Optional[int]:
            v = instance.get(key)
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _s(key: str) -> Optional[str]:
            v = instance.get(key)
            return str(v).strip() if v is not None else None

        raw: dict[str, Any] = {
            "instance_id":    _i("id"),
            "host_id":        _i("host_id"),
            "machine_id":     _i("machine_id"),
            # GPU
            "cuda_max":       _s("cuda_max_good"),
            "tflops":         _f("total_flops"),
            "mem_bw_gbps":    _f("gpu_mem_bw"),
            "dlperf":         _f("dlperf"),
            "dlperf_per_hr":  _f("dlperf_usd"),
            "num_gpus":       _i("num_gpus"),
            # Network
            "num_ports":      _i("direct_port_count"),
            "inet_down_mbps": _f("inet_down"),
            "inet_up_mbps":   _f("inet_up"),
            # CPU
            "cpu_name":       _s("cpu_name"),
            "cpu_cores":      _f("cpu_cores_effective"),
            "cpu_ram_gb":     _f("cpu_ram"),
            # Disk
            "disk_name":      _s("disk_name"),
            "disk_bw_mbps":   _f("disk_bw"),
            "disk_gb":        _f("disk_space"),
            # Motherboard / PCIe
            "mobo":           _s("mobo_name"),
            "pcie_bw_gbps":   _f("pcie_bw"),
            "pcie_gen":       _i("pcie_gen"),
            "pcie_lanes":     _i("pcie_lanes"),
        }
        # Drop None values to keep storage lean
        return {k: v for k, v in raw.items() if v is not None}

    def extract_worker_address(
        self, instance: dict[str, Any]
    ) -> Optional[tuple[str, int]]:
        """
        Extract (host, port) from a running vast.ai instance dict.

        vast.ai maps container port 8000/tcp to a random host port.
        The instance dict carries `public_ipaddr` and `ports`.
        """
        actual_status = instance.get("actual_status", "")
        if actual_status != "running":
            log.debug(
                "vast.extract_address.not_running",
                actual_status=actual_status,
                instance_id=instance.get("id"),
            )
            return None

        host = instance.get("public_ipaddr")
        ports: dict = instance.get("ports", {})
        # e.g. {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "34567"}]}
        #
        # Try the configured port first, then fall back to 8000 in case the
        # worker image still has EXPOSE 8000 from before the port change.
        candidates = [settings.vllm_port]
        if settings.vllm_port != 8000:
            candidates.append(8000)

        for candidate in candidates:
            port_key = f"{candidate}/tcp"
            mapped = ports.get(port_key, [])
            if host and mapped:
                port = int(mapped[0]["HostPort"])
                if candidate != settings.vllm_port:
                    log.warning(
                        "vast.extract_address.fallback_port",
                        configured_port=settings.vllm_port,
                        actual_port=candidate,
                        host=host,
                        instance_id=instance.get("id"),
                    )
                else:
                    log.info(
                        "vast.extract_address.ok",
                        host=host,
                        port=port,
                        instance_id=instance.get("id"),
                    )
                return host, port

        log.warning(
            "vast.extract_address.missing",
            host=host,
            ports=ports,
            tried_ports=candidates,
            instance_id=instance.get("id"),
        )
        return None
