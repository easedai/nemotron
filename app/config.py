from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Provider selection ────────────────────────────────────────────────────
    # One or more GPU providers, comma-separated.  Each name must match a key
    # registered in app/orchestrator/providers/__init__.py.
    # Example: "vastai" or "vastai,salad"
    providers: str = "vastai"

    # ── vast.ai ──────────────────────────────────────────────────────────────
    # Required for the orchestrator; also used by the LB's /admin/terminate.
    vastai_api_key: Optional[str] = None

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id:     Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    dynamodb_endpoint_url: Optional[str] = None

    # ── DynamoDB tables ───────────────────────────────────────────────────────
    dynamodb_table:  str = "eased-workers"
    history_table:   str = "eased-workers-history"
    events_table:    str = "eased-instance-events"

    # The orchestrator registers/deregisters workers here;
    # the LB reads this table to pick upstreams.
    lb_workers_table: str = "eased-lb-workers"

    # ── Admin auth ────────────────────────────────────────────────────────────
    admin_token: str

    # ── Discord ───────────────────────────────────────────────────────────────
    # Required for the orchestrator; unused by the LB.
    discord_webhook_url: Optional[str] = None

    # ── Worker image ──────────────────────────────────────────────────────────
    worker_image: str = "easedai/nemotron-vastai:latest"
    worker_disk_gb: float = 100.0
    ghcr_username: Optional[str] = None
    ghcr_pat:      Optional[str] = None

    # ── vLLM / model ──────────────────────────────────────────────────────────
    vllm_port: int = 8080
    model_id: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    hf_home: str = "/hf"
    vllm_cache_root: str = "/vllm-cache"
    vllm_max_model_len: int = 32768
    # 0.90 leaves ~2.4 GB headroom on 24 GB cards for torch.compile / CUDA
    # graph capture; 0.95 OOMs on RTX A5000 during kv-cache warmup.
    vllm_gpu_memory_utilization: float = 0.90
    # Minimum free VRAM per GPU after vLLM's allocation, in GB.
    # Used to filter out offers where (1 - utilization) × gpu_ram_gb < this value.
    # CUDA graph capture + NCCL overhead requires ~1.5 GB; 2.0 GB gives margin.
    min_gpu_overhead_gb: float = 2.0
    vllm_video_loader_backend: str = "opencv"

    # ── Bidding ───────────────────────────────────────────────────────────────
    bid_start_pct: float = 0.50
    bid_step_pct: float = 0.05
    bid_retry_interval_sec: int = 300
    bid_max_multiplier: float = 1.10
    # Hard ceiling on offer price — offers above this $/hr are excluded before
    # bidding even starts.  Set to 0.0 to disable the cap.
    max_hourly_rate: float = 0.0

    # ── GPU requirements ──────────────────────────────────────────────────────
    min_gpu_ram_gb: int = 40
    min_disk_gb: float = 100.0
    min_inet_down_mbps: float = 300.0
    min_reliability: float = 0.90
    # Minimum CUDA compute capability (vast.ai integer format: 750 = SM 7.5).
    # PyTorch in the worker image requires SM >= 7.5 (Turing / T4 and newer).
    # V100 is SM 7.0 and will crash with cudaErrorNoKernelImageForDevice.
    min_compute_cap: int = 750
    # Minimum CUDA version (vast.ai field: cuda_max_good, e.g. 13.0).
    # vLLM 0.19.0+cu130 requires CUDA 13.0; instances with older drivers
    # (e.g. 560.x / CUDA 12.6) crash on torch._C._cuda_init().
    min_cuda_version: float = 13.0

    # ── Health checking ───────────────────────────────────────────────────────
    health_check_interval_sec: int = 30
    health_check_timeout_sec: int = 10
    health_check_fail_threshold: int = 3
    instance_running_timeout_sec: int = 2400
    worker_startup_timeout_sec: int = 900
    provider_check_interval_sec: int = 60
    status_report_interval_sec: int = 1800

    # ── Instance limits ───────────────────────────────────────────────────────
    max_instances: int = 1
    # Never scale below this number of running workers.
    min_instances: int = 1
    keep_debug_instance: bool = False

    # ── Auto-scaling ──────────────────────────────────────────────────────────
    # Fraction of workers that must be leased (busy) to count as a saturated tick.
    scale_up_threshold: float = 0.8
    # Number of consecutive saturated health-check ticks before bidding for more.
    # Prevents scale-up on momentary spikes.
    scale_up_consecutive_ticks: int = 3
    # Seconds to wait after a scale-up before considering another.
    # Should be longer than a cold-start (vast.ai ~10–15 min).
    scale_up_cooldown_sec: int = 600
    # Seconds without a completed request before a worker is eligible for scale-down.
    scale_down_idle_sec: int = 600
    # Number of consecutive ticks with idle workers before actually terminating one.
    # Creates hysteresis to avoid thrashing during bursty traffic.
    scale_down_consecutive_ticks: int = 5

    # ── RunPod ───────────────────────────────────────────────────────────────
    runpod_api_key: Optional[str] = None

    # ── Lambda Labs ───────────────────────────────────────────────────────────
    lambdalabs_api_key:      Optional[str] = None
    # Name of a pre-registered SSH key in Lambda Labs to attach to new instances.
    # Register via Lambda Labs console → SSH Keys, then set this to the key name.
    lambdalabs_ssh_key_name: Optional[str] = None

    # ── TensorDock ────────────────────────────────────────────────────────────
    tensordock_api_key: Optional[str] = None
    tensordock_org_id:  Optional[str] = None
    # RAM (GB) to allocate per VM; 32 GB is sufficient for Nemotron-Nano-12B.
    tensordock_ram_gb:  int           = 32

    # ── Salad ────────────────────────────────────────────────────────────────
    salad_api_key:            Optional[str] = None
    salad_org_name:           Optional[str] = None
    salad_project_name:       Optional[str] = None
    # Priority tier: batch | low | medium | high (default: high — most reliable)
    salad_priority:           str           = "high"
    # CPU and RAM allocated per container replica (must meet vLLM requirements)
    salad_container_cpu:      int           = 4
    salad_container_memory_mb: int          = 30720   # 30 GB

    # ── SSH ───────────────────────────────────────────────────────────────────
    orchestrator_ssh_private_key: Optional[str] = None

    # ── CloudWatch metrics ────────────────────────────────────────────────────
    cloudwatch_enabled: bool = True
    latency_warn_threshold_ms: float = 30_000

    # ── Redis (worker queue) ──────────────────────────────────────────────────
    redis_url:            str = "redis://localhost:6379/0"
    # Seconds an LB holds a worker lease before Redis auto-expires it.
    # Protects against a crashed LB leaving a worker permanently checked out.
    redis_lease_ttl_sec:  int = 300
    # Stable identity for this LB instance used in lease ownership tracking.
    # Defaults to the Docker container hostname (short container ID).
    lb_instance_id:       Optional[str] = None

    # ── Identity ──────────────────────────────────────────────────────────────
    orchestrator_id: str = "prod"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
