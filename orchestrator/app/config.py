from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── vast.ai ──────────────────────────────────────────────────────────────
    vastai_api_key: str

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    # Overridden in docker-compose to point at dynamodb-local
    dynamodb_endpoint_url: Optional[str] = None
    dynamodb_table:  str = "eased-workers"
    # Permanent record of all workers — written alongside eased-workers but
    # never deleted, so terminated instances remain queryable indefinitely.
    history_table:   str = "eased-workers-history"
    events_table:    str = "eased-instance-events"

    # ── Admin auth ───────────────────────────────────────────────────────────
    # Bearer token required on all /admin/* routes.
    # Generate with: openssl rand -hex 32
    admin_token: str

    # ── Discord ───────────────────────────────────────────────────────────────
    discord_webhook_url: str

    # ── Load-balancer worker table ────────────────────────────────────────────
    # The orchestrator registers/deregisters workers here; the load-balancer
    # service reads this table to pick upstreams for each incoming request.
    lb_workers_table: str = "eased-lb-workers"

    # ── Worker image ──────────────────────────────────────────────────────────
    worker_image: str = "easedai/nemotron-vastai:latest"
    # GHCR credentials so vast.ai workers can pull a private image.
    # Set GHCR_USERNAME (GitHub org/user) and GHCR_PAT (read:packages PAT).
    ghcr_username: Optional[str] = None
    ghcr_pat: Optional[str] = None
    # Disk: model ~24 GB + vLLM image ~15 GB + OS/tmp -> 100 GB recommended
    worker_disk_gb: float = 100.0

    # ── vLLM / model ──────────────────────────────────────────────────────────
    # vast.ai maps this container port to a random host port.
    vllm_port: int = 8080
    # HuggingFace model ID passed as --model to vLLM.
    # vLLM resolves this through HF_HOME so no explicit local path is needed.
    model_id: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    # HF_HOME inside the worker container -- must match the path used in
    # worker/Dockerfile so the baked-in weights are found without a download.
    hf_home: str = "/hf"
    # vLLM compiled Triton kernel / torch.compile artifact cache.
    vllm_cache_root: str = "/vllm-cache"
    # Context length.  Model supports up to 131072 but:
    #   >= 40 GB VRAM -> use <= 32768 (default)
    #   >= 80 GB VRAM -> use up to 131072 for long-video workloads
    vllm_max_model_len: int = 32768
    vllm_gpu_memory_utilization: float = 0.95
    # Video-loader backend -- opencv is required for --video-pruning-rate
    vllm_video_loader_backend: str = "opencv"

    # ── Bidding ───────────────────────────────────────────────────────────────
    bid_start_pct: float = 0.50        # Start at 50 % of market
    bid_step_pct: float = 0.05         # +5 % each retry
    bid_retry_interval_sec: int = 300  # Retry every 5 min
    # Give up bidding above this multiple of market; fall back to on-demand
    bid_max_multiplier: float = 1.10

    # ── GPU requirements ──────────────────────────────────────────────────────
    # Nemotron-Nano-12B (BF16) uses ~24 GB for weights alone.
    # A 40 GB GPU (A100 40 GB) is the practical minimum for useful context lengths.
    min_gpu_ram_gb: int = 40
    min_disk_gb: float = 100.0
    min_inet_down_mbps: float = 300.0
    min_reliability: float = 0.90

    # ── Health checking ───────────────────────────────────────────────────────
    health_check_interval_sec: int = 30
    health_check_timeout_sec: int = 10
    # Consecutive failures before marking a worker as lost
    health_check_fail_threshold: int = 3
    # How long to wait for the vast.ai instance to reach "running" status
    # (includes image pull, which can be slow on some nodes)
    instance_running_timeout_sec: int = 2400  # 40 min
    # How long to wait for vLLM to become healthy after the container starts
    # (clock resets once the instance is running — image pull is NOT counted)
    worker_startup_timeout_sec: int = 900  # 15 min (model load only)
    # How often to cross-check active workers against the vast.ai instance list
    vast_check_interval_sec: int = 60
    # How often to post a status summary to Discord (must be >= vast_check_interval_sec)
    status_report_interval_sec: int = 1800  # 30 min

    # ── Instance limits ───────────────────────────────────────────────────────
    # Maximum number of operational worker instances.  The orchestrator will
    # not bid for more than this many workers at once.
    max_instances: int = 1

    # When True, failed instances are NOT destroyed -- they stay alive on
    # vast.ai so you can SSH in and inspect logs.  To prevent infinite
    # accumulation, at most one debug instance is kept alive at a time
    # (max_instances + 1 total).  When that slot is already taken, the
    # *newest* debug instance is destroyed to make room for the next failure.
    # Set to False (the default) in production.
    keep_debug_instance: bool = False

    # ── SSH -- orchestrator -> vast.ai worker access ──────────────────────────
    # OpenSSH-format Ed25519 private key injected by ECS from Secrets Manager
    # (ORCHESTRATOR_SSH_PRIVATE_KEY env var).  When set, the orchestrator uses
    # this stable key instead of generating an ephemeral one at startup.
    # Leave unset for local dev -- a fresh key is generated automatically.
    orchestrator_ssh_private_key: Optional[str] = None

    # ── CloudWatch metrics ────────────────────────────────────────────────────
    # Set to False to disable metric emission (e.g. local dev without AWS creds).
    cloudwatch_enabled: bool = True
    # Milliseconds above which a request is logged at WARNING level.
    # Same threshold the CloudWatch alarm should use for scale-up decisions.
    latency_warn_threshold_ms: float = 30_000  # 30 s

    # ── Identity ──────────────────────────────────────────────────────────────
    # Human-readable name for this orchestrator replica — included in all
    # Discord notifications so you can tell local dev from prod.
    # Defaults to "prod"; override in .env with ORCHESTRATOR_ID=local for dev.
    orchestrator_id: str = "prod"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
