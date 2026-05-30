from __future__ import annotations

import asyncio
import base64
import random
import re
import secrets
from datetime import datetime, timezone
from typing import Optional

import asyncssh
import boto3
import httpx
import structlog
from botocore.exceptions import ClientError

from ..config import settings
from ..worker_db import DynamoDB
from ..event_store import EventStore
from ..lb_queue import WorkerQueue
from ..lb_registry import LBRegistry
from ..models import Worker, WorkerStatus, WorkerType
from .notifications import Discord
from .providers import CreateConfig, GPUOffer, GPUProvider, InstanceInfo, get_provider

log = structlog.get_logger(__name__)

# Consecutive misses required before treating an instance as truly gone
_MISS_THRESHOLD = 3

# Seconds between SSH log checks while waiting for vLLM to start.
_STARTUP_LOG_CHECK_INTERVAL = 60

# Known fatal patterns in vLLM logs that mean the instance will never recover.
# Order matters — more specific patterns first.
_FATAL_LOG_PATTERNS: list[tuple[str, str]] = [
    (
        r"CUDA out of memory|torch\.OutOfMemoryError",
        "GPU out of memory — model too large for available VRAM",
    ),
    (
        r"The NVIDIA driver on your system is too old \(found version (\S+)\)",
        "NVIDIA driver too old for this CUDA build (need driver ≥ 575 for CUDA 13.0)",
    ),
    (
        r"torch\._C\._cuda_init\(\)",
        "CUDA init failed — driver/CUDA version mismatch",
    ),
    (
        r"Engine core initialization failed|EngineCore failed to start",
        "vLLM engine core failed to initialize",
    ),
    (
        r"RuntimeError: CUDA error:",
        "CUDA runtime error during model load",
    ),
    (
        r"No space left on device",
        "Disk full — insufficient space for model weights",
    ),
    (
        r"No module named '([^']+)'",
        "Missing Python dependency",
    ),
]


def _diagnose_vllm_logs(log_text: str) -> Optional[str]:
    """
    Scan vLLM log text for known fatal error patterns.

    Returns a short human-readable diagnosis string (suitable for Discord and the
    _fail_worker reason), or None if no recognised fatal pattern is found.
    """
    for pattern, label in _FATAL_LOG_PATTERNS:
        m = re.search(pattern, log_text)
        if m:
            detail = m.group(1) if m.lastindex else None
            return f"{label} ({detail})" if detail else label
    return None


def _jitter(base: float, pct: float = 0.20) -> float:
    """Return *base* ± *pct* fraction so polling never lands on a fixed cadence."""
    return max(1.0, base * (1 + random.uniform(-pct, pct)))


class WorkerManager:
    def __init__(self) -> None:
        self.db       = DynamoDB()
        self._providers: dict[str, GPUProvider] = {
            name: get_provider(name)
            for name in settings.provider_list
        }
        self.discord  = Discord()
        self.events   = EventStore()
        # lb and _queue are initialised in start() after the event loop is running
        self._queue: Optional[WorkerQueue] = None
        self.lb:     Optional[LBRegistry]  = None
        self._bidding_task:      Optional[asyncio.Task] = None
        self._monitor_task:      Optional[asyncio.Task] = None
        self._provider_monitor_task: Optional[asyncio.Task] = None
        # Consecutive times each worker_id has been missing from provider list.
        # Only fail the worker once this exceeds _MISS_THRESHOLD.
        self._instance_miss_counts: dict[str, int] = {}
        # Multiplier at which the last bid was won. The next campaign starts one
        # step below this so we probe whether an even cheaper bid will land.
        self._last_winning_multiplier: Optional[float] = None
        # Number of times an interruptible instance was preempted (outbid) this
        # session.  Each preemption raises the starting bid floor by one
        # bid_step_pct so repeated outbids automatically escalate the bid price.
        self._preemption_count: int = 0
        # SSH key pair generated at startup and injected into every instance so
        # we can SSH in to read vLLM logs during startup.
        self._ssh_key:        Optional[asyncssh.SSHKey] = None
        self._ssh_public_key: Optional[str] = None
        # Active _wait_for_vllm_health asyncio tasks, keyed by worker_id.
        # Prevents duplicate tasks from spawning on every reconcile/sync tick
        # when the same worker is already being polled.
        self._health_check_tasks: dict[str, asyncio.Task] = {}
        # ── Auto-scaling state ────────────────────────────────────────────────
        # Rolling window of utilization samples (one per health-check tick).
        self._utilization_history: list[float] = []
        # Timestamp of the last scale-up trigger; enforces cooldown between bids.
        self._last_scale_up_at: Optional[datetime] = None
        # Consecutive health-check ticks with idle workers; used for scale-down
        # hysteresis so a single idle sample doesn't immediately kill a worker.
        self._scale_down_ticks: int = 0

    def _ev(
        self,
        worker:     Worker,
        event_type: str,
        message:    str,
        prev_status: Optional[str] = None,
        meta:        Optional[dict] = None,
    ) -> None:
        """Convenience wrapper — records an event with worker context filled in."""
        self.events.record(
            worker_id=worker.worker_id,
            event_type=event_type,
            status=worker.status.value,
            message=message,
            instance_id=worker.instance_id,
            label=worker.label,
            prev_status=prev_status,
            meta=meta,
        )

    # ── Provider helpers ──────────────────────────────────────────────────

    def _provider_for(self, worker: Worker) -> GPUProvider:
        """Return the loaded GPUProvider that owns this worker."""
        p = self._providers.get(worker.provider)
        if p is None:
            available = ", ".join(repr(k) for k in self._providers)
            raise RuntimeError(
                f"No provider loaded for {worker.provider!r} "
                f"(worker {worker.worker_id}). Loaded: {available}"
            )
        return p

    async def _all_instances(self) -> list[InstanceInfo]:
        """Query every configured provider in parallel and aggregate results."""
        results: list[InstanceInfo] = []
        tasks = {
            name: asyncio.create_task(p.list_instances())
            for name, p in self._providers.items()
        }
        for name, task in tasks.items():
            try:
                results.extend(await task)
            except Exception as exc:
                log.warning(
                    "worker_manager.all_instances.provider_failed",
                    provider=name,
                    error=str(exc),
                )
        return results

    async def _search_all_offers(self, on_demand: bool = False) -> list[GPUOffer]:
        """Search every configured provider in parallel and return combined offers."""
        results: list[GPUOffer] = []
        tasks = {
            name: asyncio.create_task(p.search_offers(on_demand=on_demand))
            for name, p in self._providers.items()
        }
        for name, task in tasks.items():
            try:
                results.extend(await task)
            except Exception as exc:
                log.warning(
                    "worker_manager.search_offers.provider_failed",
                    provider=name,
                    error=str(exc),
                )
        results.sort(key=lambda o: o.price_per_hr)
        return results

    def _market_price(self, offers: list[GPUOffer]) -> float:
        """Median price across all offers (ignoring zero-priced placeholder entries)."""
        prices = sorted(o.price_per_hr for o in offers if o.price_per_hr > 0)
        if not prices:
            return 0.0
        mid = len(prices) // 2
        return prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2

    # ── Startup / shutdown ────────────────────────────────────────────────

    async def start(self) -> None:
        log.info("worker_manager.start")

        # Connect to Redis and wire up the LB registry shim.
        self._queue = await WorkerQueue.create(
            settings.redis_url,
            lease_ttl=settings.redis_lease_ttl_sec,
        )
        self.lb = LBRegistry(self._queue)

        # Resolve the SSH key pair used to access GPU worker instances.
        #
        # Priority:
        #   1. ORCHESTRATOR_SSH_PRIVATE_KEY env var — injected by ECS from
        #      Secrets Manager (ssh_keys.tf / Terraform).
        #   2. Fetch directly from AWS Secrets Manager — used for local dev
        #      where the container has AWS credentials mounted but the env var
        #      is not pre-populated.
        #   3. Generate an ephemeral Ed25519 key — last resort fallback; the
        #      key is not stable across restarts so SSH access won't work until
        #      the account-level key registration succeeds.
        if settings.orchestrator_ssh_private_key:
            self._ssh_key = asyncssh.import_private_key(
                settings.orchestrator_ssh_private_key
            )
            log.info("worker_manager.start.ssh_key_from_env")
        else:
            pem = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_ssh_key_from_aws
            )
            if pem:
                self._ssh_key = asyncssh.import_private_key(pem)
                log.info("worker_manager.start.ssh_key_from_secrets_manager")
            else:
                self._ssh_key = asyncssh.generate_private_key("ssh-ed25519")
                log.info("worker_manager.start.ssh_key_generated_ephemeral")
        raw_pubkey = self._ssh_key.export_public_key("openssh").decode().strip()
        self._ssh_public_key = f"{raw_pubkey} eased-orchestrator"
        log.info("worker_manager.start.ssh_key_ready", pubkey_prefix=raw_pubkey[:40])
        await self._manage_ssh_keys()

        await self._reconcile_state()
        await self._sync_redis_queue()
        await self._destroy_zombie_instances()
        await self._post_startup_audit()
        self._monitor_task = asyncio.create_task(
            self._health_monitor_loop(), name="health-monitor"
        )
        self._provider_monitor_task = asyncio.create_task(
            self._provider_monitor_loop(), name="provider-monitor"
        )
        await self._ensure_worker()

    async def stop(self) -> None:
        log.info("worker_manager.stop")

        # Cancel background tasks first so nothing races with the cleanup below
        for task in (self._bidding_task, self._monitor_task, self._provider_monitor_task):
            if task and not task.done():
                task.cancel()

        # NOTE: instances are intentionally NOT destroyed on shutdown.
        # Multiple orchestrator replicas may be running simultaneously (e.g.
        # rolling ECS deployments or local dev alongside production), so tearing
        # down instances on exit would kill workers owned by a sibling replica.
        log.info("worker_manager.stop.skipping_instance_cleanup")

        # The orchestrator SSH key is intentionally NOT removed here — it is
        # a persistent account-level credential (idempotently registered by
        # _manage_ssh_keys on startup) that other running orchestrator replicas
        # and already-launched instances still rely on.

    async def _destroy_all_instances(self) -> None:
        """
        Destroy every instance we own (active workers + debug instances kept alive by
        keep_debug_instance) and update their DB status to TERMINATED.

        Runs concurrently so multiple instances are torn down in parallel.
        destroy_instance handles 404 gracefully, so already-gone instances are safe.
        """
        all_workers = self.db.list_workers()
        to_destroy  = [w for w in all_workers if w.instance_id]

        if not to_destroy:
            log.info("worker_manager.stop.no_instances")
            return

        log.info("worker_manager.stop.destroying_all", count=len(to_destroy))
        await self.discord.send(
            f"**Orchestrator shutting down** — destroying {len(to_destroy)} instance(s): "
            + ", ".join(f"`{w.instance_id}`" for w in to_destroy),
            "warning",
        )

        async def _destroy(worker: Worker) -> None:
            log.info(
                "worker_manager.stop.destroying",
                instance_id=worker.instance_id,
                worker_id=worker.worker_id,
                status=worker.status,
            )
            try:
                await self._provider_for(worker).destroy_instance(worker.instance_id)
                self.db.delete_worker(worker.worker_id)
            except Exception as exc:
                log.error(
                    "worker_manager.stop.destroy_failed",
                    instance_id=worker.instance_id,
                    error=repr(exc),
                )

        await asyncio.gather(*(_destroy(w) for w in to_destroy), return_exceptions=True)
        log.info("worker_manager.stop.done")

    # ── Public interface used by the proxy layer ──────────────────────────

    def get_best_worker(self) -> Optional[Worker]:
        """Return the best RUNNING worker for proxying. Prefers interruptible (cheaper)."""
        workers = self.db.get_running_workers()
        if not workers:
            log.warning("worker_manager.get_best_worker.none_available")
            return None
        interruptible = [w for w in workers if w.worker_type == WorkerType.INTERRUPTIBLE]
        chosen = interruptible[0] if interruptible else workers[0]
        log.debug(
            "worker_manager.get_best_worker",
            worker_id=chosen.worker_id,
            worker_type=chosen.worker_type,
        )
        return chosen

    # ── Redis queue sync ──────────────────────────────────────────────────

    async def _sync_redis_queue(self) -> None:
        """
        Reconcile the Redis worker queue against DynamoDB after startup.

        Called after _reconcile_state() so that DynamoDB is already the
        authoritative source of truth.  Any worker that is RUNNING in DynamoDB
        but absent from Redis is re-registered; any worker in Redis that is no
        longer RUNNING in DynamoDB is deregistered.

        All discrepancies are logged at WARNING level so they are visible in
        production monitoring.
        """
        if self._queue is None:
            log.error("worker_manager.sync_redis.queue_not_ready")
            return
        running_workers = self.db.get_active_workers()
        result = await self._queue.sync(running_workers)
        if result["added"] or result["removed"]:
            log.warning(
                "worker_manager.sync_redis.repaired",
                added=result["added"],
                removed=result["removed"],
            )
        else:
            log.info("worker_manager.sync_redis.ok")

    # ── State reconciliation ──────────────────────────────────────────────

    async def _reconcile_state(self) -> None:
        """
        On startup, cross-check DynamoDB records against live provider instances.
        Repairs stale state left by a previous orchestrator crash or restart.
        Workers still pending have their _wait_for_running task resumed.
        """
        log.info("worker_manager.reconcile.start")
        db_workers = self.db.get_active_workers()
        if not db_workers:
            log.info("worker_manager.reconcile.no_active_workers")
            return

        try:
            instances    = await self._all_instances()
            live         = {i.instance_id: i for i in instances}
        except Exception as exc:
            log.error("worker_manager.reconcile.list_failed", error=str(exc))
            return

        log.info(
            "worker_manager.reconcile",
            db_workers=len(db_workers),
            live_instances=len(live),
        )

        for worker in db_workers:
            if not worker.instance_id:
                continue

            instance = live.get(worker.instance_id)
            if not instance:
                log.warning(
                    "worker_manager.reconcile.instance_gone",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    db_status=worker.status,
                )
                # Route through _fail_worker so the LB entry is deregistered,
                # an event is recorded, Discord is notified, and a new bid
                # campaign is started if no workers remain.
                await self._fail_worker(
                    worker,
                    reason="instance not found on provider during startup reconciliation",
                )
                continue

            log.info(
                "worker_manager.reconcile.instance",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
                actual_status=instance.actual_status,
                cur_state=instance.cur_state,
                status_msg=instance.status_msg,
                db_status=worker.status,
            )

            if instance.actual_status == "running":
                if instance.host:
                    host, port = instance.host, instance.port
                    if worker.status == WorkerStatus.RUNNING:
                        # Already verified before — just refresh host/port in case they changed
                        self.db.update_worker_status(
                            worker.worker_id,
                            WorkerStatus.RUNNING,
                            host=host,
                            port=port,
                        )
                        # Re-register in LB (idempotent — preserves accumulated stats)
                        await self.lb.register(worker.worker_id, worker.provider, host, port, worker.api_key, ssh_port=instance.ssh_port)
                    else:
                        # Instance is up but vLLM hasn't been health-checked yet
                        log.info(
                            "worker_manager.reconcile.vllm_health_check",
                            worker_id=worker.worker_id,
                            host=host,
                            port=port,
                        )
                        self.db.update_worker_status(
                            worker.worker_id,
                            WorkerStatus.STARTING,
                            host=host,
                            port=port,
                        )
                        self._spawn_vllm_health_task(
                            worker_id=worker.worker_id,
                            host=host,
                            port=port,
                            api_key=worker.api_key,
                            provider_name=worker.provider,
                            ssh_port=instance.ssh_port,
                        )
            elif instance.is_terminal:
                # Route through _fail_worker so outbid instances increment
                # _preemption_count, send a Discord alert, and raise the bid
                # floor for the next campaign.
                reason = (
                    f"outbid: {instance.status_msg}"
                    if instance.is_outbid
                    else f"provider reports {instance.actual_status!r} on startup: {instance.status_msg}"
                )
                await self._fail_worker(worker, reason=reason)
            elif worker.status in (WorkerStatus.PENDING, WorkerStatus.STARTING):
                # Instance is still coming up — resume the wait task
                log.info(
                    "worker_manager.reconcile.resuming_wait",
                    worker_id=worker.worker_id,
                    actual_status=instance.actual_status,
                )
                asyncio.create_task(
                    self._wait_for_running(worker),
                    name=f"wait-running-{worker.worker_id}",
                )

    # ── Startup zombie cleanup ────────────────────────────────────────────

    async def _destroy_zombie_instances(self) -> None:
        """
        Destroy any provider instances we own that have no DynamoDB record.

        _reconcile_state() runs first, so any instance the DB already knows
        about is handled.  Anything left with our label/image is a zombie from
        a previous crashed or restarted orchestrator — it's burning money with
        no one watching it.
        """
        log.info("worker_manager.destroy_zombies.start")
        try:
            all_instances = await self._all_instances()
        except Exception as exc:
            log.error("worker_manager.destroy_zombies.list_failed", error=str(exc))
            return

        known_ids   = self.db.get_known_instance_ids()
        ghcr_prefix = "ghcr.io/easedai/"
        zombies: list[InstanceInfo] = []

        for instance in all_instances:
            if not instance.instance_id or instance.instance_id in known_ids:
                continue
            if not (instance.label.startswith("eased-") or ghcr_prefix in instance.image):
                continue
            if instance.is_terminal:
                continue  # already dead

            zombies.append(instance)
            log.warning(
                "worker_manager.destroy_zombies.found",
                instance_id=instance.instance_id,
                label=instance.label,
                actual_status=instance.actual_status,
                gpu=instance.gpu_name,
            )

        if not zombies:
            log.info("worker_manager.destroy_zombies.none_found")
            return

        lines = [f"**Startup cleanup — {len(zombies)} zombie instance(s) destroyed**"]
        for z in zombies:
            lines.append(
                f"  • `{z.instance_id}` — `{z.gpu_name or '?'}` "
                f"status `{z.actual_status}` "
                f"label `{z.label or '?'}`"
            )
        await self.discord.send("\n".join(lines), "warning")

        for zombie in zombies:
            prov = self._providers.get(zombie.provider)
            if not prov:
                log.warning("worker_manager.destroy_zombies.unknown_provider",
                            instance_id=zombie.instance_id, provider=zombie.provider)
                continue
            try:
                await prov.destroy_instance(zombie.instance_id)
                log.info("worker_manager.destroy_zombies.destroyed", instance_id=zombie.instance_id)
            except Exception as exc:
                log.error(
                    "worker_manager.destroy_zombies.destroy_failed",
                    instance_id=zombie.instance_id,
                    error=repr(exc),
                )

        log.info("worker_manager.destroy_zombies.done", destroyed=len(zombies))

    # ── Startup audit ─────────────────────────────────────────────────────

    async def _post_startup_audit(self) -> None:
        """Post a one-time Discord summary of DB + provider state after startup reconciliation."""
        try:
            db_workers    = self.db.list_workers()
            all_instances = await self._all_instances()
        except Exception as exc:
            log.warning("worker_manager.startup_audit.failed", error=str(exc))
            return

        by_status: dict[str, int] = {}
        for w in db_workers:
            by_status[w.status.value] = by_status.get(w.status.value, 0) + 1

        ghcr_prefix = "ghcr.io/easedai/"
        our_instances = [
            i for i in all_instances
            if i.label.startswith("eased-") or ghcr_prefix in i.image
        ]

        log.info(
            "worker_manager.startup_audit",
            db_workers=len(db_workers),
            our_instances=len(our_instances),
            by_status=by_status,
        )

        status_parts = ", ".join(f"{s}: {n}" for s, n in sorted(by_status.items())) or "none"
        lines = [
            "**Orchestrator started**",
            f"DB workers: **{len(db_workers)}** ({status_parts})",
            f"Provider instances (ours): **{len(our_instances)}**",
        ]
        for inst in our_instances:
            lines.append(
                f"  • `{inst.instance_id}` — `{inst.gpu_name or '?'}` "
                f"`{inst.actual_status}` label `{inst.label or '?'}`"
            )
        await self.discord.send("\n".join(lines), "info")

    # ── Ensure at least one worker is active ──────────────────────────────

    async def _ensure_worker(self) -> None:
        active = self.db.get_active_workers()
        if active:
            log.info(
                "worker_manager.ensure.already_active",
                count=len(active),
                statuses=[w.status for w in active],
            )
            return
        log.info("worker_manager.ensure.no_active_workers — starting bid campaign")
        if not self._bidding_task or self._bidding_task.done():
            self._bidding_task = asyncio.create_task(
                self._bidding_campaign(), name="bid-campaign"
            )

    # ── Agentic bidding campaign ──────────────────────────────────────────

    async def _bidding_campaign(self, override: dict | None = None) -> None:
        """
        Bid for a cheap interruptible GPU instance.

        Strategy:
          • Start at bid_start_pct  (default 50 %) of median market price
          • Every bid_retry_interval_sec (default 5 min) increase by bid_step_pct (5 %)
          • Give up and fall back to on-demand once bid_max_multiplier (110 %) is exceeded
        """
        _override_provider = (override or {}).get("provider")
        _override_image    = (override or {}).get("image")
        log.info(
            "worker_manager.bid_campaign.start",
            preemption_count=self._preemption_count,
            override_provider=_override_provider,
            override_image=_override_image,
        )
        await self._enforce_debug_cap()
        preemption_note = (
            f" ({self._preemption_count}× preempted this session — "
            f"starting bid raised by {self._preemption_count} step(s))"
            if self._preemption_count
            else ""
        )
        await self.discord.send(
            f"**Bid campaign started** — searching for a cheap GPU worker.{preemption_note}",
            "info",
        )

        try:
            offers = await self._search_all_offers(on_demand=False)
        except Exception as exc:
            log.error("worker_manager.bid_campaign.search_failed", error=str(exc))
            await self.discord.send(f"Offer search failed: `{exc}`", "error")
            return

        if not offers:
            log.error("worker_manager.bid_campaign.no_offers")
            await self.discord.send(
                "No interruptible GPU offers found. Falling back to on-demand.",
                "warning",
            )
            await self._launch_on_demand()
            return

        market_price   = self._market_price(offers)
        worker_api_key = secrets.token_urlsafe(32)
        attempt        = 0

        # Each preemption raises the floor by one bid_step_pct.  This ensures
        # that after being outbid N times the campaign automatically starts N
        # steps higher than the configured minimum, converging on the real
        # market price without manual tuning.
        preemption_floor = settings.bid_start_pct + self._preemption_count * settings.bid_step_pct

        # If a previous campaign won, probe one step below that price first.
        # Floor at preemption_floor so repeated outbids aren't forgotten.
        if self._last_winning_multiplier is not None:
            start_multiplier = max(
                preemption_floor,
                self._last_winning_multiplier - settings.bid_step_pct,
            )
            log.info(
                "worker_manager.bid_campaign.probing_lower",
                last_win=f"{self._last_winning_multiplier:.0%}",
                preemption_floor=f"{preemption_floor:.0%}",
                start_multiplier=f"{start_multiplier:.0%}",
                preemption_count=self._preemption_count,
            )
        else:
            start_multiplier = preemption_floor

        while True:
            multiplier = start_multiplier + (attempt * settings.bid_step_pct)

            if multiplier > settings.bid_max_multiplier:
                log.warning(
                    "worker_manager.bid_campaign.cap_reached",
                    multiplier=f"{multiplier:.0%}",
                    market_price=market_price,
                    attempts=attempt,
                )
                await self.discord.send(
                    f"Bid cap reached after **{attempt}** attempts "
                    f"(>{settings.bid_max_multiplier:.0%} of market). "
                    "Falling back to on-demand.",
                    "warning",
                )
                await self._launch_on_demand()
                return

            bid_price = round(market_price * multiplier, 6)

            # All offers our bid can afford, cheapest first (provider filter from admin UI)
            matching_offers = [
                o for o in offers
                if o.price_per_hr <= bid_price
                and (_override_provider is None or o.provider == _override_provider)
            ]
            if not matching_offers:
                cheapest = offers[0].price_per_hr if offers else None
                log.info(
                    "worker_manager.bid_campaign.no_match",
                    attempt=attempt + 1,
                    bid_price=bid_price,
                    cheapest_available=cheapest,
                    next_retry_sec=settings.bid_retry_interval_sec,
                )
                await self.discord.send(
                    f"Bid attempt **{attempt + 1}**: **${bid_price:.4f}/hr** "
                    f"({multiplier:.0%} of market) — no match yet "
                    f"(cheapest available: ${cheapest:.4f}/hr). "
                    f"Retrying in {settings.bid_retry_interval_sec // 60} min.",
                    "info",
                )
                await asyncio.sleep(_jitter(settings.bid_retry_interval_sec))
                attempt += 1
                # Refresh offers after the wait — prices shift between retries
                try:
                    offers       = await self._search_all_offers(on_demand=False)
                    market_price = self._market_price(offers)
                except Exception as exc:
                    log.warning("worker_manager.bid_campaign.refresh_failed", error=str(exc))
                continue

            # Try each affordable offer in price order — skip unavailable ones
            # immediately rather than waiting the full retry interval per failure.
            worker_id     = secrets.token_urlsafe(8)
            label         = f"eased-{worker_id}"
            best_offer    = None
            instance_id   = ""
            for candidate in matching_offers:
                log.info(
                    "worker_manager.bid_campaign.placing",
                    attempt=attempt + 1,
                    offer_id=candidate.offer_id,
                    bid_price=bid_price,
                    gpu=candidate.gpu_name,
                    gpu_ram_gb=candidate.gpu_ram_gb,
                    label=label,
                )
                try:
                    config = CreateConfig(
                        worker_api_key=worker_api_key,
                        on_demand=False,
                        label=label,
                        price=bid_price,
                        ssh_public_key=self._ssh_public_key,
                        image_override=_override_image,
                    )
                    instance_id = await self._providers[candidate.provider].create_instance(candidate, config)
                    best_offer  = candidate
                    break
                except Exception as exc:
                    log.warning(
                        "worker_manager.bid_campaign.offer_unavailable",
                        attempt=attempt + 1,
                        offer_id=candidate.offer_id,
                        gpu=candidate.gpu_name,
                        error=str(exc),
                    )

            if not best_offer:
                log.error(
                    "worker_manager.bid_campaign.all_offers_failed",
                    attempt=attempt + 1,
                    offers_tried=len(matching_offers),
                )
                await asyncio.sleep(_jitter(settings.bid_retry_interval_sec))
                attempt += 1
                continue

            # Attach the SSH key via vast.ai's per-instance endpoint as a
            # second, reliable injection channel.  EXTRA_COMMANDS also injects
            # the key into authorized_keys at launch, but vast.ai has been
            # observed to drop EXTRA_COMMANDS on some hosts — this call closes
            # that gap.  Best-effort: a failure here is not fatal since the
            # launch-time injection may still have succeeded.
            await self._attach_ssh_key_best_effort(instance_id, best_offer.provider)

            now = datetime.now(timezone.utc)
            worker = Worker(
                worker_id=worker_id,
                instance_id=instance_id,
                label=label,
                status=WorkerStatus.PENDING,
                worker_type=WorkerType.INTERRUPTIBLE,
                provider=best_offer.provider,
                api_key=worker_api_key,
                gpu_name=best_offer.gpu_name,
                gpu_ram_gb=best_offer.gpu_ram_gb,
                num_gpus=best_offer.num_gpus,
                bid_price=bid_price,
                market_price=market_price,
                bid_attempts=attempt + 1,
                specs=best_offer.specs,
                image_pull_started_at=now,
                created_at=now,
                updated_at=now,
            )
            self.db.save_worker(worker)
            self._ev(
                worker, "worker.created",
                f"Bid accepted — {best_offer.gpu_name} "
                f"at ${bid_price:.4f}/hr ({multiplier:.0%} of market), instance {instance_id}",
                meta={"bid_price": bid_price, "market_price": market_price,
                      "attempt": attempt + 1, "gpu": best_offer.gpu_name,
                      "offer_id": best_offer.offer_id},
            )

            self._last_winning_multiplier = multiplier
            log.info(
                "worker_manager.bid_campaign.bid_placed",
                worker_id=worker_id,
                instance_id=instance_id,
                bid_price=bid_price,
                multiplier=f"{multiplier:.0%}",
                gpu=best_offer.gpu_name,
            )
            gpu_desc = (
                f"{best_offer.num_gpus}×{best_offer.gpu_name} "
                f"({best_offer.total_gpu_ram_gb:.0f} GB total, TP={best_offer.num_gpus})"
                if best_offer.num_gpus > 1
                else f"{best_offer.gpu_name} ({best_offer.gpu_ram_gb} GB)"
            )
            await self.discord.send(
                f"**Bid accepted** — `{gpu_desc}` "
                f"at **${bid_price:.4f}/hr** ({multiplier:.0%} of market).\n"
                f"Instance `{instance_id}` is pending — monitoring until ready.",
                "success",
            )

            await self._wait_for_running(worker)
            return

    # ── On-demand fallback ────────────────────────────────────────────────

    async def _launch_on_demand(self, override: dict | None = None) -> None:
        _override_provider = (override or {}).get("provider")
        _override_image    = (override or {}).get("image")
        log.info("worker_manager.on_demand.start",
                 override_provider=_override_provider, override_image=_override_image)
        await self._enforce_debug_cap()
        await self.discord.send(
            "**Launching on-demand fallback** — more expensive, but no bidding delay.",
            "warning",
        )
        try:
            offers = await self._search_all_offers(on_demand=True)
        except Exception as exc:
            log.error("worker_manager.on_demand.search_failed", error=str(exc))
            await self.discord.send(f"On-demand search failed: `{exc}`", "error")
            return

        if _override_provider:
            offers = [o for o in offers if o.provider == _override_provider]
        if not offers:
            log.error("worker_manager.on_demand.no_offers")
            await self.discord.send("No on-demand GPU offers found!", "error")
            return

        best           = offers[0]
        worker_api_key = secrets.token_urlsafe(32)
        worker_id      = secrets.token_urlsafe(8)
        label          = f"eased-{worker_id}"

        log.info(
            "worker_manager.on_demand.creating",
            offer_id=best.offer_id,
            price=best.price_per_hr,
            gpu=best.gpu_name,
            label=label,
        )

        try:
            config = CreateConfig(
                worker_api_key=worker_api_key,
                on_demand=True,
                label=label,
                price=best.price_per_hr,
                ssh_public_key=self._ssh_public_key,
                image_override=_override_image,
            )
            instance_id = await self._providers[best.provider].create_instance(best, config)
        except Exception as exc:
            log.error("worker_manager.on_demand.create_failed", error=str(exc))
            await self.discord.send(f"On-demand launch failed: `{exc}`", "error")
            return

        # Second injection channel — see note in _bid_campaign.
        await self._attach_ssh_key_best_effort(instance_id, best.provider)

        now = datetime.now(timezone.utc)
        worker = Worker(
            worker_id=worker_id,
            instance_id=instance_id,
            label=label,
            status=WorkerStatus.PENDING,
            worker_type=WorkerType.ON_DEMAND,
            provider=best.provider,
            api_key=worker_api_key,
            gpu_name=best.gpu_name,
            gpu_ram_gb=best.gpu_ram_gb,
            num_gpus=best.num_gpus,
            bid_price=best.price_per_hr,
            market_price=best.price_per_hr,
            specs=best.specs,
            image_pull_started_at=now,
            created_at=now,
            updated_at=now,
        )
        self.db.save_worker(worker)
        self._ev(
            worker, "worker.created",
            f"On-demand instance created — {best.gpu_name} "
            f"at ${best.price_per_hr:.4f}/hr, instance {instance_id}",
            meta={"price": best.price_per_hr, "gpu": best.gpu_name,
                  "offer_id": best.offer_id},
        )

        gpu_desc = (
            f"{best.num_gpus}×{best.gpu_name} "
            f"({best.total_gpu_ram_gb:.0f} GB total, TP={best.num_gpus})"
            if best.num_gpus > 1
            else f"{best.gpu_name} ({best.gpu_ram_gb} GB)"
        )
        await self.discord.send(
            f"**On-demand instance created** — `{gpu_desc}` "
            f"at **${best.price_per_hr:.4f}/hr**. Instance `{instance_id}`.",
            "warning",
        )
        await self._wait_for_running(worker)

    # ── Worker readiness waiting ──────────────────────────────────────────

    async def _wait_for_running(self, worker: Worker) -> None:
        """
        Poll the provider every 15 s until the instance reaches `running` status,
        then hand off to _wait_for_vllm_health.

        Logs cur_state / status_msg on every tick so failures are visible in logs.
        On terminal state or timeout, fetches container logs and fires _fail_worker.
        """
        log.info(
            "worker_manager.wait_for_running.start",
            worker_id=worker.worker_id,
            instance_id=worker.instance_id,
            timeout_sec=settings.instance_running_timeout_sec,
        )
        loop     = asyncio.get_event_loop()
        deadline = loop.time() + settings.instance_running_timeout_sec
        consecutive_errors = 0
        miss_count = 0
        # Tracks whether we have ever seen this instance in the provider API.
        # Before the first confirmation we do NOT count misses as failures —
        # the API can take 30-60 s to propagate a freshly created instance.
        instance_ever_seen = False

        while loop.time() < deadline:
            await asyncio.sleep(_jitter(15))
            try:
                instance = await self._provider_for(worker).get_instance(worker.instance_id)
            except Exception as exc:
                log.warning(
                    "worker_manager.wait_for_running.poll_error",
                    worker_id=worker.worker_id,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                continue

            if not instance:
                if not instance_ever_seen:
                    # API propagation lag — instance not yet visible.  Just keep
                    # waiting; the deadline will catch genuinely stuck launches.
                    log.debug(
                        "worker_manager.wait_for_running.not_yet_visible",
                        worker_id=worker.worker_id,
                        instance_id=worker.instance_id,
                    )
                    continue

                # Instance was previously confirmed alive but is now missing.
                # Count consecutive misses before giving up (transient API blip).
                miss_count += 1
                log.warning(
                    "worker_manager.wait_for_running.instance_not_found",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    consecutive_misses=miss_count,
                    threshold=_MISS_THRESHOLD,
                )
                if miss_count < _MISS_THRESHOLD:
                    continue
                log.error(
                    "worker_manager.wait_for_running.instance_gone",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="instance_gone")
                await self._fail_worker(
                    worker,
                    reason="instance disappeared from provider",
                    logs=logs,
                )
                return

            instance_ever_seen = True
            miss_count = 0  # confirmed alive — reset

            elapsed = round(loop.time() - (deadline - settings.instance_running_timeout_sec))

            log.info(
                "worker_manager.wait_for_running.poll",
                worker_id=worker.worker_id,
                actual_status=instance.actual_status,
                cur_state=instance.cur_state,
                status_msg=instance.status_msg,
                next_state=instance.next_state,
                elapsed_sec=elapsed,
            )

            # Detect repeated Docker pull / image errors in status_msg — fail fast
            # rather than waiting for the full startup timeout.
            if instance.status_msg.lower().startswith("error"):
                consecutive_errors += 1
                log.warning(
                    "worker_manager.wait_for_running.status_msg_error",
                    worker_id=worker.worker_id,
                    status_msg=instance.status_msg,
                    consecutive_errors=consecutive_errors,
                )
                if consecutive_errors >= 3:
                    logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="status_msg_error")
                    await self._fail_worker(
                        worker,
                        reason=f"repeated error in provider status: {instance.status_msg}",
                        logs=logs,
                    )
                    return
            else:
                consecutive_errors = 0

            if instance.actual_status == "running":
                if instance.host:
                    host, port = instance.host, instance.port

                    # ── Image pull duration ───────────────────────────────
                    pull_duration: Optional[float] = None
                    if worker.image_pull_started_at is not None:
                        pull_duration = round(
                            (datetime.now(timezone.utc) - worker.image_pull_started_at).total_seconds(),
                            1,
                        )
                        log.info(
                            "worker_manager.wait_for_running.image_pull_complete",
                            worker_id=worker.worker_id,
                            instance_id=worker.instance_id,
                            image_pull_duration_sec=pull_duration,
                            gpu=worker.gpu_name,
                        )

                    self.db.update_worker_status(
                        worker.worker_id,
                        WorkerStatus.STARTING,
                        host=host,
                        port=port,
                        **({"image_pull_duration_sec": pull_duration} if pull_duration is not None else {}),
                    )
                    self.events.record(
                        worker_id=worker.worker_id,
                        event_type="status.changed",
                        status=WorkerStatus.STARTING.value,
                        prev_status=worker.status.value,
                        message=f"Provider instance running — waiting for vLLM health at {host}:{port}",
                        instance_id=worker.instance_id,
                        label=worker.label,
                        meta={"host": host, "port": port,
                              "cur_state": instance.cur_state, "elapsed_sec": elapsed,
                              "image_pull_duration_sec": pull_duration},
                    )
                    pull_note = (
                        f" Container started in **{pull_duration:.0f}s**."
                        if pull_duration is not None
                        else ""
                    )
                    await self.discord.send(
                        f"**Instance up** `{worker.instance_id}` — "
                        f"container running at `{host}:{port}`."
                        f"{pull_note} "
                        f"Waiting for vLLM to finish loading (up to "
                        f"{settings.worker_startup_timeout_sec // 60} min).",
                        "info",
                    )
                    # vast.ai silently drops our Docker -e env dict and
                    # EXTRA_COMMANDS on many hosts, so we can't rely on the API
                    # to inject VLLM_API_KEY / TENSOR_PARALLEL_SIZE / etc.
                    # Instead, SSH in once the container is up, write
                    # /etc/vllm-env.sh, and restart vllm so it picks up our
                    # config.  If SSH isn't ready yet, the health-check loop
                    # will retry the inject on each periodic SSH check interval.
                    inject_ok = await self._ssh_inject_env_and_restart_vllm(instance, worker)
                    self._spawn_vllm_health_task(
                        worker_id=worker.worker_id,
                        host=host,
                        port=port,
                        api_key=worker.api_key,
                        provider_name=worker.provider,
                        ssh_port=instance.ssh_port,
                        ssh_inject_done=inject_ok,
                    )
                    return
                else:
                    # Container is running but port isn't mapped yet — log it
                    # so it's visible; it usually resolves on the next poll.
                    log.warning(
                        "worker_manager.wait_for_running.no_address",
                        worker_id=worker.worker_id,
                        instance_id=worker.instance_id,
                        elapsed_sec=elapsed,
                    )

            elif instance.is_terminal:
                log.error(
                    "worker_manager.wait_for_running.terminal_state",
                    worker_id=worker.worker_id,
                    actual_status=instance.actual_status,
                    cur_state=instance.cur_state,
                    status_msg=instance.status_msg,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="terminal_state")
                await self._fail_worker(
                    worker,
                    reason=f"provider status=`{instance.actual_status}` cur_state=`{instance.cur_state}` — {instance.status_msg}",
                    logs=logs,
                )
                return

        # Deadline exceeded
        log.error(
            "worker_manager.wait_for_running.timeout",
            worker_id=worker.worker_id,
            timeout_sec=settings.instance_running_timeout_sec,
        )
        logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="startup_timeout")
        await self._fail_worker(
            worker,
            reason=f"timed out after {settings.instance_running_timeout_sec // 60} min waiting for worker to come up",
            logs=logs,
        )

    def _spawn_vllm_health_task(
        self,
        worker_id: str,
        host: str,
        port: int,
        api_key: str,
        provider_name: str = "vastai",
        ssh_port: Optional[int] = None,
        ssh_inject_done: bool = False,
    ) -> None:
        """
        Start a _wait_for_vllm_health task for *worker_id* if one is not already
        running.  Stores the task in self._health_check_tasks and registers a
        done-callback that removes it so the slot is freed when the task exits.

        *ssh_inject_done* should be True when the caller already successfully
        injected /etc/vllm-env.sh (via _ssh_inject_env_and_restart_vllm) before
        spawning this task.  When False, the health loop will retry the injection
        periodically until it succeeds — this covers the case where SSH was not yet
        available when the instance first came up, as well as orchestrator restarts
        where the inject was never attempted.
        """
        existing = self._health_check_tasks.get(worker_id)
        if existing and not existing.done():
            log.debug(
                "worker_manager.health_task.already_running",
                worker_id=worker_id,
            )
            return
        task = asyncio.create_task(
            self._wait_for_vllm_health(
                worker_id=worker_id,
                host=host,
                port=port,
                api_key=api_key,
                provider_name=provider_name,
                ssh_port=ssh_port,
                ssh_inject_done=ssh_inject_done,
            ),
            name=f"vllm-health-{worker_id}",
        )
        self._health_check_tasks[worker_id] = task
        task.add_done_callback(
            lambda t: self._health_check_tasks.pop(worker_id, None)
        )
        log.info(
            "worker_manager.health_task.spawned",
            worker_id=worker_id,
            host=host,
            port=port,
        )

    async def _wait_for_vllm_health(
        self,
        worker_id: str,
        host: str,
        port: int,
        api_key: str,
        provider_name: str = "vastai",
        ssh_port: Optional[int] = None,
        ssh_inject_done: bool = False,
    ) -> None:
        """
        Poll the worker's /health endpoint until vLLM responds 200.

        The provider is treated as ground truth: on every iteration we also check
        the instance status.  If it is gone or in a terminal state we fail
        immediately instead of waiting out the full startup timeout.

        If *ssh_inject_done* is False, the loop will attempt to SSH-inject
        /etc/vllm-env.sh (including TENSOR_PARALLEL_SIZE) on each periodic SSH
        check interval until the inject succeeds.  This provides fault-tolerant
        multi-GPU configuration: even if SSH wasn't ready when the instance first
        came up, or the orchestrator restarted after the container was already
        running, vLLM will eventually receive the correct env and be restarted.
        """
        url = f"https://{host}/health" if port == 443 else f"http://{host}:{port}/health"
        loop = asyncio.get_event_loop()
        deadline = loop.time() + settings.worker_startup_timeout_sec
        _ssh_inject_done = ssh_inject_done  # mutable local copy

        log.info(
            "worker_manager.wait_for_vllm_health.start",
            worker_id=worker_id,
            url=url,
            timeout_sec=settings.worker_startup_timeout_sec,
        )

        vast_miss_count = 0
        last_log_check  = loop.time()  # SSH log check cadence

        while loop.time() < deadline:
            # ── 1. Check provider first — ground truth ────────────────────
            worker = self.db.get_worker(worker_id)
            if worker and worker.instance_id:
                try:
                    prov = self._providers.get(provider_name) or self._providers.get(worker.provider)
                    instance = await prov.get_instance(worker.instance_id) if prov else None
                    gone = not instance or instance.is_terminal

                    if gone:
                        vast_miss_count += 1
                        log.warning(
                            "worker_manager.wait_for_vllm_health.instance_gone",
                            worker_id=worker_id,
                            instance_id=worker.instance_id,
                            actual_status=instance.actual_status if instance else "missing",
                            consecutive_misses=vast_miss_count,
                            threshold=_MISS_THRESHOLD,
                        )
                        if vast_miss_count >= _MISS_THRESHOLD:
                            reason = (
                                "provider instance gone (no record)"
                                if not instance
                                else f"provider status={instance.actual_status!r}"
                            )
                            logs = await self._fetch_worker_logs(
                                worker.instance_id, worker=worker, trigger="instance_gone_during_health"
                            )
                            await self._fail_worker(worker, reason=reason, logs=logs)
                            return
                    else:
                        vast_miss_count = 0  # instance confirmed alive — reset
                except Exception as exc:
                    log.debug(
                        "worker_manager.wait_for_vllm_health.provider_check_failed",
                        worker_id=worker_id,
                        error=str(exc),
                    )

            # ── 2. Periodic SSH log check + inject retry ──────────────────
            if (
                self._ssh_key is not None
                and ssh_port is not None
                and (loop.time() - last_log_check) >= _STARTUP_LOG_CHECK_INTERVAL
            ):
                last_log_check = loop.time()
                worker = self.db.get_worker(worker_id)
                if worker and worker.instance_id:
                    try:
                        prov2 = self._providers.get(provider_name) or self._providers.get(worker.provider)
                        instance = await prov2.get_instance(worker.instance_id) if prov2 else None
                        if instance:
                            # Retry SSH env inject until it succeeds.  This is the
                            # fault-tolerant multi-GPU path: if SSH wasn't ready when
                            # the container first came up, or the orchestrator
                            # restarted after the instance was already running, we
                            # re-inject /etc/vllm-env.sh (with TENSOR_PARALLEL_SIZE)
                            # and kick supervisord so vLLM starts with the right config.
                            if not _ssh_inject_done:
                                ok = await self._ssh_inject_env_and_restart_vllm(instance, worker)
                                if ok:
                                    _ssh_inject_done = True
                                    log.info(
                                        "worker_manager.wait_for_vllm_health.ssh_inject_retry_ok",
                                        worker_id=worker_id,
                                        num_gpus=worker.num_gpus,
                                    )
                            log_text = await self._fetch_vllm_logs_ssh(instance, worker)
                            if log_text:
                                diagnosis = _diagnose_vllm_logs(log_text)
                                if diagnosis:
                                    log.warning(
                                        "worker_manager.wait_for_vllm_health.fatal_log_detected",
                                        worker_id=worker_id,
                                        diagnosis=diagnosis,
                                    )
                                    await self._fail_worker(
                                        worker,
                                        reason=diagnosis,
                                        logs=log_text,
                                    )
                                    return
                    except Exception as exc:
                        log.debug(
                            "worker_manager.wait_for_vllm_health.log_check_failed",
                            worker_id=worker_id,
                            error=repr(exc),
                        )

            # ── 3. Ping vLLM health endpoint ──────────────────────────────
            try:
                async with httpx.AsyncClient(timeout=settings.health_check_timeout_sec) as c:
                    r = await c.get(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if r.status_code != 200:
                        log.debug(
                            "worker_manager.wait_for_vllm_health.not_ready",
                            worker_id=worker_id,
                            status=r.status_code,
                        )
                        await asyncio.sleep(_jitter(15))
                        continue
            except Exception as exc:
                log.debug(
                    "worker_manager.wait_for_vllm_health.ping_failed",
                    worker_id=worker_id,
                    error=str(exc),
                )
                # Trigger an SSH log check on the next iteration so we see
                # vLLM output immediately when the instance isn't responding.
                # Using min() means we only advance the check if it's overdue —
                # the check itself resets last_log_check so this won't spam.
                last_log_check = min(
                    last_log_check,
                    loop.time() - _STARTUP_LOG_CHECK_INTERVAL,
                )
                await asyncio.sleep(_jitter(15))
                continue

            # ── 4. Smoke-test: send a real inference request ──────────────
            # /health can return 200 while CUDA graphs are still being captured.
            # A successful chat completion confirms the model is truly serving.
            smoke_ok = await self._smoke_test(host, port, api_key, worker_id)
            if not smoke_ok:
                await asyncio.sleep(_jitter(15))
                continue

            # ── 5. Mark worker RUNNING and register in LB ─────────────────
            now = datetime.now(timezone.utc)
            self.db.update_worker_status(
                worker_id,
                WorkerStatus.RUNNING,
                running_since=now,
            )
            # Register in the load-balancer pool so it starts
            # receiving traffic immediately.
            await self.lb.register(worker_id, provider_name, host, port, api_key, ssh_port=ssh_port)
            # Calculate total time from bid accepted → vLLM ready
            fresh = self.db.get_worker(worker_id)
            elapsed_sec = (
                round((now - fresh.created_at).total_seconds())
                if fresh and fresh.created_at
                else None
            )
            self.events.record(
                worker_id=worker_id,
                event_type="health.ready",
                status=WorkerStatus.RUNNING.value,
                prev_status=WorkerStatus.STARTING.value,
                message=f"vLLM health check and smoke test passed — serving at {host}:{port}",
                meta={"host": host, "port": port, "startup_sec": elapsed_sec},
            )
            elapsed_str = (
                f" (ready in **{elapsed_sec // 60}m {elapsed_sec % 60}s**)"
                if elapsed_sec is not None
                else ""
            )
            log.info(
                "worker_manager.worker_ready",
                worker_id=worker_id,
                url=url,
                startup_sec=elapsed_sec,
            )
            await self.discord.send(
                f"**Worker ready** `{worker_id}` — "
                f"vLLM is serving at `{host}:{port}`{elapsed_str}.",
                "success",
            )
            return

        # Deadline exceeded — SSH logs are more detailed; try them first.
        worker = self.db.get_worker(worker_id)
        logs: str = ""
        if worker and worker.instance_id:
            try:
                prov3 = self._providers.get(provider_name) or self._providers.get(worker.provider)
                instance = await prov3.get_instance(worker.instance_id) if prov3 else None
                if instance:
                    logs = await self._fetch_vllm_logs_ssh(instance, worker) or ""
                    if logs:
                        log.info(
                            "worker_manager.wait_for_vllm_health.ssh_logs_ok",
                            worker_id=worker_id,
                            bytes=len(logs),
                        )
            except Exception as exc:
                log.debug(
                    "worker_manager.wait_for_vllm_health.ssh_log_fetch_failed",
                    worker_id=worker_id,
                    error=repr(exc),
                )

        if not logs:
            logs = await self._fetch_worker_logs(
                worker.instance_id if worker else None,
                worker=worker,
                trigger="vllm_timeout",
            )

        # Diagnose from logs; fall back to a generic timeout message.
        reason = _diagnose_vllm_logs(logs) if logs else None
        if not reason:
            reason = (
                f"vLLM health check timed out after "
                f"{settings.worker_startup_timeout_sec // 60} min — model may have failed to load"
            )
        await self._fail_worker(worker, reason=reason, logs=logs)

    # ── Smoke test ────────────────────────────────────────────────────────

    async def _smoke_test(
        self,
        host: str,
        port: int,
        api_key: str,
        worker_id: str,
    ) -> bool:
        """
        Send a minimal chat-completion request to confirm the model is truly
        serving.  /health can return 200 while CUDA graphs are still being
        captured; a real inference call catches that gap.

        Returns True on a successful 2xx response, False on any failure.
        The caller should retry after a short sleep on False.
        """
        url = f"https://{host}/v1/chat/completions" if port == 443 else f"http://{host}:{port}/v1/chat/completions"
        payload = {
            "model":       settings.model_id,
            "messages":    [{"role": "user", "content": "Hello"}],
            "max_tokens":  5,
            "temperature": 1.0,
        }
        log.info("worker_manager.smoke_test.start", worker_id=worker_id, url=url)
        try:
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if r.status_code == 200:
                log.info(
                    "worker_manager.smoke_test.passed",
                    worker_id=worker_id,
                    status=r.status_code,
                )
                return True
            log.warning(
                "worker_manager.smoke_test.failed",
                worker_id=worker_id,
                status=r.status_code,
                body=r.text[:200],
            )
            return False
        except Exception as exc:
            log.warning(
                "worker_manager.smoke_test.error",
                worker_id=worker_id,
                error=str(exc),
            )
            return False

    # ── Provider cross-check loop ─────────────────────────────────────────

    async def _provider_monitor_loop(self) -> None:
        """
        Background task — every provider_check_interval_sec:
          • Reconcile DB workers against live provider instances (sync)
          • Every status_report_interval_sec post a summary to Discord
        """
        log.info(
            "worker_manager.provider_monitor.start",
            interval_sec=settings.provider_check_interval_sec,
        )
        ticks_per_report = max(1, settings.status_report_interval_sec // settings.provider_check_interval_sec)
        tick = 0
        while True:
            try:
                await asyncio.sleep(_jitter(settings.provider_check_interval_sec))
                await self._sync_with_provider()
                tick += 1
                if tick % ticks_per_report == 0:
                    await self._post_status_report()
            except asyncio.CancelledError:
                log.info("worker_manager.provider_monitor.cancelled")
                break
            except Exception as exc:
                log.error("worker_manager.provider_monitor.error", error=str(exc))

    async def _sync_with_provider(self) -> None:
        """
        Cross-check every active DB worker against the live provider instance list.

        Three responsibilities:
          1. Detect active workers whose instance is gone / terminal → fail them
          2. Detect PENDING/STARTING workers whose instance is now running → start health check
          3. Detect orphaned instances (our label, not in our DB) → destroy them
        """
        workers = self.db.get_active_workers()

        try:
            all_instances = await self._all_instances()
            live          = {i.instance_id: i for i in all_instances}
        except Exception as exc:
            log.warning("worker_manager.provider_sync.list_failed", error=str(exc))
            return

        log.debug(
            "worker_manager.provider_sync.tick",
            active_workers=len(workers),
            live_instances=len(live),
        )

        # ── 1 & 2: reconcile active DB workers against provider ───────────
        for worker in workers:
            if not worker.instance_id:
                continue

            instance = live.get(worker.instance_id)

            if not instance:
                miss = self._instance_miss_counts.get(worker.worker_id, 0) + 1
                self._instance_miss_counts[worker.worker_id] = miss
                log.warning(
                    "worker_manager.provider_sync.instance_missing",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    db_status=worker.status,
                    consecutive_misses=miss,
                    threshold=_MISS_THRESHOLD,
                )
                if miss < _MISS_THRESHOLD:
                    continue  # transient — wait for next sync tick
                # Confirmed gone after _MISS_THRESHOLD consecutive misses
                self._instance_miss_counts.pop(worker.worker_id, None)
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="instance_missing")
                await self._fail_worker(
                    worker,
                    reason=f"instance no longer exists on provider after {miss} consecutive checks (reclaimed or deleted)",
                    logs=logs,
                )
                continue

            # Instance found — reset miss counter
            self._instance_miss_counts.pop(worker.worker_id, None)

            log.debug(
                "worker_manager.provider_sync.instance",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
                actual_status=instance.actual_status,
                cur_state=instance.cur_state,
                status_msg=instance.status_msg,
                db_status=worker.status,
            )

            if instance.is_terminal:
                log.warning(
                    "worker_manager.provider_sync.terminal",
                    worker_id=worker.worker_id,
                    actual_status=instance.actual_status,
                    status_msg=instance.status_msg,
                    outbid=instance.is_outbid,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="terminal_state")
                reason = (
                    f"outbid: {instance.status_msg}"
                    if instance.is_outbid
                    else f"provider reports {instance.actual_status!r}: {instance.status_msg}"
                )
                await self._fail_worker(worker, reason=reason, logs=logs)

            elif (
                instance.cur_state in {"stopped", "exited", "failed"}
                and worker.status in (WorkerStatus.PENDING, WorkerStatus.STARTING)
            ):
                # Container stopped during startup (e.g. outbid while pulling image).
                # _wait_for_running won't catch this if get_instance can't see the
                # instance, so _sync_with_provider acts as a safety net.
                log.warning(
                    "worker_manager.provider_sync.stopped_during_startup",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    actual_status=instance.actual_status,
                    cur_state=instance.cur_state,
                    db_status=worker.status,
                    outbid=instance.is_outbid,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="stopped_during_startup")
                reason = (
                    f"outbid: {instance.status_msg}"
                    if instance.is_outbid
                    else f"container stopped during startup (cur_state={instance.cur_state!r}, actual={instance.actual_status!r}): {instance.status_msg}"
                )
                await self._fail_worker(worker, reason=reason, logs=logs)

            elif instance.actual_status != "running" and worker.status in (
                WorkerStatus.RUNNING, WorkerStatus.UNHEALTHY
            ):
                # Instance was healthy but provider no longer reports it as running
                # (e.g. reclaimed, restarting). Fail immediately — provider is ground truth.
                log.warning(
                    "worker_manager.provider_sync.no_longer_running",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    actual_status=instance.actual_status,
                    db_status=worker.status,
                    outbid=instance.is_outbid,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="no_longer_running")
                reason = (
                    f"outbid: {instance.status_msg}"
                    if instance.is_outbid
                    else f"provider reports {instance.actual_status!r} (was RUNNING in DB): {instance.status_msg}"
                )
                await self._fail_worker(worker, reason=reason, logs=logs)

            elif (
                instance.actual_status == "running"
                and worker.status in (WorkerStatus.PENDING, WorkerStatus.STARTING)
            ):
                # Instance came up while the orchestrator was restarting or the
                # _wait_for_running task didn't fire. Kick off vLLM health check.
                if instance.host:
                    host, port = instance.host, instance.port
                    log.info(
                        "worker_manager.provider_sync.recovered_running",
                        worker_id=worker.worker_id,
                        host=host,
                        port=port,
                    )
                    self.db.update_worker_status(
                        worker.worker_id,
                        WorkerStatus.STARTING,
                        host=host,
                        port=port,
                    )
                    self._spawn_vllm_health_task(
                        worker_id=worker.worker_id,
                        host=host,
                        port=port,
                        api_key=worker.api_key,
                        provider_name=worker.provider,
                        ssh_port=instance.ssh_port,
                    )

        # ── 3: destroy orphaned instances ─────────────────────────────────
        # An instance is "ours" if it has an eased-* label OR its image is from
        # our GHCR org (catches pre-label instances created by older orchestrator runs).
        # If it's ours but has no DB record, it's an orphan — destroy it.
        known_ids = self.db.get_known_instance_ids()
        ghcr_prefix = "ghcr.io/easedai/"
        for instance_id, instance in live.items():
            is_ours = instance.label.startswith("eased-") or ghcr_prefix in instance.image
            if not is_ours:
                continue
            if instance_id in known_ids:
                continue
            log.warning(
                "worker_manager.provider_sync.orphan",
                instance_id=instance_id,
                label=instance.label,
                image=instance.image,
                actual_status=instance.actual_status,
            )
            await self.discord.send(
                f"**Orphaned instance** `{instance_id}` "
                f"(label: `{instance.label}`, image: `{instance.image}`, status: `{instance.actual_status}`) "
                "has no DB record — destroying.",
                "warning",
            )
            # Record to event log using the label-derived worker_id if possible
            orphan_worker_id = (
                instance.label[6:]
                if instance.label.startswith("eased-") and len(instance.label) > 6
                else instance_id
            )
            self.events.record(
                worker_id=orphan_worker_id,
                event_type="orphan.destroyed",
                status="terminated",
                message=f"Orphaned instance {instance_id} (label: {instance.label!r}, status: {instance.actual_status!r}) — no DB record, destroying",
                instance_id=instance_id,
                label=instance.label or None,
                meta={"image": instance.image, "actual_status": instance.actual_status},
            )
            try:
                orphan_prov = self._providers.get(instance.provider)
                if orphan_prov:
                    await orphan_prov.destroy_instance(instance_id)
            except Exception as exc:
                log.error(
                    "worker_manager.provider_sync.orphan_destroy_failed",
                    instance_id=instance_id,
                    error=str(exc),
                )

    # ── HTTP health monitoring loop ───────────────────────────────────────

    async def _health_monitor_loop(self) -> None:
        """
        Background task — pings all RUNNING workers every health_check_interval_sec.
        Marks workers lost after health_check_fail_threshold consecutive failures
        and triggers a new bid campaign if no workers remain.
        """
        log.info(
            "worker_manager.health_monitor.start",
            interval_sec=settings.health_check_interval_sec,
            fail_threshold=settings.health_check_fail_threshold,
        )
        while True:
            try:
                await asyncio.sleep(_jitter(settings.health_check_interval_sec))
                await self._check_all_workers()
                if self._queue is not None:
                    await self._queue.reclaim_orphaned()
            except asyncio.CancelledError:
                log.info("worker_manager.health_monitor.cancelled")
                break
            except Exception as exc:
                log.error("worker_manager.health_monitor.error", error=str(exc))

    async def _check_all_workers(self) -> None:
        # Check both RUNNING and UNHEALTHY workers — either can recover or need killing
        workers = self.db.get_running_workers() + [
            w for w in self.db.get_active_workers()
            if w.status == WorkerStatus.UNHEALTHY
        ]
        log.debug(
            "worker_manager.health_check.tick",
            workers=len(workers),
        )

        for worker in workers:
            if not worker.base_url:
                continue
            healthy = await self._ping_worker(worker)

            # Log the last 5 lines of vLLM output for each monitored instance
            if worker.instance_id:
                try:
                    instance = await self._provider_for(worker).get_instance(worker.instance_id)
                    if instance:
                        tail = await self._fetch_vllm_logs_ssh(instance, worker, lines=5)
                        if tail and tail.strip():
                            log.debug(
                                "worker_manager.health_check.vllm_log_tail",
                                worker_id=worker.worker_id,
                                instance_id=worker.instance_id,
                                log_tail=tail.strip(),
                            )
                except Exception as exc:
                    log.debug(
                        "worker_manager.health_check.vllm_log_tail_failed",
                        worker_id=worker.worker_id,
                        error=str(exc),
                    )

            if healthy:
                if worker.status != WorkerStatus.RUNNING or worker.consecutive_failures > 0:
                    log.info(
                        "worker_manager.health_check.recovered",
                        worker_id=worker.worker_id,
                        prev_status=worker.status,
                        prev_failures=worker.consecutive_failures,
                    )
                    extra: dict = {"consecutive_failures": 0}
                    if worker.running_since is None:
                        extra["running_since"] = datetime.now(timezone.utc)
                    self.db.update_worker_status(
                        worker.worker_id,
                        WorkerStatus.RUNNING,
                        **extra,
                    )
                    # Re-register with LB now that the worker is healthy again.
                    # (provider_sync will also do this on its next cycle, but
                    # registering here closes the gap.)
                    if worker.host and worker.port:
                        await self.lb.register(
                            worker.worker_id, worker.provider,
                            worker.host, worker.port, worker.api_key,
                        )
                    self._ev(
                        worker, "health.recovered",
                        f"Health check passed — recovered from {worker.status.value} "
                        f"(had {worker.consecutive_failures} failure(s))",
                        prev_status=worker.status.value,
                    )
            else:
                new_failures = worker.consecutive_failures + 1
                log.warning(
                    "worker_manager.health_check.failure",
                    worker_id=worker.worker_id,
                    status=worker.status,
                    consecutive_failures=new_failures,
                    threshold=settings.health_check_fail_threshold,
                )
                self._ev(
                    worker, "health.fail",
                    f"Health check failed ({new_failures}/{settings.health_check_fail_threshold})",
                    meta={"consecutive_failures": new_failures,
                          "threshold": settings.health_check_fail_threshold},
                )
                if new_failures >= settings.health_check_fail_threshold:
                    # Threshold hit — fetch logs and kill
                    logs = await self._fetch_worker_logs(worker.instance_id, worker=worker)
                    await self._fail_worker(
                        worker,
                        reason=f"{new_failures} consecutive HTTP health check failures",
                        logs=logs,
                    )
                else:
                    # Deregister from LB immediately on first failure so requests
                    # stop being routed to this worker.  _fail_worker will also
                    # call deregister when the threshold is reached; it is
                    # idempotent so the double-call is harmless.
                    await self.lb.deregister(worker.worker_id)
                    self.db.update_worker_status(
                        worker.worker_id,
                        WorkerStatus.UNHEALTHY,
                        consecutive_failures=new_failures,
                    )

        await self._maybe_scale()

    # ── Auto-scaling ──────────────────────────────────────────────────────────

    async def _maybe_scale(self) -> None:
        """
        Evaluate current queue utilization and trigger scale-up or scale-down
        if sustained thresholds are met.

        Called at the end of every health-check tick so scaling decisions ride
        the same cadence as health monitoring (health_check_interval_sec).
        """
        if self._queue is None:
            return

        # Honour manual scale requests from the admin UI (highest priority).
        signal_kind, signal_cfg = await self._queue.pop_control_signal()
        if signal_kind:
            log.info(
                "worker_manager.autoscale.manual_signal",
                kind=signal_kind,
                provider=signal_cfg.get("provider"),
                image=signal_cfg.get("image"),
            )
            if signal_kind == "on_demand":
                asyncio.create_task(
                    self._launch_on_demand(override=signal_cfg),
                    name="on-demand-manual",
                )
            else:
                if not self._bidding_task or self._bidding_task.done():
                    self._bidding_task = asyncio.create_task(
                        self._bidding_campaign(override=signal_cfg),
                        name="bid-campaign-manual",
                    )
            return

        available, leased, utilization = await self._queue.get_utilization()
        total = available + leased

        # Read and reset the 503 counter the LB increments on every no-worker response.
        pending_503s = await self._queue.pop_503_count()

        log.info(
            "worker_manager.autoscale.tick",
            available=available,
            leased=leased,
            total=total,
            utilization=round(utilization, 2),
            pending_503s=pending_503s,
        )

        # Maintain rolling window for scale-up (smooths over momentary spikes)
        self._utilization_history.append(utilization)
        if len(self._utilization_history) > settings.scale_up_consecutive_ticks:
            self._utilization_history.pop(0)

        if total > 0:
            await self._maybe_scale_up(available, leased, pending_503s)
            await self._maybe_scale_down()
        else:
            # Queue is empty — no utilization to compute.
            # Defensively call _ensure_worker so that if no bid is already
            # in progress (e.g. after a Redis flush or first-boot with a slow
            # vLLM start), a new campaign starts automatically.
            self._utilization_history.clear()
            self._scale_down_ticks = 0
            if pending_503s:
                log.warning(
                    "worker_manager.autoscale.no_workers_under_load",
                    pending_503s=pending_503s,
                    note="queue empty while requests are failing — ensuring a bid is in flight",
                )
            await self._ensure_worker()

    async def _maybe_scale_up(self, available: int, leased: int, pending_503s: int = 0) -> None:
        """
        Bid for an additional worker when utilization has been sustained above
        ``scale_up_threshold`` for ``scale_up_consecutive_ticks`` ticks,
        OR when requests are actively failing (503s) with all workers leased.

        Guards:
          - Cooldown period after last scale-up (cold starts take 10–15 min)
          - Already at max_instances
          - A bid campaign is already in progress
        """
        avg = (
            sum(self._utilization_history) / len(self._utilization_history)
            if self._utilization_history
            else 0.0
        )

        # Fast-path: all workers are leased AND requests are actively failing.
        # Don't wait for the rolling window to fill — the queue is already saturated.
        under_load = pending_503s > 0 and available == 0 and leased > 0
        window_full = len(self._utilization_history) >= settings.scale_up_consecutive_ticks

        if not under_load:
            if not window_full:
                return  # window not full yet, no immediate pressure
            if avg < settings.scale_up_threshold:
                return

        running_workers = self.db.get_running_workers()
        running_count   = len(running_workers)
        if running_count >= settings.max_instances:
            log.info(
                "worker_manager.autoscale.scale_up.at_max",
                running=running_count,
                max_instances=settings.max_instances,
                avg_utilization=round(avg, 2),
            )
            return

        now = datetime.now(timezone.utc)
        if self._last_scale_up_at is not None:
            elapsed = (now - self._last_scale_up_at).total_seconds()
            if elapsed < settings.scale_up_cooldown_sec:
                log.debug(
                    "worker_manager.autoscale.scale_up.cooldown",
                    cooldown_remaining_sec=round(settings.scale_up_cooldown_sec - elapsed),
                )
                return

        if self._bidding_task and not self._bidding_task.done():
            log.debug("worker_manager.autoscale.scale_up.bid_already_in_progress")
            return

        trigger = (
            f"{pending_503s} failing requests with all workers leased"
            if under_load
            else f"utilization {avg:.0%} over {settings.scale_up_consecutive_ticks} ticks"
        )
        log.warning(
            "worker_manager.autoscale.scale_up.triggered",
            trigger=trigger,
            avg_utilization=round(avg, 2),
            threshold=settings.scale_up_threshold,
            pending_503s=pending_503s,
            available=available,
            leased=leased,
            running=running_count,
            max_instances=settings.max_instances,
        )
        cooldown_min = settings.scale_up_cooldown_sec // 60
        await self.discord.send(
            title="Scaling Up",
            message=(
                f"Trigger: **{trigger}**\n"
                f"Launching a new bid campaign for worker "
                f"**{running_count + 1} / {settings.max_instances}**."
            ),
            level="warning",
            fields=[
                {
                    "name":   "Queue",
                    "value":  f"{available} available · {leased} leased",
                    "inline": True,
                },
                {
                    "name":   "Fleet",
                    "value":  f"{running_count} / {settings.max_instances} workers",
                    "inline": True,
                },
                {
                    "name":   "Cooldown after this",
                    "value":  f"{cooldown_min} min",
                    "inline": True,
                },
            ],
        )
        self._last_scale_up_at = now
        self._utilization_history.clear()
        self._bidding_task = asyncio.create_task(
            self._bidding_campaign(), name="bid-campaign-autoscale"
        )

    async def _maybe_scale_down(self) -> None:
        """
        Terminate the most idle worker when the fleet is over-provisioned.

        Only fires when:
          - running_count > min_instances
          - At least one worker has been idle > scale_down_idle_sec
          - This condition has persisted for scale_down_consecutive_ticks ticks
            (hysteresis prevents thrashing during bursty-but-light traffic)
        """
        running_workers = self.db.get_running_workers()
        if len(running_workers) <= settings.min_instances:
            self._scale_down_ticks = 0
            return

        idle_workers = await self._queue.get_idle_workers(settings.scale_down_idle_sec)
        if not idle_workers:
            self._scale_down_ticks = 0
            return

        self._scale_down_ticks += 1
        log.debug(
            "worker_manager.autoscale.scale_down.accumulating",
            idle_workers=[wid for wid, _ in idle_workers],
            ticks=self._scale_down_ticks,
            needed=settings.scale_down_consecutive_ticks,
        )
        if self._scale_down_ticks < settings.scale_down_consecutive_ticks:
            return

        # Find the most idle worker that is still in the RUNNING state
        idle_by_id = {wid: secs for wid, secs in idle_workers}
        candidates = [w for w in running_workers if w.worker_id in idle_by_id]
        if not candidates:
            self._scale_down_ticks = 0
            return

        candidates.sort(key=lambda w: idle_by_id[w.worker_id], reverse=True)
        target    = candidates[0]
        idle_secs = idle_by_id[target.worker_id]

        log.warning(
            "worker_manager.autoscale.scale_down.triggered",
            worker_id=target.worker_id,
            idle_sec=round(idle_secs),
            idle_min=round(idle_secs / 60, 1),
            running=len(running_workers),
            min_instances=settings.min_instances,
        )
        self._scale_down_ticks = 0
        await self._scale_down_worker(target, idle_secs)

    async def _scale_down_worker(self, worker: Worker, idle_secs: float) -> None:
        """
        Gracefully terminate an idle worker for scale-down.

        Unlike ``_fail_worker``, this path:
          - Does NOT attempt SSH log fetching (the worker is healthy, not broken)
          - Does NOT trigger a new bid campaign after termination (intentional)
          - Logs at INFO not ERROR (this is expected, not a failure)
        """
        idle_min = round(idle_secs / 60, 1)
        log.info(
            "worker_manager.scale_down_worker",
            worker_id=worker.worker_id,
            instance_id=worker.instance_id,
            idle_min=idle_min,
        )
        self.db.update_worker_status(
            worker.worker_id,
            WorkerStatus.TERMINATED,
            terminated_reason=f"idle scale-down ({idle_min}m idle)",
        )
        await self.lb.deregister(worker.worker_id)
        self._ev(
            worker, "worker.scale_down",
            f"Worker terminated by auto scale-down after {idle_min}m idle",
        )
        if worker.instance_id:
            try:
                await self._provider_for(worker).destroy_instance(worker.instance_id)
            except Exception as exc:
                log.warning(
                    "worker_manager.scale_down_worker.destroy_failed",
                    instance_id=worker.instance_id,
                    error=str(exc),
                )
            self.db.delete_worker(worker.worker_id)

        remaining = len(self.db.get_running_workers())
        await self.discord.send(
            title="Scaling Down",
            message=(
                f"Worker `{worker.worker_id}` was idle for **{idle_min} min** "
                f"(threshold: {settings.scale_down_idle_sec // 60} min) and has been terminated."
            ),
            level="info",
            fields=[
                {
                    "name":   "GPU",
                    "value":  worker.gpu_name or "unknown",
                    "inline": True,
                },
                {
                    "name":   "Instance",
                    "value":  f"`{worker.instance_id}`" if worker.instance_id else "n/a",
                    "inline": True,
                },
                {
                    "name":   "Fleet",
                    "value":  f"{remaining} / {settings.max_instances} workers remaining",
                    "inline": True,
                },
            ],
        )

    async def _ping_worker(self, worker: Worker) -> bool:
        url = f"{worker.base_url}/health"
        try:
            async with httpx.AsyncClient(
                timeout=settings.health_check_timeout_sec
            ) as c:
                r = await c.get(
                    url,
                    headers={"Authorization": f"Bearer {worker.api_key}"},
                )
                ok = r.status_code == 200
                log.debug(
                    "worker_manager.ping",
                    worker_id=worker.worker_id,
                    status=r.status_code,
                    ok=ok,
                )
                return ok
        except Exception as exc:
            log.debug(
                "worker_manager.ping.exception",
                worker_id=worker.worker_id,
                error=str(exc),
            )
            return False

    # ── Consolidated failure handler ──────────────────────────────────────

    async def _fail_worker(
        self,
        worker: Optional[Worker],
        reason: str,
        logs: str = "",
    ) -> None:
        """
        Mark a worker as TERMINATED, destroy its instance, and post a Discord alert
        with reason + log tail.

        If the worker was outbid, first attempts to raise the bid on the existing
        instance.  Only falls through to full termination if the instance is already
        gone or the bid cap is exceeded.
        """
        if worker is None:
            log.error("worker_manager.fail_worker.no_worker", reason=reason)
            return

        # ── Outbid: try raising the bid before giving up ──────────────────────
        # The provider may accept a higher bid on the same instance before it is
        # fully reclaimed, keeping the worker alive without a full restart.
        if (
            reason.startswith("outbid")
            and worker.instance_id
            and worker.bid_price is not None
            and worker.market_price is not None
        ):
            new_bid = round(
                worker.bid_price + worker.market_price * settings.bid_step_pct, 6
            )
            cap = round(worker.market_price * settings.bid_max_multiplier, 6)
            if new_bid <= cap:
                log.info(
                    "worker_manager.outbid.attempting_rebid",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    old_bid=worker.bid_price,
                    new_bid=new_bid,
                    cap=cap,
                )
                try:
                    accepted = await self._provider_for(worker).change_bid(worker.instance_id, new_bid)
                    if accepted:
                        self.db.update_worker_status(
                            worker.worker_id, worker.status, bid_price=new_bid
                        )
                        await self.discord.send(
                            f"Outbid — raised bid on `{worker.gpu_name}` "
                            f"from **${worker.bid_price:.4f}** to **${new_bid:.4f}/hr**. "
                            "Instance retained.",
                            "warning",
                        )
                        log.info(
                            "worker_manager.outbid.rebid_accepted",
                            worker_id=worker.worker_id,
                            new_bid=new_bid,
                        )
                        return  # Instance stays alive — do not terminate
                    # Instance already gone: fall through to normal fail path
                    log.info(
                        "worker_manager.outbid.rebid_declined_instance_gone",
                        worker_id=worker.worker_id,
                    )
                except Exception as exc:
                    log.warning(
                        "worker_manager.outbid.rebid_error",
                        worker_id=worker.worker_id,
                        error=str(exc),
                    )
            else:
                log.warning(
                    "worker_manager.outbid.rebid_cap_exceeded",
                    worker_id=worker.worker_id,
                    new_bid=new_bid,
                    cap=cap,
                )

        # Re-read from DB to guard against concurrent failure paths (e.g.
        # _wait_for_running and _sync_with_provider both detecting the same
        # missing instance and racing to call _fail_worker).
        current = self.db.get_worker(worker.worker_id)
        if current and current.status == WorkerStatus.TERMINATED:
            log.debug(
                "worker_manager.fail_worker.already_terminated",
                worker_id=worker.worker_id,
                reason=reason,
            )
            return

        log.error(
            "worker_manager.fail_worker",
            worker_id=worker.worker_id,
            instance_id=worker.instance_id,
            db_status=worker.status,
            reason=reason,
        )
        prev = worker.status.value
        self.db.update_worker_status(
            worker.worker_id,
            WorkerStatus.TERMINATED,
            terminated_reason=reason,
        )
        # Remove from LB pool immediately so no new requests are routed to it.
        await self.lb.deregister(worker.worker_id)
        self._ev(
            worker, "worker.terminated",
            f"Worker terminated: {reason}",
            prev_status=prev,
            meta={"reason": reason},
        )

        def _logs_useful(text: str) -> bool:
            return bool(text.strip()) and "(logs not yet available" not in text and "(log request failed" not in text

        # If logs weren't passed in (or came back as a placeholder), fetch them
        # now — while the instance still exists — so we have something useful.
        if not _logs_useful(logs) and worker.instance_id:
            logs = await self._fetch_worker_logs(
                worker.instance_id, worker=worker, trigger="pre_destroy"
            )

        if worker.instance_id:
            is_outbid = reason.startswith("outbid")
            if settings.keep_debug_instance and not is_outbid:
                # Leave the instance alive so it can be SSH'd into for debugging.
                # _enforce_debug_cap() will evict the newest debug instance
                # before the next bid campaign to prevent unbounded accumulation.
                # Outbid instances are never kept: the provider already reclaimed the
                # hardware, so there is nothing to SSH into.
                # DB record is removed now — the instance_id is still logged in the
                # event store if you need to find it later.
                log.info(
                    "worker_manager.fail_worker.keeping_for_debug",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                )
                self.db.delete_worker(worker.worker_id)
            else:
                if is_outbid:
                    log.info(
                        "worker_manager.fail_worker.outbid_cleanup",
                        worker_id=worker.worker_id,
                        instance_id=worker.instance_id,
                    )
                try:
                    await self._provider_for(worker).destroy_instance(worker.instance_id)
                except Exception as exc:
                    log.warning(
                        "worker_manager.fail_worker.destroy_failed",
                        instance_id=worker.instance_id,
                        error=str(exc),
                    )
                self.db.delete_worker(worker.worker_id)
                log.info(
                    "worker_manager.fail_worker.deleted_from_db",
                    worker_id=worker.worker_id,
                )

        log_section = (
            f"\n**Last logs:**\n```\n{logs[:1400]}\n```"
            if _logs_useful(logs)
            else ""
        )
        if reason.startswith("outbid"):
            self._preemption_count += 1
            next_floor = settings.bid_start_pct + self._preemption_count * settings.bid_step_pct
            log.warning(
                "worker_manager.fail_worker.preempted",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
                provider=worker.provider,
                gpu=worker.gpu_name,
                bid_price=worker.bid_price,
                market_price=worker.market_price,
                preemption_count=self._preemption_count,
                next_bid_floor=f"{next_floor:.0%}",
            )
            # Extract the provider's raw status message from the reason string
            # ("outbid: <provider status_msg>").
            provider_msg = reason[len("outbid:"):].strip() or "higher bid won"
            total_vram = (worker.gpu_ram_gb or 0) * worker.num_gpus
            gpu_desc = (
                f"{worker.num_gpus}×{worker.gpu_name} ({total_vram:.0f} GB total)"
                if worker.num_gpus > 1
                else f"{worker.gpu_name} ({worker.gpu_ram_gb or '?'} GB)"
            )
            await self.discord.send(
                title="Outbid",
                message=(
                    f"Instance `{worker.instance_id}` was claimed by a higher bidder "
                    "and is no longer available. Starting a new bid campaign."
                ),
                level="warning",
                fields=[
                    {
                        "name":   "GPU",
                        "value":  gpu_desc,
                        "inline": True,
                    },
                    {
                        "name":   "Provider",
                        "value":  worker.provider,
                        "inline": True,
                    },
                    {
                        "name":   "Bid price",
                        "value":  f"${worker.bid_price:.4f}/hr" if worker.bid_price else "n/a",
                        "inline": True,
                    },
                    {
                        "name":   "Market price",
                        "value":  f"${worker.market_price:.4f}/hr" if worker.market_price else "n/a",
                        "inline": True,
                    },
                    {
                        "name":   "Preemptions this session",
                        "value":  str(self._preemption_count),
                        "inline": True,
                    },
                    {
                        "name":   "Next bid floor",
                        "value":  f"{next_floor:.0%} of market",
                        "inline": True,
                    },
                    {
                        "name":   "Provider status",
                        "value":  provider_msg,
                        "inline": False,
                    },
                ],
            )
        else:
            await self.discord.send(
                f"**Worker failed** `{worker.worker_id}` (`{worker.gpu_name}`)\n"
                f"Reason: {reason}{log_section}",
                "error",
            )

        remaining = self.db.get_active_workers()
        if remaining:
            log.info(
                "worker_manager.fail_worker.workers_remain",
                count=len(remaining),
            )
            return

        log.info("worker_manager.fail_worker.no_workers_remain — starting bid campaign")
        await self.discord.send("No workers remaining. Starting a new bid campaign.", "warning")

        # _fail_worker may be called from *inside* the bid campaign task
        # (via _wait_for_running → _fail_worker). In that case _bidding_task.done()
        # returns False even though the campaign is about to exit, so the normal
        # check would silently skip the restart. Detect this by comparing against
        # the current running task and always start fresh when that's the case.
        current_task = asyncio.current_task()
        inside_bid_task = (
            self._bidding_task is not None
            and self._bidding_task is current_task
        )
        if inside_bid_task or not self._bidding_task or self._bidding_task.done():
            if self._bidding_task and not self._bidding_task.done() and not inside_bid_task:
                self._bidding_task.cancel()
            self._bidding_task = asyncio.create_task(
                self._bidding_campaign(), name="bid-campaign-recovery"
            )

    # ── Debug instance cap ────────────────────────────────────────────────

    async def _enforce_debug_cap(self) -> None:
        """
        When keep_debug_instance=True, ensure total alive instances ≤
        max_instances + 1 (one slot reserved for a debug instance).

        Called before every new bid/on-demand launch so we never accumulate
        more than one debug instance.
        """
        if not settings.keep_debug_instance:
            return

        cap = settings.max_instances + 1

        try:
            live_ids = {i.instance_id for i in await self._all_instances()}
        except Exception as exc:
            log.warning("worker_manager.debug_cap.list_failed", error=repr(exc))
            return

        active_workers  = self.db.get_active_workers()
        # TERMINATED workers whose instance is still alive
        terminated      = self.db.list_workers(status=WorkerStatus.TERMINATED)
        debug_instances = [
            w for w in terminated
            if w.instance_id and w.instance_id in live_ids
        ]

        alive_count    = len(active_workers) + len(debug_instances)
        # How many to evict so there is room for one new instance (alive < cap)
        to_evict_count = alive_count - (cap - 1)

        log.debug(
            "worker_manager.debug_cap.check",
            alive=alive_count,
            cap=cap,
            active=len(active_workers),
            debug=len(debug_instances),
            to_evict=max(0, to_evict_count),
        )

        if to_evict_count <= 0:
            return

        # Sort ascending by created_at so the newest entries are at the tail.
        # Evict from the tail — the *newest* debug instance(s), not the old one
        # the user is actively SSH'd into.
        debug_instances.sort(key=lambda w: w.created_at)

        # Never evict an instance that is still within its startup window —
        # it may be in the middle of pulling the container image.
        now = datetime.now(timezone.utc)
        evictable = [
            w for w in debug_instances
            if (now - w.created_at).total_seconds() >= settings.worker_startup_timeout_sec
        ]
        if not evictable:
            log.info(
                "worker_manager.debug_cap.all_too_young",
                debug=len(debug_instances),
                startup_timeout_sec=settings.worker_startup_timeout_sec,
            )
            return
        to_evict = evictable[-to_evict_count:]

        for w in to_evict:
            age_min = (datetime.now(timezone.utc) - w.created_at).total_seconds() / 60
            log.info(
                "worker_manager.debug_cap.evicting",
                worker_id=w.worker_id,
                instance_id=w.instance_id,
                gpu=w.gpu_name,
                age_min=round(age_min, 1),
            )
            await self.discord.send(
                f"**Debug cap** — evicting newest debug instance `{w.instance_id}` "
                f"(`{w.gpu_name}`, age {age_min:.0f} min). "
                f"Cap is {cap} (max_instances={settings.max_instances} + 1 debug slot). "
                "Oldest instance preserved for debugging.",
                "warning",
            )
            try:
                await self._provider_for(w).destroy_instance(w.instance_id)
            except Exception as exc:
                log.error(
                    "worker_manager.debug_cap.destroy_failed",
                    instance_id=w.instance_id,
                    error=repr(exc),
                )

    # ── SSH key management ────────────────────────────────────────────────

    def _fetch_ssh_key_from_aws(self) -> Optional[str]:
        """
        Fetch the orchestrator SSH private key from AWS Secrets Manager.
        Returns the PEM string on success, or None if unavailable.
        Called in a thread executor to avoid blocking the event loop.
        """
        secret_name = "nemotron-vllm/orchestrator-ssh-private-key"
        try:
            client = boto3.client("secretsmanager", region_name=settings.aws_region)
            resp = client.get_secret_value(SecretId=secret_name)
            pem = resp.get("SecretString")
            if pem:
                log.info(
                    "worker_manager.start.ssh_key_fetched",
                    secret=secret_name,
                )
                return pem
        except ClientError as exc:
            log.warning(
                "worker_manager.start.ssh_key_fetch_failed",
                secret=secret_name,
                error=repr(exc),
            )
        return None

    async def _manage_ssh_keys(self) -> None:
        """Register the orchestrator SSH key on every SSH-capable provider (e.g. vastai)."""
        if not self._ssh_public_key:
            return
        for p in self._providers.values():
            if p.supports_ssh:
                await self._manage_ssh_keys_for(p)

    async def _manage_ssh_keys_for(self, provider: GPUProvider) -> None:
        """
        Ensure the orchestrator's SSH public key is registered on *provider*.
        Idempotent: skips the create call when a matching key already exists.
        """
        if not self._ssh_public_key:
            return

        # OpenSSH pubkeys are "<algo> <base64-body> [<comment>]".  Two keys are
        # the same iff algo + body match — comments are ignored.
        our_parts = self._ssh_public_key.split()
        if len(our_parts) < 2:
            log.warning("worker_manager.ssh_keys.malformed_pubkey")
            return
        our_fingerprint = (our_parts[0], our_parts[1])

        try:
            existing = await provider.list_ssh_keys()
            for key in existing:
                pubkey_str = (key.get("public_key") or "").strip()
                parts = pubkey_str.split()
                if len(parts) >= 2 and (parts[0], parts[1]) == our_fingerprint:
                    log.info(
                        "worker_manager.ssh_keys.already_registered",
                        key_id=key.get("id"),
                        key_prefix=pubkey_str[:40],
                    )
                    return
            result = await provider.add_ssh_key(self._ssh_public_key)
            log.info("worker_manager.ssh_keys.registered", result=result)
        except Exception as exc:
            log.warning("worker_manager.ssh_keys.manage_failed", error=repr(exc))

    async def _attach_ssh_key_best_effort(self, instance_id: str, provider_name: str = "vastai") -> None:
        """
        Attach the orchestrator's SSH public key to a freshly created instance
        via the provider's per-instance attach endpoint.  No-ops for providers
        that don't support SSH.
        """
        if not self._ssh_public_key or not instance_id:
            return
        prov = self._providers.get(provider_name)
        if not prov or not prov.supports_ssh:
            return
        try:
            await prov.attach_ssh_key(instance_id, self._ssh_public_key)
            log.info(
                "worker_manager.ssh_keys.attached",
                instance_id=instance_id,
            )
        except Exception as exc:
            log.warning(
                "worker_manager.ssh_keys.attach_failed",
                instance_id=instance_id,
                error=repr(exc),
            )

    # ── SSH log fetching ──────────────────────────────────────────────────

    async def _ssh_inject_env_and_restart_vllm(
        self,
        instance: InstanceInfo,
        worker: Worker,
    ) -> bool:
        """
        SSH into the instance, write /etc/vllm-env.sh with the orchestrator's
        runtime config, and restart the vllm supervisor program so it picks up
        the new env.

        This is our primary config-injection path: vast.ai's API silently drops
        the `env` dict (Docker -e vars) and EXTRA_COMMANDS on many hosts, so
        values like VLLM_API_KEY, TENSOR_PARALLEL_SIZE and CUDA_VISIBLE_DEVICES
        never reach the container via the API.  We bypass the API entirely by
        using our SSH key (already injected into authorized_keys at launch) to
        write the env file ourselves and kick supervisor.

        Returns True on success.  Failure is non-fatal — the health-check loop
        will still try, log diagnostics, and eventually time out if the vllm
        process never becomes healthy.
        """
        if self._ssh_key is None:
            log.debug(
                "worker_manager.ssh_inject.no_key",
                worker_id=worker.worker_id,
            )
            return False
        if not (instance.ssh_host and instance.ssh_port):
            log.debug(
                "worker_manager.ssh_inject.no_ssh_port",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
            )
            return False

        # Build env from the same settings the vast.ai client would have
        # baked into its (ignored) env dict.
        # Prefer worker.num_gpus (set at bid time from the offer) over the live
        # instance.specs value — the API response for a running instance may omit
        # num_gpus, and the offer is the authoritative source of truth.
        num_gpus = worker.num_gpus or int(instance.specs.get("num_gpus") or 1)
        cuda_devices = ",".join(str(i) for i in range(num_gpus))
        env_overrides: dict[str, str] = {
            "VLLM_API_KEY":                worker.api_key,
            "MODEL_ID":                    settings.model_id,
            "HF_HOME":                     settings.hf_home,
            "VLLM_CACHE_ROOT":             settings.vllm_cache_root,
            "HF_HUB_ENABLE_HF_TRANSFER":   "1",
            "VLLM_PORT":                   str(settings.vllm_port),
            "VLLM_MAX_MODEL_LEN":          str(settings.vllm_max_model_len),
            "VLLM_GPU_MEMORY_UTILIZATION": str(settings.vllm_gpu_memory_utilization),
            "VLLM_VIDEO_LOADER_BACKEND":   settings.vllm_video_loader_backend,
            "TENSOR_PARALLEL_SIZE":        str(num_gpus),
            "DATA_PARALLEL_SIZE":          "1",
            "CUDA_VISIBLE_DEVICES":        cuda_devices,
        }
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env_overrides.items())
        env_script = f"#!/bin/bash\n{env_lines}\n"
        env_b64 = base64.b64encode(env_script.encode()).decode()

        # One-shot remote command: write env file, supervisorctl restart vllm.
        # Uses base64 to avoid any quoting hazards across the SSH transport.
        remote_cmd = (
            f"echo {env_b64} | base64 -d > /etc/vllm-env.sh"
            " && chmod 644 /etc/vllm-env.sh"
            " && (supervisorctl restart vllm 2>&1 || true)"
            " && echo OK"
        )

        host, port = instance.ssh_host, instance.ssh_port
        log.info(
            "worker_manager.ssh_inject.connecting",
            worker_id=worker.worker_id,
            host=host,
            port=port,
            num_gpus=num_gpus,
        )
        try:
            async with asyncssh.connect(
                host,
                port=port,
                username="root",
                client_keys=[self._ssh_key],
                known_hosts=None,
                connect_timeout=20,
            ) as conn:
                result = await conn.run(remote_cmd, timeout=30)
                ok = (result.stdout or "").strip().endswith("OK")
                log.info(
                    "worker_manager.ssh_inject.done",
                    worker_id=worker.worker_id,
                    ok=ok,
                    stdout_tail=(result.stdout or "").strip()[-200:],
                    stderr_tail=(result.stderr or "").strip()[-200:],
                )
                return ok
        except asyncssh.PermissionDenied:
            log.info(
                "worker_manager.ssh_inject.permission_denied_reinjecting",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
            )
            await self._attach_ssh_key_best_effort(worker.instance_id, worker.provider)
            return False
        except Exception as exc:
            log.warning(
                "worker_manager.ssh_inject.failed",
                worker_id=worker.worker_id,
                host=host,
                port=port,
                error=repr(exc),
            )
            return False

    async def _fetch_vllm_logs_ssh(
        self,
        instance: InstanceInfo,
        worker: Worker,
        lines: int = 150,
    ) -> Optional[str]:
        """
        SSH into the instance and collect debug logs.

        Fetches (in a single connection):
          • /var/log/onstart.log  — startup log: EXTRA_COMMANDS output,
                                    vLLM patch application, entrypoint invocation.
          • /tmp/vllm.log         — vLLM process stdout/stderr (tee'd by onstart.sh).

        Returns the combined log text, or None if SSH is unavailable
        (no key generated, instance not reachable, or port not mapped yet).
        """
        if self._ssh_key is None:
            return None
        if not (instance.ssh_host and instance.ssh_port):
            log.debug(
                "worker_manager.ssh_logs.no_ssh_port",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
            )
            return None

        host, port = instance.ssh_host, instance.ssh_port
        log.info(
            "worker_manager.ssh_logs.connecting",
            worker_id=worker.worker_id,
            host=host,
            port=port,
        )
        # Collect logs from all known locations in one SSH connection.
        # /var/log/portal/vllm.log — supervisor log-tee (always present on vastai/base-image)
        # /tmp/vllm.log            — tee'd by our onstart.sh (present when EXTRA_COMMANDS ran)
        # /var/log/onstart.log     — EXTRA_COMMANDS output + entrypoint invocation
        cmd = (
            f"echo '=== vllm log (last {lines} lines) ===';"
            f" tail -n {lines} /var/log/portal/vllm.log 2>/dev/null"
            f" || tail -n {lines} /tmp/vllm.log 2>/dev/null"
            " || echo '(no vllm log found yet)';"
            f" echo; echo '=== onstart.log (last 50 lines) ===';"
            " tail -n 50 /var/log/onstart.log 2>/dev/null"
            " || echo '(no onstart.log yet)'"
        )
        try:
            async with asyncssh.connect(
                host,
                port=port,
                username="root",
                client_keys=[self._ssh_key],
                known_hosts=None,       # vast.ai hosts rotate — fingerprint pinning not useful
                connect_timeout=20,
            ) as conn:
                result = await conn.run(cmd, timeout=20)
                text = result.stdout
                log.info(
                    "worker_manager.ssh_logs.ok",
                    worker_id=worker.worker_id,
                    bytes=len(text),
                )
                return text or None
        except asyncssh.PermissionDenied:
            # Key not in authorized_keys — re-inject it via the provider API.
            # Happens on instances that were provisioned before this key was
            # registered (e.g. orchestrator restart with a new stable key).
            log.info(
                "worker_manager.ssh_logs.permission_denied_reinjecting",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
            )
            await self._attach_ssh_key_best_effort(worker.instance_id, worker.provider)
            return None
        except Exception as exc:
            log.debug(
                "worker_manager.ssh_logs.failed",
                worker_id=worker.worker_id,
                host=host,
                port=port,
                error=repr(exc),
            )
            return None

    # ── Periodic status report ────────────────────────────────────────────

    async def _post_status_report(self) -> None:
        """Post a fleet-summary to Discord every status_report_interval_sec."""
        try:
            all_workers = self.db.list_workers()
        except Exception as exc:
            log.warning("worker_manager.status_report.db_failed", error=str(exc))
            return

        by_status: dict[str, list] = {}
        for w in all_workers:
            by_status.setdefault(w.status.value, []).append(w)

        running    = by_status.get("running", [])
        unhealthy  = by_status.get("unhealthy", [])
        starting   = by_status.get("starting", [])
        pending    = by_status.get("pending", [])
        terminated = by_status.get("terminated", [])

        now = datetime.now(timezone.utc)

        def _hours_running(w: Worker) -> float:
            if w.running_since:
                return (now - w.running_since).total_seconds() / 3600
            return 0.0

        def _instance_cost(w: Worker) -> float:
            if w.bid_price and w.running_since:
                return w.bid_price * _hours_running(w)
            return 0.0

        active_rate = sum(
            w.bid_price for w in all_workers
            if w.bid_price and w.status.value in ("running", "unhealthy", "starting", "pending")
        )
        total_spent = sum(_instance_cost(w) for w in all_workers)

        lines = ["**Fleet status report**"]

        def _specs_summary(w: Worker) -> str:
            s = w.specs or {}
            parts = []
            if s.get("cuda_max"):
                parts.append(f"CUDA {s['cuda_max']}")
            if s.get("tflops"):
                parts.append(f"{s['tflops']:.1f} TFLOPS")
            if s.get("mem_bw_gbps"):
                parts.append(f"{s['mem_bw_gbps']:.0f} GB/s")
            if s.get("dlperf"):
                parts.append(f"DLPerf {s['dlperf']:.1f}")
            if s.get("inet_down_mbps"):
                parts.append(f"↓{s['inet_down_mbps']:.0f} Mbps")
            if s.get("machine_id"):
                parts.append(f"machine#{s['machine_id']}")
            return f" ({', '.join(parts)})" if parts else ""

        if running:
            lines.append(f"🟢 Running: **{len(running)}**")
            for w in running:
                hrs   = _hours_running(w)
                spent = _instance_cost(w)
                rate  = w.bid_price or 0.0
                lines.append(
                    f"  • `{w.worker_id}` — `{w.gpu_name}`{_specs_summary(w)} "
                    f"${rate:.4f}/hr · {hrs:.1f}h up · **${spent:.4f} spent**"
                )
        else:
            lines.append("🟢 Running: **0**")

        if unhealthy:
            lines.append(f"🟡 Unhealthy: **{len(unhealthy)}**")
            for w in unhealthy:
                spent = _instance_cost(w)
                rate  = w.bid_price or 0.0
                lines.append(f"  • `{w.worker_id}` — `{w.gpu_name}`{_specs_summary(w)} ${rate:.4f}/hr · **${spent:.4f} spent**")
        if starting:
            lines.append(f"🔵 Starting: **{len(starting)}**")
        if pending:
            lines.append(f"⏳ Pending: **{len(pending)}**")
        if terminated:
            lines.append(f"🔴 Terminated this session: **{len(terminated)}**")

        lines.append(
            f"💰 Spend rate: **${active_rate:.4f}/hr** | "
            f"Total spent this session: **${total_spent:.4f}**"
        )
        if self._preemption_count:
            next_floor = settings.bid_start_pct + self._preemption_count * settings.bid_step_pct
            lines.append(
                f"⚡ Preemptions this session: **{self._preemption_count}** — "
                f"bid floor raised to **{next_floor:.0%}** of market"
            )

        await self.discord.send("\n".join(lines), "info")
        log.info(
            "worker_manager.status_report.sent",
            running=len(running),
            unhealthy=len(unhealthy),
            pending=len(pending),
            terminated=len(terminated),
        )

    # ── Log fetching ──────────────────────────────────────────────────────

    async def _fetch_worker_logs(
        self,
        instance_id: Optional[str],
        worker: Optional[Worker] = None,
        trigger: str = "manual",
    ) -> str:
        """
        Fetch the last ~100 lines of container logs from the provider, cache the full
        text in the event store, and return a short excerpt for Discord embeds.
        """
        if not instance_id:
            return "(no instance ID)"
        prov = (
            self._providers.get(worker.provider)
            if worker
            else next(iter(self._providers.values()), None)
        )
        if prov is None:
            return "(no provider available)"
        try:
            text = await prov.get_instance_logs(instance_id, tail=100)
        except Exception as exc:
            log.warning(
                "worker_manager.fetch_logs_failed",
                instance_id=instance_id,
                error=str(exc),
            )
            return f"(failed to fetch logs: {exc})"

        # Cache full log text in the event store
        if worker and text.strip():
            self.events.record_logs(
                worker_id=worker.worker_id,
                status=worker.status.value,
                log_text=text,
                instance_id=instance_id,
                label=worker.label,
                trigger=trigger,
            )

        # Return a short excerpt for Discord
        if len(text) > 1400:
            return "…(truncated)\n" + text[-1400:]
        return text
