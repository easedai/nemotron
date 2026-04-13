from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timezone
from typing import Optional

import asyncssh
import httpx
import structlog

from .config import settings
from .db import DynamoDB
from .event_store import EventStore
from .lb_registry import LBRegistry
from .models import Worker, WorkerStatus, WorkerType
from .notifications import Discord
from .vast_client import VastAIClient

log = structlog.get_logger(__name__)

# vast.ai instance states that are unrecoverable — worker is gone
_TERMINAL_STATES = frozenset({"exited", "offline", "deleted", "failed", "inactive"})
# Consecutive misses required before treating an instance as truly gone
_MISS_THRESHOLD = 3
# Keywords in status_msg / cur_state that indicate the instance was preempted by a higher bid
_OUTBID_KEYWORDS = ("outbid", "preempted", "overbid")


def _is_outbid(instance: dict) -> bool:
    """Return True if the vast.ai instance was terminated because of a higher bid."""
    if not instance:
        return False
    haystack = (
        (instance.get("status_msg") or "")
        + " "
        + (instance.get("cur_state") or "")
    ).lower()
    return any(kw in haystack for kw in _OUTBID_KEYWORDS)


class WorkerManager:
    def __init__(self) -> None:
        self.db      = DynamoDB()
        self.vast    = VastAIClient()
        self.discord = Discord()
        self.lb      = LBRegistry()
        self.events  = EventStore()
        self._bidding_task:      Optional[asyncio.Task] = None
        self._monitor_task:      Optional[asyncio.Task] = None
        self._vast_monitor_task: Optional[asyncio.Task] = None
        # Consecutive times each worker_id has been missing from vast.ai list.
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
        # Prevents duplicate tasks from spawning on every _reconcile_state /
        # _sync_with_vast tick when the same worker is already being polled.
        self._health_check_tasks: dict[str, asyncio.Task] = {}

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

    # ── Startup / shutdown ────────────────────────────────────────────────

    async def start(self) -> None:
        log.info("worker_manager.start")

        # Resolve the SSH key pair used to access vast.ai worker instances.
        #
        # ECS deployment: ORCHESTRATOR_SSH_PRIVATE_KEY is injected from Secrets
        # Manager (ssh_keys.tf / Terraform).  Using a stable key avoids the
        # "remove old eased-orchestrator keys" scan on every restart and
        # guarantees the same key survives task restarts.
        #
        # Local dev: no env var → generate an ephemeral Ed25519 key.
        if settings.orchestrator_ssh_private_key:
            self._ssh_key = asyncssh.import_private_key(
                settings.orchestrator_ssh_private_key
            )
            log.info("worker_manager.start.ssh_key_from_secret")
        else:
            self._ssh_key = asyncssh.generate_private_key("ssh-ed25519")
            log.info("worker_manager.start.ssh_key_generated_ephemeral")
        raw_pubkey = self._ssh_key.export_public_key("openssh").decode().strip()
        self._ssh_public_key = f"{raw_pubkey} eased-orchestrator"
        log.info("worker_manager.start.ssh_key_ready", pubkey_prefix=raw_pubkey[:40])
        await self._manage_vast_ssh_keys()

        await self._reconcile_state()
        await self._destroy_zombie_instances()
        await self._post_startup_audit()
        self._monitor_task = asyncio.create_task(
            self._health_monitor_loop(), name="health-monitor"
        )
        self._vast_monitor_task = asyncio.create_task(
            self._vast_monitor_loop(), name="vast-monitor"
        )
        await self._ensure_worker()

    async def stop(self) -> None:
        log.info("worker_manager.stop")

        # Cancel background tasks first so nothing races with the cleanup below
        for task in (self._bidding_task, self._monitor_task, self._vast_monitor_task):
            if task and not task.done():
                task.cancel()

        # NOTE: instances are intentionally NOT destroyed on shutdown.
        # Multiple orchestrator replicas may be running simultaneously (e.g.
        # rolling ECS deployments or local dev alongside production), so tearing
        # down instances on exit would kill workers owned by a sibling replica.
        log.info("worker_manager.stop.skipping_instance_cleanup")

        # Best-effort: remove our SSH key from the vast.ai account
        if self._ssh_public_key:
            try:
                existing = await self.vast.list_ssh_keys()
                for key in existing:
                    if "eased-orchestrator" in (key.get("public_key") or ""):
                        await self.vast.delete_ssh_key(int(key["id"]))
                        log.info("worker_manager.stop.ssh_key_removed", key_id=key["id"])
            except Exception as exc:
                log.warning("worker_manager.stop.ssh_key_removal_failed", error=repr(exc))

    async def _destroy_all_instances(self) -> None:
        """
        Destroy every vast.ai instance we own (active workers + debug instances
        kept alive by keep_debug_instance) and update their DB status to TERMINATED.

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
                await self.vast.destroy_instance(worker.instance_id)
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

    # ── State reconciliation ──────────────────────────────────────────────

    async def _reconcile_state(self) -> None:
        """
        On startup, cross-check DynamoDB records against live vast.ai instances.
        Repairs stale state left by a previous orchestrator crash or restart.
        Workers still pending have their _wait_for_running task resumed.
        """
        log.info("worker_manager.reconcile.start")
        db_workers = self.db.get_active_workers()
        if not db_workers:
            log.info("worker_manager.reconcile.no_active_workers")
            return

        try:
            live_instances = {
                str(i["id"]): i for i in await self.vast.list_instances()
            }
        except Exception as exc:
            log.error("worker_manager.reconcile.list_failed", error=str(exc))
            return

        log.info(
            "worker_manager.reconcile",
            db_workers=len(db_workers),
            live_instances=len(live_instances),
        )

        for worker in db_workers:
            if not worker.instance_id:
                continue

            live = live_instances.get(str(worker.instance_id))
            if not live:
                log.warning(
                    "worker_manager.reconcile.instance_gone",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                )
                self.db.delete_worker(worker.worker_id)
                continue

            actual     = live.get("actual_status", "unknown")
            cur_state  = live.get("cur_state", "")
            status_msg = live.get("status_msg", "")
            log.info(
                "worker_manager.reconcile.instance",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
                actual_status=actual,
                cur_state=cur_state,
                status_msg=status_msg,
                db_status=worker.status,
            )

            if actual == "running":
                addr = self.vast.extract_worker_address(live)
                if addr:
                    host, port = addr
                    if worker.status == WorkerStatus.RUNNING:
                        # Already verified before — just refresh host/port in case they changed
                        self.db.update_worker_status(
                            worker.worker_id,
                            WorkerStatus.RUNNING,
                            host=host,
                            port=port,
                        )
                        # Re-register in LB (idempotent — preserves accumulated stats)
                        # in case the LB table entry was lost while we were offline.
                        self.lb.register(worker.worker_id, "vastai", host, port, worker.api_key)
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
                        )
            elif actual in _TERMINAL_STATES:
                # Route through _fail_worker so outbid instances increment
                # _preemption_count, send a Discord alert, and raise the bid
                # floor for the next campaign — same path as _sync_with_vast.
                outbid = _is_outbid(live)
                reason = (
                    f"outbid: {status_msg}"
                    if outbid
                    else f"vast.ai reports {actual!r} on startup: {status_msg}"
                )
                await self._fail_worker(worker, reason=reason)
            elif worker.status in (WorkerStatus.PENDING, WorkerStatus.STARTING):
                # Instance is still coming up — resume the wait task
                log.info(
                    "worker_manager.reconcile.resuming_wait",
                    worker_id=worker.worker_id,
                    actual_status=actual,
                )
                asyncio.create_task(
                    self._wait_for_running(worker),
                    name=f"wait-running-{worker.worker_id}",
                )

    # ── Startup zombie cleanup ────────────────────────────────────────────

    async def _destroy_zombie_instances(self) -> None:
        """
        Destroy any vast.ai instances we own that have no DynamoDB record.

        _reconcile_state() runs first, so any instance the DB already knows
        about is handled.  Anything left with our label/image is a zombie from
        a previous crashed or restarted orchestrator — it's burning money with
        no one watching it.
        """
        log.info("worker_manager.destroy_zombies.start")
        try:
            all_instances = await self.vast.list_instances()
        except Exception as exc:
            log.error("worker_manager.destroy_zombies.list_failed", error=str(exc))
            return

        known_ids   = self.db.get_known_instance_ids()
        ghcr_prefix = "ghcr.io/easedai/"
        zombies: list[dict] = []

        for instance in all_instances:
            iid = str(instance.get("id", ""))
            if not iid or iid in known_ids:
                continue
            label  = instance.get("label", "") or ""
            image  = instance.get("image", "") or ""
            if not (label.startswith("eased-") or ghcr_prefix in image):
                continue
            if instance.get("actual_status", "") in _TERMINAL_STATES:
                continue  # already dead

            zombies.append(instance)
            log.warning(
                "worker_manager.destroy_zombies.found",
                instance_id=iid,
                label=label,
                actual_status=instance.get("actual_status"),
                gpu=instance.get("gpu_name"),
            )

        if not zombies:
            log.info("worker_manager.destroy_zombies.none_found")
            return

        lines = [f"**Startup cleanup — {len(zombies)} zombie instance(s) destroyed**"]
        for z in zombies:
            lines.append(
                f"  • `{z.get('id')}` — `{z.get('gpu_name', '?')}` "
                f"status `{z.get('actual_status', '?')}` "
                f"label `{z.get('label', '?')}`"
            )
        await self.discord.send("\n".join(lines), "warning")

        for zombie in zombies:
            iid = str(zombie.get("id", ""))
            try:
                await self.vast.destroy_instance(iid)
                log.info("worker_manager.destroy_zombies.destroyed", instance_id=iid)
            except Exception as exc:
                log.error(
                    "worker_manager.destroy_zombies.destroy_failed",
                    instance_id=iid,
                    error=repr(exc),
                )

        log.info("worker_manager.destroy_zombies.done", destroyed=len(zombies))

    # ── Startup audit ─────────────────────────────────────────────────────

    async def _post_startup_audit(self) -> None:
        """Post a one-time Discord summary of DB + vast.ai state after startup reconciliation."""
        try:
            db_workers    = self.db.list_workers()
            all_instances = await self.vast.list_instances()
        except Exception as exc:
            log.warning("worker_manager.startup_audit.failed", error=str(exc))
            return

        by_status: dict[str, int] = {}
        for w in db_workers:
            by_status[w.status.value] = by_status.get(w.status.value, 0) + 1

        ghcr_prefix = "ghcr.io/easedai/"
        our_instances = [
            i for i in all_instances
            if (i.get("label", "") or "").startswith("eased-")
            or ghcr_prefix in (i.get("image", "") or "")
        ]

        log.info(
            "worker_manager.startup_audit",
            db_workers=len(db_workers),
            our_vast_instances=len(our_instances),
            by_status=by_status,
        )

        status_parts = ", ".join(f"{s}: {n}" for s, n in sorted(by_status.items())) or "none"
        lines = [
            "**Orchestrator started**",
            f"DB workers: **{len(db_workers)}** ({status_parts})",
            f"vast.ai instances (ours): **{len(our_instances)}**",
        ]
        for inst in our_instances:
            lines.append(
                f"  • `{inst.get('id')}` — `{inst.get('gpu_name', '?')}` "
                f"`{inst.get('actual_status', '?')}` label `{inst.get('label', '?')}`"
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

    async def _bidding_campaign(self) -> None:
        """
        Bid for a cheap interruptible GPU instance on vast.ai.

        Strategy:
          • Start at bid_start_pct  (default 70 %) of median market price
          • Every bid_retry_interval_sec (default 5 min) increase by bid_step_pct (5 %)
          • Give up and fall back to on-demand once bid_max_multiplier (110 %) is exceeded
        """
        log.info(
            "worker_manager.bid_campaign.start",
            preemption_count=self._preemption_count,
        )
        await self._enforce_debug_cap()
        preemption_note = (
            f" ({self._preemption_count}× preempted this session — "
            f"starting bid raised by {self._preemption_count} step(s))"
            if self._preemption_count
            else ""
        )
        await self.discord.send(
            f"**Bid campaign started** — searching for a cheap GPU worker on vast.ai.{preemption_note}",
            "info",
        )

        try:
            offers = await self.vast.search_offers(on_demand=False)
        except Exception as exc:
            log.error("worker_manager.bid_campaign.search_failed", error=str(exc))
            await self.discord.send(f"vast.ai offer search failed: `{exc}`", "error")
            return

        if not offers:
            log.error("worker_manager.bid_campaign.no_offers")
            await self.discord.send(
                "No interruptible GPU offers found on vast.ai. Falling back to on-demand.",
                "warning",
            )
            await self._launch_on_demand()
            return

        market_price   = self.vast.get_market_price(offers)
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

            # Pick the cheapest offer our bid can beat
            best_offer = next(
                (o for o in offers if o.get("dph_base", float("inf")) <= bid_price),
                None,
            )
            if not best_offer:
                cheapest = offers[0].get("dph_base") if offers else None
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
                await asyncio.sleep(settings.bid_retry_interval_sec)
                attempt += 1
                # Refresh offers after the wait — prices shift between retries
                try:
                    offers       = await self.vast.search_offers(on_demand=False)
                    market_price = self.vast.get_market_price(offers)
                except Exception as exc:
                    log.warning("worker_manager.bid_campaign.refresh_failed", error=str(exc))
                continue

            worker_id = secrets.token_urlsafe(8)
            label     = f"eased-{worker_id}"
            log.info(
                "worker_manager.bid_campaign.placing",
                attempt=attempt + 1,
                offer_id=best_offer["id"],
                bid_price=bid_price,
                gpu=best_offer.get("gpu_name"),
                gpu_ram_gb=round(best_offer.get("gpu_ram", 0) / 1024, 1),
                label=label,
            )

            try:
                result = await self.vast.create_instance(
                    offer_id=best_offer["id"],
                    price=bid_price,
                    worker_api_key=worker_api_key,
                    worker_type=WorkerType.INTERRUPTIBLE,
                    label=label,
                    ssh_public_key=self._ssh_public_key,
                )
                instance_id = str(
                    result.get("new_contract") or result.get("id") or ""
                )
                if not instance_id:
                    raise ValueError(f"No instance ID in response: {result}")
            except Exception as exc:
                log.error(
                    "worker_manager.bid_campaign.create_failed",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                await asyncio.sleep(settings.bid_retry_interval_sec)
                attempt += 1
                continue

            now = datetime.now(timezone.utc)
            worker = Worker(
                worker_id=worker_id,
                instance_id=instance_id,
                label=label,
                status=WorkerStatus.PENDING,
                worker_type=WorkerType.INTERRUPTIBLE,
                api_key=worker_api_key,
                gpu_name=best_offer.get("gpu_name"),
                gpu_ram_gb=round(best_offer.get("gpu_ram", 0) / 1024, 1),
                bid_price=bid_price,
                market_price=market_price,
                bid_attempts=attempt + 1,
                specs=self.vast.extract_instance_specs(best_offer),
                image_pull_started_at=now,
                created_at=now,
                updated_at=now,
            )
            self.db.save_worker(worker)
            self._ev(
                worker, "worker.created",
                f"Bid accepted — {best_offer.get('gpu_name')} "
                f"at ${bid_price:.4f}/hr ({multiplier:.0%} of market), instance {instance_id}",
                meta={"bid_price": bid_price, "market_price": market_price,
                      "attempt": attempt + 1, "gpu": best_offer.get("gpu_name"),
                      "offer_id": best_offer["id"]},
            )

            self._last_winning_multiplier = multiplier
            log.info(
                "worker_manager.bid_campaign.bid_placed",
                worker_id=worker_id,
                instance_id=instance_id,
                bid_price=bid_price,
                multiplier=f"{multiplier:.0%}",
                gpu=best_offer.get("gpu_name"),
            )
            await self.discord.send(
                f"**Bid accepted** — `{best_offer.get('gpu_name')}` "
                f"({round(best_offer.get('gpu_ram', 0) / 1024, 1)} GB VRAM) "
                f"at **${bid_price:.4f}/hr** ({multiplier:.0%} of market).\n"
                f"Instance `{instance_id}` is pending — monitoring until ready.",
                "success",
            )

            await self._wait_for_running(worker)
            return

    # ── On-demand fallback ────────────────────────────────────────────────

    async def _launch_on_demand(self) -> None:
        log.info("worker_manager.on_demand.start")
        await self._enforce_debug_cap()
        await self.discord.send(
            "**Launching on-demand fallback** — more expensive, but no bidding delay.",
            "warning",
        )
        try:
            offers = await self.vast.search_offers(on_demand=True)
        except Exception as exc:
            log.error("worker_manager.on_demand.search_failed", error=str(exc))
            await self.discord.send(f"On-demand search failed: `{exc}`", "error")
            return

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
            offer_id=best["id"],
            price=best.get("dph_base"),
            gpu=best.get("gpu_name"),
            label=label,
        )

        try:
            result = await self.vast.create_instance(
                offer_id=best["id"],
                price=best["dph_base"],
                worker_api_key=worker_api_key,
                worker_type=WorkerType.ON_DEMAND,
                label=label,
                ssh_public_key=self._ssh_public_key,
            )
            instance_id = str(result.get("new_contract") or result.get("id") or "")
            if not instance_id:
                raise ValueError(f"No instance ID in response: {result}")
        except Exception as exc:
            log.error("worker_manager.on_demand.create_failed", error=str(exc))
            await self.discord.send(f"On-demand launch failed: `{exc}`", "error")
            return

        now = datetime.now(timezone.utc)
        worker = Worker(
            worker_id=worker_id,
            instance_id=instance_id,
            label=label,
            status=WorkerStatus.PENDING,
            worker_type=WorkerType.ON_DEMAND,
            api_key=worker_api_key,
            gpu_name=best.get("gpu_name"),
            gpu_ram_gb=round(best.get("gpu_ram", 0) / 1024, 1),
            bid_price=best.get("dph_base"),
            market_price=best.get("dph_base"),
            specs=self.vast.extract_instance_specs(best),
            image_pull_started_at=now,
            created_at=now,
            updated_at=now,
        )
        self.db.save_worker(worker)
        self._ev(
            worker, "worker.created",
            f"On-demand instance created — {best.get('gpu_name')} "
            f"at ${best.get('dph_base', 0):.4f}/hr, instance {instance_id}",
            meta={"price": best.get("dph_base"), "gpu": best.get("gpu_name"),
                  "offer_id": best["id"]},
        )

        await self.discord.send(
            f"**On-demand instance created** — `{best.get('gpu_name')}` "
            f"at **${best.get('dph_base', 0):.4f}/hr**. Instance `{instance_id}`.",
            "warning",
        )
        await self._wait_for_running(worker)

    # ── Worker readiness waiting ──────────────────────────────────────────

    async def _wait_for_running(self, worker: Worker) -> None:
        """
        Poll vast.ai every 15 s until the instance reaches `running` status,
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
        # Tracks whether we have ever seen this instance in the vast.ai API.
        # Before the first confirmation we do NOT count misses as failures —
        # the API can take 30-60 s to propagate a freshly created instance.
        instance_ever_seen = False

        while loop.time() < deadline:
            await asyncio.sleep(15)
            try:
                instance = await self.vast.get_instance(worker.instance_id)
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
                    reason="instance disappeared from vast.ai",
                    logs=logs,
                )
                return

            instance_ever_seen = True
            miss_count = 0  # confirmed alive — reset

            actual     = instance.get("actual_status", "unknown")
            cur_state  = instance.get("cur_state", "")
            status_msg = instance.get("status_msg", "") or ""
            next_state = instance.get("next_state", "")
            elapsed    = round(loop.time() - (deadline - settings.instance_running_timeout_sec))

            log.info(
                "worker_manager.wait_for_running.poll",
                worker_id=worker.worker_id,
                actual_status=actual,
                cur_state=cur_state,
                status_msg=status_msg,
                next_state=next_state,
                elapsed_sec=elapsed,
            )

            # Detect repeated Docker pull / image errors in status_msg — fail fast
            # rather than waiting for the full startup timeout.
            if status_msg.lower().startswith("error"):
                consecutive_errors += 1
                log.warning(
                    "worker_manager.wait_for_running.status_msg_error",
                    worker_id=worker.worker_id,
                    status_msg=status_msg,
                    consecutive_errors=consecutive_errors,
                )
                if consecutive_errors >= 3:
                    logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="status_msg_error")
                    await self._fail_worker(
                        worker,
                        reason=f"repeated error in vast.ai status: {status_msg}",
                        logs=logs,
                    )
                    return
            else:
                consecutive_errors = 0

            if actual == "running":
                addr = self.vast.extract_worker_address(instance)
                if addr:
                    host, port = addr

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
                        message=f"vast.ai instance running — waiting for vLLM health at {host}:{port}",
                        instance_id=worker.instance_id,
                        label=worker.label,
                        meta={"host": host, "port": port,
                              "cur_state": cur_state, "elapsed_sec": elapsed,
                              "image_pull_duration_sec": pull_duration},
                    )
                    pull_note = (
                        f" Image pulled in **{pull_duration:.0f}s**."
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
                    self._spawn_vllm_health_task(
                        worker_id=worker.worker_id,
                        host=host,
                        port=port,
                        api_key=worker.api_key,
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
                        ports=instance.get("ports"),
                    )

            elif actual in _TERMINAL_STATES:
                log.error(
                    "worker_manager.wait_for_running.terminal_state",
                    worker_id=worker.worker_id,
                    actual_status=actual,
                    cur_state=cur_state,
                    status_msg=status_msg,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="terminal_state")
                await self._fail_worker(
                    worker,
                    reason=f"vast.ai status=`{actual}` cur_state=`{cur_state}` — {status_msg}",
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
    ) -> None:
        """
        Start a _wait_for_vllm_health task for *worker_id* if one is not already
        running.  Stores the task in self._health_check_tasks and registers a
        done-callback that removes it so the slot is freed when the task exits.
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
    ) -> None:
        """
        Poll the worker's /health endpoint until vLLM responds 200.

        vast.ai is treated as ground truth: on every iteration we also check
        the instance status.  If it is gone or in a terminal state we fail
        immediately instead of waiting out the full startup timeout.
        """
        url  = f"http://{host}:{port}/health"
        loop = asyncio.get_event_loop()
        deadline = loop.time() + settings.worker_startup_timeout_sec

        log.info(
            "worker_manager.wait_for_vllm_health.start",
            worker_id=worker_id,
            url=url,
            timeout_sec=settings.worker_startup_timeout_sec,
        )

        vast_miss_count = 0

        while loop.time() < deadline:
            # ── 1. Check vast.ai first — ground truth ────────────────────
            worker = self.db.get_worker(worker_id)
            if worker and worker.instance_id:
                try:
                    instance = await self.vast.get_instance(worker.instance_id)
                    actual   = instance.get("actual_status", "") if instance else ""

                    if not instance or actual in _TERMINAL_STATES:
                        vast_miss_count += 1
                        log.warning(
                            "worker_manager.wait_for_vllm_health.instance_gone",
                            worker_id=worker_id,
                            instance_id=worker.instance_id,
                            actual_status=actual,
                            consecutive_misses=vast_miss_count,
                            threshold=_MISS_THRESHOLD,
                        )
                        if vast_miss_count >= _MISS_THRESHOLD:
                            reason = (
                                "vast.ai instance gone (no record)"
                                if not instance
                                else f"vast.ai status={actual!r}"
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
                        "worker_manager.wait_for_vllm_health.vast_check_failed",
                        worker_id=worker_id,
                        error=str(exc),
                    )

            # ── 2. Ping vLLM ─────────────────────────────────────────────
            try:
                async with httpx.AsyncClient(timeout=settings.health_check_timeout_sec) as c:
                    r = await c.get(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if r.status_code == 200:
                        now = datetime.now(timezone.utc)
                        self.db.update_worker_status(
                            worker_id,
                            WorkerStatus.RUNNING,
                            running_since=now,
                        )
                        # Register in the load-balancer pool so it starts
                        # receiving traffic immediately.
                        self.lb.register(worker_id, "vastai", host, port, api_key)
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
                            message=f"vLLM health check passed — serving at {host}:{port}",
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
                    log.debug(
                        "worker_manager.wait_for_vllm_health.not_ready",
                        worker_id=worker_id,
                        status=r.status_code,
                    )
            except Exception as exc:
                log.debug(
                    "worker_manager.wait_for_vllm_health.ping_failed",
                    worker_id=worker_id,
                    error=str(exc),
                )
            await asyncio.sleep(15)

        # Deadline exceeded — re-fetch worker, try SSH logs first (more detailed
        # than the vast.ai API log endpoint), then fall back to the API.
        worker = self.db.get_worker(worker_id)
        logs: str = ""
        if worker and worker.instance_id:
            try:
                instance = await self.vast.get_instance(worker.instance_id)
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
        await self._fail_worker(
            worker,
            reason=f"vLLM health check timed out after {settings.worker_startup_timeout_sec // 60} min — model may have failed to load",
            logs=logs,
        )

    # ── vast.ai cross-check loop ──────────────────────────────────────────

    async def _vast_monitor_loop(self) -> None:
        """
        Background task — every vast_check_interval_sec:
          • Reconcile DB workers against live vast.ai instances (sync)
          • Every status_report_interval_sec post a summary to Discord
        """
        log.info(
            "worker_manager.vast_monitor.start",
            interval_sec=settings.vast_check_interval_sec,
        )
        ticks_per_report = max(1, settings.status_report_interval_sec // settings.vast_check_interval_sec)
        tick = 0
        while True:
            try:
                await asyncio.sleep(settings.vast_check_interval_sec)
                await self._sync_with_vast()
                tick += 1
                if tick % ticks_per_report == 0:
                    await self._post_status_report()
            except asyncio.CancelledError:
                log.info("worker_manager.vast_monitor.cancelled")
                break
            except Exception as exc:
                log.error("worker_manager.vast_monitor.error", error=str(exc))

    async def _sync_with_vast(self) -> None:
        """
        Cross-check every active DB worker against the live vast.ai instance list.

        Three responsibilities:
          1. Detect active workers whose vast.ai instance is gone / terminal → fail them
          2. Detect PENDING/STARTING workers whose instance is now running → start health check
          3. Detect orphaned vast.ai instances (our label, not in our DB) → destroy them
        """
        workers = self.db.get_active_workers()

        try:
            all_instances = await self.vast.list_instances()
            live = {str(i["id"]): i for i in all_instances}
        except Exception as exc:
            log.warning("worker_manager.vast_sync.list_failed", error=str(exc))
            return

        log.debug(
            "worker_manager.vast_sync.tick",
            active_workers=len(workers),
            live_instances=len(live),
        )

        # ── 1 & 2: reconcile active DB workers against vast.ai ────────────
        for worker in workers:
            if not worker.instance_id:
                continue

            instance = live.get(str(worker.instance_id))

            if not instance:
                miss = self._instance_miss_counts.get(worker.worker_id, 0) + 1
                self._instance_miss_counts[worker.worker_id] = miss
                log.warning(
                    "worker_manager.vast_sync.instance_missing",
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
                    reason=f"instance no longer exists on vast.ai after {miss} consecutive checks (reclaimed or deleted)",
                    logs=logs,
                )
                continue

            # Instance found — reset miss counter
            self._instance_miss_counts.pop(worker.worker_id, None)

            actual     = instance.get("actual_status", "")
            cur_state  = instance.get("cur_state", "")
            status_msg = instance.get("status_msg", "")

            log.debug(
                "worker_manager.vast_sync.instance",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
                actual_status=actual,
                cur_state=cur_state,
                status_msg=status_msg,
                db_status=worker.status,
            )

            if actual in _TERMINAL_STATES:
                outbid = _is_outbid(instance)
                log.warning(
                    "worker_manager.vast_sync.terminal",
                    worker_id=worker.worker_id,
                    actual_status=actual,
                    status_msg=status_msg,
                    outbid=outbid,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="terminal_state")
                reason = (
                    f"outbid: {status_msg}"
                    if outbid
                    else f"vast.ai reports {actual!r}: {status_msg}"
                )
                await self._fail_worker(worker, reason=reason, logs=logs)

            elif (
                cur_state in {"stopped", "exited", "failed"}
                and worker.status in (WorkerStatus.PENDING, WorkerStatus.STARTING)
            ):
                # Container stopped during startup (e.g. outbid while pulling image).
                # _wait_for_running won't catch this if get_instance can't see the
                # instance, so _sync_with_vast acts as a safety net.
                outbid = _is_outbid(instance)
                log.warning(
                    "worker_manager.vast_sync.stopped_during_startup",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    actual_status=actual,
                    cur_state=cur_state,
                    db_status=worker.status,
                    outbid=outbid,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="stopped_during_startup")
                reason = (
                    f"outbid: {status_msg}"
                    if outbid
                    else f"container stopped during startup (cur_state={cur_state!r}, actual={actual!r}): {status_msg}"
                )
                await self._fail_worker(worker, reason=reason, logs=logs)

            elif actual != "running" and worker.status in (
                WorkerStatus.RUNNING, WorkerStatus.UNHEALTHY
            ):
                # Instance was healthy but vast.ai no longer reports it as running
                # (e.g. reclaimed, restarting). Fail immediately — vast.ai is ground truth.
                outbid = _is_outbid(instance)
                log.warning(
                    "worker_manager.vast_sync.no_longer_running",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    actual_status=actual,
                    db_status=worker.status,
                    outbid=outbid,
                )
                logs = await self._fetch_worker_logs(worker.instance_id, worker=worker, trigger="no_longer_running")
                reason = (
                    f"outbid: {status_msg}"
                    if outbid
                    else f"vast.ai reports {actual!r} (was RUNNING in DB): {status_msg}"
                )
                await self._fail_worker(worker, reason=reason, logs=logs)

            elif (
                actual == "running"
                and worker.status in (WorkerStatus.PENDING, WorkerStatus.STARTING)
            ):
                # Instance came up while the orchestrator was restarting or the
                # _wait_for_running task didn't fire. Kick off vLLM health check.
                addr = self.vast.extract_worker_address(instance)
                if addr:
                    host, port = addr
                    log.info(
                        "worker_manager.vast_sync.recovered_running",
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
                    )

        # ── 3: destroy orphaned instances ─────────────────────────────────
        # An instance is "ours" if it has an eased-* label OR its image is from
        # our GHCR org (catches pre-label instances created by older orchestrator runs).
        # If it's ours but has no DB record, it's an orphan — destroy it.
        known_ids = self.db.get_known_instance_ids()
        ghcr_prefix = "ghcr.io/easedai/"
        for instance_id, instance in live.items():
            label = instance.get("label", "") or ""
            image = instance.get("image", "") or ""
            is_ours = label.startswith("eased-") or ghcr_prefix in image
            if not is_ours:
                continue
            if instance_id in known_ids:
                continue
            actual = instance.get("actual_status", "")
            log.warning(
                "worker_manager.vast_sync.orphan",
                instance_id=instance_id,
                label=label,
                image=image,
                actual_status=actual,
            )
            await self.discord.send(
                f"**Orphaned instance** `{instance_id}` (label: `{label}`, image: `{image}`, status: `{actual}`) "
                "has no DB record — destroying.",
                "warning",
            )
            # Record to event log using the label-derived worker_id if possible
            orphan_worker_id = label[6:] if label.startswith("eased-") and len(label) > 6 else instance_id
            self.events.record(
                worker_id=orphan_worker_id,
                event_type="orphan.destroyed",
                status="terminated",
                message=f"Orphaned instance {instance_id} (label: {label!r}, status: {actual!r}) — no DB record, destroying",
                instance_id=instance_id,
                label=label or None,
                meta={"image": image, "actual_status": actual},
            )
            try:
                await self.vast.destroy_instance(instance_id)
            except Exception as exc:
                log.error(
                    "worker_manager.vast_sync.orphan_destroy_failed",
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
                await asyncio.sleep(settings.health_check_interval_sec)
                await self._check_all_workers()
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
                    instance = await self.vast.get_instance(worker.instance_id)
                    if instance:
                        tail = await self._fetch_vllm_logs_ssh(instance, worker, lines=5)
                        if tail and tail.strip():
                            log.info(
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
                    # Mark UNHEALTHY so it stops receiving traffic but isn't killed yet
                    self.db.update_worker_status(
                        worker.worker_id,
                        WorkerStatus.UNHEALTHY,
                        consecutive_failures=new_failures,
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
        Mark a worker as TERMINATED, fetch container logs, destroy the vast.ai
        instance, and post a Discord alert with reason + log tail.

        Logs are fetched (or re-fetched if the caller's attempt came back empty)
        BEFORE the instance is destroyed so they are still available on the host.
        """
        if worker is None:
            log.error("worker_manager.fail_worker.no_worker", reason=reason)
            return

        # Re-read from DB to guard against concurrent failure paths (e.g.
        # _wait_for_running and _vast_monitor_loop both detecting the same
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
        self.lb.deregister(worker.worker_id)
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
                # Outbid instances are never kept: vast.ai already reclaimed the
                # hardware, so there is nothing to SSH into.
                log.info(
                    "worker_manager.fail_worker.keeping_for_debug",
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                )
            else:
                if is_outbid:
                    log.info(
                        "worker_manager.fail_worker.outbid_cleanup",
                        worker_id=worker.worker_id,
                        instance_id=worker.instance_id,
                    )
                try:
                    await self.vast.destroy_instance(worker.instance_id)
                    self.db.delete_worker(worker.worker_id)
                    log.info(
                        "worker_manager.fail_worker.deleted_from_db",
                        worker_id=worker.worker_id,
                    )
                except Exception as exc:
                    log.warning(
                        "worker_manager.fail_worker.destroy_failed",
                        instance_id=worker.instance_id,
                        error=str(exc),
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
                preemption_count=self._preemption_count,
                next_bid_floor=f"{next_floor:.0%}",
            )
            await self.discord.send(
                f"**Outbid** — `{worker.gpu_name}` instance `{worker.instance_id}` was claimed by a "
                f"higher bidder and is no longer available.\n"
                f"Preemption count this session: **{self._preemption_count}** — "
                f"next campaign starts at **{next_floor:.0%}** of market price. "
                "Starting a new bid campaign.",
                "warning",
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
        current = asyncio.current_task()
        inside_bid_task = (
            self._bidding_task is not None
            and self._bidding_task is current
        )
        if inside_bid_task or not self._bidding_task or self._bidding_task.done():
            # If we're inside the old bid task, it will finish naturally once
            # _fail_worker returns. Schedule the new campaign to start after
            # the current event-loop iteration so there's no overlap.
            if self._bidding_task and not self._bidding_task.done() and not inside_bid_task:
                self._bidding_task.cancel()
            self._bidding_task = asyncio.create_task(
                self._bidding_campaign(), name="bid-campaign-recovery"
            )

    # ── Debug instance cap ────────────────────────────────────────────────

    async def _enforce_debug_cap(self) -> None:
        """
        When keep_debug_instance=True, ensure total alive vast.ai instances ≤
        max_instances + 1 (one slot reserved for a debug instance).

        Called before every new bid/on-demand launch so we never accumulate
        more than one debug instance.

        Algorithm:
          1. List alive instances from vast.ai.
          2. Find TERMINATED workers in DB whose instance_id is still alive
             (these are the kept-for-debug instances).
          3. alive_count = active_workers + debug_instances
          4. If alive_count >= cap (max_instances + 1), evict the NEWEST debug
             instance(s) — i.e. the shortest-lived one — because that is NOT the
             instance being actively debugged (the old failure is).
        """
        if not settings.keep_debug_instance:
            return

        cap = settings.max_instances + 1

        try:
            live_ids = {str(i["id"]) for i in await self.vast.list_instances()}
        except Exception as exc:
            log.warning("worker_manager.debug_cap.list_failed", error=repr(exc))
            return

        active_workers  = self.db.get_active_workers()
        # TERMINATED workers whose vast.ai instance is still alive
        terminated      = self.db.list_workers(status=WorkerStatus.TERMINATED)
        debug_instances = [
            w for w in terminated
            if w.instance_id and str(w.instance_id) in live_ids
        ]

        alive_count     = len(active_workers) + len(debug_instances)
        # How many to evict so there is room for one new instance (alive < cap)
        to_evict_count  = alive_count - (cap - 1)

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
            age_min = (
                datetime.now(timezone.utc) - w.created_at
            ).total_seconds() / 60
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
                await self.vast.destroy_instance(w.instance_id)
            except Exception as exc:
                log.error(
                    "worker_manager.debug_cap.destroy_failed",
                    instance_id=w.instance_id,
                    error=repr(exc),
                )

    # ── SSH key management ────────────────────────────────────────────────

    async def _manage_vast_ssh_keys(self) -> None:
        """
        Remove any stale eased-orchestrator SSH keys from the vast.ai account,
        then register the current session's public key.

        Called once on startup.  This ensures old keys left behind by previous
        orchestrator runs don't accumulate, while the new instance gets SSH access
        injected via EXTRA_COMMANDS on every create_instance call.
        """
        if not self._ssh_public_key:
            return
        try:
            existing = await self.vast.list_ssh_keys()
            for key in existing:
                pubkey_str = key.get("public_key", "") or ""
                if "eased-orchestrator" in pubkey_str:
                    key_id = key.get("id")
                    if key_id is not None:
                        await self.vast.delete_ssh_key(int(key_id))
                        log.info(
                            "worker_manager.ssh_keys.removed_old",
                            key_id=key_id,
                            key_prefix=pubkey_str[:40],
                        )
            result = await self.vast.add_ssh_key(self._ssh_public_key)
            log.info("worker_manager.ssh_keys.registered", result=result)
        except Exception as exc:
            log.warning("worker_manager.ssh_keys.manage_failed", error=repr(exc))

    # ── SSH log fetching ──────────────────────────────────────────────────

    async def _fetch_vllm_logs_ssh(
        self,
        instance: dict,
        worker: Worker,
        lines: int = 150,
    ) -> Optional[str]:
        """
        SSH into the instance and collect debug logs.

        Fetches (in a single connection):
          • /var/log/onstart.log  — vast.ai startup log: EXTRA_COMMANDS output,
                                    vLLM patch application, entrypoint invocation.
          • /tmp/vllm.log         — vLLM process stdout/stderr (tee'd by onstart.sh).

        Returns the combined log text, or None if SSH is unavailable
        (no key generated, instance not reachable, or port not mapped yet).
        """
        if self._ssh_key is None:
            return None

        addr = self.vast.extract_ssh_address(instance)
        if not addr:
            log.debug(
                "worker_manager.ssh_logs.no_ssh_port",
                worker_id=worker.worker_id,
                instance_id=worker.instance_id,
            )
            return None

        host, port = addr
        log.info(
            "worker_manager.ssh_logs.connecting",
            worker_id=worker.worker_id,
            host=host,
            port=port,
        )
        # Single command: tail both log files and label each section clearly.
        cmd = (
            f"echo '=== onstart.log (last {lines} lines) ===';"
            f" tail -n {lines} /var/log/onstart.log 2>/dev/null"
            " || echo '(no /var/log/onstart.log yet)';"
            f" echo; echo '=== vllm.log (last {lines} lines) ===';"
            f" tail -n {lines} /tmp/vllm.log 2>/dev/null"
            " || echo '(no /tmp/vllm.log — entrypoint may not have started yet)'"
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

        running   = by_status.get("running", [])
        unhealthy = by_status.get("unhealthy", [])
        starting  = by_status.get("starting", [])
        pending   = by_status.get("pending", [])
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
        Fetch the last ~100 lines of container logs from vast.ai, cache the full
        text in the event store, and return a short excerpt for Discord embeds.
        """
        if not instance_id:
            return "(no instance ID)"
        try:
            text = await self.vast.get_instance_logs(instance_id, tail=100)
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
