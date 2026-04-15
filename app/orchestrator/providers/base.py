from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Normalized offer / instance data ─────────────────────────────────────────

@dataclass
class GPUOffer:
    """Provider-agnostic description of a GPU rental offer."""
    offer_id:     str
    provider:     str
    price_per_hr: float
    gpu_name:     str
    gpu_ram_gb:   float          # VRAM per GPU in GB
    num_gpus:     int = 1        # number of GPUs in this offer
    specs:        dict[str, Any] = field(default_factory=dict)
    raw:          dict[str, Any] = field(default_factory=dict)

    @property
    def total_gpu_ram_gb(self) -> float:
        """Total VRAM across all GPUs in the offer."""
        return self.gpu_ram_gb * self.num_gpus


@dataclass
class InstanceInfo:
    """
    Provider-agnostic snapshot of a running (or transitioning) GPU instance.

    *host* and *port* are the vLLM endpoint (None until the container is up and
    the provider has mapped the port).  *ssh_host* / *ssh_port* are the SSH
    endpoint (None when SSH is not available or not yet mapped).
    """
    instance_id:   str
    provider:      str
    actual_status: str           # e.g. "running", "exited", "offline"
    cur_state:     str  = ""     # provider-specific sub-state
    status_msg:    str  = ""     # human-readable status / error text
    next_state:    str  = ""     # where the instance is transitioning to
    gpu_name:      Optional[str]   = None
    gpu_ram_gb:    Optional[float] = None
    label:         str  = ""
    image:         str  = ""
    host:          Optional[str] = None
    port:          Optional[int] = None
    ssh_host:      Optional[str] = None
    ssh_port:      Optional[int] = None
    specs:         dict[str, Any] = field(default_factory=dict)
    raw:           dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """True when the instance is unrecoverably gone."""
        return self.actual_status in {"exited", "offline", "deleted", "failed", "inactive"}

    @property
    def is_outbid(self) -> bool:
        """True when the instance was stopped because a higher bid won."""
        keywords = ("outbid", "preempted", "overbid")
        haystack = (self.status_msg + " " + self.cur_state).lower()
        return any(kw in haystack for kw in keywords)


@dataclass
class CreateConfig:
    """Parameters the orchestrator passes to a provider when creating an instance."""
    worker_api_key: str
    on_demand:      bool           # True = on-demand / reserved, False = interruptible / spot
    label:          str
    price:          float
    ssh_public_key: Optional[str] = None


# ── Abstract provider ─────────────────────────────────────────────────────────

class GPUProvider(ABC):
    """
    Abstract GPU provider.  Implement this to add a new marketplace (RunPod,
    TensorDock, etc.) without touching the orchestrator's core logic.
    """
    name: str = ""
    # Set to True in providers that support SSH key management and SSH log
    # fetching (currently only vastai).
    supports_ssh: bool = False

    # ── Offer discovery ───────────────────────────────────────────────────

    @abstractmethod
    async def search_offers(self, on_demand: bool = False) -> list[GPUOffer]:
        """Return GPU offers sorted ascending by price."""

    @abstractmethod
    def get_market_price(self, offers: list[GPUOffer]) -> float:
        """Return a reference market price (e.g. median) from a list of offers."""

    # ── Instance lifecycle ────────────────────────────────────────────────

    @abstractmethod
    async def create_instance(self, offer: GPUOffer, config: CreateConfig) -> str:
        """
        Rent the given offer and return the new instance_id.
        Raises on failure — caller must handle.
        """

    @abstractmethod
    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        """Return current state for one instance, or None if it no longer exists."""

    @abstractmethod
    async def list_instances(self) -> list[InstanceInfo]:
        """Return all instances owned by this account."""

    @abstractmethod
    async def destroy_instance(self, instance_id: str) -> None:
        """Terminate and release the instance.  Must handle 404 / already-gone gracefully."""

    @abstractmethod
    async def change_bid(self, instance_id: str, new_price: float) -> bool:
        """
        Attempt to raise the spot bid on an existing instance.
        Returns True if accepted, False if the instance is already gone.
        Only meaningful for interruptible providers; on-demand implementations
        may simply return False.
        """

    @abstractmethod
    async def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        """Fetch recent container stdout/stderr.  Returns descriptive string on failure."""

    # ── Optional SSH support (default: not available) ─────────────────────

    async def list_ssh_keys(self) -> list[dict[str, Any]]:
        return []

    async def add_ssh_key(self, pubkey_text: str) -> dict[str, Any]:
        return {}

    async def delete_ssh_key(self, key_id: int) -> None:
        pass

    async def attach_ssh_key(self, instance_id: str, pubkey_text: str) -> dict[str, Any]:
        """Attach a public key to a running instance's authorized_keys (provider-side)."""
        return {}
