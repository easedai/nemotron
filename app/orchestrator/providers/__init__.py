from .base import CreateConfig, GPUOffer, GPUProvider, InstanceInfo
from .lambdalabs import LambdaLabsProvider
from .runpod import RunPodProvider
from .salad import SaladProvider
from .tensordock import TensorDockProvider
from .vastai import VastAIProvider

# Registry mapping provider name → factory callable.
# To add a new provider: import its class and add an entry here.
_REGISTRY: dict[str, type[GPUProvider]] = {
    "lambdalabs": LambdaLabsProvider,
    "runpod":     RunPodProvider,
    "salad":      SaladProvider,
    "tensordock": TensorDockProvider,
    "vastai":     VastAIProvider,
}


def get_provider(name: str = "vastai") -> GPUProvider:
    """Return a GPUProvider implementation by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        supported = ", ".join(repr(k) for k in _REGISTRY)
        raise ValueError(f"Unknown GPU provider: {name!r}. Supported: {supported}")
    return cls()


__all__ = [
    "get_provider",
    "GPUProvider",
    "GPUOffer",
    "InstanceInfo",
    "CreateConfig",
    "LambdaLabsProvider",
    "RunPodProvider",
    "SaladProvider",
    "TensorDockProvider",
    "VastAIProvider",
]
