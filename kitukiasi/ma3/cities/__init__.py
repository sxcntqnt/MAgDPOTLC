from __future__ import annotations

from .base import CityAdapter, CityContext
from .nairobi import NairobiAdapter

_REGISTRY = {
    "nairobi": NairobiAdapter,
}


def get_adapter(name: str, **kwargs) -> CityAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown city adapter: {name}; known: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def register_adapter(name: str, cls) -> None:
    _REGISTRY[name] = cls


__all__ = ["CityAdapter", "CityContext", "NairobiAdapter", "get_adapter", "register_adapter"]
