"""Concrete model providers for local, cloud, and fallback routing."""

from core.models.providers.local import DummyLocalProvider
from core.models.providers.cloud import DummyCloudProvider
from core.models.providers.fallback import FallbackProvider

__all__ = ["DummyLocalProvider", "DummyCloudProvider", "FallbackProvider"]
