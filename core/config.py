"""Configuration helpers for the platform.

Reads optional env vars for model provider endpoints, cloud training
hints, and LOW_RAM_MODE preferences. Validates defensively and never
logs secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from core.errors import ConfigError
from core.memory import DEFAULT_LIMITS


@dataclass
class Config:
    """Runtime configuration surface.

    All secrets must be supplied via env vars set through the platform UI
    (Settings -> Environment) or the workspace .env / .env.local files.
    This module never defaults any secret and never logs them.
    """

    # Model provider hints (optional)
    local_provider_name: str = "default_local"
    cloud_provider_name: str = "default_cloud"

    # Cloud training (optional backend)
    cloud_training_endpoint: Optional[str] = None
    cloud_training_enabled: bool = False

    # Cost / latency guards
    max_inference_latency_ms: int = 200
    max_cloud_cost_usd_per_1k_tokens: float = 1.0

    # LOW_RAM_MODE preference
    low_ram_mode_forced: bool = False
    low_ram_mode_auto_detect: bool = True

    def load_from_env(self) -> "Config":
        """Populate optional settings from env vars without touching secrets."""
        c = Config()

        name = os.environ.get("FORTNITE_AI_LOCAL_PROVIDER")
        if name:
            c.local_provider_name = name

        name = os.environ.get("FORTNITE_AI_CLOUD_PROVIDER")
        if name:
            c.cloud_provider_name = name

        endpoint = os.environ.get("FORTNITE_AI_CLOUD_TRAINING_ENDPOINT")
        if endpoint:
            c.cloud_training_endpoint = endpoint
            c.cloud_training_enabled = True

        latency = os.environ.get("FORTNITE_AI_MAX_INFERENCE_LATENCY_MS")
        if latency:
            try:
                c.max_inference_latency_ms = int(latency)
            except ValueError:
                raise ConfigError("FORTNITE_AI_MAX_INFERENCE_LATENCY_MS must be an integer")

        cost = os.environ.get("FORTNITE_AI_MAX_CLOUD_COST_USD_PER_1K")
        if cost:
            try:
                c.max_cloud_cost_usd_per_1k_tokens = float(cost)
            except ValueError:
                raise ConfigError("FORTNITE_AI_MAX_CLOUD_COST_USD_PER_1K must be a float")

        forced = os.environ.get("FORTNITE_AI_LOW_RAM_MODE")
        if forced is not None:
            c.low_ram_mode_forced = forced.strip().lower() in ("1", "true", "yes")

        auto = os.environ.get("FORTNITE_AI_LOW_RAM_MODE_AUTO")
        if auto is not None:
            c.low_ram_mode_auto_detect = auto.strip().lower() not in ("0", "false", "no")

        if c.max_inference_latency_ms <= 0:
            raise ConfigError("max_inference_latency_ms must be > 0")
        if c.max_cloud_cost_usd_per_1k_tokens < 0:
            raise ConfigError("max_cloud_cost_usd_per_1k_tokens must be >= 0")

        return c

    def low_ram_limits(self) -> dict[str, dict[str, int | float]]:
        """Return aggressive memory limits when LOW_RAM_MODE is active."""
        from core.low_ram import DEFAULT_LOW_RAM_LIMITS
        return dict(DEFAULT_LOW_RAM_LIMITS)

    def is_low_ram_mode(self, detected: bool) -> bool:
        return self.low_ram_mode_forced or (self.low_ram_mode_auto_detect and detected)


def validate_config(cfg: Config) -> None:
    """Raise ConfigError for invalid configuration."""
    if cfg.max_inference_latency_ms <= 0:
        raise ConfigError("max_inference_latency_ms must be > 0")
    if cfg.max_cloud_cost_usd_per_1k_tokens < 0:
        raise ConfigError("max_cloud_cost_usd_per_1k_tokens must be >= 0")
