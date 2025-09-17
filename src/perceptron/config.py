from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any
import os


@dataclass
class Settings:
    """SDK configuration with environment overlay.

    Provider options: currently only "fal" is bundled, but additional providers can be
    registered by extending `_PROVIDER_CONFIG` in `perceptron.client`.
    """

    base_url: str | None = None
    api_key: str | None = None
    provider: str | None = None  # currently "fal"; extensible for custom transports

    timeout: float = 60.0
    retries: int = 3

    strict: bool = False
    allow_multiple: bool = False
    warn_on_implicit_anchor: bool = True

    # generation defaults
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int | None = None

    # parsing/streaming knobs
    max_buffer_bytes: int | None = None

    # image handling
    resize_max_side: int | None = None
    auto_coerce_paths: bool = False


_global_settings = Settings()
_stack: list[Settings] = []


def _from_env(s: Settings) -> Settings:
    base_url = os.getenv("PERCEPTRON_BASE_URL", s.base_url)
    api_key = os.getenv("PERCEPTRON_API_KEY", s.api_key)
    provider = os.getenv("PERCEPTRON_PROVIDER", s.provider)
    # Providers
    if provider is None and (os.getenv("FAL_KEY") or os.getenv("PERCEPTRON_API_KEY")):
        provider = "fal"
    return Settings(
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        timeout=s.timeout,
        retries=s.retries,
        strict=s.strict,
        allow_multiple=s.allow_multiple,
        warn_on_implicit_anchor=s.warn_on_implicit_anchor,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        top_p=s.top_p,
        top_k=s.top_k,
        max_buffer_bytes=s.max_buffer_bytes,
        resize_max_side=s.resize_max_side,
        auto_coerce_paths=s.auto_coerce_paths,
    )


def configure(**kwargs: Any) -> None:
    """Configure global SDK defaults.

    Example:
        configure(provider="fal", timeout=60)
    """
    global _global_settings
    for k, v in kwargs.items():
        if not hasattr(_global_settings, k):
            raise AttributeError(f"Unknown setting: {k}")
        setattr(_global_settings, k, v)


@contextmanager
def config(**kwargs: Any):
    """Temporarily apply settings within a context."""
    global _global_settings
    _stack.append(Settings(**asdict(_global_settings)))
    try:
        configure(**kwargs)
        yield
    finally:
        prev = _stack.pop()
        _global_settings = prev


def settings() -> Settings:
    """Return the effective merged settings (env overlaid on current)."""
    return _from_env(_global_settings)
