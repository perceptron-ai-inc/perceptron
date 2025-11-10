from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Settings:
    """SDK configuration with environment overlay.

    Provider options: currently only "fal" is bundled, but additional providers can be
    registered by extending `_PROVIDER_CONFIG` in `perceptron.client`.
    """

    base_url: str | None = None
    api_key: str | None = None
    provider: str | None = None  # currently "fal"; extensible for custom transports
    model: str | None = None

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
_stack: list[tuple[Settings, set[str]]] = []
_defaults = Settings()  # Track default values
_explicit_fields: set[str] = set()  # Track which fields have been explicitly configured


def _from_env(s: Settings) -> Settings:
    # Only read from environment for fields that haven't been explicitly configured
    base_url = s.base_url if "base_url" in _explicit_fields else os.getenv("PERCEPTRON_BASE_URL", s.base_url)
    api_key = s.api_key if "api_key" in _explicit_fields else os.getenv("PERCEPTRON_API_KEY", s.api_key)
    provider = s.provider if "provider" in _explicit_fields else os.getenv("PERCEPTRON_PROVIDER", s.provider)
    model = s.model if "model" in _explicit_fields else os.getenv("PERCEPTRON_MODEL", s.model)

    # Providers - only apply default logic if provider wasn't explicitly set
    if (
        provider is None
        and "provider" not in _explicit_fields
        and (os.getenv("FAL_KEY") or os.getenv("PERCEPTRON_API_KEY"))
    ):
        provider = "fal"

    return Settings(
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        model=model,
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
    global _global_settings, _explicit_fields
    for k, v in kwargs.items():
        if not hasattr(_global_settings, k):
            raise AttributeError(f"Unknown setting: {k}")
        setattr(_global_settings, k, v)
        # Track that this field was explicitly configured
        _explicit_fields.add(k)


@contextmanager
def config(**kwargs: Any):
    """Temporarily apply settings within a context."""
    global _global_settings, _explicit_fields
    _stack.append((Settings(**asdict(_global_settings)), _explicit_fields.copy()))
    try:
        configure(**kwargs)
        yield
    finally:
        prev_settings, prev_explicit = _stack.pop()
        _global_settings = prev_settings
        _explicit_fields = prev_explicit


def settings() -> Settings:
    """Return the effective merged settings (env overlaid on current)."""
    return _from_env(_global_settings)
