from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Settings:
    """SDK configuration with environment overlay.

    Providers: ``"perceptron"`` (the Perceptron API, ``https://api.perceptron.inc/v1``; default model
    ``perceptron-mk1.5``) and ``"fal"`` (``isaac-0.1`` only). The registry is ``perceptron._providers._PROVIDER_CONFIG``.

    One rule picks the provider for every surface (the helpers, ``perceive``, ``Client``, the message API, files,
    models, multilook and the CLI): the provider you chose (``Client(provider=...)``, ``configure``/``config``,
    ``PERCEPTRON_PROVIDER``, a per-call ``provider=`` or ``--provider``); otherwise ``"fal"`` only when ``FAL_KEY`` is
    set and ``PERCEPTRON_API_KEY`` is not; otherwise ``"perceptron"`` (see :func:`_default_provider`).

    ``api_key`` is the key you set in code, else ``PERCEPTRON_API_KEY``. Provider ``"fal"`` authenticates with a key
    set in code or ``FAL_KEY``, never with one read from ``PERCEPTRON_API_KEY``. Files, models and multilook exist only
    on provider ``"perceptron"``. A configured ``base_url`` (``PERCEPTRON_BASE_URL``) replaces the provider's base URL
    on every surface.
    """

    base_url: str | None = None  # every surface uses it when set; for provider "perceptron" include the /v1 prefix
    api_key: str | None = None
    provider: str | None = None  # "perceptron" or "fal"; None = the default rule (see above)
    model: str | None = None  # None = the provider's default model

    timeout: float = 125.0  # seconds per request, past the gateway's 120 s chat budget (multilook waits >= 305 s)
    retries: int = 3  # accepted for compatibility; the SDK does not retry requests

    strict: bool = False
    allow_multiple: bool = False
    warn_on_implicit_anchor: bool = True

    # generation defaults (None = use API server defaults)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # parsing/streaming knobs
    max_buffer_bytes: int | None = None

    # image handling
    resize_max_side: int | None = None
    auto_coerce_paths: bool = False

    # Not a dataclass field: the environment variable `settings()` read `api_key` from, or None when it was set in
    # code. A provider only receives such a key when it reads that variable too, so PERCEPTRON_API_KEY never reaches
    # fal (see `_providers.provider_api_key`).
    _api_key_env = None


_global_settings = Settings()
_stack: list[tuple[Settings, set[str]]] = []
_defaults = Settings()  # Track default values
_explicit_fields: set[str] = set()  # Track which fields have been explicitly configured


def _default_provider() -> str:
    """The provider when none was chosen: ``"fal"`` only when ``FAL_KEY`` is set and ``PERCEPTRON_API_KEY`` is not."""
    if os.getenv("FAL_KEY") and not os.getenv("PERCEPTRON_API_KEY"):
        return "fal"
    return "perceptron"


def _from_env(s: Settings) -> Settings:
    # Only read from environment for fields that haven't been explicitly configured
    base_url = s.base_url if "base_url" in _explicit_fields else os.getenv("PERCEPTRON_BASE_URL", s.base_url)
    env_api_key = None if "api_key" in _explicit_fields else os.getenv("PERCEPTRON_API_KEY")
    api_key = s.api_key if env_api_key is None else env_api_key
    provider = s.provider if "provider" in _explicit_fields else os.getenv("PERCEPTRON_PROVIDER", s.provider)
    model = s.model if "model" in _explicit_fields else os.getenv("PERCEPTRON_MODEL", s.model)

    merged = Settings(
        base_url=base_url,
        api_key=api_key,
        provider=provider or _default_provider(),
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
        frequency_penalty=s.frequency_penalty,
        presence_penalty=s.presence_penalty,
        max_buffer_bytes=s.max_buffer_bytes,
        resize_max_side=s.resize_max_side,
        auto_coerce_paths=s.auto_coerce_paths,
    )
    if env_api_key is not None:
        merged._api_key_env = "PERCEPTRON_API_KEY"
    return merged


def configure(**kwargs: Any) -> None:
    """Configure global SDK defaults. A configured field wins over its environment variable.

    Example:
        configure(api_key="sk_live_...", model="perceptron-mk1.5", timeout=180)
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
