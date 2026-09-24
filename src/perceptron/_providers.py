"""Provider and model registry, and how each surface picks its provider and model.

The legacy surfaces (``perceive``, the helpers, ``Client.generate/stream``) use ``settings().provider``, which keeps the
fal auto-detect of ``config._from_env``. The message API, files, models and multilook use :func:`surface_provider_cfg`:
the provider the caller chose explicitly, otherwise ``perceptron``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from .errors import INVALID_REASONING_EFFORT, MODEL_RENAMED, UNSUPPORTED_PROVIDER_FEATURE, BadRequestError

# The config module itself: `perceptron.config` is shadowed by the `config()` function on the package, and its
# `_explicit_fields` / `_global_settings` are rebound by `config()`, so read them through the module at call time.
_config = importlib.import_module(".config", __package__)

PERCEPTRON_PROVIDER = "perceptron"

# The `reasoning_effort` tiers the API accepts, sent as the top-level request field.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")


def _normalize_reasoning_effort(value: Any) -> str | None:
    """Lower-case a `reasoning_effort` tier, or raise before any request for a value the API rejects."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in REASONING_EFFORTS:
        raise BadRequestError(
            f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}; got {value!r}.",
            code=INVALID_REASONING_EFFORT,
        )
    return normalized


_PROVIDER_CONFIG = {
    "fal": {
        "base_url": "https://fal.run",
        "path": "/perceptron/isaac-01/openai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Key ",
        "env_keys": ["FAL_KEY", "PERCEPTRON_API_KEY"],
        "default_model": "isaac-0.1",
        "supported_models": ["isaac-0.1"],
        "models": {
            "isaac-0.1": {"reasoning": False, "skip_structured_hints": False},
        },
        "stream": True,
    },
    "perceptron": {
        "base_url": "https://api.perceptron.inc/v1",
        "path": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "env_keys": ["PERCEPTRON_API_KEY"],
        "default_model": "perceptron-mk1.5",
        # The ids the SDK knows capabilities for. The gateway is authoritative (some ids are feature-gated), so other
        # ids pass through and get permissive capability defaults.
        "supported_models": [
            "isaac-0.1",
            "isaac-0.2-1b",
            "isaac-0.2-2b-preview",
            "isaac-0.3-fast",
            "perceptron-mk1",
            "perceptron-mk1.5",
        ],
        "accepts_unknown_models": True,
        "models": {
            "isaac-0.1": {"reasoning": False, "skip_structured_hints": False},
            "isaac-0.2-1b": {"reasoning": True, "skip_structured_hints": False},
            "isaac-0.2-2b-preview": {"reasoning": True, "skip_structured_hints": False},
            "isaac-0.3-fast": {"reasoning": True, "skip_structured_hints": False},
            "perceptron-mk1": {"reasoning": True, "skip_structured_hints": False},
            "perceptron-mk1.5": {"reasoning": True, "skip_structured_hints": False},
        },
        "stream": True,
    },
}

# Ids renamed before release. The gateway no longer resolves the old id, so fail clearly instead of aliasing.
_RENAMED_MODELS = {"perceptron-mk1.5-preview": "perceptron-mk1.5"}

_SELECT_PERCEPTRON = 'configure(provider="perceptron") or PERCEPTRON_PROVIDER=perceptron'


def _select_model(
    provider_cfg: dict[str, Any],
    requested_model: str | None,
    *,
    provider_name: str | None = None,
) -> str | None:
    model = requested_model or provider_cfg.get("default_model")
    renamed = _RENAMED_MODELS.get(model) if isinstance(model, str) else None
    if renamed:
        raise BadRequestError(f"Model '{model}' was renamed to {renamed}; use model='{renamed}'.", code=MODEL_RENAMED)
    supported = provider_cfg.get("supported_models")
    provider_label = provider_name or provider_cfg.get("name") or "unknown"
    if supported and model and model not in supported and not provider_cfg.get("accepts_unknown_models"):
        message = f"Model '{model}' is not supported for provider='{provider_label}'. Allowed: {', '.join(supported)}"
        if model in _PROVIDER_CONFIG[PERCEPTRON_PROVIDER]["supported_models"]:
            message += (
                f". '{model}' is served by the Perceptron API: select provider 'perceptron' with {_SELECT_PERCEPTRON}."
            )
        raise BadRequestError(message)
    return model


def _pop_and_resolve_model(provider_cfg: dict[str, Any], gen_kwargs: dict[str, Any]) -> str:
    requested_model = gen_kwargs.pop("model", None)
    resolved = _select_model(provider_cfg, requested_model)
    if resolved:
        return resolved
    default_model = provider_cfg.get("default_model")
    if default_model:
        return default_model
    provider_label = provider_cfg.get("name") or "unknown"
    raise BadRequestError(
        f"No model configured for provider '{provider_label}'. Specify a model explicitly or configure a default."
    )


def _resolve_provider(provider: str | None) -> dict:
    provider = provider or "fal"
    provider_lc = provider.lower() if isinstance(provider, str) else provider
    if provider_lc not in _PROVIDER_CONFIG:
        raise BadRequestError(f"Unsupported provider: {provider}")
    return {"name": provider_lc, **_PROVIDER_CONFIG[provider_lc]}


def explicit_provider(override: str | None = None) -> str | None:
    """The provider the caller chose, ignoring the legacy fal auto-detect.

    Order: a ``Client(provider=...)`` override, ``configure(provider=...)`` / ``config(provider=...)``, then the
    ``PERCEPTRON_PROVIDER`` environment variable. None when nothing chose one.
    """
    if override:
        return override
    if "provider" in _config._explicit_fields:
        return _config._global_settings.provider or None
    return os.getenv("PERCEPTRON_PROVIDER") or None


def surface_provider_cfg(client: Any, *, feature: str | None = None) -> dict[str, Any]:
    """Provider config for the message API, files, models and multilook.

    Uses the explicit provider chosen when the client was built (like its settings), otherwise ``perceptron``.
    ``settings.base_url`` applies only when it was configured for this provider: an explicit provider, or settings whose
    provider is ``perceptron``. With ``feature`` set (e.g. ``"Files"``), a non-``perceptron`` provider raises
    ``unsupported_provider_feature``.
    """
    settings = client._settings
    # Clients record their explicit provider when built, alongside `_settings`; other objects resolve it now.
    explicit = client._provider_override if hasattr(client, "_provider_override") else explicit_provider()
    cfg = _resolve_provider(explicit or PERCEPTRON_PROVIDER)
    if feature is not None and cfg["name"] != PERCEPTRON_PROVIDER:
        raise BadRequestError(
            f"{feature} is only available on provider 'perceptron' (the configured provider is '{cfg['name']}'). "
            f"Select it with {_SELECT_PERCEPTRON}.",
            code=UNSUPPORTED_PROVIDER_FEATURE,
        )
    configured_provider = settings.provider.lower() if isinstance(settings.provider, str) else None
    if settings.base_url and (explicit is not None or configured_provider == PERCEPTRON_PROVIDER):
        cfg["base_url"] = settings.base_url
    return cfg


def surface_model(provider_cfg: dict[str, Any], model: str | None, settings: Any) -> str | None:
    """``model`` ?? ``settings.model`` ?? the provider default, checked against the registry."""
    requested = model if model is not None else settings.model
    return _select_model(provider_cfg, requested, provider_name=provider_cfg.get("name"))
