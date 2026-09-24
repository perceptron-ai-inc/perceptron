"""Provider and model registry, and how each surface picks its provider, model and API key.

Every surface resolves the provider with one rule (``config._from_env``, :func:`_provider_key`): the provider the caller
chose (``Client(provider=...)``, ``configure``/``config``, ``PERCEPTRON_PROVIDER``, a per-call ``provider=``, the CLI's
``--provider``), otherwise ``fal`` only when ``FAL_KEY`` is set and ``PERCEPTRON_API_KEY`` is not, otherwise
``perceptron``. :func:`provider_api_key` never gives fal a key read from ``PERCEPTRON_API_KEY``. Files, models and
multilook exist only on ``perceptron`` (:func:`surface_provider_cfg`). A configured ``base_url`` applies to every surface.
"""

from __future__ import annotations

import os
from typing import Any

from .config import _default_provider
from .errors import INVALID_REASONING_EFFORT, MODEL_RENAMED, UNSUPPORTED_PROVIDER_FEATURE, BadRequestError

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
        "env_keys": ["FAL_KEY"],
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
                f". '{model}' is served by the Perceptron API: select provider 'perceptron' with {_SELECT_PERCEPTRON}"
            )
            if provider_label == "fal":
                message += " (fal is used when you choose it, or when FAL_KEY is set and PERCEPTRON_API_KEY is not)"
            message += "."
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


def _provider_key(provider: str | None) -> str:
    """The registry key of ``provider`` (case-insensitive).

    No provider means the default rule, ``config._default_provider``: ``fal`` only when ``FAL_KEY`` is set and
    ``PERCEPTRON_API_KEY`` is not, otherwise ``perceptron``.
    """
    provider = provider or _default_provider()
    return provider.lower() if isinstance(provider, str) else provider


def _resolve_provider(provider: str | None) -> dict:
    provider_lc = _provider_key(provider)
    if provider_lc not in _PROVIDER_CONFIG:
        raise BadRequestError(f"Unsupported provider: {provider}")
    return {"name": provider_lc, **_PROVIDER_CONFIG[provider_lc]}


def provider_api_key(settings: Any, provider_cfg: dict[str, Any]) -> str | None:
    """The API key to send to the provider: ``settings.api_key``, else the provider's ``env_keys``.

    A key ``settings()`` read from an environment variable only counts when the provider reads that variable too, so a
    ``PERCEPTRON_API_KEY`` is never sent to fal (whose only env key is ``FAL_KEY``); a key set in code
    (``configure(api_key=...)``, ``Client(api_key=...)``) goes to whichever provider is in use.
    """
    env_keys = provider_cfg.get("env_keys", [])
    source = getattr(settings, "_api_key_env", None)
    token = settings.api_key if source is None or source in env_keys else None
    for env in env_keys:
        token = token or os.getenv(env)
    return token


def missing_api_key_message(provider_cfg: dict[str, Any]) -> str:
    """Why no key was found for the provider, and how to set one."""
    name = provider_cfg.get("name")
    env_keys = provider_cfg.get("env_keys", [])
    sources = [*env_keys, "configure(api_key=...)"]
    message = f"No API key for provider '{name}'. Set {' or '.join(sources)}."
    if "PERCEPTRON_API_KEY" not in env_keys and os.getenv("PERCEPTRON_API_KEY"):
        message += (
            f" PERCEPTRON_API_KEY is only sent to the Perceptron API, never to provider '{name}'; to use it, "
            f"select provider 'perceptron' with {_SELECT_PERCEPTRON}."
        )
    return message


def surface_provider_cfg(client: Any, *, feature: str | None = None) -> dict[str, Any]:
    """Provider config for the message API, files, models and multilook.

    Uses the provider the client resolved when it was built (``client._settings.provider``, see
    :func:`_provider_key`), so one client never switches provider or sends its key to another host. A configured
    ``settings.base_url`` (``Client(base_url=...)``, ``configure``/``config``, ``PERCEPTRON_BASE_URL``) replaces the
    provider's base URL, as it does for ``Client.generate``: requests and the API key go where the caller pointed them.
    With ``feature`` set (e.g. ``"Files"``), a provider other than ``perceptron`` (chosen or auto-selected) raises
    ``unsupported_provider_feature``.
    """
    settings = client._settings
    cfg = _resolve_provider(settings.provider)
    if feature is not None and cfg["name"] != PERCEPTRON_PROVIDER:
        raise BadRequestError(
            f"{feature} is only available on provider 'perceptron'; this client uses provider '{cfg['name']}'. "
            f"Select the Perceptron API with {_SELECT_PERCEPTRON}.",
            code=UNSUPPORTED_PROVIDER_FEATURE,
        )
    if settings.base_url:
        cfg["base_url"] = settings.base_url
    return cfg


def surface_model(provider_cfg: dict[str, Any], model: str | None, settings: Any) -> str | None:
    """``model`` ?? ``settings.model`` ?? the provider default, checked against the registry."""
    requested = model if model is not None else settings.model
    return _select_model(provider_cfg, requested, provider_name=provider_cfg.get("name"))
