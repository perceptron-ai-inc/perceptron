"""Models API: ``client.models`` (provider ``perceptron`` only; on ``fal`` it raises ``unsupported_provider_feature``).

The SDK never calls it on its own: the endpoint allows 30 requests per minute per organization, so fetch what you need
once and keep it. ``extended=True`` adds capabilities, modalities, limits and pricing (:class:`ModelInfo`)::

    info = client.models.retrieve("perceptron-mk1.5", extended=True)
    info.supports("tool_calling"), info.max_output_tokens

The capability, modality, output-format and sampling-parameter lists are open: unknown names are kept, not rejected.
Multilook support is not listed; ``perceptron-mk1`` and ``perceptron-mk1.5`` have it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import _transport
from ._providers import surface_provider_cfg
from .chat import _int_or_none, _str_or_none

__all__ = ["AsyncModels", "Model", "ModelInfo", "ModelReasoning", "Models"]

_FEATURE = "The Models API"


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _str_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _model_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": _str_or_none(data.get("id")) or "",
        "created": _int_or_none(data.get("created")),
        "owned_by": _str_or_none(data.get("owned_by")),
        "object": _str_or_none(data.get("object")) or "model",
    }


@dataclass
class Model:
    """A model id with its creation time (Unix seconds) and owner."""

    id: str
    created: int | None = None
    owned_by: str | None = None
    object: str = "model"

    @classmethod
    def from_dict(cls, data: Any) -> Model:
        return cls(**_model_fields(data if isinstance(data, Mapping) else {}))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "object": self.object, "created": self.created, "owned_by": self.owned_by}


@dataclass
class ModelReasoning:
    """Whether the model can reason, and whether it always does."""

    supported: bool | None = None
    always_enabled: bool | None = None

    @classmethod
    def from_dict(cls, data: Any) -> ModelReasoning | None:
        if not isinstance(data, Mapping):
            return None
        return cls(
            supported=_bool_or_none(data.get("supported")), always_enabled=_bool_or_none(data.get("always_enabled"))
        )

    def to_dict(self) -> dict[str, Any]:
        return {"supported": self.supported, "always_enabled": self.always_enabled}


@dataclass
class ModelInfo(Model):
    """A model's extended metadata (``extended=True``). ``credits_per_million_*_tokens / 1e6`` is USD per million
    tokens; ``raw`` is the response object, unknown keys included."""

    name: str | None = None
    description: str | None = None
    capabilities: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    output_formats: list[str] = field(default_factory=list)
    sampling_parameters: list[str] = field(default_factory=list)
    reasoning: ModelReasoning | None = None
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    credits_per_million_input_tokens: int | None = None
    credits_per_million_output_tokens: int | None = None
    credits_per_million_cache_read_tokens: int | None = None
    early_access: bool | None = None
    raw: dict | None = None

    def supports(self, capability: str) -> bool:
        """Whether ``capability`` (e.g. ``"tool_calling"``, ``"regex"``, ``"response_format_json_schema"``) is listed."""
        return capability in self.capabilities

    @classmethod
    def from_dict(cls, data: Any) -> ModelInfo:
        data = data if isinstance(data, Mapping) else {}
        return cls(
            **_model_fields(data),
            name=_str_or_none(data.get("name")),
            description=_str_or_none(data.get("description")),
            capabilities=_str_list(data.get("capabilities")),
            modalities=_str_list(data.get("modalities")),
            output_formats=_str_list(data.get("output_formats")),
            sampling_parameters=_str_list(data.get("sampling_parameters")),
            reasoning=ModelReasoning.from_dict(data.get("reasoning")),
            max_context_tokens=_int_or_none(data.get("max_context_tokens")),
            max_output_tokens=_int_or_none(data.get("max_output_tokens")),
            credits_per_million_input_tokens=_int_or_none(data.get("credits_per_million_input_tokens")),
            credits_per_million_output_tokens=_int_or_none(data.get("credits_per_million_output_tokens")),
            credits_per_million_cache_read_tokens=_int_or_none(data.get("credits_per_million_cache_read_tokens")),
            early_access=_bool_or_none(data.get("early_access")),
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out.update(
            name=self.name,
            capabilities=list(self.capabilities),
            modalities=list(self.modalities),
            output_formats=list(self.output_formats),
            sampling_parameters=list(self.sampling_parameters),
            reasoning=self.reasoning.to_dict() if self.reasoning is not None else None,
            max_context_tokens=self.max_context_tokens,
            max_output_tokens=self.max_output_tokens,
        )
        if self.description is not None:
            out["description"] = self.description
        out.update(
            credits_per_million_input_tokens=self.credits_per_million_input_tokens,
            credits_per_million_output_tokens=self.credits_per_million_output_tokens,
            credits_per_million_cache_read_tokens=self.credits_per_million_cache_read_tokens,
            early_access=self.early_access,
        )
        return out


def _model_path(model_id: Any) -> str:
    return _transport.resource_path("models", model_id, "model_id")


def _params(extended: bool) -> dict[str, str] | None:
    return {"extended": "true"} if extended else None


def _parse(data: Any, extended: bool) -> Model:
    return ModelInfo.from_dict(data) if extended else Model.from_dict(data)


def _parse_list(payload: Any, extended: bool) -> list[Model]:
    items = payload.get("data") if isinstance(payload, Mapping) else None
    return [_parse(item, extended) for item in items] if isinstance(items, list) else []


class Models:
    """``client.models``: the models this organization can use."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def list(self, *, extended: bool = False) -> list[Model]:
        """The available models (``GET /models``), as :class:`ModelInfo` with ``extended=True``, else :class:`Model`."""
        cfg = surface_provider_cfg(self._client, feature=_FEATURE)
        payload, _ = _transport.request_json(self._client, "GET", "/models", params=_params(extended), provider_cfg=cfg)
        return _parse_list(payload, extended)

    def retrieve(self, model_id: str, *, extended: bool = False) -> Model:
        """One model (``GET /models/{model_id}``); ``NotFoundError`` (code ``model_not_found``) for an unknown id."""
        cfg = surface_provider_cfg(self._client, feature=_FEATURE)
        path = _model_path(model_id)
        payload, _ = _transport.request_json(self._client, "GET", path, params=_params(extended), provider_cfg=cfg)
        return _parse(payload, extended)


class AsyncModels:
    """``AsyncClient.models``: the async :class:`Models`."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(self, *, extended: bool = False) -> list[Model]:
        """Async :meth:`Models.list`."""
        cfg = surface_provider_cfg(self._client, feature=_FEATURE)
        payload, _ = await _transport.arequest_json(
            self._client, "GET", "/models", params=_params(extended), provider_cfg=cfg
        )
        return _parse_list(payload, extended)

    async def retrieve(self, model_id: str, *, extended: bool = False) -> Model:
        """Async :meth:`Models.retrieve`."""
        cfg = surface_provider_cfg(self._client, feature=_FEATURE)
        path = _model_path(model_id)
        payload, _ = await _transport.arequest_json(
            self._client, "GET", path, params=_params(extended), provider_cfg=cfg
        )
        return _parse(payload, extended)
