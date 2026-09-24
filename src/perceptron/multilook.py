"""Multilook: several prompts against one shared context in one request (``POST /chat/completions/multilook``).

Called through ``client.chat.completions.multilook(context=[...], prompts=[...])``. Each prompt is answered as
``[*context, {"role": "user", "content": prompt}]``; prompts never see each other, and the shared context is prefilled
once (``usage.cached_tokens``). Put media shared by every prompt in ``context``: ``asset_idx`` numbers the context's
assets first, then the prompt's own (``completion.asset_count``). DSL tags in the context must hold for every prompt,
so they anchor as if the prompt with the most media followed them: when that makes more than one asset, give each an
``image=``/``asset=``.

A failed prompt carries ``result.error`` while the others succeed; the request as a whole fails (an HTTP error, mapped
like any other) only when every prompt fails. Available on ``perceptron-mk1`` and ``perceptron-mk1.5``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import _transport
from ._lowering import count_assets
from ._providers import _normalize_reasoning_effort, surface_model, surface_provider_cfg
from .chat import (
    ChatCompletionMessage,
    Usage,
    _content_ledger,
    _generation_params,
    _int_or_none,
    _lower_content,
    _merge_extra_body,
    _normalize_messages,
    _normalize_vision_config,
    _str_or_none,
    _validate_temperature,
)
from .dsl.nodes import DSLNode
from .errors import INVALID_PARAMETER, INVALID_RESPONSE, UNSUPPORTED_PARAMETER, BadRequestError, ServerError
from .pointing.parser import AnnotationCollection, resolve_asset_idx

__all__ = [
    "MultilookCompletion",
    "MultilookPromptError",
    "MultilookResponse",
    "MultilookResult",
    "MultilookUsage",
    "acreate",
    "create",
]

MULTILOOK_PATH = "/chat/completions/multilook"
MAX_PROMPTS = 16
MAX_N = 8
MAX_COMPLETIONS = 64
# The gateway gives multilook 300 s (chat gets 120 s), so by default wait a little longer than that.
MIN_TIMEOUT = 305.0

_FEATURE = "Multilook"
# `interrupted` (a force-closed think budget) still carries a full answer.
_COMPLETE_FINISH_REASONS = frozenset({"stop", "interrupted"})


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass
class MultilookUsage(Usage):
    """Request-wide usage: the shared context counts once per prompt in ``prompt_tokens``; ``cached_tokens`` is the part
    served from the shared prefill (billed at the cache-read rate)."""


@dataclass
class MultilookPromptError:
    """Why one prompt failed. Branch on its presence, not on ``type``/``code``."""

    message: str
    type: str | None = None
    code: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> MultilookPromptError | None:
        if not isinstance(data, Mapping):
            return None
        return cls(
            message=_str_or_none(data.get("message")) or "",
            type=_str_or_none(data.get("type")),
            code=_str_or_none(data.get("code")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"message": self.message}
        if self.type is not None:
            out["type"] = self.type
        if self.code is not None:
            out["code"] = self.code
        return out


@dataclass
class MultilookCompletion:
    """One sampled answer to a prompt. ``asset_count`` is the number of media assets the model saw (the context's plus
    the prompt's): the ``asset_idx`` space of annotations in ``message.content``."""

    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None
    asset_count: int | None = None

    @property
    def text(self) -> str | None:
        return self.message.content

    @property
    def reasoning(self) -> str | None:
        return self.message.reasoning_content

    @property
    def complete(self) -> bool:
        """True for ``stop`` and ``interrupted``; ``length`` means the answer may be cut off."""
        return self.finish_reason in _COMPLETE_FINISH_REASONS

    def annotations(self, *, expects: str | None = None, strict: bool = False) -> AnnotationCollection:
        """The annotations in the answer (see :meth:`ChatCompletionMessage.annotations`)."""
        return self.message.annotations(expects=expects, strict=strict)

    def resolve_asset_idx(self, annotation: Any) -> int | None:
        """The asset an annotation from this answer refers to, among the context's and this prompt's: its own (or
        inherited) ``asset_idx``, else the last one (``asset_count - 1``). ``ValueError`` when the ``asset_idx`` is out
        of range; None without assets. Pass flattened annotations (from :meth:`annotations`)."""
        return resolve_asset_idx(annotation, self.asset_count)

    @classmethod
    def from_dict(cls, data: Any, position: int = 0, *, asset_count: int | None = None) -> MultilookCompletion:
        data = data if isinstance(data, Mapping) else {}
        index = _int_or_none(data.get("index"))
        return cls(
            index=position if index is None else index,
            message=ChatCompletionMessage.from_dict(data.get("message")),
            finish_reason=_str_or_none(data.get("finish_reason")),
            asset_count=asset_count,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"index": self.index, "message": self.message.to_dict()}
        if self.finish_reason is not None:
            out["finish_reason"] = self.finish_reason
        return out


@dataclass
class MultilookResult:
    """The outcome of one prompt: ``completions`` (``n`` of them) and ``usage`` on success, ``error`` on failure."""

    prompt_index: int
    completions: list[MultilookCompletion] | None = None
    usage: Usage | None = None
    error: MultilookPromptError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.completions is not None

    @classmethod
    def from_dict(cls, data: Any, position: int = 0, *, asset_counts: list[int] | None = None) -> MultilookResult:
        data = data if isinstance(data, Mapping) else {}
        prompt_index = _int_or_none(data.get("prompt_index"))
        prompt_index = position if prompt_index is None else prompt_index
        asset_count = asset_counts[prompt_index] if asset_counts and 0 <= prompt_index < len(asset_counts) else None
        raw_completions = data.get("completions")
        return cls(
            prompt_index=prompt_index,
            completions=(
                [MultilookCompletion.from_dict(c, j, asset_count=asset_count) for j, c in enumerate(raw_completions)]
                if isinstance(raw_completions, list)
                else None
            ),
            usage=Usage.from_dict(data.get("usage")),
            error=MultilookPromptError.from_dict(data.get("error")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"prompt_index": self.prompt_index}
        if self.completions is not None:
            out["completions"] = [completion.to_dict() for completion in self.completions]
        if self.usage is not None:
            # Per-prompt usage carries only `completion_tokens` on the wire; absent keys stay absent.
            out["usage"] = {key: value for key, value in self.usage.to_dict().items() if value is not None}
        if self.error is not None:
            out["error"] = self.error.to_dict()
        return out


@dataclass
class MultilookResponse:
    """A multilook response: one result per prompt, in prompt order. ``request_id`` is the ``x-trace-id`` response
    header and ``raw`` the response JSON. Retry only the ``failed`` prompts (by ``prompt_index``) if you need to."""

    id: str | None
    object: str | None
    model: str | None
    results: list[MultilookResult]
    usage: MultilookUsage | None = None
    request_id: str | None = None
    raw: dict | None = None

    @property
    def succeeded(self) -> list[MultilookResult]:
        return [result for result in self.results if result.ok]

    @property
    def failed(self) -> list[MultilookResult]:
        return [result for result in self.results if not result.ok]

    @classmethod
    def from_dict(
        cls, data: Any, *, request_id: str | None = None, asset_counts: list[int] | None = None
    ) -> MultilookResponse:
        data = data if isinstance(data, Mapping) else {}
        raw_results = data.get("results")
        return cls(
            id=_str_or_none(data.get("id")),
            object=_str_or_none(data.get("object")),
            model=_str_or_none(data.get("model")),
            results=(
                [MultilookResult.from_dict(r, i, asset_counts=asset_counts) for i, r in enumerate(raw_results)]
                if isinstance(raw_results, list)
                else []
            ),
            usage=MultilookUsage.from_dict(data.get("usage")),
            request_id=request_id,
            raw=data if isinstance(data, dict) else dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        """The wire shape of the response."""
        out: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "results": [result.to_dict() for result in self.results],
        }
        if self.usage is not None:
            out["usage"] = self.usage.to_dict()
        return out


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass
class _MultilookRequest:
    provider_cfg: dict[str, Any]
    body: dict[str, Any]
    timeout: float | None
    asset_counts: list[int]


def _reject_tool_traffic(context: list[dict[str, Any]]) -> None:
    for i, message in enumerate(context):
        role = message.get("role")
        if role == "tool" or (role == "assistant" and message.get("tool_calls")):
            raise BadRequestError(
                "Tool calling is not supported on multilook: context may not hold tool results or tool calls.",
                code=UNSUPPORTED_PARAMETER,
                param=f"context[{i}]",
            )


def _prompt_content(item: Any, index: int) -> list | tuple:
    """The content list of a non-str prompt: ``{"content": [...]}``, a list of parts / DSL nodes, or one DSL node."""
    if isinstance(item, (list, tuple)):
        return item
    if isinstance(item, DSLNode):
        return [item]
    if not isinstance(item, Mapping):
        raise TypeError(
            f"prompts[{index}] must be a str, a list of content parts / DSL nodes, or {{'content': [...]}}; "
            f"got {type(item).__name__}"
        )
    unexpected = [key for key in item if key != "content"]
    if unexpected:
        # A prompt is always the final user turn, so there is no role (or anything else) to set.
        raise TypeError(f"prompts[{index}] takes only a 'content' key; got {', '.join(map(repr, unexpected))}")
    content = item.get("content")
    if not isinstance(content, (list, tuple)):
        raise TypeError(
            f"prompts[{index}]['content'] must be a list of content parts (pass text as a plain str prompt)"
        )
    return content


def _prompt_items(prompts: Any) -> list[Any]:
    """Each prompt as a str or its content list (not lowered yet)."""
    if isinstance(prompts, (str, bytes, Mapping)) or not isinstance(prompts, (list, tuple)):
        raise TypeError("prompts must be a list of prompts (str, content-part lists or {'content': [...]})")
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise BadRequestError(
            f"Invalid prompts: expected between 1 and {MAX_PROMPTS} prompts, got {len(prompts)}.",
            code=INVALID_PARAMETER,
            param="prompts",
        )
    return [item if isinstance(item, str) else _prompt_content(item, i) for i, item in enumerate(prompts)]


def _media_count(item: Any) -> int:
    """The media assets a prompt item adds: DSL media nodes and media part dicts (a str adds none)."""
    if isinstance(item, str):
        return 0
    ledger = _content_ledger([{"content": item}])
    return count_assets([{"content": item}]) if ledger is None else ledger.count


def _normalize_prompts(
    items: list[Any], *, context: list | tuple, context_assets: int, base_url: str | None, provider_name: str | None
) -> tuple[list[Any], list[int]]:
    """The wire prompts (a str stays a str; parts and DSL nodes become ``{"content": [...]}``) and each one's media
    asset count.

    Each prompt is answered after the context, so its tags anchor among the context's ``context_assets`` media first,
    then its own. ``context`` is the caller's context (not the lowered one) so ``image=``/``asset=`` anchors find its
    DSL nodes by identity."""
    wire: list[Any] = []
    assets: list[int] = []
    for i, parts in enumerate(items):
        if isinstance(parts, str):
            wire.append(parts)
            assets.append(0)
            continue
        lowered = _lower_content(
            parts,
            i,
            base_url=base_url,
            provider_name=provider_name,
            name="prompts",
            ledger=_content_ledger([*context, {"content": parts}]),
            assets_before=context_assets,
        )
        prompt = {"content": list(parts) if lowered is None else lowered}
        wire.append(prompt)
        assets.append(count_assets([prompt]))
    return wire, assets


def _check_n(n: Any, prompt_count: int) -> int:
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_N):
        raise BadRequestError(
            f"Invalid n: {n!r}. Expected an integer between 1 and {MAX_N}.", code=INVALID_PARAMETER, param="n"
        )
    samples = 1 if n is None else n
    if prompt_count * samples > MAX_COMPLETIONS:
        raise BadRequestError(
            f"len(prompts) * n = {prompt_count * samples} exceeds the per-call limit of {MAX_COMPLETIONS} completions.",
            code=INVALID_PARAMETER,
            param="n",
        )
    return samples


def _check_temperature(temperature: Any, samples: int) -> None:
    """A set ``temperature`` is a finite number >= 0, and ``n > 1`` needs one above 0 (the gateway reads unset as 0)."""
    _validate_temperature(temperature)
    if samples > 1 and (temperature is None or temperature <= 0):
        raise BadRequestError(
            f"Invalid request: n = {samples} requires temperature > 0.", code=INVALID_PARAMETER, param="temperature"
        )


def _build_request(  # noqa: PLR0913 - mirrors the multilook signature
    client: Any,
    *,
    context: Any,
    prompts: Any,
    model: str | None,
    n: int | None,
    max_completion_tokens: int | None,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    reasoning_effort: str | None,
    vision_config: dict[str, Any] | None,
    extra_body: Any,
    timeout: float | None,
) -> _MultilookRequest:
    """Validate and build the request (``vision_config`` arrives normalized)."""
    settings = client._settings
    cfg = surface_provider_cfg(client, feature=_FEATURE)
    # Shared with create(): the max_tokens alias, and `configure()` generation defaults count as set.
    params = _generation_params(
        settings,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    lowering = {"base_url": cfg.get("base_url"), "provider_name": cfg["name"]}
    resolved_model = surface_model(cfg, model, settings)
    prompt_items = _prompt_items(prompts)
    # The context is shared, so its tags must hold in every prompt's asset space (the context's media, then that
    # prompt's): they anchor in the widest one.
    widest = max(_media_count(item) for item in prompt_items)
    wire_context = _normalize_messages(context, name="context", extra_assets=widest, **lowering)
    _reject_tool_traffic(wire_context)
    context_assets = count_assets(wire_context)
    wire_prompts, prompt_assets = _normalize_prompts(
        prompt_items, context=context, context_assets=context_assets, **lowering
    )
    samples = _check_n(n, len(wire_prompts))
    _check_temperature(params["temperature"], samples)

    body: dict[str, Any] = {"model": resolved_model, "context": wire_context, "prompts": wire_prompts}
    optional: dict[str, Any] = {
        "n": n,
        **params,
        "reasoning_effort": _normalize_reasoning_effort(reasoning_effort),
        "vision_config": vision_config,
    }
    body.update((key, value) for key, value in optional.items() if value is not None)
    _merge_extra_body(body, extra_body)

    if timeout is None and settings.timeout is not None:
        timeout = max(settings.timeout, MIN_TIMEOUT)
    return _MultilookRequest(
        provider_cfg=cfg,
        body=body,
        timeout=timeout,
        asset_counts=[context_assets + count for count in prompt_assets],
    )


def _response_from_payload(payload: Any, headers: Any, asset_counts: list[int]) -> MultilookResponse:
    request_id = _transport.header_value(headers, _transport.TRACE_ID_HEADER)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        raise ServerError(
            "The server returned a multilook response without results.",
            code=INVALID_RESPONSE,
            request_id=request_id,
            details={"response": payload},
        )
    return MultilookResponse.from_dict(payload, request_id=request_id, asset_counts=asset_counts)


def create(  # noqa: PLR0913 - the multilook request parameters
    client: Any,
    *,
    context: list[Any],
    prompts: list[Any],
    model: str | None = None,
    n: int | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: str | None = None,
    vision_config: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> MultilookResponse:
    """Answer 1-16 ``prompts`` against one shared ``context`` (``POST /chat/completions/multilook``).

    ``context`` is a list of messages, taken like ``create()``'s ``messages`` (``[]`` is allowed; tool results and
    assistant tool calls are not). Each prompt is a str, a list of content parts / DSL nodes, or
    ``{"content": [...]}``. ``n`` (1-8) completions per prompt, at most 64 in all; ``n > 1`` needs
    ``temperature > 0``. Only the parameters you set are sent (``configure()`` generation defaults count as set);
    ``max_tokens`` is an alias of ``max_completion_tokens``; ``extra_body`` is merged last, unvalidated. There is no
    streaming, and no tools, ``response_format``, ``regex`` or ``stop``.

    ``timeout`` defaults to ``max(settings.timeout, 305)`` seconds, to outlast the gateway's 300 s budget. Invalid
    values raise ``BadRequestError`` before any request. Provider ``perceptron`` only.
    """
    request = _build_request(
        client,
        context=context,
        prompts=prompts,
        model=model,
        n=n,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        reasoning_effort=reasoning_effort,
        # Normalized here so its deprecation warning points at the caller, as create()'s does.
        vision_config=_normalize_vision_config(vision_config),
        extra_body=extra_body,
        timeout=timeout,
    )
    payload, headers = _transport.request_json(
        client,
        "POST",
        MULTILOOK_PATH,
        json=request.body,
        timeout=request.timeout,
        provider_cfg=request.provider_cfg,
    )
    return _response_from_payload(payload, headers, request.asset_counts)


async def acreate(  # noqa: PLR0913 - the multilook request parameters
    client: Any,
    *,
    context: list[Any],
    prompts: list[Any],
    model: str | None = None,
    n: int | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: str | None = None,
    vision_config: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> MultilookResponse:
    """Async :func:`create`."""
    request = _build_request(
        client,
        context=context,
        prompts=prompts,
        model=model,
        n=n,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        reasoning_effort=reasoning_effort,
        # Normalized here so its deprecation warning points at the caller, as create()'s does.
        vision_config=_normalize_vision_config(vision_config),
        extra_body=extra_body,
        timeout=timeout,
    )
    payload, headers = await _transport.arequest_json(
        client,
        "POST",
        MULTILOOK_PATH,
        json=request.body,
        timeout=request.timeout,
        provider_cfg=request.provider_cfg,
    )
    return _response_from_payload(payload, headers, request.asset_counts)
