"""HTTP client for executing compiled Tasks against supported providers.

Providers
- perceptron: the Perceptron API (the default)
- fal: Fal-hosted endpoint (OpenAI-compatible); selected when you choose it, or when `FAL_KEY` is set and neither
  `PERCEPTRON_API_KEY` nor a key set in code is (see `perceptron.config.Settings`)

Additional transports can be registered by extending `_PROVIDER_CONFIG`.

`Client.stream` reads the SSE stream and yields events:
- reasoning.delta / text.delta: reasoning and answer text as they arrive
- tool_call.delta: one tool-call fragment (`index`; `id`/`name` on the fragment that first carries them; `arguments`)
- points.delta: emitted once per leaf of the expected kind when it closes, even inside a still-open collection or
  track; `context` holds the `mention`/`t`/`asset_idx` it gets from its containers and the `container` kind. Deltas
  are provisional (later markup can invalidate a leaf's container); `final` is authoritative
- final: the result, with the `generate()` keys (`raw` is None); the last event of a finished stream
- error: a terminal error (HTTP error, error event, cut or malformed stream, or malformed markup with `strict=True`)
  with its `details` and what arrived before it (`partial`); no `final` follows

Connections: each `Client` sends every request (all surfaces) through one pooled `httpx.Client` (HTTP/2), created on
first use, and each `AsyncClient` through one `httpx.AsyncClient`; close them with `close()` / `await aclose()` or a
`with` / `async with` block. Pass `http_client=` to use your own httpx client instead; the SDK never closes it.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, fields
from functools import cached_property
from types import ModuleType
from typing import Any, TypedDict

import httpx

from . import _transport

# Registry, error mapping and lowering moved to these modules; the old names stay importable from here.
from ._lowering import _is_url_payload, count_assets, task_to_messages  # noqa: F401
from ._providers import (  # noqa: F401
    _PROVIDER_CONFIG,
    PERCEPTRON_PROVIDER,
    REASONING_EFFORTS,
    _normalize_reasoning_effort,
    _pop_and_resolve_model,
    _resolve_provider,
    _select_model,
)
from ._transport import _extract_error_metadata, _first_nonempty, http_error_from_response  # noqa: F401
from .chat import (
    AsyncChat,
    Chat,
    ChatCompletion,
    ChatStreamAccumulator,
    _decode_chunk,
    _normalize_vision_config,
    _validate_request_body,
)
from .config import Settings, _settings_with

# The error classes, INVALID_REASONING_EFFORT, parse_text, extract_points and extract_clips stay importable from here,
# as in earlier releases.
from .errors import (  # noqa: F401
    INVALID_REASONING_EFFORT,
    INVALID_RESPONSE,
    STREAM_INCOMPLETE,
    STREAM_TRUNCATED,
    AuthError,
    BadRequestError,
    IncompleteStreamError,
    ParseError,
    RateLimitError,
    SDKError,
    ServerError,
    TimeoutError,
    TransportError,
)
from .expectations import STRUCTURED_EXPECTATIONS
from .pointing.parser import (  # noqa: F401
    _scan_leaves,
    collect_annotations,
    extract_clips,
    extract_points,
    parse_text,
    scan_leaves,
)

# Maps each structured `expects` value to its PerceiveResult bucket (an `AnnotationCollection` attribute).
_BUCKET_BY_EXPECTS = {"point": "points", "box": "boxes", "polygon": "polygons", "clip": "clips"}

# Retired Mk1 arguments; passing one raises a TypeError that says so.
_RETIRED_ARGUMENTS = ("focus", "visual_reasoning")


def _unexpected_keyword(func: str, name: str) -> TypeError:
    """The ``TypeError`` for an unknown keyword argument of ``func``; a retired Focus argument says it was removed."""
    message = f"{func}() got an unexpected keyword argument {name!r}"
    if name in _RETIRED_ARGUMENTS:
        message += f': "{name}" was removed: Focus controls are retired (see "Migrate from Mk1")'
    return TypeError(message)


def _reject_unexpected_kwargs(func: str, kwargs: Mapping[str, Any]) -> None:
    if kwargs:
        raise _unexpected_keyword(func, next(iter(kwargs)))


# ---------------------------------------------------------------------------
# Response format types for constrained decoding
# ---------------------------------------------------------------------------


class JsonSchemaSpec(TypedDict, total=False):
    """JSON Schema specification object containing name, schema, and optional strict flag."""

    name: str
    schema: dict[str, Any]
    strict: bool


class JsonSchemaFormat(TypedDict, total=False):
    """JSON Schema response format specification."""

    type: str  # Must be "json_schema"
    json_schema: JsonSchemaSpec


class RegexFormat(TypedDict, total=False):
    """Regex response format specification."""

    type: str  # Must be "regex"
    regex: str  # The regex pattern to constrain output


# Union type for response_format parameter
ResponseFormat = JsonSchemaFormat | RegexFormat | dict[str, Any]


@dataclass
class _PreparedInvocation:
    body: dict[str, Any]
    expects: str | None
    provider_cfg: dict[str, Any]  # with the effective base_url
    asset_count: int  # media assets in the lowered messages (the `asset_idx` space)


def _build_response_format(
    response_format: ResponseFormat | None,
) -> tuple[str, dict[str, Any] | str] | None:
    """Validate and normalize response_format for the API request.

    Returns:
        None if response_format is None, otherwise a tuple of (field_name, value):
        - For text: ("response_format", {"type": "text"})
        - For json_schema: ("response_format", {"type": "json_schema", "json_schema": {...}})
        - For regex: ("regex", "pattern_string")
    """
    if response_format is None:
        return None

    fmt_type = response_format.get("type")
    if fmt_type == "text":
        return ("response_format", {"type": "text"})

    if fmt_type == "json_schema":
        schema_spec = response_format.get("json_schema")
        if not isinstance(schema_spec, dict):
            raise ValueError("json_schema response_format requires a 'json_schema' dict with 'name' and 'schema'")
        return ("response_format", {"type": "json_schema", "json_schema": schema_spec})

    if fmt_type == "regex":
        regex_pattern = response_format.get("regex")
        if not isinstance(regex_pattern, str):
            raise ValueError("regex response_format requires a 'regex' string pattern")
        return ("regex", regex_pattern)

    raise ValueError(f"Unknown response_format type: {fmt_type!r}. Supported types: 'text', 'json_schema', 'regex'")


def _add_buckets(
    result: dict[str, Any], content: str, expects: str, errors: list[dict[str, Any]], *, strict: bool = False
) -> None:
    """Set the ``expects`` bucket, ``tracks`` and ``parsed`` from ``content``.

    Buckets are flattened: collection children and track waypoints appear with the context their markup gives them
    (own ?? track ?? collection); ``tracks`` (every track, whatever its kind) and ``parsed`` keep the tree. Lenient: a
    malformed element stays text and adds an ``errors`` entry, and an unclosed container (truncated output) is kept
    with ``complete=False`` and reported as ``incomplete_annotation``. ``strict`` raises the first problem as
    ``ParseError``. For ``clip``, whose parse leaves spatial markup as text, ``tracks`` come from a lenient parse of
    that markup whose problems are not reported (like other markup outside the ``expects`` family).
    """
    found = collect_annotations(content, expects=expects, strict=strict)
    result[_BUCKET_BY_EXPECTS[expects]] = getattr(found, _BUCKET_BY_EXPECTS[expects])
    result["tracks"] = collect_annotations(content).tracks if expects == "clip" else found.tracks
    result["parsed"] = found.parsed
    errors.extend(found.errors)


def _result_metadata(completion: ChatCompletion, usage: Any) -> dict[str, Any]:
    """The completion metadata shared by ``generate()`` results and the stream's ``final`` result."""
    return {
        "finish_reason": completion.finish_reason,
        "tool_calls": completion.tool_calls,
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "id": completion.id,
        "model": completion.model,
        "request_id": completion.request_id,
        "complete": completion.complete,
        "asset_count": completion.asset_count,
    }


def _error_event(
    exc: SDKError, *, partial: dict[str, Any] | None = None, request_id: str | None = None
) -> dict[str, Any]:
    """The terminal ``error`` event. ``details`` is the error's ``details`` dict (the server's error object, when there
    is one, with ``request_id`` mirrored in as on raised errors); ``partial`` is what arrived before the error (None
    before the stream opened)."""
    request_id = exc.request_id if exc.request_id is not None else request_id
    details = dict(exc.details)
    if request_id is not None:
        details.setdefault("request_id", request_id)
    return {
        "type": "error",
        "message": str(exc),
        "code": exc.code,
        "error_type": exc.error_type,
        "param": exc.param,
        "status": exc.status_code,
        "request_id": request_id,
        "details": details,
        "partial": partial,
    }


def _error_events(exc: SDKError) -> Iterator[dict[str, Any]]:
    yield _error_event(exc)


async def _aerror_events(exc: SDKError) -> AsyncIterator[dict[str, Any]]:
    yield _error_event(exc)


class _StreamProcessor:
    """Turns stream payloads into events and, at the end, the ``final`` result or a terminal ``error`` event."""

    def __init__(  # noqa: PLR0913 - keyword-only stream settings
        self,
        *,
        client_core: _ClientCore,
        expects: str | None,
        parse_points: bool,
        max_buffer_bytes: int | None,
        request_id: str | None = None,
        n_assets: int | None = None,
        strict: bool = False,
    ) -> None:
        self._client_core = client_core
        self._expects = expects
        self._parse_points = parse_points and expects in _BUCKET_BY_EXPECTS
        self._max_buffer_bytes = max_buffer_bytes
        self._strict = strict  # malformed markup in the final answer ends the stream with an `error` event
        self._cumulative: str = ""
        self._reasoning: str = ""
        self._emitted_spans: set[tuple[int, int]] = set()
        self._scan_start = 0  # the answer before this offset holds only complete top-level elements, already scanned
        self._parsing_enabled = True
        self._accumulator = ChatStreamAccumulator(request_id=request_id, asset_count=n_assets)
        self._announced_ids: set[int] = set()
        self._announced_names: set[int] = set()

    @property
    def done(self) -> bool:
        """True once ``[DONE]`` arrived."""
        return self._accumulator.done

    def feed(self, data: Any) -> list[dict[str, Any]]:
        """Events for one SSE payload (``_transport.DONE`` for ``[DONE]``).

        Raises the mapped ``SDKError`` for an error event and ``ServerError(code="invalid_stream_chunk")`` for a
        malformed chunk.
        """
        if data is _transport.DONE:
            self._accumulator.mark_done()
            return []
        obj = _decode_chunk(data)
        if obj.get("error") is not None:
            raise _transport.stream_error_from_event(obj["error"], request_id=self._accumulator.request_id)
        return self.handle_payload(obj)

    def handle_payload(self, obj: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not isinstance(obj, dict):
            return events

        answer_chars = len(self._cumulative)
        # Usage, finish_reason, id/model and tool calls accumulate here (`delta: null` and `choices: []` included).
        chunk = self._accumulator.feed(obj)
        for choice in chunk.choices:
            delta = choice.delta

            # Process reasoning content
            reasoning = delta.reasoning_content
            if reasoning:
                self._reasoning += reasoning
                events.append(
                    {
                        "type": "reasoning.delta",
                        "chunk": reasoning,
                        "total_chars": len(self._reasoning),
                    }
                )

            # Process answer content
            content = delta.content
            if content:
                self._cumulative += content
                events.append(
                    {
                        "type": "text.delta",
                        "chunk": content,
                        "total_chars": len(self._cumulative),
                    }
                )

            events.extend(self._tool_call_events(delta.tool_calls))

        # Check buffer limits
        if self._parsing_enabled and self._max_buffer_bytes is not None:
            if len(self._cumulative.encode("utf-8")) > self._max_buffer_bytes:
                self._parsing_enabled = False

        # Parse points only when this chunk brought a `>`: every leaf and clip ends with one, so no new leaf can close
        # without it. Each scan starts after the last complete top-level element (see `_point_events`).
        if self._parse_points and self._parsing_enabled and ">" in self._cumulative[answer_chars:]:
            events.extend(self._point_events())

        return events

    def _tool_call_events(self, pieces: list[Any] | None) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for piece in pieces or []:
            index = piece.index
            call_id = piece.id if piece.id and index not in self._announced_ids else None
            if call_id is not None:
                self._announced_ids.add(index)
            function = piece.function
            name = function.name if function is not None else None
            if not name or index in self._announced_names:
                name = None
            else:
                self._announced_names.add(index)
            arguments = function.arguments if function is not None else None
            events.append(
                {"type": "tool_call.delta", "index": index, "id": call_id, "name": name, "arguments": arguments or ""}
            )
        return events

    def partial(self) -> dict[str, Any]:
        """What arrived so far; never a complete answer, so its tool calls must not be run."""
        return {
            "text": self._cumulative or None,
            "reasoning": self._reasoning or None,
            "tool_calls": self._accumulator.tool_calls,
            "finish_reason": self._accumulator.finish_reason,
        }

    def error_event(self, exc: SDKError) -> dict[str, Any]:
        return _error_event(exc, partial=self.partial(), request_id=self._accumulator.request_id)

    def terminal_event(self) -> dict[str, Any]:
        """``final`` once ``[DONE]`` arrived; a ``stream_truncated`` error event when the body ended without it."""
        if not self.done:
            return self.error_event(IncompleteStreamError("The stream ended before [DONE].", code=STREAM_TRUNCATED))
        return self.finalize()

    def finalize(self) -> dict[str, Any]:
        """The ``final`` event. Never raises: malformed markup (including containers the answer left unclosed, which
        are never auto-closed) and a missing ``finish_reason`` become ``errors``. With ``strict``, malformed markup
        gives the terminal ``error`` event (the ``ParseError`` code, with ``partial``) instead of ``final``."""
        content = self._cumulative or None
        reasoning = self._reasoning or None
        result: dict[str, Any] = {"text": content, "reasoning": reasoning, "raw": None}
        issues: list[dict[str, Any]] = []
        expects = self._expects
        if expects in _BUCKET_BY_EXPECTS and self._parsing_enabled and isinstance(content, str):
            try:
                _add_buckets(result, content, expects, issues, strict=self._strict)
            except ParseError as exc:
                return self.error_event(exc)
        if not self._parsing_enabled:
            issues.append(
                {
                    "code": "stream_buffer_overflow",
                    "message": "parsing disabled due to buffer limit",
                }
            )
        completion = self._accumulator.snapshot()
        if completion.finish_reason is None:
            issues.append(
                {
                    "code": STREAM_INCOMPLETE,
                    "message": "The stream ended without a finish_reason, so the answer may be incomplete.",
                }
            )
        return {
            "type": "final",
            "result": {
                **result,
                **_result_metadata(completion, self._accumulator.usage),
                "errors": issues,
            },
        }

    def _point_events(self) -> list[dict[str, Any]]:
        """One ``points.delta`` per newly closed leaf of the expected kind, keyed by its span so each is emitted once.

        Leaves inside still-open collections and tracks count; ``context`` carries what the open containers give them
        (``mention``, ``t``, ``asset_idx``, and ``container``: ``"collection"``, ``"track"`` or None). Leaves of a
        container that is already invalid are not emitted. Deltas are provisional: one cannot be retracted when later
        markup invalidates its container, so the ``final`` result is authoritative.
        """
        events: list[dict[str, Any]] = []
        # Rescan only from the element still arriving, not the whole answer on every chunk.
        leaves, self._scan_start = _scan_leaves(self._cumulative, self._expects, self._scan_start)
        for leaf in leaves:
            span = (leaf["span"]["start"], leaf["span"]["end"])
            if span in self._emitted_spans:
                continue
            self._emitted_spans.add(span)
            events.append(
                {"type": "points.delta", "points": [leaf["value"]], "span": leaf["span"], "context": leaf["context"]}
            )
        return events


def _task_to_openai_messages(
    task: dict, *, base_url: str | None = None, provider_name: str | None = None
) -> list[dict[str, Any]]:
    """Lower a compiled task to chat messages (see `_lowering.task_to_messages`)."""
    return task_to_messages(task, base_url=base_url, provider_name=provider_name)


def _model_entry(model_name: str | None, provider_cfg: dict[str, Any] | None) -> dict | None:
    models_cfg = provider_cfg.get("models") if isinstance(provider_cfg, dict) else None
    if isinstance(models_cfg, dict):
        entry = models_cfg.get(model_name)
        if isinstance(entry, dict):
            return entry
    return None


def _model_capabilities(model_name: str | None, provider_cfg: dict[str, Any] | None) -> tuple[bool, bool, bool]:
    entry = _model_entry(model_name, provider_cfg) or {}
    supports_reasoning = bool(entry.get("reasoning", True))
    requires_reasoning = bool(entry.get("only_reasoning", False))
    skip_hints = bool(entry.get("skip_structured_hints", False))
    return supports_reasoning, requires_reasoning, skip_hints


def _build_hint_content(expects: str | None, include_reasoning: bool) -> str | None:
    tokens: list[str] = []
    if expects and expects.lower() in STRUCTURED_EXPECTATIONS:
        tokens.append(expects.upper())
    if include_reasoning:
        tokens.append("THINK")
    if not tokens:
        return None
    return f"<hint>{' '.join(sorted(tokens))}</hint>"


def _inject_expectation_hint(
    task: dict,
    expects: str | None,
    *,
    model_name: str | None,
    provider_cfg: dict[str, Any] | None,
    include_reasoning: bool,
) -> dict:
    _, _, skip_hints = _model_capabilities(model_name, provider_cfg)
    if skip_hints:
        content = task.get("content") or []
        filtered = [
            entry
            for entry in content
            if not (
                isinstance(entry, dict)
                and entry.get("type") == "text"
                and isinstance(entry.get("content"), str)
                and "<hint" in entry.get("content", "").lower()
            )
        ]
        new_task = dict(task)
        new_task["content"] = filtered
        return new_task

    hint = _build_hint_content(expects, include_reasoning)
    if hint is None:
        return task

    content = task.get("content") or []
    if any(entry.get("content") == hint for entry in content if isinstance(entry, dict)):
        return task
    # The perceptron AI gateway only honors `<hint>...</hint>` when it
    # arrives as a system-role message. Prepending it inside the user
    # message (the previous behavior) was silently ignored, so flags like
    # `reasoning=True` produced no `reasoning_content` in the response.
    # Other providers (fal, nebius, modal) keep the original user-content
    # behavior since their backends handled it correctly.
    is_perceptron = isinstance(provider_cfg, dict) and provider_cfg.get("name") == "perceptron"
    if is_perceptron:
        new_content: list[dict[str, Any]] = [
            {"type": "text", "role": "system", "content": hint},
            *content,
        ]
    else:
        new_content = []
        inserted = False
        for entry in content:
            if not inserted and entry.get("role") != "system":
                new_content.append({"type": "text", "role": "user", "content": hint})
                inserted = True
            new_content.append(entry)
        if not inserted:
            new_content.append({"type": "text", "role": "user", "content": hint})
    new_task = dict(task)
    new_task["content"] = new_content
    return new_task


def _apply_reasoning_and_hints(
    *,
    task: dict,
    expects: str | None,
    model_name: str | None,
    provider_cfg: dict[str, Any] | None,
    reasoning_flag: bool | None,
) -> tuple[dict, bool]:
    supports, requires, _ = _model_capabilities(model_name, provider_cfg)

    final_reasoning = reasoning_flag  # None means "auto"

    # If the model requires reasoning, force it on.
    if requires and final_reasoning is not True:
        final_reasoning = True

    # If caller didn't specify, enable when expects==think or a THINK hint will be injected.
    if final_reasoning is None and expects and expects.lower() == "think":
        final_reasoning = True

    # Disable if model lacks support.
    if final_reasoning is True and not supports:
        final_reasoning = False

    include_reasoning_hint = bool((final_reasoning is True) or requires or (expects and expects.lower() == "think"))
    task_with_hint = _inject_expectation_hint(
        task,
        expects,
        model_name=model_name,
        provider_cfg=provider_cfg,
        include_reasoning=include_reasoning_hint,
    )

    return task_with_hint, final_reasoning


# Kept under its old name; `http_error_from_response` also reads x-trace-id, Retry-After on 503 and quota errors.
_map_http_error = http_error_from_response


# The real httpx client classes (tests may replace this module's `httpx` with a stub namespace).
_HTTPX_CLIENT = httpx.Client
_HTTPX_ASYNC_CLIENT = httpx.AsyncClient


def _http_client(timeout: float | None) -> httpx.Client:
    """The pooled HTTP client a :class:`Client` creates on first use; looked up at call time so tests can patch it."""
    return httpx.Client(timeout=timeout, http2=True)


def _async_http_client(timeout: float | None) -> httpx.AsyncClient:
    """The pooled HTTP client an :class:`AsyncClient` creates on first use; looked up at call time so tests can patch
    it."""
    if not isinstance(httpx, ModuleType):  # compat: a test stub namespace in place of httpx (it takes no `http2`)
        return httpx.AsyncClient(timeout=timeout)
    return httpx.AsyncClient(timeout=timeout, http2=True)


class _StubSession:
    """Compat for hand-rolled test stand-ins for an httpx client, until the tests all use ``httpx.MockTransport``.

    The stand-ins take no per-request ``timeout`` and have no ``close``/``aclose``; everything else passes through.
    """

    def __init__(self, stub: Any) -> None:
        self._stub = stub

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._stub, name)

        def call(*args: Any, timeout: Any = None, **kwargs: Any) -> Any:
            return method(*args, **kwargs)

        return call

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


class _ClientCore:
    _HTTP_CLIENT_TYPE: type = _HTTPX_CLIENT  # what `http_client=` must be

    def __init__(self, *, http_client: Any = None, **overrides: Any) -> None:
        known = {f.name for f in fields(Settings)}
        for k in overrides:
            if k not in known:
                raise TypeError(f"{type(self).__name__}() got an unexpected keyword argument {k!r}")
        if http_client is not None and not isinstance(http_client, self._HTTP_CLIENT_TYPE):
            raise TypeError(
                f"http_client must be an httpx.{self._HTTP_CLIENT_TYPE.__name__}; got {type(http_client).__name__}"
            )
        # The keyword arguments count as configured settings, so the provider rule sees a key passed here.
        self._settings = _settings_with(overrides)
        self._http: Any = http_client  # the pooled HTTP client; created on first use unless one was passed
        self._owns_http = http_client is None  # only an HTTP client created here is closed by close()/aclose()
        self._closed = False
        self._lock = threading.Lock()

    def _session(self) -> Any:
        """The HTTP client every request goes through: ``http_client``, else one created on first use (by the
        subclass's ``_new_session``) and reused until the client is closed."""
        with self._lock:
            if self._closed:
                raise RuntimeError(f"This {type(self).__name__} is closed; create a new one to send requests.")
            if self._http is None:
                self._http = self._new_session(self._settings.timeout)
            return self._http

    def _detach_session(self) -> Any:
        """Mark the client closed; returns the HTTP client to close (None when there is none or it was passed in)."""
        with self._lock:
            self._closed = True
            session, self._http = self._http, None
        return session if self._owns_http else None

    def _prepare_invocation(
        self,
        task: dict,
        *,
        expects: str | None,
        stream: bool,
        gen_kwargs: dict[str, Any],
    ) -> _PreparedInvocation:
        """Resolve provider and model, apply the `<hint>` encoding, lower the task and build the request body.

        Only parameters the caller set (or configured defaults) are sent. Tool parameters and structured-output controls
        go through the message API's validators before any request; ``extra_body`` is merged last, unvalidated.
        """
        s = self._settings
        stream_options = gen_kwargs.get("stream_options")
        if stream_options is not None and not isinstance(stream_options, Mapping):
            raise TypeError("stream_options must be a dict, e.g. {'include_usage': True}")
        extra_body = gen_kwargs.get("extra_body")
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise TypeError("extra_body must be a dict")

        def _option(name: str) -> Any:
            # An explicit argument, else the configured default.
            value = gen_kwargs.get(name)
            return getattr(s, name) if value is None else value

        reasoning_flag = gen_kwargs.get("reasoning")
        reasoning_effort = _normalize_reasoning_effort(gen_kwargs.get("reasoning_effort"))
        enable_audio_in_video = gen_kwargs.get("enable_audio_in_video")
        # The message API's `vision_config` check: a non-bool raises (coercing would send "false" as true).
        vision_config = (
            None
            if enable_audio_in_video is None
            else _normalize_vision_config({"enable_audio_in_video": enable_audio_in_video})
        )
        provider_cfg = _resolve_provider(_option("provider"))
        provider_name = provider_cfg["name"]
        model = _pop_and_resolve_model(provider_cfg, {"model": _option("model")})
        task_with_hint, reasoning_flag = _apply_reasoning_and_hints(
            task=task,
            expects=expects,
            model_name=model,
            provider_cfg=provider_cfg,
            reasoning_flag=reasoning_flag,
        )
        if stream and not provider_cfg.get("stream", True):
            raise BadRequestError(f"Streaming is not supported for provider='{provider_name}'")
        base_url = s.base_url or provider_cfg.get("base_url")
        messages = _task_to_openai_messages(task_with_hint, base_url=base_url, provider_name=provider_name)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        temperature = _option("temperature")
        if temperature is not None:
            body["temperature"] = temperature
        max_tokens = _option("max_tokens")
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        top_p = _option("top_p")
        if top_p is not None:
            body["top_p"] = top_p
        top_k = _option("top_k")
        if top_k is not None:
            body["top_k"] = top_k
        frequency_penalty = _option("frequency_penalty")
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        presence_penalty = _option("presence_penalty")
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if reasoning_flag:
            # Reasoning + geometry expectations are signaled to the perceptron
            # backend via the `<hint>...</hint>` system message that
            # `_inject_expectation_hint` prepends. Non-perceptron providers
            # keep the original top-level `reasoning` field until each
            # backend's exact format is verified.
            if provider_name != "perceptron":
                body["reasoning"] = True
        if stream:
            body["stream"] = True
            # Usage arrives on perceptron streams only when requested; fal's support is unverified.
            if stream_options is None and provider_name == PERCEPTRON_PROVIDER:
                stream_options = {"include_usage": True}
            if stream_options is not None:
                body["stream_options"] = dict(stream_options)
        if reasoning_effort is not None:
            # Top-level, as the API defines it. Independent of the `<hint>` THINK encoding of
            # `reasoning=True`: the API turns reasoning on for any tier other than `none`.
            body["reasoning_effort"] = reasoning_effort
        if vision_config is not None:
            body["vision_config"] = vision_config

        # Add constrained decoding field (json_schema → response_format, regex → regex)
        format_result = _build_response_format(gen_kwargs.get("response_format"))
        if format_result is not None:
            field_name, field_value = format_result
            body[field_name] = field_value

        tools = gen_kwargs.get("tools")
        if tools is not None:
            body["tools"] = list(tools) if isinstance(tools, tuple) else tools
        for name in ("tool_choice", "parallel_tool_calls"):
            if gen_kwargs.get(name) is not None:
                body[name] = gen_kwargs[name]
        _validate_request_body(body)
        if extra_body:
            body.update(extra_body)  # merged last, unvalidated

        return _PreparedInvocation(
            body=body,
            expects=expects,
            provider_cfg={**provider_cfg, "base_url": base_url},
            asset_count=count_assets(messages),
        )

    def _build_result(
        self,
        data: dict[str, Any],
        expects: str | None,
        *,
        headers: Any = None,
        n_assets: int | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Normalize a chat completion response.

        Keys: ``text``, ``reasoning``, ``raw``, the ``expects`` bucket, ``tracks`` and ``parsed`` (see
        ``_add_buckets``), ``finish_reason``, ``tool_calls`` (``list[ToolCall]`` or None), ``usage`` (the server's
        usage object as sent, or None), ``id``, ``model``, ``request_id`` (the ``x-trace-id`` header), ``complete``,
        ``asset_count`` (``n_assets``, the ``asset_idx`` space) and ``errors``. Malformed markup in the answer becomes an
        ``errors`` entry; with ``strict`` it raises ``ParseError`` carrying ``request_id`` (also in ``details``) and the
        answer as ``partial`` (``text``, ``reasoning``, ``tool_calls``, ``finish_reason``), like the strict stream's
        ``error`` event. A response without choices raises ``ServerError``.
        """
        request_id = _transport.header_value(headers, _transport.TRACE_ID_HEADER)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ServerError(
                "The server returned a chat completion without choices.",
                code=INVALID_RESPONSE,
                request_id=request_id,
                details={"response": data},
            )
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}

        reasoning_content = message.get("reasoning_content")
        content = message.get("content")

        result: dict[str, Any] = {"text": content, "reasoning": reasoning_content, "raw": data}
        errors: list[dict[str, Any]] = []
        completion = ChatCompletion.from_dict(data, request_id=request_id, asset_count=n_assets)
        if expects in _BUCKET_BY_EXPECTS and isinstance(content, str):
            try:
                _add_buckets(result, content, expects, errors, strict=strict)
            except ParseError as exc:
                exc.request_id = request_id
                if request_id is not None:
                    exc.details.setdefault("request_id", request_id)
                exc.partial = {
                    "text": content,
                    "reasoning": completion.reasoning,
                    "tool_calls": completion.tool_calls,
                    "finish_reason": completion.finish_reason,
                }
                raise
        result.update(_result_metadata(completion, data.get("usage")))
        result["errors"] = errors
        return result

    def _stream_processor(
        self, invocation: _PreparedInvocation, response: Any, *, parse_points: bool, strict: bool
    ) -> _StreamProcessor:
        return _StreamProcessor(
            client_core=self,
            expects=invocation.expects,
            parse_points=parse_points,
            max_buffer_bytes=self._settings.max_buffer_bytes,
            request_id=_transport.request_id_of(response),
            n_assets=invocation.asset_count,
            strict=strict,
        )


class Client(_ClientCore):
    """Runs compiled tasks (``generate``/``stream``) and hosts the message API (``chat``), ``files`` and ``models``.

    Keyword arguments override the settings for this client (``Client(provider="fal", api_key=...)``). The provider is
    resolved when the client is built, with the rule of :class:`~perceptron.config.Settings` (``perceptron`` unless you
    choose one, or ``FAL_KEY`` is your only key; an ``api_key=`` without a ``provider=`` goes to the Perceptron API),
    and every surface uses it; ``files``, ``models`` and multilook need provider ``perceptron``. ``generate``/``stream``
    also take a per-call ``provider=``.

    ``generate``/``stream`` send only the parameters you set (or configured defaults). ``reasoning=True`` adds the
    ``<hint>THINK</hint>`` encoding and ``expects`` the ``<hint>BOX</hint>``-style one (a system message on provider
    ``perceptron``); ``reasoning_effort`` is sent as the top-level field. They are independent: the server treats a
    THINK hint as reasoning on, so ``reasoning=True, reasoning_effort="none"`` still reasons; use ``reasoning_effort``
    alone to pick a tier. ``tools``, ``tool_choice`` (``"auto"``/``"none"``) and ``parallel_tool_calls`` are checked
    like ``client.chat.completions.create``; ``extra_body`` is merged into the body last, unvalidated. Malformed markup
    in the answer is an ``errors`` entry; ``strict=True`` raises ``ParseError`` instead (streams end with an ``error``
    event). Unknown keyword arguments raise ``TypeError``; the retired ``focus`` and ``visual_reasoning`` say so.

    Every request goes through one pooled HTTP client (``httpx.Client`` with HTTP/2), created on first use and reused
    by ``generate``/``stream``, ``chat``, ``files``, ``models`` and multilook. Close it with :meth:`close` or a ``with
    Client() as client:`` block; a closed client cannot send requests. ``http_client=`` supplies your own
    ``httpx.Client`` (proxies, custom transports, limits): the client uses it as is and never closes it, and each
    request still carries the SDK's timeout (``timeout``, or a per-call one).
    """

    def _new_session(self, timeout: float | None) -> Any:
        session = _http_client(timeout)
        return session if isinstance(session, _HTTPX_CLIENT) else _StubSession(session)  # compat: test stand-ins

    def close(self) -> None:
        """Close the HTTP client this client created (an ``http_client`` you passed stays open); streams still
        reading from it then end with an error. Safe to call more than once."""
        session = self._detach_session()
        if session is not None:
            session.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @cached_property
    def chat(self) -> Chat:
        """The message-level API: ``client.chat.completions.create(messages=[...])``."""
        return Chat(self)

    @cached_property
    def files(self) -> Any:
        """The Files API (upload, list, retrieve, content, download, delete)."""
        from .files import Files  # noqa: PLC0415

        return Files(self)

    @cached_property
    def models(self) -> Any:
        """The Models API (list, retrieve)."""
        from .models import Models  # noqa: PLC0415

        return Models(self)

    def generate(  # noqa: PLR0913 - the generation parameters
        self,
        task: dict,
        *,
        expects: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        reasoning: bool | None = None,
        reasoning_effort: str | None = None,
        enable_audio_in_video: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Run ``task`` and return the normalized result (see ``_build_result`` for its keys)."""
        _reject_unexpected_kwargs("generate", kwargs)
        invocation = self._prepare_invocation(
            task,
            expects=expects,
            stream=False,
            gen_kwargs={
                "model": model,
                "provider": provider,
                "reasoning": reasoning,
                "reasoning_effort": reasoning_effort,
                "enable_audio_in_video": enable_audio_in_video,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "response_format": response_format,
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "extra_body": extra_body,
            },
        )
        cfg = invocation.provider_cfg
        payload, headers = _transport.request_json(self, "POST", cfg["path"], json=invocation.body, provider_cfg=cfg)
        return self._build_result(payload, expects, headers=headers, n_assets=invocation.asset_count, strict=strict)

    def stream(  # noqa: PLR0913 - the generation parameters
        self,
        task: dict,
        *,
        expects: str | None = None,
        parse_points: bool = False,
        model: str | None = None,
        provider: str | None = None,
        reasoning: bool | None = None,
        reasoning_effort: str | None = None,
        enable_audio_in_video: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream ``task`` as events (see the module docstring); ends with one ``final`` or one ``error`` event.

        Keyword arguments are checked when called (``TypeError``); other errors before the request (lowering, model
        resolution, validation) arrive as the single ``error`` event. Streams to provider ``perceptron`` request usage
        (``stream_options={"include_usage": True}``) unless you pass ``stream_options``.
        """
        _reject_unexpected_kwargs("stream", kwargs)
        try:
            invocation = self._prepare_invocation(
                task,
                expects=expects,
                stream=True,
                gen_kwargs={
                    "model": model,
                    "provider": provider,
                    "reasoning": reasoning,
                    "reasoning_effort": reasoning_effort,
                    "enable_audio_in_video": enable_audio_in_video,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "response_format": response_format,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": parallel_tool_calls,
                    "extra_body": extra_body,
                    "stream_options": stream_options,
                },
            )
        except SDKError as exc:
            return _error_events(exc)
        return self._stream_events(invocation, parse_points=parse_points, strict=strict)

    def _stream_events(
        self, invocation: _PreparedInvocation, *, parse_points: bool, strict: bool
    ) -> Iterator[dict[str, Any]]:
        cfg = invocation.provider_cfg
        try:
            response, closer = _transport.open_stream(self, "POST", cfg["path"], json=invocation.body, provider_cfg=cfg)
        except SDKError as exc:  # HTTP errors carry the full body (it is read before mapping)
            yield _error_event(exc)
            return
        processor = self._stream_processor(invocation, response, parse_points=parse_points, strict=strict)
        with closer:
            try:
                for data in _transport.iter_sse_data(_transport.iter_response_lines(response)):
                    yield from processor.feed(data)
                    if processor.done:
                        break
            except SDKError as exc:
                yield processor.error_event(exc)
                return
        yield processor.terminal_event()


class AsyncClient(_ClientCore):
    """Asynchronous variant of :class:`Client` (same parameters) over one pooled ``httpx.AsyncClient`` (HTTP/2).

    Close it with ``await client.aclose()`` or an ``async with AsyncClient() as client:`` block; use it on one event
    loop. ``http_client=`` takes your own ``httpx.AsyncClient``, which the SDK never closes.
    """

    _HTTP_CLIENT_TYPE = _HTTPX_ASYNC_CLIENT

    def _new_session(self, timeout: float | None) -> Any:
        session = _async_http_client(timeout)
        return session if isinstance(session, _HTTPX_ASYNC_CLIENT) else _StubSession(session)  # compat: test stand-ins

    async def aclose(self) -> None:
        """Close the HTTP client this client created (an ``http_client`` you passed stays open); streams still
        reading from it then end with an error. Safe to call more than once."""
        session = self._detach_session()
        if session is not None:
            await session.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    @cached_property
    def chat(self) -> AsyncChat:
        """The async message-level API: ``await client.chat.completions.create(messages=[...])``."""
        return AsyncChat(self)

    @cached_property
    def files(self) -> Any:
        """The async Files API."""
        from .files import AsyncFiles  # noqa: PLC0415

        return AsyncFiles(self)

    @cached_property
    def models(self) -> Any:
        """The async Models API."""
        from .models import AsyncModels  # noqa: PLC0415

        return AsyncModels(self)

    async def generate(  # noqa: PLR0913 - the generation parameters
        self,
        task: dict,
        *,
        expects: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        reasoning: bool | None = None,
        reasoning_effort: str | None = None,
        enable_audio_in_video: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Async :meth:`Client.generate`."""
        _reject_unexpected_kwargs("generate", kwargs)
        invocation = self._prepare_invocation(
            task,
            expects=expects,
            stream=False,
            gen_kwargs={
                "model": model,
                "provider": provider,
                "reasoning": reasoning,
                "reasoning_effort": reasoning_effort,
                "enable_audio_in_video": enable_audio_in_video,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "response_format": response_format,
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "extra_body": extra_body,
            },
        )
        cfg = invocation.provider_cfg
        payload, headers = await _transport.arequest_json(
            self, "POST", cfg["path"], json=invocation.body, provider_cfg=cfg
        )
        return self._build_result(payload, expects, headers=headers, n_assets=invocation.asset_count, strict=strict)

    def stream(  # noqa: PLR0913 - the generation parameters
        self,
        task: dict,
        *,
        expects: str | None = None,
        parse_points: bool = False,
        model: str | None = None,
        provider: str | None = None,
        reasoning: bool | None = None,
        reasoning_effort: str | None = None,
        enable_audio_in_video: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async :meth:`Client.stream`: returns an async iterator of the same events."""
        _reject_unexpected_kwargs("stream", kwargs)
        try:
            invocation = self._prepare_invocation(
                task,
                expects=expects,
                stream=True,
                gen_kwargs={
                    "model": model,
                    "provider": provider,
                    "reasoning": reasoning,
                    "reasoning_effort": reasoning_effort,
                    "enable_audio_in_video": enable_audio_in_video,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "response_format": response_format,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": parallel_tool_calls,
                    "extra_body": extra_body,
                    "stream_options": stream_options,
                },
            )
        except SDKError as exc:
            return _aerror_events(exc)
        return self._stream_events(invocation, parse_points=parse_points, strict=strict)

    async def _stream_events(
        self, invocation: _PreparedInvocation, *, parse_points: bool, strict: bool
    ) -> AsyncIterator[dict[str, Any]]:
        cfg = invocation.provider_cfg
        try:
            response, closer = await _transport.aopen_stream(
                self, "POST", cfg["path"], json=invocation.body, provider_cfg=cfg
            )
        except SDKError as exc:  # HTTP errors carry the full body (it is read before mapping)
            yield _error_event(exc)
            return
        processor = self._stream_processor(invocation, response, parse_points=parse_points, strict=strict)
        async with closer:
            try:
                async for data in _transport.aiter_sse_data(_transport.aiter_response_lines(response)):
                    for event in processor.feed(data):
                        yield event
                    if processor.done:
                        break
            except SDKError as exc:
                yield processor.error_event(exc)
                return
        yield processor.terminal_event()


# ---------------------------------------------------------------------------
# Response format helpers for constrained decoding
# ---------------------------------------------------------------------------


def json_schema_format(
    schema: dict[str, Any],
    *,
    name: str = "response",
    strict: bool | None = None,
) -> JsonSchemaFormat:
    """Create a JSON schema response format for constrained decoding.

    Args:
        schema: JSON Schema object defining the expected output structure.
        name: A name for this schema (used for identification in logs/errors).
        strict: If True, enforce strict schema validation. Defaults to None (provider default).

    Returns:
        A response_format dict suitable for passing to generate/stream/perceive.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        ...     "required": ["name", "age"],
        ... }
        >>> result = client.generate(task, response_format=json_schema_format(schema))
    """
    json_schema_spec: JsonSchemaSpec = {"name": name, "schema": schema}
    if strict is not None:
        json_schema_spec["strict"] = strict
    return {"type": "json_schema", "json_schema": json_schema_spec}


def regex_format(pattern: str) -> RegexFormat:
    """Create a regex response format for constrained decoding.

    Args:
        pattern: A regular expression pattern that the output must match.

    Returns:
        A response_format dict suitable for passing to generate/stream/perceive.

    Example:
        >>> # Constrain output to a valid email address format
        >>> result = client.generate(
        ...     task, response_format=regex_format(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
        ... )
    """
    return {"type": "regex", "regex": pattern}


def pydantic_format(
    model: type,
    *,
    name: str | None = None,
    strict: bool | None = None,
) -> JsonSchemaFormat:
    """Create a JSON schema response format from a Pydantic model.

    This is a convenience wrapper that extracts the JSON schema from a Pydantic
    model class and passes it to the constrained decoding engine.

    Args:
        model: A Pydantic model class (subclass of pydantic.BaseModel).
        name: Optional name for the schema. Defaults to the model's class name.
        strict: If True, enforce strict schema validation. Defaults to None (provider default).

    Returns:
        A response_format dict suitable for passing to generate/stream/perceive.

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     email: str | None = None
        >>>
        >>> result = client.generate(task, response_format=pydantic_format(Person))
        >>> person = Person.model_validate_json(result.text)

    Note:
        Requires pydantic to be installed. The model must be a Pydantic v2 model
        (subclass of pydantic.BaseModel with model_json_schema method).
    """
    # Check if it's a Pydantic model
    if not hasattr(model, "model_json_schema"):
        raise TypeError(
            f"Expected a Pydantic model class with model_json_schema method, got {type(model).__name__}. "
            "Make sure you're using Pydantic v2."
        )

    # Extract JSON schema from the Pydantic model
    schema = model.model_json_schema()

    # Use model class name as default schema name
    schema_name = name if name is not None else model.__name__

    return json_schema_format(schema, name=schema_name, strict=strict)
