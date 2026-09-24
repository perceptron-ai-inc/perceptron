"""Message-level API: ``client.chat.completions.create(messages=[...])``.

OpenAI-compatible and lossless: messages are sent verbatim (never merged), only the fields the caller set are sent,
and results are typed dataclasses with ``to_dict()``. Tool calling round trip::

    completion = client.chat.completions.create(messages=messages, tools=[function_tool("get_weather", ...)])
    messages.append(completion.message)  # replayed verbatim (reasoning_content, tool_calls, arguments)
    for call in completion.tool_calls or []:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": run(call.name, call.parse_arguments())})

``create(stream=True)`` sends the request eagerly (HTTP errors raise before it returns) and returns a stream of
:class:`ChatCompletionChunk` objects that ends in a :class:`ChatCompletion` (``get_final_completion()``).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import warnings
import weakref
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from . import _transport
from ._lowering import count_assets, entry_to_part
from ._providers import PERCEPTRON_PROVIDER, _normalize_reasoning_effort, surface_model, surface_provider_cfg
from .dsl.nodes import DSLNode
from .errors import (
    ANCHOR_AMBIGUOUS,
    CONFLICTING_STRUCTURED_OUTPUT_CONTROLS,
    CONFLICTING_TOOLS,
    DUPLICATE_TOOL_NAME,
    INVALID_PARAMETER,
    INVALID_REGEX,
    INVALID_RESPONSE,
    INVALID_RESPONSE_FORMAT,
    INVALID_STREAM_CHUNK,
    INVALID_TEMPERATURE,
    INVALID_TOOL_ARGUMENTS,
    INVALID_TOOLS,
    INVALID_VISION_CONFIG,
    RESERVED_TOOL_NAME,
    STREAM_TRUNCATED,
    UNSUPPORTED_PARAMETER,
    UNSUPPORTED_RESPONSE_FORMAT,
    UNSUPPORTED_TOOL_CHOICE,
    UNSUPPORTED_TOOL_TYPE,
    UNSUPPORTED_TOOLS_COMBINATION,
    BadRequestError,
    IncompleteStreamError,
    ParseError,
    SDKError,
    ServerError,
)
from .pointing.parser import AnnotationCollection, collect_annotations, resolve_asset_idx

__all__ = [
    "AsyncChat",
    "AsyncChatCompletionStream",
    "AsyncChatCompletions",
    "Chat",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "ChatCompletionStream",
    "ChatCompletions",
    "Choice",
    "ChoiceDelta",
    "ChunkChoice",
    "FunctionCall",
    "FunctionCallDelta",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
    "function_tool",
]


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _arguments_string(value: Any) -> str:
    # The gateway always sends a string; other OpenAI-compatible servers may send an object.
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value)


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass
class FunctionCall:
    """The function a tool call names, with its ``arguments`` exactly as the model wrote them (a JSON string)."""

    name: str
    arguments: str

    @classmethod
    def from_dict(cls, data: Any) -> FunctionCall:
        data = data if isinstance(data, Mapping) else {}
        return cls(name=_str_or_none(data.get("name")) or "", arguments=_arguments_string(data.get("arguments")))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    """One call the model asked the caller to run. Execute calls only when ``finish_reason == "tool_calls"``."""

    id: str
    function: FunctionCall
    type: str = "function"

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def arguments(self) -> str:
        return self.function.arguments

    def parse_arguments(self) -> Any:
        """``json.loads(arguments)``; raises ``ParseError(code="invalid_tool_arguments")`` for invalid JSON."""
        try:
            return json.loads(self.function.arguments)
        except (TypeError, ValueError) as exc:
            raise ParseError(
                f"Tool call {self.id!r} ({self.name}) has arguments that are not valid JSON.",
                code=INVALID_TOOL_ARGUMENTS,
                details={"tool_call_id": self.id, "name": self.name, "arguments": self.arguments},
            ) from exc

    @classmethod
    def from_dict(cls, data: Any) -> ToolCall:
        data = data if isinstance(data, Mapping) else {}
        return cls(
            id=_str_or_none(data.get("id")) or "",
            function=FunctionCall.from_dict(data.get("function")),
            type=_str_or_none(data.get("type")) or "function",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function.to_dict()}


@dataclass
class Usage:
    """Token usage. ``prompt_tokens_details`` holds e.g. ``audio_tokens``; any field may be None."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: dict | None = None

    @property
    def audio_tokens(self) -> int | None:
        """``prompt_tokens_details.audio_tokens``; None when the server did not report it."""
        return self._detail("audio_tokens")

    @property
    def cached_tokens(self) -> int | None:
        """``prompt_tokens_details.cached_tokens``; None when the server did not report it."""
        return self._detail("cached_tokens")

    def _detail(self, name: str) -> int | None:
        details = self.prompt_tokens_details
        return _int_or_none(details.get(name)) if isinstance(details, Mapping) else None

    @classmethod
    def from_dict(cls, data: Any) -> Usage | None:
        if not isinstance(data, Mapping):
            return None
        details = data.get("prompt_tokens_details")
        return cls(
            prompt_tokens=_int_or_none(data.get("prompt_tokens")),
            completion_tokens=_int_or_none(data.get("completion_tokens")),
            total_tokens=_int_or_none(data.get("total_tokens")),
            prompt_tokens_details=dict(details) if isinstance(details, Mapping) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.prompt_tokens_details is not None:
            out["prompt_tokens_details"] = dict(self.prompt_tokens_details)
        return out


@dataclass
class ChatCompletionMessage:
    """An assistant message. ``to_dict()`` is its replay form: append it to ``messages`` unchanged."""

    role: str
    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @classmethod
    def from_dict(cls, data: Any) -> ChatCompletionMessage:
        data = data if isinstance(data, Mapping) else {}
        content = data.get("content")
        if isinstance(content, list):  # text parts from other OpenAI-compatible servers
            content = "".join(
                p.get("text") or "" for p in content if isinstance(p, Mapping) and p.get("type") == "text"
            )
        raw_calls = data.get("tool_calls")
        calls = [ToolCall.from_dict(c) for c in raw_calls] if isinstance(raw_calls, list) else []
        return cls(
            role=_str_or_none(data.get("role")) or "assistant",
            content=_str_or_none(content),
            reasoning_content=_str_or_none(data.get("reasoning_content")),
            tool_calls=calls or None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            out["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            out["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return out

    def annotations(self, *, expects: str | None = None, strict: bool = False) -> AnnotationCollection:
        """The annotation markup in ``content``, flattened into ``points``/``boxes``/``polygons``/``clips`` (collection
        children and track waypoints included, with the context their markup gives them) plus ``tracks`` and
        ``parsed``. Lenient: malformed elements stay text and are listed in ``errors``; ``strict=True`` raises
        ``ParseError``. ``expects`` (``"point"``, ``"box"``, ``"polygon"``, ``"clip"``) keeps one tag family; any
        other value raises ``ValueError``."""
        return _annotations(self.content, expects, strict)


def _annotations(content: str | None, expects: str | None, strict: bool) -> AnnotationCollection:
    if expects is not None and expects not in _ANNOTATION_FORMATS:
        raise ValueError(f"expects must be one of {', '.join(map(repr, _ANNOTATION_FORMATS))} or None, not {expects!r}")
    return collect_annotations(content or "", expects=expects, strict=strict)


@dataclass
class Choice:
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None

    @classmethod
    def from_dict(cls, data: Any, position: int = 0) -> Choice:
        data = data if isinstance(data, Mapping) else {}
        index = _int_or_none(data.get("index"))
        return cls(
            index=position if index is None else index,
            message=ChatCompletionMessage.from_dict(data.get("message")),
            finish_reason=_str_or_none(data.get("finish_reason")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "message": self.message.to_dict(), "finish_reason": self.finish_reason}


# `interrupted` (a force-closed think budget) still carries a full answer; `length` and the rest do not.
_COMPLETE_FINISH_REASONS = frozenset({"stop", "tool_calls", "interrupted"})


@dataclass
class ChatCompletion:
    """A chat completion.

    ``request_id`` is the ``x-trace-id`` response header, ``raw`` the response JSON (None for streams), ``asset_count``
    the number of media assets in the request (the ``asset_idx`` space), and ``done`` False for a stream that ended
    before ``[DONE]``.
    """

    id: str | None
    object: str | None
    created: int | None
    model: str | None
    choices: list[Choice]
    usage: Usage | None = None
    request_id: str | None = None
    raw: dict | None = None
    asset_count: int | None = None
    done: bool = True

    @property
    def message(self) -> ChatCompletionMessage | None:
        return self.choices[0].message if self.choices else None

    @property
    def text(self) -> str | None:
        message = self.message
        return message.content if message is not None else None

    @property
    def reasoning(self) -> str | None:
        message = self.message
        return message.reasoning_content if message is not None else None

    @property
    def tool_calls(self) -> list[ToolCall] | None:
        message = self.message
        return message.tool_calls if message is not None else None

    @property
    def finish_reason(self) -> str | None:
        return self.choices[0].finish_reason if self.choices else None

    @property
    def complete(self) -> bool:
        """True for a finished answer: ``stop``/``tool_calls``/``interrupted``, with tool calls present exactly when
        ``finish_reason == "tool_calls"`` (and, for a stream, ``[DONE]`` seen). ``length`` is incomplete."""
        finish_reason = self.finish_reason
        return (
            self.done
            and finish_reason in _COMPLETE_FINISH_REASONS
            and (finish_reason == "tool_calls") == bool(self.tool_calls)
        )

    def annotations(self, *, expects: str | None = None, strict: bool = False) -> AnnotationCollection:
        """The annotations in the answer (see :meth:`ChatCompletionMessage.annotations`)."""
        return _annotations(self.text, expects, strict)

    def resolve_asset_idx(self, annotation: Any) -> int | None:
        """The request asset an annotation from this answer refers to: its own (or inherited) ``asset_idx``, else the
        last asset (``asset_count - 1``). ``ValueError`` when the ``asset_idx`` is out of range; None without assets.
        Pass flattened annotations (from :meth:`annotations`) so container selectors are applied."""
        return resolve_asset_idx(annotation, self.asset_count)

    @classmethod
    def from_dict(cls, data: Any, *, request_id: str | None = None, asset_count: int | None = None) -> ChatCompletion:
        data = data if isinstance(data, Mapping) else {}
        raw_choices = data.get("choices")
        choices = [Choice.from_dict(c, i) for i, c in enumerate(raw_choices)] if isinstance(raw_choices, list) else []
        return cls(
            id=_str_or_none(data.get("id")),
            object=_str_or_none(data.get("object")),
            created=_int_or_none(data.get("created")),
            model=_str_or_none(data.get("model")),
            choices=choices,
            usage=Usage.from_dict(data.get("usage")),
            request_id=request_id,
            raw=data if isinstance(data, dict) else dict(data),
            asset_count=asset_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """The wire shape of the response."""
        out: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.to_dict() for choice in self.choices],
        }
        if self.usage is not None:
            out["usage"] = self.usage.to_dict()
        return out


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


@dataclass
class FunctionCallDelta:
    name: str | None = None
    arguments: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> FunctionCallDelta:
        data = data if isinstance(data, Mapping) else {}
        arguments = data.get("arguments")
        return cls(
            name=_str_or_none(data.get("name")), arguments=None if arguments is None else _arguments_string(arguments)
        )

    def to_dict(self) -> dict[str, Any]:
        return _without_none({"name": self.name, "arguments": self.arguments})


@dataclass
class ToolCallDelta:
    """A tool-call fragment: the first piece for an ``index`` carries ``id``/``type``/``function.name``; later pieces
    carry ``function.arguments`` to append."""

    index: int
    id: str | None = None
    type: str | None = None
    function: FunctionCallDelta | None = None

    @classmethod
    def from_dict(cls, data: Any, position: int = 0) -> ToolCallDelta:
        data = data if isinstance(data, Mapping) else {}
        index = _int_or_none(data.get("index"))
        function = data.get("function")
        return cls(
            index=position if index is None else index,
            id=_str_or_none(data.get("id")),
            type=_str_or_none(data.get("type")),
            function=FunctionCallDelta.from_dict(function) if isinstance(function, Mapping) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out = _without_none({"index": self.index, "id": self.id, "type": self.type})
        if self.function is not None:
            out["function"] = self.function.to_dict()
        return out


@dataclass
class ChoiceDelta:
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None

    @classmethod
    def from_dict(cls, data: Any) -> ChoiceDelta:
        data = data if isinstance(data, Mapping) else {}  # absent and `null` deltas are empty
        raw_calls = data.get("tool_calls")
        calls = [ToolCallDelta.from_dict(c, i) for i, c in enumerate(raw_calls)] if isinstance(raw_calls, list) else []
        return cls(
            role=_str_or_none(data.get("role")),
            content=_str_or_none(data.get("content")),
            reasoning_content=_str_or_none(data.get("reasoning_content")),
            tool_calls=calls or None,
        )

    def to_dict(self) -> dict[str, Any]:
        out = _without_none({"role": self.role, "content": self.content, "reasoning_content": self.reasoning_content})
        if self.tool_calls:
            out["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return out


@dataclass
class ChunkChoice:
    index: int
    delta: ChoiceDelta
    finish_reason: str | None = None

    @classmethod
    def from_dict(cls, data: Any, position: int = 0) -> ChunkChoice:
        data = data if isinstance(data, Mapping) else {}
        index = _int_or_none(data.get("index"))
        return cls(
            index=position if index is None else index,
            delta=ChoiceDelta.from_dict(data.get("delta")),
            finish_reason=_str_or_none(data.get("finish_reason")),
        )

    def to_dict(self) -> dict[str, Any]:
        return _without_none({"index": self.index, "delta": self.delta.to_dict(), "finish_reason": self.finish_reason})


@dataclass
class ChatCompletionChunk:
    """One stream event. The trailing usage chunk has ``choices == []``."""

    id: str | None
    object: str | None
    created: int | None
    model: str | None
    choices: list[ChunkChoice]
    usage: Usage | None = None

    @classmethod
    def from_dict(cls, data: Any) -> ChatCompletionChunk:
        data = data if isinstance(data, Mapping) else {}
        raw_choices = data.get("choices")
        choices = (
            [ChunkChoice.from_dict(c, i) for i, c in enumerate(raw_choices)] if isinstance(raw_choices, list) else []
        )
        return cls(
            id=_str_or_none(data.get("id")),
            object=_str_or_none(data.get("object")),
            created=_int_or_none(data.get("created")),
            model=_str_or_none(data.get("model")),
            choices=choices,
            usage=Usage.from_dict(data.get("usage")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.to_dict() for choice in self.choices],
        }
        if self.usage is not None:
            out["usage"] = self.usage.to_dict()
        return out


def function_tool(
    name: str,
    *,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> dict[str, Any]:
    """A function tool declaration for ``tools=[...]``. ``parameters`` (a JSON Schema object) is sent as given, key
    order included; the model sees properties in that order."""
    function: dict[str, Any] = {"name": name}
    if description is not None:
        function["description"] = description
    if parameters is not None:
        function["parameters"] = parameters
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


# ---------------------------------------------------------------------------
# Stream accumulation
# ---------------------------------------------------------------------------


class ChatStreamAccumulator:
    """Folds stream chunk dicts into one :class:`ChatCompletion`.

    Tolerates ``delta: null``/absent and ``choices: []``; reads ``usage`` before the choices (last non-null wins);
    assembles tool calls by ``index`` (``id``/``type``/``name`` from the first non-empty value, ``arguments``
    concatenated, ordered by index). :meth:`mark_done` records ``[DONE]``; a snapshot without it is incomplete.
    """

    def __init__(self, *, request_id: str | None = None, asset_count: int | None = None) -> None:
        self.request_id = request_id
        self.asset_count = asset_count
        self.id: str | None = None
        self.model: str | None = None
        self.created: int | None = None
        self.role: str | None = None
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.done = False
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, dict[str, Any]] = {}

    def feed(self, obj: Mapping[str, Any]) -> ChatCompletionChunk:
        """Fold one decoded chunk in and return it parsed."""
        chunk = ChatCompletionChunk.from_dict(obj)
        self.id = self.id or chunk.id
        self.model = self.model or chunk.model
        if self.created is None:
            self.created = chunk.created
        usage = obj.get("usage") if isinstance(obj, Mapping) else None
        if isinstance(usage, Mapping):
            self.usage = dict(usage)
        for choice in chunk.choices:
            delta = choice.delta
            if delta.role:
                self.role = delta.role
            if delta.content:
                self._content.append(delta.content)
            if delta.reasoning_content:
                self._reasoning.append(delta.reasoning_content)
            for piece in delta.tool_calls or []:
                call = self._calls.setdefault(piece.index, {"id": None, "type": None, "name": None, "arguments": []})
                call["id"] = call["id"] or piece.id
                call["type"] = call["type"] or piece.type
                if piece.function is not None:
                    call["name"] = call["name"] or piece.function.name
                    if piece.function.arguments:
                        call["arguments"].append(piece.function.arguments)
            if choice.finish_reason is not None:
                self.finish_reason = choice.finish_reason
        return chunk

    def mark_done(self) -> None:
        """Record the ``[DONE]`` sentinel."""
        self.done = True

    @property
    def text(self) -> str | None:
        return "".join(self._content) if self._content else None

    @property
    def reasoning(self) -> str | None:
        return "".join(self._reasoning) if self._reasoning else None

    @property
    def tool_calls(self) -> list[ToolCall] | None:
        if not self._calls:
            return None
        return [
            ToolCall(
                id=call["id"] or "",
                function=FunctionCall(name=call["name"] or "", arguments="".join(call["arguments"])),
                type=call["type"] or "function",
            )
            for _, call in sorted(self._calls.items())
        ]

    def snapshot(self) -> ChatCompletion:
        """The completion so far (``done`` False until ``[DONE]``, so an error's ``partial`` is never complete)."""
        message = ChatCompletionMessage(
            role=self.role or "assistant",
            content=self.text,
            reasoning_content=self.reasoning,
            tool_calls=self.tool_calls,
        )
        return ChatCompletion(
            id=self.id,
            object="chat.completion",
            created=self.created,
            model=self.model,
            choices=[Choice(index=0, message=message, finish_reason=self.finish_reason)],
            usage=Usage.from_dict(self.usage),
            request_id=self.request_id,
            raw=None,
            asset_count=self.asset_count,
            done=self.done,
        )


def _decode_chunk(data: str) -> dict[str, Any]:
    try:
        obj = json.loads(data)
    except ValueError as exc:
        raise ServerError(f"Malformed stream chunk: {data[:200]!r}", code=INVALID_STREAM_CHUNK) from exc
    if not isinstance(obj, dict):
        raise ServerError(f"Malformed stream chunk: {data[:200]!r}", code=INVALID_STREAM_CHUNK)
    return obj


def _attach_stream_context(exc: SDKError, accumulator: ChatStreamAccumulator) -> None:
    if exc.partial is None:
        exc.partial = accumulator.snapshot()
    if exc.request_id is None and accumulator.request_id is not None:
        exc.request_id = accumulator.request_id
        exc.details.setdefault("request_id", accumulator.request_id)


class _StreamBase:
    def __init__(self, response: Any, closer: Any, *, request_id: str | None, asset_count: int | None) -> None:
        self.response = response
        self.request_id = request_id
        self.completion: ChatCompletion | None = None
        self._closer = closer
        self._error: SDKError | None = None
        self._accumulator = ChatStreamAccumulator(request_id=request_id, asset_count=asset_count)

    def _handle_payload(self, data: Any) -> ChatCompletionChunk | None:
        """Fold one SSE payload in; None at ``[DONE]``. Raises for error events and malformed chunks."""
        if data is _transport.DONE:
            self._accumulator.mark_done()
            return None
        obj = _decode_chunk(data)
        if obj.get("error") is not None:
            raise _transport.stream_error_from_event(obj["error"], request_id=self.request_id)
        return self._accumulator.feed(obj)

    def _finish(self) -> None:
        if not self._accumulator.done:
            raise IncompleteStreamError("The stream ended before [DONE].", code=STREAM_TRUNCATED)
        self.completion = self._accumulator.snapshot()

    def _fail(self, exc: SDKError) -> None:
        _attach_stream_context(exc, self._accumulator)
        self._error = exc

    def _final(self) -> ChatCompletion:
        if self.completion is not None:
            return self.completion
        if self._error is not None:
            raise self._error
        raise IncompleteStreamError(
            "The stream was closed before it finished.",
            code=STREAM_TRUNCATED,
            request_id=self.request_id,
            partial=self._accumulator.snapshot(),
        )


class ChatCompletionStream(_StreamBase):
    """A streaming chat completion: iterate :class:`ChatCompletionChunk` objects.

    ``.completion`` holds the :class:`ChatCompletion` once the stream is exhausted; ``get_final_completion()`` consumes
    the rest and returns it. Error events, malformed chunks and a missing ``[DONE]`` raise mapped errors carrying
    ``.partial``. The stream owns its connection and closes it when exhausted, on error, on ``close()``, on exit, or
    when it is garbage collected unfinished (never iterated, or left mid-way).
    """

    def __init__(self, response: Any, closer: Any, *, request_id: str | None = None, asset_count: int | None = None):
        super().__init__(response, closer, request_id=request_id, asset_count=asset_count)
        # Holds the closer, not the stream, so a dropped stream can still be collected (and then closed).
        self._finalizer = weakref.finalize(self, closer.close)
        self._iterator = self._iterate()

    def _iterate(self) -> Iterator[ChatCompletionChunk]:
        try:
            for data in _transport.iter_sse_data(_transport.iter_response_lines(self.response)):
                chunk = self._handle_payload(data)
                if chunk is None:
                    break
                yield chunk
            self._finish()
        except SDKError as exc:
            self._fail(exc)
            raise
        finally:
            self._release()

    def _release(self) -> None:
        self._finalizer()  # closes the connection the first time; later calls, and garbage collection, do nothing

    def __iter__(self) -> ChatCompletionStream:
        return self

    def __next__(self) -> ChatCompletionChunk:
        return next(self._iterator)

    def __enter__(self) -> ChatCompletionStream:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """Stop reading and release the connection."""
        self._iterator.close()
        self._release()

    def get_final_completion(self) -> ChatCompletion:
        """Consume the rest of the stream and return the final :class:`ChatCompletion`."""
        for _ in self:
            pass
        return self._final()


class AsyncChatCompletionStream(_StreamBase):
    """Async :class:`ChatCompletionStream` (``async for``, ``async with``, ``await close()``,
    ``await get_final_completion()``).

    Use ``async with`` (or ``await close()``) when you may stop early: closing needs the event loop. A stream garbage
    collected unfinished is closed on its event loop while that loop runs; after the loop stops it can only be
    reported (``ResourceWarning``), as asyncio does for unclosed transports.
    """

    def __init__(self, response: Any, closer: Any, *, request_id: str | None = None, asset_count: int | None = None):
        super().__init__(response, closer, request_id=request_id, asset_count=asset_count)
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()  # the loop the connection belongs to
        except RuntimeError:
            loop = None
        # Holds the closer and the loop, not the stream, so a dropped stream can still be collected.
        self._finalizer = weakref.finalize(self, _close_dropped_async_stream, loop, closer)
        self._iterator = self._aiterate()

    async def _aiterate(self) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async for data in _transport.aiter_sse_data(_transport.aiter_response_lines(self.response)):
                chunk = self._handle_payload(data)
                if chunk is None:
                    break
                yield chunk
            self._finish()
        except SDKError as exc:
            self._fail(exc)
            raise
        finally:
            await self._release()

    async def _release(self) -> None:
        if self._finalizer.detach() is not None:  # the first release; garbage collection then does nothing
            await self._closer.aclose()

    def __aiter__(self) -> AsyncChatCompletionStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        return await self._iterator.__anext__()

    async def __aenter__(self) -> AsyncChatCompletionStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Stop reading and release the connection."""
        await self._iterator.aclose()
        await self._release()

    async def get_final_completion(self) -> ChatCompletion:
        """Consume the rest of the stream and return the final :class:`ChatCompletion`."""
        async for _ in self:
            pass
        return self._final()


# Closes of dropped async streams scheduled on their loop, kept referenced until they finish.
_PENDING_CLOSES: set[asyncio.Task] = set()


def _close_dropped_async_stream(loop: asyncio.AbstractEventLoop | None, closer: Any) -> None:
    """Finalizer of an :class:`AsyncChatCompletionStream` collected unclosed; it must not reference the stream.

    It cannot await: while the stream's loop runs, the close is scheduled there. Otherwise nothing can be closed
    synchronously (httpx closes async clients and responses only on their loop; asyncio closes the sockets when their
    transports are collected), so it warns, like asyncio does for unclosed transports.
    """
    if loop is not None and loop.is_running():
        with suppress(RuntimeError):  # the loop closed meanwhile
            loop.call_soon_threadsafe(_schedule_close, loop, closer)
            return
    warnings.warn(
        "An AsyncChatCompletionStream was garbage collected unclosed while its event loop was not running, so its "
        "connection was not released; use 'async with' or 'await stream.close()'.",
        ResourceWarning,
        stacklevel=1,  # raised during garbage collection: there is no caller to point at
    )


def _schedule_close(loop: asyncio.AbstractEventLoop, closer: Any) -> None:
    task = loop.create_task(closer.aclose())
    _PENDING_CLOSES.add(task)
    task.add_done_callback(_close_done)


def _close_done(task: asyncio.Task) -> None:
    _PENDING_CLOSES.discard(task)
    if not task.cancelled():
        task.exception()  # retrieved: nobody awaits this close, so a failure must not be logged as unretrieved


# ---------------------------------------------------------------------------
# Request validation (shared with the legacy Client.generate/stream path)
# ---------------------------------------------------------------------------

# OpenAI parameters the gateway silently drops (or rejects); refuse them instead of sending nothing.
_UNSUPPORTED_OPENAI_PARAMS = {
    "stop": "",
    "seed": "",
    "logprobs": "",
    "top_logprobs": "",
    "logit_bias": "",
    "user": "",
    "metadata": "",
    "store": "",
    "service_tier": "",
    "modalities": "",
    "audio": " (send audio as input_audio/audio_url content parts)",
    "prediction": "",
    "web_search_options": "",
    "verbosity": "",
    "prompt_cache_key": "",
    "safety_identifier": "",
    "cache_salt": "",
    "functions": "; declare tools=[function_tool(...)] instead",
    "function_call": "; use tool_choice instead",
}

_ANNOTATION_FORMATS = ("point", "box", "polygon", "clip")
_VISION_CONFIG_KEYS = ("annotation_format", "enable_audio_in_video", "enable_thinking")
_TOOL_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]+")
_RESERVED_TOOL_RECIPIENT = "functions.parallel"


def _reject_unknown_kwargs(method: str, kwargs: Mapping[str, Any]) -> None:
    for name in kwargs:
        if name in _UNSUPPORTED_OPENAI_PARAMS:
            raise TypeError(f"{name!r} is not supported by the Perceptron API{_UNSUPPORTED_OPENAI_PARAMS[name]}.")
        raise TypeError(f"{method}() got an unexpected keyword argument {name!r}")


def _sanitize_tool_name_part(value: str) -> str:
    return _TOOL_NAME_UNSAFE.sub("_", value.strip()).strip("_") or "tool"


def tool_recipient(name: str) -> str:
    """The name the model sees for a declared function: ``ns.name`` for a dotted name, else ``functions.<name>``, with
    runs of characters outside ``[A-Za-z0-9_.-]`` replaced by ``_``. Two tools may not share a recipient."""
    namespace, sep, rest = name.strip().partition(".")
    if sep and namespace and rest:
        return f"{_sanitize_tool_name_part(namespace)}.{_sanitize_tool_name_part(rest)}"
    return f"functions.{_sanitize_tool_name_part(name)}"


def _validate_tools(tools: Any) -> None:
    if tools is None:
        return
    if not isinstance(tools, (list, tuple)):
        raise TypeError("tools must be a list of tool dicts (see function_tool())")
    recipients: dict[str, str] = {}
    has_focus = False
    for i, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise TypeError(f"tools[{i}] must be a dict; got {type(tool).__name__}")
        tool_type = tool.get("type")
        if tool_type == "perceptron.FOCUS":
            has_focus = True
            continue
        if tool_type != "function":
            raise BadRequestError(
                f"tools[{i}].type must be 'function'; got {tool_type!r}.",
                code=UNSUPPORTED_TOOL_TYPE,
                param=f"tools[{i}].type",
            )
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name.strip():
            raise BadRequestError(
                f"tools[{i}].function.name must be a non-empty string.",
                code=INVALID_TOOLS,
                param=f"tools[{i}].function.name",
            )
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise BadRequestError(
                f"tools[{i}].function.parameters must be a JSON Schema object.",
                code=INVALID_TOOLS,
                param=f"tools[{i}].function.parameters",
            )
        recipient = tool_recipient(name)
        if recipient == _RESERVED_TOOL_RECIPIENT:
            raise BadRequestError(
                f"Tool name {name!r} is reserved (it resolves to {_RESERVED_TOOL_RECIPIENT}).",
                code=RESERVED_TOOL_NAME,
                param=f"tools[{i}].function.name",
            )
        if recipient in recipients:
            raise BadRequestError(
                f"Tool names {recipients[recipient]!r} and {name!r} both resolve to {recipient!r}.",
                code=DUPLICATE_TOOL_NAME,
                param=f"tools[{i}].function.name",
            )
        recipients[recipient] = name
    if has_focus and len(tools) > 1:
        raise BadRequestError(
            "`perceptron.FOCUS` must be the only tool in `tools`.", code=CONFLICTING_TOOLS, param="tools"
        )


def _validate_temperature(temperature: Any) -> None:
    """A set ``temperature`` must be a finite number >= 0, as the API requires (its code: ``invalid_temperature``)."""
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or (isinstance(temperature, float) and not math.isfinite(temperature))
        or temperature < 0
    ):
        raise BadRequestError(
            f"temperature must be a finite number >= 0; got {temperature!r}.",
            code=INVALID_TEMPERATURE,
            param="temperature",
        )


def _validate_request_body(body: Mapping[str, Any]) -> None:
    """Checks over a request body that both ``create()`` and the legacy ``Client.generate/stream`` path run.

    ``temperature`` (a finite number >= 0), tools (shape, the recipient rule's reserved and duplicate names),
    ``tool_choice`` (``auto``/``none`` only), ``parallel_tool_calls``, and the structured-output rules:
    ``response_format`` is ``text`` or ``json_schema``; non-empty ``tools`` exclude ``regex`` and ``json_schema``;
    ``regex`` excludes any ``response_format``.
    """
    _validate_temperature(body.get("temperature"))
    tools = body.get("tools")
    _validate_tools(tools)
    tool_choice = body.get("tool_choice")
    if tool_choice is not None and tool_choice not in ("auto", "none"):
        raise BadRequestError(
            "Only tool_choice 'auto' and 'none' are supported; forcing a tool call is not supported yet.",
            code=UNSUPPORTED_TOOL_CHOICE,
            param="tool_choice",
        )
    parallel_tool_calls = body.get("parallel_tool_calls")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise BadRequestError(
            "parallel_tool_calls must be a bool.", code=INVALID_PARAMETER, param="parallel_tool_calls"
        )

    response_format = body.get("response_format")
    format_type = None
    if response_format is not None:
        format_type = response_format.get("type") if isinstance(response_format, Mapping) else None
        if format_type not in ("text", "json_schema"):
            raise BadRequestError(
                f"response_format type must be 'text' or 'json_schema'; got {format_type!r}.",
                code=UNSUPPORTED_RESPONSE_FORMAT,
                param="response_format.type",
            )
        if format_type == "json_schema" and not isinstance(response_format.get("json_schema"), Mapping):
            raise BadRequestError(
                "A json_schema response_format needs a 'json_schema' dict with 'name' and 'schema'.",
                code=INVALID_RESPONSE_FORMAT,
                param="response_format.json_schema",
            )
    regex = body.get("regex")
    if regex is not None and (not isinstance(regex, str) or not regex):
        raise BadRequestError("regex must be a non-empty string.", code=INVALID_REGEX, param="regex")
    if tools and (regex is not None or format_type == "json_schema"):
        raise BadRequestError(
            "`tools` cannot be combined with a `json_schema` `response_format` or `regex`.",
            code=UNSUPPORTED_TOOLS_COMBINATION,
            param="tools",
        )
    if regex is not None and response_format is not None:
        raise BadRequestError(
            "Send either `regex` or `response_format`, not both.",
            code=CONFLICTING_STRUCTURED_OUTPUT_CONTROLS,
            param="regex",
        )


def _normalize_vision_config(vision_config: Any) -> dict[str, Any] | None:
    if vision_config is None:
        return None
    if not isinstance(vision_config, Mapping):
        raise TypeError("vision_config must be a dict")
    normalized: dict[str, Any] = {}
    for key, value in vision_config.items():
        if key not in _VISION_CONFIG_KEYS:
            hint = " (Focus controls are retired; see 'Migrate from Mk1')" if key == "internal_tools" else ""
            raise BadRequestError(
                f"vision_config.{key} is not supported{hint}; supported keys: {', '.join(_VISION_CONFIG_KEYS)}.",
                code=UNSUPPORTED_PARAMETER,
                param=f"vision_config.{key}",
            )
        if key == "annotation_format" and value is not None:
            value = value.strip().lower() if isinstance(value, str) else value  # noqa: PLW2901
            if value not in _ANNOTATION_FORMATS:
                raise BadRequestError(
                    f"vision_config.annotation_format must be one of {', '.join(_ANNOTATION_FORMATS)}; "
                    f"got {vision_config[key]!r}.",
                    code=INVALID_VISION_CONFIG,
                    param="vision_config.annotation_format",
                )
        elif value is not None and not isinstance(value, bool):
            raise BadRequestError(
                f"vision_config.{key} must be a bool.", code=INVALID_VISION_CONFIG, param=f"vision_config.{key}"
            )
        if key == "enable_thinking":
            warnings.warn(
                "vision_config.enable_thinking is deprecated; use reasoning_effort instead.",
                DeprecationWarning,
                stacklevel=4,
            )
        normalized[key] = value
    return normalized


def _generation_params(  # noqa: PLR0913 - the sampling parameters
    settings: Any,
    *,
    max_completion_tokens: Any,
    max_tokens: Any,
    temperature: Any,
    top_p: Any,
    top_k: Any,
    frequency_penalty: Any,
    presence_penalty: Any,
) -> dict[str, Any]:
    """The generation parameters of ``create()`` and ``multilook()``, None where unset: ``max_tokens`` is an alias of
    ``max_completion_tokens`` (not both), and ``configure()`` defaults fill what the caller left unset."""
    if max_completion_tokens is not None and max_tokens is not None:
        raise TypeError("Pass max_completion_tokens or its alias max_tokens, not both.")
    if max_completion_tokens is None:
        max_completion_tokens = max_tokens if max_tokens is not None else settings.max_tokens
    if max_completion_tokens is not None and (
        not isinstance(max_completion_tokens, int)
        or isinstance(max_completion_tokens, bool)
        or max_completion_tokens < 1
    ):
        raise BadRequestError(
            f"max_completion_tokens must be an integer >= 1; got {max_completion_tokens!r}.",
            code=INVALID_PARAMETER,
            param="max_completion_tokens",
        )
    return {
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature if temperature is not None else settings.temperature,
        "top_p": top_p if top_p is not None else settings.top_p,
        "top_k": top_k if top_k is not None else settings.top_k,
        "frequency_penalty": frequency_penalty if frequency_penalty is not None else settings.frequency_penalty,
        "presence_penalty": presence_penalty if presence_penalty is not None else settings.presence_penalty,
    }


def _merge_extra_body(body: dict[str, Any], extra_body: Any) -> None:
    """Merge ``extra_body`` into ``body``: last, and unvalidated."""
    if extra_body is None:
        return
    if not isinstance(extra_body, Mapping):
        raise TypeError("extra_body must be a dict")
    body.update(extra_body)


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def _message_dict(item: Any, index: int, name: str = "messages") -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if isinstance(item, ChatCompletionMessage):
        return item.to_dict()
    message = getattr(item, "message", None)  # ChatCompletion, PerceiveResult
    if isinstance(message, ChatCompletionMessage):
        return message.to_dict()
    raise TypeError(
        f"{name}[{index}] must be a dict, a ChatCompletionMessage, a ChatCompletion or a PerceiveResult with a "
        f"message; got {type(item).__name__}"
    )


def _content_ledger(messages: list[Any] | tuple[Any, ...]) -> Any:
    """The media assets of the message dicts' content lists, numbered across every message (a tag's ``asset_idx``
    counts the whole request); None when no content list holds a DSL node. Message objects carry text only."""
    items: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, (list, tuple)):
            items.extend(content)
    if not any(isinstance(item, DSLNode) for item in items):
        return None
    from .dsl.perceive import _asset_ledger  # noqa: PLC0415

    return _asset_ledger(items)


# Compiled entries that are whole messages (``agent(..., tool_calls=...)`` and ``tool_result(...)``).
_TURN_ENTRY_TYPES = ("assistant_turn", "tool_result")

_PACKAGE_DIR = os.path.dirname(__file__) + os.sep  # this package: frames here are the SDK, not the caller


def _warn_caller(message: str, category: type[Warning]) -> None:
    """Warn at the caller's line: the first frame outside this package, however deep the SDK noticed the problem."""
    frame = sys._getframe(1)
    stacklevel = 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, category, stacklevel=stacklevel)


def _lower_dsl_node(  # noqa: PLR0913 - the lowering target, the request's assets and the error location
    node: DSLNode,
    *,
    base_url: str | None,
    provider_name: str | None,
    ledger: Any = None,
    first_asset: int = 0,
    where: str = "content",
) -> list[dict[str, Any]]:
    """A DSL node from message content as wire parts. Tags anchor in the request's ``asset_idx`` space (``ledger``,
    in which this node's first asset is ``first_asset``). There is no issue list here: an ``anchor_ambiguous`` tag (its
    ``image=``/``asset=`` node appears more than once, as when a replayed conversation reuses an image node) still
    names one asset and is sent with a ``UserWarning``, as ``perceive()`` records it; any other issue the DSL reports
    would send markup that does not mean what the caller wrote, so it raises ``BadRequestError`` naming ``where``."""
    # dsl.perceive imports the client, which imports this module.
    from .dsl.perceive import _compile  # noqa: PLC0415

    task, issues = _compile(node, expects=None, strict=False, ledger=ledger, first_asset=first_asset)
    for issue in issues:
        if issue["code"] != ANCHOR_AMBIGUOUS:
            raise BadRequestError(f"{where}: {issue['message']}", code=issue["code"], details=issue, param=where)
    for issue in issues:
        _warn_caller(f"{where}: {issue['message']}", UserWarning)
    entries = task.get("content") or []
    if any(entry.get("role", "user") != "user" or entry.get("type") in _TURN_ENTRY_TYPES for entry in entries):
        # Lowering them to parts would drop their role into the enclosing message.
        raise TypeError(
            f"{where}: system(), agent() and tool_result() are turns of their own, not content parts; send each as "
            "its own message"
        )
    return [entry_to_part(entry, base_url=base_url, provider_name=provider_name) for entry in entries]


def _lower_content(  # noqa: PLR0913 - the lowering target and the request's assets
    content: list | tuple,
    index: int,
    *,
    base_url: str | None,
    provider_name: str | None,
    name: str = "messages",
    ledger: Any = None,
    assets_before: int = 0,
) -> list[Any] | None:
    """The content list with strings and DSL nodes lowered to wire parts; None when it holds only part dicts.

    ``name`` is the argument named in errors (``messages``, or multilook's ``context``/``prompts``). ``ledger`` numbers
    the request's media assets (by default this content list's) and ``assets_before`` counts those before this content
    list, so tags get request-relative ``asset_idx`` values.
    """
    if all(isinstance(part, dict) for part in content):
        return None
    if ledger is None:
        ledger = _content_ledger([{"content": content}])
    parts: list[Any] = []
    for j, part in enumerate(content):
        if isinstance(part, dict):
            parts.append(part)
        elif isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, DSLNode):
            first_asset = assets_before + count_assets([{"content": parts}])  # media before this item
            parts.extend(
                _lower_dsl_node(
                    part,
                    base_url=base_url,
                    provider_name=provider_name,
                    ledger=ledger,
                    first_asset=first_asset,
                    where=f"{name}[{index}].content[{j}]",
                )
            )
        else:
            raise TypeError(
                f"{name}[{index}].content[{j}] must be a content-part dict, a str or a DSL node "
                f"(text(), image(), video(), audio()); got {type(part).__name__}"
            )
    return parts


def _normalize_messages(
    messages: Any, *, base_url: str | None, provider_name: str | None, name: str = "messages", extra_assets: int = 0
) -> list[dict[str, Any]]:
    """Messages as sent: dicts verbatim (every key, order kept, never merged; ``ToolCall`` objects in ``tool_calls`` via
    ``to_dict()``); message objects via ``to_dict()``; strings and DSL nodes inside a content list lowered to parts,
    with tags anchored across every message. ``name`` is the argument named in errors; ``extra_assets`` counts media
    that follow these messages in the request (multilook's prompts), so tags anchor in the whole request's
    ``asset_idx`` space."""
    if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, (list, tuple)):
        raise TypeError(f"{name} must be a list of message dicts")
    ledger = _content_ledger(messages)
    if ledger is not None:
        ledger.count += extra_assets
    normalized = []
    assets_before = 0
    for i, item in enumerate(messages):
        message = _message_dict(item, i, name)
        content = message.get("content")
        if isinstance(content, DSLNode):
            raise TypeError(f"{name}[{i}].content must be a str or a list; put DSL nodes in a list, e.g. [image(...)]")
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, (list, tuple)) and any(isinstance(call, ToolCall) for call in tool_calls):
            # A replayed turn built from `completion.tool_calls`, as `agent(tool_calls=...)` accepts them.
            calls = [call.to_dict() if isinstance(call, ToolCall) else call for call in tool_calls]
            message = {**message, "tool_calls": calls}
        if isinstance(content, (list, tuple)):
            lowered = _lower_content(
                content,
                i,
                base_url=base_url,
                provider_name=provider_name,
                name=name,
                ledger=ledger,
                assets_before=assets_before,
            )
            if lowered is not None:
                message = {**message, "content": lowered}
        normalized.append(message)
        assets_before += count_assets([message])
    return normalized


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@dataclass
class _ChatRequest:
    provider_cfg: dict[str, Any]
    body: dict[str, Any]
    timeout: float | None
    asset_count: int


def _build_request(  # noqa: PLR0913 - mirrors the create() signature
    client: Any,
    *,
    messages: Any,
    model: str | None,
    stream: bool,
    max_completion_tokens: int | None,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    response_format: Any,
    regex: str | None,
    reasoning_effort: str | None,
    vision_config: Any,
    tools: Any,
    tool_choice: Any,
    parallel_tool_calls: bool | None,
    stream_options: Any,
    n: int | None,
    extra_body: Any,
    timeout: float | None,
    kwargs: Mapping[str, Any],
) -> _ChatRequest:
    _reject_unknown_kwargs("create", kwargs)
    if stream_options is not None and not isinstance(stream_options, Mapping):
        raise TypeError("stream_options must be a dict, e.g. {'include_usage': True}")

    settings = client._settings
    cfg = surface_provider_cfg(client)
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
    provider_name = cfg["name"]
    resolved_model = surface_model(cfg, model, settings)
    normalized_messages = _normalize_messages(messages, base_url=cfg.get("base_url"), provider_name=provider_name)

    # `True == 1` and `1.0 == 1`, but the gateway parses `n` as an integer.
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n != 1):
        raise BadRequestError(f"Only n = 1 is supported; got n={n!r}.", code=INVALID_PARAMETER, param="n")
    if isinstance(response_format, Mapping) and response_format.get("type") == "regex":
        # SDK convenience (regex_format()): the API takes the pattern as top-level `regex`.
        if regex is not None:
            raise BadRequestError(
                "Send either `regex` or `response_format`, not both.",
                code=CONFLICTING_STRUCTURED_OUTPUT_CONTROLS,
                param="regex",
            )
        regex = response_format.get("regex")
        if regex is None:
            raise BadRequestError(
                "A regex response_format needs a 'regex' pattern.", code=INVALID_REGEX, param="response_format.regex"
            )
        response_format = None
    if stream and stream_options is None and provider_name == PERCEPTRON_PROVIDER:
        stream_options = {"include_usage": True}

    body: dict[str, Any] = {"model": resolved_model, "messages": normalized_messages}
    if stream:
        body["stream"] = True
    optional: dict[str, Any] = {
        "stream_options": dict(stream_options) if stream_options is not None else None,
        **params,
        "response_format": response_format,
        "regex": regex,
        "reasoning_effort": _normalize_reasoning_effort(reasoning_effort),
        "vision_config": _normalize_vision_config(vision_config),
        "tools": list(tools) if isinstance(tools, tuple) else tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "n": n,
    }
    body.update((key, value) for key, value in optional.items() if value is not None)
    _validate_request_body(body)
    _merge_extra_body(body, extra_body)
    return _ChatRequest(provider_cfg=cfg, body=body, timeout=timeout, asset_count=count_assets(normalized_messages))


def _completion_from_payload(payload: Any, headers: Any, asset_count: int) -> ChatCompletion:
    request_id = _transport.header_value(headers, _transport.TRACE_ID_HEADER)
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ServerError(
            "The server returned a chat completion without choices.",
            code=INVALID_RESPONSE,
            request_id=request_id,
            details={"response": payload},
        )
    return ChatCompletion.from_dict(payload, request_id=request_id, asset_count=asset_count)


class ChatCompletions:
    """``client.chat.completions``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(  # noqa: PLR0913 - the OpenAI chat-completions parameters
        self,
        *,
        messages: list[Any],
        model: str | None = None,
        stream: bool = False,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        regex: str | None = None,
        reasoning_effort: str | None = None,
        vision_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        n: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | ChatCompletionStream:
        """Create a chat completion (``POST /chat/completions``).

        ``messages`` are sent verbatim: dicts (any OpenAI message, including ``tool`` results and assistant
        ``tool_calls``/``reasoning_content``/``content: None``), or returned messages/completions. Inside a dict's
        ``content`` list, strings and DSL nodes (``text()``, ``image()``, ``video()``, ``audio()``, ``video_frames()``,
        tags) become parts. A tag's ``asset_idx`` counts the media of every message. A tag whose ``image=``/``asset=``
        node appears more than once (e.g. a replayed conversation that reuses an image node) is anchored to the node's
        latest use before the tag (else its first use after it) with a ``UserWarning``; any other tag the DSL would
        flag (no ``image=``/``asset=`` in a multi-asset request, an anchor outside the request, a coordinate off the
        0-1000 grid, ...) raises ``BadRequestError``.
        Only the parameters you set are sent (``configure()`` generation defaults count as set); ``max_tokens`` is an
        alias of ``max_completion_tokens``; ``response_format={"type": "regex", "regex": p}`` is sent as ``regex``.
        ``extra_body`` is merged into the body last, unvalidated. Streams to provider ``perceptron`` request usage
        (``stream_options={"include_usage": True}``) unless you pass ``stream_options``.

        Uses the provider you chose explicitly (``Client(provider=...)``, ``configure(provider=...)``,
        ``PERCEPTRON_PROVIDER``), otherwise ``perceptron``. Invalid values raise ``BadRequestError`` before any
        request; unknown or unsupported OpenAI parameters raise ``TypeError``.

        Returns a :class:`ChatCompletion`, or with ``stream=True`` a :class:`ChatCompletionStream` (the request is
        sent before it returns, so HTTP errors raise here).
        """
        request = _build_request(
            self._client,
            messages=messages,
            model=model,
            stream=stream,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            regex=regex,
            reasoning_effort=reasoning_effort,
            vision_config=vision_config,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream_options=stream_options,
            n=n,
            extra_body=extra_body,
            timeout=timeout,
            kwargs=kwargs,
        )
        cfg = request.provider_cfg
        send = {"json": request.body, "timeout": request.timeout, "provider_cfg": cfg}
        if stream:
            response, closer = _transport.open_stream(self._client, "POST", cfg["path"], **send)
            return ChatCompletionStream(
                response, closer, request_id=_transport.request_id_of(response), asset_count=request.asset_count
            )
        payload, headers = _transport.request_json(self._client, "POST", cfg["path"], **send)
        return _completion_from_payload(payload, headers, request.asset_count)

    def multilook(  # noqa: PLR0913 - the multilook request parameters
        self,
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
    ) -> Any:
        """Answer several prompts against one shared ``context`` in one request (``POST
        /chat/completions/multilook``). See :func:`perceptron.multilook.create`."""
        from .multilook import create as multilook_create  # noqa: PLC0415

        return multilook_create(
            self._client,
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
            vision_config=vision_config,
            extra_body=extra_body,
            timeout=timeout,
        )


class AsyncChatCompletions:
    """``AsyncClient.chat.completions``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def create(  # noqa: PLR0913 - the OpenAI chat-completions parameters
        self,
        *,
        messages: list[Any],
        model: str | None = None,
        stream: bool = False,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        regex: str | None = None,
        reasoning_effort: str | None = None,
        vision_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        n: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncChatCompletionStream:
        """Async :meth:`ChatCompletions.create`; ``stream=True`` returns an :class:`AsyncChatCompletionStream`."""
        request = _build_request(
            self._client,
            messages=messages,
            model=model,
            stream=stream,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            regex=regex,
            reasoning_effort=reasoning_effort,
            vision_config=vision_config,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream_options=stream_options,
            n=n,
            extra_body=extra_body,
            timeout=timeout,
            kwargs=kwargs,
        )
        cfg = request.provider_cfg
        send = {"json": request.body, "timeout": request.timeout, "provider_cfg": cfg}
        if stream:
            response, closer = await _transport.aopen_stream(self._client, "POST", cfg["path"], **send)
            return AsyncChatCompletionStream(
                response, closer, request_id=_transport.request_id_of(response), asset_count=request.asset_count
            )
        payload, headers = await _transport.arequest_json(self._client, "POST", cfg["path"], **send)
        return _completion_from_payload(payload, headers, request.asset_count)

    async def multilook(  # noqa: PLR0913 - the multilook request parameters
        self,
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
    ) -> Any:
        """Async :meth:`ChatCompletions.multilook`. See :func:`perceptron.multilook.acreate`."""
        from .multilook import acreate as multilook_acreate  # noqa: PLC0415

        return await multilook_acreate(
            self._client,
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
            vision_config=vision_config,
            extra_body=extra_body,
            timeout=timeout,
        )


class Chat:
    """``client.chat``: the message-level API."""

    def __init__(self, client: Any) -> None:
        self.completions = ChatCompletions(client)


class AsyncChat:
    """``AsyncClient.chat``: the async message-level API."""

    def __init__(self, client: Any) -> None:
        self.completions = AsyncChatCompletions(client)
