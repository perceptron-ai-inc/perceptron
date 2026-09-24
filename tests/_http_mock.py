"""Route the client's session seams through `httpx.MockTransport`.

Patches `perceptron.client._http_client` (sync) and `perceptron.client.httpx` (async: a namespace whose `AsyncClient`
uses the mock transport and whose exception classes are the real httpx ones). Response bodies are unread byte streams,
like a real socket, so error bodies must be read before mapping.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from perceptron import client as client_mod

_REAL_ASYNC_CLIENT = httpx.AsyncClient


class Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    """An unread response body that records whether it was closed."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def __iter__(self):
        yield self._data

    async def __aiter__(self):
        yield self._data

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class FailingBody(Body):
    """A body that yields ``data`` and then raises ``exc`` (a timeout or cut connection mid-body)."""

    def __init__(self, data: bytes, exc: Exception) -> None:
        super().__init__(data)
        self._exc = exc

    def __iter__(self):
        yield self._data
        raise self._exc

    async def __aiter__(self):
        yield self._data
        raise self._exc


class Recorder:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


def install(monkeypatch, handler) -> Recorder:
    recorder = Recorder(handler)
    transport = httpx.MockTransport(recorder)
    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: httpx.Client(transport=transport, timeout=timeout))
    async_httpx = SimpleNamespace(
        AsyncClient=lambda timeout: _REAL_ASYNC_CLIENT(transport=transport, timeout=timeout),
        TimeoutException=httpx.TimeoutException,
        HTTPError=httpx.HTTPError,
    )
    monkeypatch.setattr(client_mod, "httpx", async_httpx)
    return recorder


def json_response(payload, status: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, stream=Body(json.dumps(payload).encode()), headers=headers or {})


def text_response(text: str, status: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, stream=Body(text.encode()), headers=headers or {})


def sse_body(events, *, done: bool = True) -> bytes:
    """`events`: dicts (JSON-encoded) or raw strings (sent verbatim as whole lines)."""
    lines = []
    for event in events:
        lines.append(event if isinstance(event, str) else f"data: {json.dumps(event)}")
    if done:
        lines.append("data: [DONE]")
    return "".join(f"{line}\n\n" for line in lines).encode()


def sse_response(events, *, done: bool = True, headers: dict | None = None, body: Body | None = None):
    stream = body or Body(sse_body(events, done=done))
    return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream", **(headers or {})})


def completion(content="Hello", *, finish_reason="stop", tool_calls=None, reasoning=None, usage=None, **extra):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": "perceptron-mk1.5",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        **extra,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def chunk(delta=None, *, finish_reason=None, usage=None, choices=True, omit_delta=False):
    obj = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1790000000, "model": "perceptron-mk1.5"}
    if choices:
        choice = {"index": 0}
        if not omit_delta:
            choice["delta"] = delta
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        obj["choices"] = [choice]
    else:
        obj["choices"] = []
    if usage is not None:
        obj["usage"] = usage
    return obj
