"""`async_perceive` over `httpx.MockTransport` (see `_http_mock`): each call runs on its own `AsyncClient`."""

import asyncio
import json

import pytest
from _http_mock import install, json_response, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import async_perceive
from perceptron.dsl.nodes import image, text
from perceptron.errors import AuthError

FAL_URL = "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"
COMPLETION = {"choices": [{"message": {"content": "hello async"}}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_API_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def http(monkeypatch):
    def _handler(request):
        if json.loads(request.content).get("stream"):
            return sse_response([{"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}])
        return json_response(COMPLETION)

    return install(monkeypatch, _handler)


def _sent_once(http):
    """The body of the only request, a POST to fal (``FAL_KEY`` is the only key); the call's client was closed."""
    assert [(request.method, str(request.url)) for request in http.requests] == [("POST", FAL_URL)]
    assert [client.is_closed for client in http.clients] == [True]
    return http.last_body


def test_async_perceive_generate(monkeypatch, http):
    monkeypatch.setenv("FAL_KEY", "test")

    @async_perceive()
    def describe(img):
        return image(img) + text("Hello")

    res = asyncio.run(describe(PNG_BYTES))
    assert res.text == "hello async"
    assert res.raw == COMPLETION
    assert res.errors == []
    assert "stream" not in _sent_once(http)


def test_async_perceive_stream(monkeypatch, http):
    monkeypatch.setenv("FAL_KEY", "test")

    @async_perceive(expects="point", stream=True)
    def locate(img):
        return image(img) + text("Locate point")

    async def _collect():
        events_local = []
        async for ev in locate(PNG_BYTES):
            events_local.append(ev)
        return events_local

    events = asyncio.run(_collect())

    assert [ev["type"] for ev in events] == ["text.delta", "final"]
    assert events[0]["chunk"] == "hi"
    assert events[-1]["result"]["text"] == "hi"
    assert events[-1]["result"]["errors"] == []
    assert _sent_once(http)["stream"] is True


def test_async_perceive_compile_only(http):
    # No API key: the call raises with the compiled task before creating a client or sending anything.
    @async_perceive()
    def describe(img):
        return image(img) + text("Hello")

    with pytest.raises(AuthError) as excinfo:
        asyncio.run(describe(PNG_BYTES))

    details = excinfo.value.details or {}
    task = details.get("task")
    assert task and isinstance(task, dict)
    assert task["content"][0]["type"] == "image"
    errors = details.get("errors") or []
    assert any(err.get("code") == "credentials_missing" for err in errors)
    assert http.requests == []
    assert http.clients == []


def test_async_perceive_supports_async_function(monkeypatch, http):
    monkeypatch.setenv("FAL_KEY", "test")

    @async_perceive()
    async def describe(img):
        await asyncio.sleep(0)
        return image(img) + text("Hello")

    res = asyncio.run(describe(PNG_BYTES))
    assert res.text == "hello async"
    body = _sent_once(http)
    assert [part["type"] for part in body["messages"][0]["content"]] == ["image_url", "text"]
