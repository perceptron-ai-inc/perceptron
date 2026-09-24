"""`perceive` / `async_perceive` send the `model` they were given, over `httpx.MockTransport` (see `_http_mock`)."""

import asyncio
import json

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import async_perceive, perceive
from perceptron.dsl.nodes import image, text
from perceptron.errors import AuthError

URL = "https://api.perceptron.inc/v1/chat/completions"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    # The Perceptron API passes model ids it does not know through to the gateway.
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


@pytest.fixture
def http(monkeypatch):
    def _handler(request):
        if json.loads(request.content).get("stream"):
            return sse_response([chunk({"content": "ok"}), chunk({}, finish_reason="stop")])
        return json_response(completion("ok"))

    return install(monkeypatch, _handler)


def _sent_once(http):
    """The body of the only request, a POST to the Perceptron API; the call's client was closed."""
    assert [(request.method, str(request.url)) for request in http.requests] == [("POST", URL)]
    assert [client.is_closed for client in http.clients] == [True]
    return http.last_body


def test_perceive_passes_model_to_client(http):
    @perceive(model="custom-model")
    def describe(img):
        return image(img) + text("Describe")

    result = describe(PNG_BYTES)
    assert result.text == "ok"
    body = _sent_once(http)
    assert body["model"] == "custom-model"
    assert "stream" not in body


def test_perceive_stream_passes_model(http):
    @perceive(model="stream-model", stream=True)
    def describe(img):
        return image(img) + text("Describe")

    events = list(describe(PNG_BYTES))
    assert events[-1]["type"] == "final"
    assert events[-1]["result"]["text"] == "ok"
    body = _sent_once(http)
    assert body["model"] == "stream-model"
    assert body["stream"] is True


def test_async_perceive_passes_model(http):
    @async_perceive(model="async-model")
    def describe(img):
        return image(img) + text("Describe")

    res = asyncio.run(describe(PNG_BYTES))
    assert res.text == "ok"
    body = _sent_once(http)
    assert body["model"] == "async-model"
    assert "stream" not in body


def test_async_perceive_stream_passes_model(http):
    @async_perceive(model="async-stream", stream=True)
    def describe(img):
        return image(img) + text("Describe")

    async def _collect():
        events_local = []
        async for ev in describe(PNG_BYTES):
            events_local.append(ev)
        return events_local

    collected = asyncio.run(_collect())
    assert collected[-1]["type"] == "final"
    assert collected[-1]["result"]["text"] == "ok"
    body = _sent_once(http)
    assert body["model"] == "async-stream"
    assert body["stream"] is True


def test_perceive_missing_credentials_raises(monkeypatch, http):
    monkeypatch.delenv("PERCEPTRON_API_KEY")

    @perceive()
    def describe(img):
        return image(img) + text("Describe")

    with pytest.raises(AuthError) as excinfo:
        describe(b"bytes")

    details = excinfo.value.details or {}
    task = details.get("task")
    assert task and isinstance(task, dict)
    assert task["content"][0]["type"] == "image"
    assert not http.requests  # raised before sending
