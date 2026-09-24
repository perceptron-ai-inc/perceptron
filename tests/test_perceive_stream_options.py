"""`stream_options` on `perceive`, `async_perceive` and the helpers (forwarded to the client's stream)."""

from __future__ import annotations

import asyncio

import pytest
from _http_mock import chunk, install, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import async_perceive, caption, image, perceive, question, text
from perceptron.dsl.nodes import audio
from perceptron.errors import INVALID_PARAMETER, BadRequestError

USAGE = {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14, "prompt_tokens_details": {"audio_tokens": 9}}
OFF = {"include_usage": False}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


@pytest.fixture
def http(monkeypatch):
    events = [
        chunk({"role": "assistant", "content": "A dog."}),
        chunk({}, finish_reason="stop"),
        chunk(None, choices=False, usage=USAGE),
    ]
    return install(monkeypatch, lambda request: sse_response(events))


def _final(events):
    events = list(events)
    assert events[-1]["type"] == "final", events[-1]
    return events[-1]["result"]


async def _acollect(stream):
    return [event async for event in stream]


def test_perceive_forwards_stream_options(http):
    _final(perceive(image(PNG_BYTES) + text("What?"), provider="perceptron", stream=True, stream_options=OFF))
    assert http.last_body["stream_options"] == OFF


def test_perceive_default_requests_usage(http):
    result = _final(perceive(image(PNG_BYTES) + text("What?"), provider="perceptron", stream=True))

    assert http.last_body["stream_options"] == {"include_usage": True}
    assert result["usage"] == USAGE


def test_decorated_perceive_forwards_stream_options(http):
    @perceive(provider="perceptron", stream=True, stream_options={"include_usage": True})
    def describe(data):
        return audio(data) + text("Transcribe.")

    result = _final(describe("https://example.com/a.wav"))

    assert http.last_body["stream_options"] == {"include_usage": True}
    assert result["usage"]["prompt_tokens_details"] == {"audio_tokens": 9}


def test_async_perceive_forwards_stream_options(http):
    @async_perceive(provider="perceptron", stream=True, stream_options=OFF)
    async def describe():
        return image(PNG_BYTES) + text("What?")

    events = asyncio.run(_acollect(describe()))

    assert events[-1]["type"] == "final"
    assert http.last_body["stream_options"] == OFF


@pytest.mark.parametrize(
    "call",
    [
        lambda **kw: question(image(PNG_BYTES), "What?", **kw),
        lambda **kw: caption(audio("https://example.com/a.wav"), **kw),
    ],
    ids=["question", "caption"],
)
def test_helpers_forward_stream_options(http, call):
    _final(call(provider="perceptron", stream=True, stream_options=OFF))
    assert http.last_body["stream_options"] == OFF


@pytest.mark.parametrize(
    "call",
    [
        lambda: perceive(image(PNG_BYTES) + text("What?"), provider="perceptron", stream_options=OFF),
        lambda: perceive(provider="perceptron", stream_options=OFF),
        lambda: async_perceive(provider="perceptron", stream_options=OFF),
        lambda: question(image(PNG_BYTES), "What?", provider="perceptron", stream_options=OFF),
    ],
    ids=["perceive", "decorator", "async_perceive", "question"],
)
def test_stream_options_without_streaming_raise_before_any_request(http, call):
    with pytest.raises(BadRequestError) as excinfo:
        call()
    assert excinfo.value.code == INVALID_PARAMETER and excinfo.value.param == "stream_options"
    assert http.requests == []
