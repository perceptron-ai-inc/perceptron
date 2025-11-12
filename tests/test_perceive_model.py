import asyncio

import pytest

from perceptron import async_perceive, perceive
from perceptron.dsl.nodes import image, text
from perceptron.errors import AuthError


class _StubClient:
    last_kwargs: dict | None = None

    def __init__(self, **overrides):  # pylint: disable=unused-argument
        pass

    def generate(self, task, **kwargs):  # pylint: disable=unused-argument
        type(self).last_kwargs = kwargs
        return {
            "text": "ok",
            "points": None,
            "parsed": None,
            "raw": {"choices": [{"message": {"content": "ok"}}]},
        }


class _StubStreamClient(_StubClient):
    def stream(self, task, **kwargs):  # pylint: disable=unused-argument
        type(self).last_kwargs = kwargs

        def _gen():
            yield {
                "type": "final",
                "result": {
                    "text": "ok",
                    "points": None,
                    "parsed": None,
                    "usage": None,
                    "errors": [],
                    "raw": None,
                },
            }

        return _gen()


class _StubAsyncClient:
    last_kwargs: dict | None = None

    def __init__(self, **overrides):  # pylint: disable=unused-argument
        pass

    async def generate(self, task, **kwargs):  # pylint: disable=unused-argument
        type(self).last_kwargs = kwargs
        return {
            "text": "async",
            "points": None,
            "parsed": None,
            "raw": {"choices": [{"message": {"content": "async"}}]},
        }


class _StubAsyncStreamClient(_StubAsyncClient):
    def stream(self, task, **kwargs):  # pylint: disable=unused-argument
        type(self).last_kwargs = kwargs

        async def _agen():
            yield {
                "type": "final",
                "result": {
                    "text": "async",
                    "points": None,
                    "parsed": None,
                    "usage": None,
                    "errors": [],
                    "raw": None,
                },
            }

        return _agen()


def test_perceive_passes_model_to_client(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.Client", _StubClient)

    @perceive(model="custom-model")
    def describe(img):
        return image(img) + text("Describe")

    result = describe(b"bytes")
    assert result.text == "ok"
    assert _StubClient.last_kwargs["model"] == "custom-model"


def test_perceive_stream_passes_model(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.Client", _StubStreamClient)

    @perceive(model="stream-model", stream=True)
    def describe(img):
        return image(img) + text("Describe")

    events = list(describe(b"bytes"))
    assert events[-1]["type"] == "final"
    assert _StubStreamClient.last_kwargs["model"] == "stream-model"


def test_async_perceive_passes_model(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.AsyncClient", _StubAsyncClient)

    @async_perceive(model="async-model")
    def describe(img):
        return image(img) + text("Describe")

    res = asyncio.run(describe(b"bytes"))
    assert res.text == "async"
    assert _StubAsyncClient.last_kwargs["model"] == "async-model"


def test_async_perceive_stream_passes_model(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.AsyncClient", _StubAsyncStreamClient)

    @async_perceive(model="async-stream", stream=True)
    def describe(img):
        return image(img) + text("Describe")

    async def _collect():
        events_local = []
        async for ev in describe(b"bytes"):
            events_local.append(ev)
        return events_local

    collected = asyncio.run(_collect())
    assert collected[-1]["type"] == "final"
    assert _StubAsyncStreamClient.last_kwargs["model"] == "async-stream"


def test_perceive_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("PERCEPTRON_PROVIDER", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("PERCEPTRON_API_KEY", raising=False)

    @perceive()
    def describe(img):
        return image(img) + text("Describe")

    with pytest.raises(AuthError) as excinfo:
        describe(b"bytes")

    details = excinfo.value.details or {}
    task = details.get("task")
    assert task and isinstance(task, dict)
    assert task["content"][0]["type"] == "image"
