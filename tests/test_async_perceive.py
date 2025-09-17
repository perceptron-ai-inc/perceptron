import asyncio

from perceptron import async_perceive
from perceptron.dsl.nodes import image, text


class _StubAsyncClient:
    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        pass

    async def generate(self, task, **kwargs):  # pylint: disable=unused-argument
        return {
            "text": "hello async",
            "points": None,
            "parsed": None,
            "raw": {"choices": [{"message": {"content": "hello async"}}]},
        }


class _StubStreamAsyncClient(_StubAsyncClient):
    def stream(self, task, **kwargs):  # pylint: disable=unused-argument
        async def _gen():
            yield {"type": "text.delta", "chunk": "hi"}
            yield {
                "type": "final",
                "result": {"text": "hi", "points": None, "parsed": None, "usage": None, "errors": [], "raw": None},
            }

        return _gen()


def test_async_perceive_generate(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.AsyncClient", _StubAsyncClient)

    @async_perceive()
    def describe(img):
        return image(img) + text("Hello")

    res = asyncio.run(describe(b"fake"))
    assert res.text == "hello async"
    assert res.errors == []

def test_async_perceive_stream(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.AsyncClient", _StubStreamAsyncClient)

    @async_perceive(expects="point", stream=True)
    def locate(img):
        return image(img) + text("Locate point")

    async def _collect():
        events_local = []
        async for ev in locate(b"fake"):
            events_local.append(ev)
        return events_local

    events = asyncio.run(_collect())

    assert any(ev.get("type") == "text.delta" for ev in events)
    assert events[-1]["type"] == "final"


def test_async_perceive_compile_only(monkeypatch):
    monkeypatch.delenv("PERCEPTRON_PROVIDER", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("PERCEPTRON_API_KEY", raising=False)
    monkeypatch.setattr("perceptron.dsl.perceive.AsyncClient", _StubAsyncClient)

    @async_perceive()
    def describe(img):
        return image(img) + text("Hello")

    res = asyncio.run(describe(b"bytes"))
    assert res.text is None
    assert isinstance(res.raw, dict)
    assert res.raw["content"][0]["type"] == "image"
    assert any(err.get("code") == "credentials_missing" for err in res.errors)


def test_async_perceive_supports_async_function(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.AsyncClient", _StubAsyncClient)

    @async_perceive()
    async def describe(img):
        await asyncio.sleep(0)
        return image(img) + text("Hello")

    res = asyncio.run(describe(b"bytes"))
    assert res.text == "hello async"
