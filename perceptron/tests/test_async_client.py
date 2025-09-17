import asyncio
import json
from types import SimpleNamespace

import pytest

from perceptron import AsyncClient
from perceptron import client as client_mod


class _StubResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {}

    def json(self):
        return self._payload


class _StubStreamResponse:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status
        self.headers = {}

    def json(self):  # pragma: no cover - parity helper
        return {}

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StubStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_async_client(response, stream_response):
    class _AsyncClient:
        def __init__(self, timeout):  # pylint: disable=unused-argument
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, content=None):  # pragma: no cover - passthrough
            return response

        def stream(self, method, url, headers=None, content=None):  # pragma: no cover
            return _StubStreamContext(stream_response)

    return _AsyncClient


def _patch_httpx(monkeypatch, response, stream_response):
    stub_module = SimpleNamespace(
        AsyncClient=_make_async_client(response, stream_response),
        TimeoutException=Exception,
        HTTPError=Exception,
    )
    monkeypatch.setattr(client_mod, "httpx", stub_module, raising=False)
    monkeypatch.setenv("FAL_KEY", "test-fal-key")


def test_async_generate(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": "Hello async!",
                    "reasoning_content": None,
                    "tool_calls": None,
                }
            }
        ]
    }
    _patch_httpx(monkeypatch, _StubResponse(payload), _StubStreamResponse([]))

    async def _run():
        client = AsyncClient()
        task = {"content": [{"type": "text", "role": "user", "content": "Hi"}]}
        result = await client.generate(task)
        assert result["text"] == "Hello async!"

    asyncio.run(_run())


def test_async_stream(monkeypatch):
    stream_lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
        "data: "
        + json.dumps({"choices": [{"delta": {"content": "<point> (1,2) </point>"}}]}),
        "data: [DONE]",
    ]
    response = _StubResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": "Hello <point> (1,2) </point>",
                        "reasoning_content": None,
                        "tool_calls": None,
                    }
                }
            ]
        }
    )
    _patch_httpx(monkeypatch, response, _StubStreamResponse(stream_lines))

    async def _run():
        client = AsyncClient()
        task = {"content": [{"type": "text", "role": "user", "content": "Hi"}]}
        events = []
        async for ev in client.stream(task, expects="point", parse_points=True):
            events.append(ev)
        kinds = [ev.get("type") for ev in events]
        assert "text.delta" in kinds
        assert kinds[-1] == "final"
        final = next(ev for ev in events if ev["type"] == "final")
        assert final["result"]["points"]

    asyncio.run(_run())
