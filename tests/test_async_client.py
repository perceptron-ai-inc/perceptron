"""`AsyncClient.generate` / `AsyncClient.stream` over `httpx.MockTransport` (see `_http_mock`)."""

import asyncio
import json

import pytest
from _http_mock import install, json_response, sse_response

from perceptron import AsyncClient

FAL_URL = "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"
TASK = {"content": [{"type": "text", "role": "user", "content": "Hi"}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # FAL_KEY is the only key, so the provider is fal.
    for key in ("PERCEPTRON_API_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
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
    http = install(monkeypatch, lambda request: json_response(payload))

    async def _run():
        async with AsyncClient() as client:
            return await client.generate(TASK)

    result = asyncio.run(_run())

    assert result["text"] == "Hello async!"
    assert result["raw"] == payload
    request = http.last
    assert (request.method, str(request.url)) == ("POST", FAL_URL)
    assert request.headers["authorization"] == "Key test-fal-key"
    # The async path sends the body as pre-encoded JSON.
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {"model": "isaac-0.1", "messages": [{"role": "user", "content": "Hi"}]}


def test_async_stream(monkeypatch):
    chunks = [
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "<point> (1,2) </point>"}}]},
    ]
    http = install(monkeypatch, lambda request: sse_response(chunks))

    async def _run():
        async with AsyncClient() as client:
            return [ev async for ev in client.stream(TASK, expects="point", parse_points=True)]

    events = asyncio.run(_run())

    kinds = [ev["type"] for ev in events]
    assert kinds == ["text.delta", "text.delta", "points.delta", "final"]
    final = events[-1]["result"]
    assert final["text"] == "Hello <point> (1,2) </point>"
    assert [(point.x, point.y) for point in final["points"]] == [(1, 2)]
    assert events[2]["points"] == final["points"]
    assert (http.last.method, str(http.last.url)) == ("POST", FAL_URL)
    assert http.last_body["stream"] is True
