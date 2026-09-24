"""`perceive(stream=True)` over `httpx.MockTransport` (see `_http_mock`): text, reasoning and point deltas, usage, and
an HTTP error as the stream's only event."""

import pytest
from _http_mock import install, json_response, sse_response
from _image_fixtures import PNG_BYTES
from PIL import Image as PILImage

from perceptron import image, perceive, text

FAL_URL = "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"
PERCEPTRON_URL = "https://api.perceptron.inc/v1/chat/completions"


@pytest.fixture(autouse=True)
def _set_fal_key(monkeypatch):
    for key in ("PERCEPTRON_API_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FAL_KEY", "test-fal-key")


def _sent_stream(http, url):
    """The body of the only request, a streaming POST to ``url``."""
    assert [(request.method, str(request.url)) for request in http.requests] == [("POST", url)]
    body = http.last_body
    assert body["stream"] is True
    return body


def test_stream_text_and_points(monkeypatch):
    # Build a small function with streaming enabled
    @perceive(expects="point", stream=True)
    def fn(img):
        return image(img) + text("Find point")

    chunks = [
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "<point> (1,2) </point>!"}}]},
        {"usage": {"prompt_tokens": 12, "completion_tokens": 3}},
    ]
    http = install(monkeypatch, lambda request: sse_response(chunks))

    events = list(fn(PILImage.new("RGB", (8, 8))))

    # FAL_KEY is the only key, so the stream goes to fal.
    _sent_stream(http, FAL_URL)
    types = [e["type"] for e in events]
    # Two text deltas, then the point once its tag closes, then the final result
    assert types == ["text.delta", "text.delta", "points.delta", "final"]
    assert [(point.x, point.y) for point in events[2]["points"]] == [(1, 2)]
    final = events[-1]["result"]
    assert final["text"] == "Hello <point> (1,2) </point>!"
    assert final["usage"]["prompt_tokens"] == 12


def test_stream_reasoning_delta(monkeypatch):
    @perceive(stream=True, reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def fn(img):
        return image(img) + text("Think about this")

    chunks = [
        {"choices": [{"delta": {"reasoning_content": "Let me think"}}]},
        {"choices": [{"delta": {"reasoning_content": " step by step"}}]},
        {"choices": [{"delta": {"content": "The answer"}}]},
        {"choices": [{"delta": {"content": " is 42"}}]},
    ]
    http = install(monkeypatch, lambda request: sse_response(chunks))
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    events = list(fn(PILImage.new("RGB", (8, 8))))
    types = [e["type"] for e in events]

    assert types.count("reasoning.delta") == 2
    assert types.count("text.delta") == 2
    assert types[-1] == "final"

    final = events[-1]["result"]
    assert final["reasoning"] == "Let me think step by step"
    assert final["text"] == "The answer is 42"
    body = _sent_stream(http, PERCEPTRON_URL)
    assert body["model"] == "isaac-0.2-1b"
    assert http.last.headers["authorization"] == "Bearer test-key"


def test_stream_http_error(monkeypatch):
    @perceive(stream=True)
    def fn(img):
        return image(img) + text("Hello")

    error = {"error": {"message": "upstream failed", "type": "server_error", "param": None, "code": None}}
    http = install(monkeypatch, lambda request: json_response(error, 500, headers={"x-trace-id": "t500"}))

    events = list(fn(PNG_BYTES))

    # The HTTP error is the only event; the body was read, so its message and the trace id reach it.
    assert [e["type"] for e in events] == ["error"]
    event = events[0]
    assert (event["status"], event["message"], event["code"]) == (500, "upstream failed", "server_error")
    assert event["request_id"] == "t500"
    assert event["partial"] is None
    _sent_stream(http, FAL_URL)
