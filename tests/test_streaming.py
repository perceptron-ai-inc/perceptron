import json

import pytest

from perceptron import perceive, image, text


class _MockResp:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line


def _sse(obj):
    return f"data: {json.dumps(obj)}"


@pytest.fixture(autouse=True)
def _set_fal_key(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test-fal-key")


def test_stream_text_and_points(monkeypatch):
    # Build a small function with streaming enabled
    @perceive(expects="point", stream=True)
    def fn(img):
        return image(img) + text("Find point")

    # Mock requests.post to stream SSE lines
    from perceptron import client as client_mod

    def _mock_post(url, headers=None, data=None, timeout=None, stream=False):
        # two chunks: greeting and a completed point tag
        chunks = [
            _sse({"choices": [{"delta": {"content": "Hello "}}]}),
            _sse({"choices": [{"delta": {"content": "<point> (1,2) </point>!"}}]}),
            _sse({"usage": {"prompt_tokens": 12, "completion_tokens": 3}}),
            "data: [DONE]",
        ]
        return _MockResp(chunks, status=200)

    monkeypatch.setattr(client_mod.requests, "post", _mock_post)

    # Provide an 8x8 image via PIL if available; else bytes
    try:
        from PIL import Image as PILImage  # type: ignore

        img = PILImage.new("RGB", (8, 8))
    except Exception:
        img = b"\x89PNG\r\n\x1a\n" + b"0" * 10

    stream_it = fn(img)
    events = list(stream_it)
    # Should contain at least two text.delta and one final
    types = [e.get("type") for e in events]
    assert types.count("text.delta") >= 2
    assert types[-1] == "final"
    # Ensure points.delta emitted once after tag closes
    assert any(e.get("type") == "points.delta" for e in events)
    assert events[-1]["result"]["usage"]["prompt_tokens"] == 12


def test_stream_http_error(monkeypatch):
    @perceive(stream=True)
    def fn(img):
        return image(img) + text("Hello")

    from perceptron import client as client_mod

    def _mock_post(url, headers=None, data=None, timeout=None, stream=False):
        return _MockResp(["data: [DONE]"], status=500)

    monkeypatch.setattr(client_mod.requests, "post", _mock_post)

    events = list(fn(b"\x89PNG\r\n\x1a\nxxxxxxxxxx"))
    assert events and events[0]["type"] == "error"
