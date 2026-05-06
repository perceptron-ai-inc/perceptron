import json

import pytest

from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron import image, perceive, text


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
        yield from self._lines


def _sse(obj):
    return f"data: {json.dumps(obj)}"


@pytest.fixture(autouse=True)
def _set_fal_key(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test-fal-key")


def test_stream_parsing_buffer_overflow(monkeypatch):
    @perceive(expects="point", stream=True)
    def fn(img):
        return image(img) + text("Find point")

    # Construct many small deltas to exceed buffer
    chunks = []
    for _ in range(50):
        chunks.append(_sse({"choices": [{"delta": {"content": "x"}}]}))
    # Append a tag to see that parsing is disabled by then
    chunks.append(_sse({"choices": [{"delta": {"content": "<point> (1,2) </point>"}}]}))
    chunks.append("data: [DONE]")

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *_, **__):  # pragma: no cover
            raise AssertionError

        def stream(self, method, url, headers=None, json=None):
            return _MockResp(chunks, status=200)

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())

    with cfg(max_buffer_bytes=40):
        events = list(fn(b"\x89PNG\r\n\x1a\nHEADERONLY"))
    # Final event should include buffer overflow issue
    finals = [e for e in events if e.get("type") == "final"]
    assert finals, "missing final event"
    issues = finals[0]["result"]["errors"]
    assert any(i.get("code") == "stream_buffer_overflow" for i in issues)
