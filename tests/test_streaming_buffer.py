import json

import pytest

from perceptron import perceive, image, text, config as cfg


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


def test_stream_parsing_buffer_overflow(monkeypatch):
    @perceive(expects="point", stream=True)
    def fn(img):
        return image(img) + text("Find point")

    from perceptron import client as client_mod

    # Construct many small deltas to exceed buffer
    chunks = []
    for _ in range(50):
        chunks.append(_sse({"choices": [{"delta": {"content": "x"}}]}))
    # Append a tag to see that parsing is disabled by then
    chunks.append(_sse({"choices": [{"delta": {"content": "<point> (1,2) </point>"}}]}))
    chunks.append("data: [DONE]")

    def _mock_post(url, headers=None, data=None, timeout=None, stream=False):
        return _MockResp(chunks, status=200)

    monkeypatch.setattr(client_mod.requests, "post", _mock_post)

    with cfg(max_buffer_bytes=40):
        events = list(fn(b"\x89PNG\r\n\x1a\nHEADERONLY"))
    # Final event should include buffer overflow issue
    finals = [e for e in events if e.get("type") == "final"]
    assert finals, "missing final event"
    issues = finals[0]["result"]["errors"]
    assert any(i.get("code") == "stream_buffer_overflow" for i in issues)
