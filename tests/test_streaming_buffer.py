import pytest
from _http_mock import install, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import config as cfg
from perceptron import image, perceive, text


@pytest.fixture(autouse=True)
def _set_fal_key(monkeypatch):
    for key in ("PERCEPTRON_API_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FAL_KEY", "test-fal-key")


def test_stream_parsing_buffer_overflow(monkeypatch):
    @perceive(expects="point", stream=True)
    def fn(img):
        return image(img) + text("Find point")

    # Construct many small deltas to exceed buffer
    chunks = [{"choices": [{"delta": {"content": "x"}}]} for _ in range(50)]
    # Append a tag to see that parsing is disabled by then
    chunks.append({"choices": [{"delta": {"content": "<point> (1,2) </point>"}}]})
    http = install(monkeypatch, lambda request: sse_response(chunks))

    with cfg(max_buffer_bytes=40):
        events = list(fn(PNG_BYTES))
    # Final event should include buffer overflow issue
    finals = [e for e in events if e.get("type") == "final"]
    assert finals, "missing final event"
    assert events[-1] is finals[0]
    issues = finals[0]["result"]["errors"]
    assert any(i.get("code") == "stream_buffer_overflow" for i in issues)
    # The text still arrives in full, but the tag after the limit is not parsed.
    assert finals[0]["result"]["text"] == "x" * 50 + "<point> (1,2) </point>"
    assert "points.delta" not in [e["type"] for e in events]
    assert "points" not in finals[0]["result"]
    assert len(http.requests) == 1
