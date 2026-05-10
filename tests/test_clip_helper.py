"""Coverage for the find_clips() high-level helper (perceptron.highlevel.find_clips)."""

from __future__ import annotations

import pytest
from _image_fixtures import PNG_BYTES

from perceptron import find_clips, image, video
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.errors import BadRequestError


@pytest.fixture(autouse=True)
def _stub_generate(monkeypatch):
    """Echo the compiled task back via .raw so tests can introspect it."""

    def _echo(self, task, **kwargs):  # pylint: disable=unused-argument
        return {"text": "", "points": None, "parsed": None, "raw": task}

    monkeypatch.setattr(client_mod.Client, "generate", _echo)


# Minimal mp4 magic-byte stub (matches tests/test_video.py fixture).
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42" + b"\x00" * 16


def _user_text(content) -> list[str]:
    return [entry["content"] for entry in content if entry.get("type") == "text" and entry.get("role") == "user"]


def test_find_clips_rejects_image_node():
    with pytest.raises(BadRequestError, match="requires a video"):
        with cfg(api_key="test-key", provider="perceptron"):
            find_clips(image(PNG_BYTES), "the goal")


def test_find_clips_rejects_raw_string():
    with pytest.raises(BadRequestError, match="requires a video"):
        with cfg(api_key="test-key", provider="perceptron"):
            find_clips("https://example.com/clip.mp4", "the goal")


def test_find_clips_prepends_clip_verb_to_query():
    with cfg(api_key="test-key", provider="perceptron"):
        res = find_clips(video("https://example.com/clip.mp4"), "every save attempt")

    user_messages = _user_text(res.raw["content"])
    assert "Clip every save attempt." in user_messages


def test_find_clips_sets_expects_clip_in_task():
    with cfg(api_key="test-key", provider="perceptron"):
        res = find_clips(video("https://example.com/clip.mp4"), "the goal")

    assert res.raw["expects"] == "clip"


def test_find_clips_accepts_video_node_with_bytes():
    with cfg(api_key="test-key", provider="perceptron"):
        res = find_clips(video(MP4_BYTES), "any motion")

    parts = [p for p in res.raw["content"] if p.get("type") == "video"]
    assert len(parts) == 1


def test_find_clips_multiple_false_keeps_expects_clip():
    """When multiple=False, the task still requests expects=clip; multiplicity is parsed downstream."""
    captured: dict = {}

    def _capture_perceive(self, task, **kwargs):  # pylint: disable=unused-argument
        captured["task"] = task
        captured["kwargs"] = kwargs
        return {"text": "", "points": None, "parsed": None, "raw": task}

    import perceptron.client as cm
    cm.Client.generate = _capture_perceive

    with cfg(api_key="test-key", provider="perceptron"):
        find_clips(video("https://example.com/clip.mp4"), "the winning shot", multiple=False)

    assert captured["task"]["expects"] == "clip"


def test_find_clips_exported_from_top_level():
    """Sanity check: the helper is importable from `perceptron` directly."""
    import perceptron

    assert callable(perceptron.find_clips)
    assert "find_clips" in perceptron.__all__


def test_clip_factory_remains_at_top_level():
    """The Clip-annotation factory (Max's #61) is preserved as `perceptron.clip`."""
    from perceptron import Clip, ClipTimestamp, clip

    annotation = clip(at=1.0, until=2.5, mention="moment")
    assert isinstance(annotation, Clip)
    assert isinstance(annotation.timestamp, ClipTimestamp)
    assert annotation.timestamp.at == 1.0
    assert annotation.timestamp.until == 2.5
    assert annotation.mention == "moment"
