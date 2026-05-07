"""Video support: DSL node, wire format, highlevel routing, OCR rejection."""

from __future__ import annotations

import pytest
from _image_fixtures import PNG_BYTES

from perceptron import caption, ocr, question, video
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.client import _task_to_openai_messages
from perceptron.dsl.nodes import Video as VideoNode
from perceptron.dsl.perceive import _compile, _detect_video_format
from perceptron.errors import BadRequestError

# Minimal mp4 / webm magic-byte fixtures. These are byte-stub level (not full
# decodable streams) — enough to exercise format sniffing and the wire path.
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42" + b"\x00" * 16
WEBM_BYTES = b"\x1a\x45\xdf\xa3" + b"\x00" * 32


@pytest.fixture(autouse=True)
def _stub_generate(monkeypatch):
    def _echo(self, task, **kwargs):  # pylint: disable=unused-argument
        return {"text": "", "points": None, "parsed": None, "raw": task}

    monkeypatch.setattr(client_mod.Client, "generate", _echo)


# ---- Format detection -----------------------------------------------------


def test_detect_video_format_mp4():
    assert _detect_video_format(MP4_BYTES) == "mp4"


def test_detect_video_format_webm():
    assert _detect_video_format(WEBM_BYTES) == "webm"


def test_detect_video_format_unknown_returns_none():
    assert _detect_video_format(b"not a video") is None


# ---- DSL node -------------------------------------------------------------


def test_video_factory_returns_video_node():
    assert isinstance(video("https://example.com/v.mp4"), VideoNode)


def test_video_url_compiles_with_passthrough():
    seq = video("https://example.com/v.mp4")
    task, issues = _compile(seq, expects=None, strict=False)
    assert issues == []
    parts = [p for p in task["content"] if p.get("type") == "video"]
    assert len(parts) == 1
    assert parts[0]["url"] is True
    assert parts[0]["content"] == "https://example.com/v.mp4"


def test_video_bytes_detect_mp4_format():
    seq = video(MP4_BYTES)
    task, _ = _compile(seq, expects=None, strict=False)
    parts = [p for p in task["content"] if p.get("type") == "video"]
    assert parts[0]["format"] == "mp4"


def test_video_bytes_detect_webm_format():
    seq = video(WEBM_BYTES)
    task, _ = _compile(seq, expects=None, strict=False)
    parts = [p for p in task["content"] if p.get("type") == "video"]
    assert parts[0]["format"] == "webm"


def test_unsniffable_video_bytes_raise():
    with pytest.raises(BadRequestError) as excinfo:
        seq = video(b"not a video")
        _compile(seq, expects=None, strict=False)
    assert excinfo.value.code == "invalid_video"


# ---- Wire format ----------------------------------------------------------


def test_wire_video_url_emits_video_url_part():
    seq = video("https://example.com/v.mp4")
    task, _ = _compile(seq, expects=None, strict=False)
    msgs = _task_to_openai_messages(task)
    part = msgs[0]["content"][0]
    assert part["type"] == "video_url"
    assert part["video_url"]["url"] == "https://example.com/v.mp4"


def test_wire_video_base64_emits_data_url():
    seq = video(MP4_BYTES)
    task, _ = _compile(seq, expects=None, strict=False)
    msgs = _task_to_openai_messages(task)
    part = msgs[0]["content"][0]
    assert part["type"] == "video_url"
    assert part["video_url"]["url"].startswith("data:video/mp4;base64,")


# ---- High-level routing ---------------------------------------------------


def test_caption_accepts_video():
    with cfg(api_key="test", provider="fal"):
        res = caption(video("https://x.com/v.mp4"))
    parts = [p for p in res.raw["content"] if p.get("type") == "video"]
    assert len(parts) == 1


def test_question_accepts_video():
    with cfg(api_key="test", provider="fal"):
        res = question(video("https://x.com/v.mp4"), "what happens?")
    parts = [p for p in res.raw["content"] if p.get("type") == "video"]
    assert len(parts) == 1


def test_caption_requires_wrapped_input():
    """Loose input is rejected — caller must wrap with image() or video()."""

    with pytest.raises(TypeError):
        with cfg(api_key="test", provider="fal"):
            caption(PNG_BYTES)


def test_ocr_requires_wrapped_input():
    with pytest.raises(TypeError):
        with cfg(api_key="test", provider="fal"):
            ocr(PNG_BYTES)
