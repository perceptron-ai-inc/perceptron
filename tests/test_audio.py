"""Audio support: DSL node, wire format, highlevel routing, CLI media detection."""

from __future__ import annotations

import pytest

from perceptron import audio, caption, question
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.cli import _make_media_node
from perceptron.client import _task_to_openai_messages
from perceptron.dsl.nodes import Audio as AudioNode
from perceptron.dsl.nodes import Image as ImageNode
from perceptron.dsl.nodes import Video as VideoNode
from perceptron.dsl.perceive import _compile, _detect_audio_format
from perceptron.errors import BadRequestError

# Minimal wav / mp3 / flac magic-byte fixtures. These are byte-stub level (not
# decodable streams) — enough to exercise format sniffing and the wire path.
WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24
MP3_ID3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
MP3_FRAME_BYTES = b"\xff\xfb\x90\x00" + b"\x00" * 16
FLAC_BYTES = b"fLaC\x00\x00\x00\x22" + b"\x00" * 34


@pytest.fixture(autouse=True)
def _stub_generate(monkeypatch):
    def _echo(self, task, **kwargs):  # pylint: disable=unused-argument
        return {"text": "", "points": None, "parsed": None, "raw": task}

    monkeypatch.setattr(client_mod.Client, "generate", _echo)


# ---- Format detection -----------------------------------------------------


def test_detect_audio_format_wav():
    assert _detect_audio_format(WAV_BYTES) == "wav"


def test_detect_audio_format_mp3_id3():
    assert _detect_audio_format(MP3_ID3_BYTES) == "mp3"


def test_detect_audio_format_mp3_frame_sync():
    assert _detect_audio_format(MP3_FRAME_BYTES) == "mp3"


def test_detect_audio_format_flac():
    assert _detect_audio_format(FLAC_BYTES) == "flac"


def test_detect_audio_format_unknown_returns_none():
    assert _detect_audio_format(b"not audio") is None


def test_detect_audio_format_webp_riff_is_not_wav():
    assert _detect_audio_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") is None


# ---- DSL node -------------------------------------------------------------


def test_audio_factory_returns_audio_node():
    assert isinstance(audio("https://example.com/a.wav"), AudioNode)


def test_audio_url_compiles_with_passthrough():
    seq = audio("https://example.com/a.wav")
    task, issues = _compile(seq, expects=None, strict=False)
    assert issues == []
    parts = [p for p in task["content"] if p.get("type") == "audio"]
    assert len(parts) == 1
    assert parts[0]["url"] is True
    assert parts[0]["content"] == "https://example.com/a.wav"


@pytest.mark.parametrize(
    ("data", "fmt"),
    [(WAV_BYTES, "wav"), (MP3_ID3_BYTES, "mp3"), (FLAC_BYTES, "flac")],
)
def test_audio_bytes_detect_format(data, fmt):
    seq = audio(data)
    task, _ = _compile(seq, expects=None, strict=False)
    parts = [p for p in task["content"] if p.get("type") == "audio"]
    assert parts[0]["format"] == fmt
    assert parts[0]["url"] is False


def test_audio_file_path_reads_and_encodes(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(WAV_BYTES)

    seq = audio(str(clip))
    task, _ = _compile(seq, expects=None, strict=False)
    parts = [p for p in task["content"] if p.get("type") == "audio"]
    assert len(parts) == 1
    assert parts[0]["format"] == "wav"
    assert parts[0]["url"] is False


def test_unsniffable_audio_bytes_raise():
    with pytest.raises(BadRequestError) as excinfo:
        _compile(audio(b"not audio"), expects=None, strict=False)
    assert excinfo.value.code == "invalid_audio"


def test_unsniffable_audio_file_raises_with_origin(tmp_path):
    clip = tmp_path / "clip.bin"
    clip.write_bytes(b"not audio")
    with pytest.raises(BadRequestError) as excinfo:
        _compile(audio(str(clip)), expects=None, strict=False)
    assert excinfo.value.code == "invalid_audio"
    assert excinfo.value.details == {"origin": str(clip)}


# ---- Wire format ----------------------------------------------------------


def test_wire_audio_url_emits_audio_url_part():
    task, _ = _compile(audio("https://example.com/a.wav"), expects=None, strict=False)
    msgs = _task_to_openai_messages(task)
    part = msgs[0]["content"][0]
    assert part == {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}


def test_wire_audio_base64_emits_input_audio_part():
    task, _ = _compile(audio(FLAC_BYTES), expects=None, strict=False)
    msgs = _task_to_openai_messages(task)
    part = msgs[0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "flac"
    assert part["input_audio"]["data"] == task["content"][0]["content"]


def test_wire_audio_missing_format_raises():
    task = {"content": [{"type": "audio", "role": "user", "content": "AAAA", "url": False}], "expects": None}
    with pytest.raises(BadRequestError) as excinfo:
        _task_to_openai_messages(task)
    assert excinfo.value.code == "invalid_audio_format"


# ---- High-level routing ---------------------------------------------------


def test_caption_accepts_audio():
    with cfg(api_key="test", provider="fal"):
        res = caption(audio("https://x.com/a.mp3"))
    parts = [p for p in res.raw["content"] if p.get("type") == "audio"]
    assert len(parts) == 1


def test_caption_audio_defaults_to_text():
    with cfg(api_key="test", provider="fal"):
        res = caption(audio("https://x.com/a.mp3"))
    assert res.raw["expects"] is None


def test_question_accepts_audio():
    with cfg(api_key="test", provider="fal"):
        res = question(audio("https://x.com/a.mp3"), "what is said?")
    parts = [p for p in res.raw["content"] if p.get("type") == "audio"]
    assert len(parts) == 1


def test_caption_audio_uses_audio_modality_prompt():
    with cfg(api_key="test", provider="fal"):
        res = caption(audio("https://x.com/a.mp3"), style="concise")
    user_msgs = [e["content"] for e in res.raw["content"] if e.get("type") == "text" and e.get("role") == "user"]
    assert any("upcoming audio" in m for m in user_msgs)


# ---- CLI media detection --------------------------------------------------


@pytest.mark.parametrize("name", ["clip.wav", "clip.mp3", "clip.flac", "https://x.com/a.wav?sig=1"])
def test_cli_routes_audio_extensions(name):
    assert isinstance(_make_media_node(name, "https://x.com/a.wav"), AudioNode)


def test_cli_routes_video_and_image_extensions():
    assert isinstance(_make_media_node("clip.mp4", "https://x.com/v.mp4"), VideoNode)
    assert isinstance(_make_media_node("photo.png", "https://x.com/p.png"), ImageNode)
