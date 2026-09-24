"""Audio & reasoning parity (DESIGN §13.4): audio inputs, ``enable_audio_in_video``, ``reasoning_effort`` and mixed-media
order reach the wire the same way through every surface.

Mocked HTTP, provider ``perceptron``, default model ``perceptron-mk1.5``. Legacy surfaces (``perceive``, the helpers,
``Client.generate/stream``) select the provider explicitly (a ``PERCEPTRON_API_KEY``-only env auto-detects fal there);
the message API and multilook default to ``perceptron`` on their own (§12.1).
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import struct
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import (
    AsyncClient,
    Client,
    async_perceive,
    audio,
    block,
    caption,
    image,
    perceive,
    question,
    text,
    video,
)
from perceptron._lowering import MEDIA_PART_TYPES
from perceptron.dsl.nodes import video_frames
from perceptron.dsl.perceive import _compile
from perceptron.errors import INVALID_REASONING_EFFORT, INVALID_VISION_CONFIG, BadRequestError

CHAT_URL = "https://api.perceptron.inc/v1/chat/completions"
PROMPT = "What is said in the clip?"
AUDIO_URL = "https://example.com/clip.wav"
VIDEO_URL = "https://example.com/clip.mp4"
FILE_ID = "file-A1b2C3d4E5f6G7h8I9j0K1"
USAGE = {
    "prompt_tokens": 1412,
    "completion_tokens": 42,
    "total_tokens": 1454,
    "prompt_tokens_details": {"audio_tokens": 318},
}
STREAM = [
    chunk({"role": "assistant", "content": "Hello"}),
    chunk({}, finish_reason="stop"),
    chunk(choices=False, usage=USAGE),
]
MULTILOOK = {
    "id": "mlcmpl-1",
    "object": "chat.completion.multilook",
    "model": "perceptron-mk1.5",
    "results": [
        {
            "prompt_index": 0,
            "completions": [
                {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"},
            ],
        }
    ],
    "usage": USAGE,
}


# ---------------------------------------------------------------------------
# Tiny but well-formed audio files
# ---------------------------------------------------------------------------


def _wav() -> bytes:
    """16 kHz mono 16-bit PCM with four silent samples."""
    samples = b"\x00\x00" * 4
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _mp3(*, id3: bool = True) -> bytes:
    """One silent MPEG-1 Layer III frame (128 kbps, 44.1 kHz), optionally after an empty ID3v2.4 tag."""
    frame = b"\xff\xfb\x90\x64" + b"\x00" * 413
    return (b"ID3\x04\x00\x00\x00\x00\x00\x00" if id3 else b"") + frame


def _flac() -> bytes:
    """The ``fLaC`` marker and a final STREAMINFO block (16 kHz, mono, 16-bit, no samples)."""
    packed = (16000 << 44) | (0 << 41) | (15 << 36)
    streaminfo = struct.pack(">HH", 4096, 4096) + b"\x00" * 6 + packed.to_bytes(8, "big") + b"\x00" * 16
    return b"fLaC" + bytes([0x80]) + len(streaminfo).to_bytes(3, "big") + streaminfo


WAV, MP3, FLAC = _wav(), _mp3(), _flac()
AUDIO = {"wav": WAV, "mp3": MP3, "flac": FLAC}
DATA_URL_MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _input_audio(data: bytes, fmt: str) -> dict:
    return {"type": "input_audio", "input_audio": {"data": _b64(data), "format": fmt}}


def _audio_url(url: str) -> dict:
    return {"type": "audio_url", "audio_url": {"url": url}}


SOURCES = [
    *((fmt, kind) for fmt in AUDIO for kind in ("path", "pathlib", "bytes", "https", "http", "data_url")),
    ("wav", "file_id"),
]


def _audio_source(fmt: str, kind: str, tmp_path: Path):
    """A factory for the ``audio(...)`` node of one source, and the wire part it must become."""
    data = AUDIO[fmt]
    if kind in ("path", "pathlib"):
        path = tmp_path / f"clip.{fmt}"
        path.write_bytes(data)
        obj = str(path) if kind == "path" else path
        return (lambda: audio(obj)), _input_audio(data, fmt)
    if kind == "bytes":
        return (lambda: audio(data)), _input_audio(data, fmt)
    if kind in ("https", "http"):
        url = f"{kind}://example.com/clip.{fmt}?sig=1"
        return (lambda: audio(url)), _audio_url(url)
    if kind == "data_url":
        url = f"data:{DATA_URL_MIME[fmt]};base64,{_b64(data)}"
        return (lambda: audio(url)), _audio_url(url)
    assert kind == "file_id"
    return (lambda: audio(file_id=FILE_ID)), {"type": "audio_file_id", "audio_file_id": {"file_id": FILE_ID}}


# ---------------------------------------------------------------------------
# Fixtures and surfaces
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


@pytest.fixture
def http(monkeypatch):
    def handler(request):
        if request.url.path.endswith("/multilook"):
            return json_response(MULTILOOK)
        if json.loads(request.content).get("stream"):
            return sse_response(STREAM)
        return json_response(completion("Hello", usage=USAGE))

    return install(monkeypatch, handler)


def _acollect(stream) -> list:
    async def _run():
        return [event async for event in stream]

    return asyncio.run(_run())


def _final(events) -> dict:
    events = list(events)
    assert events[-1]["type"] == "final", events[-1]
    assert [event["type"] for event in events].count("final") == 1
    return events[-1]["result"]


def _outcome(value) -> dict:
    """What every surface reports, from a result dict (``generate`` or a ``final`` event), a ``PerceiveResult`` or a
    ``ChatCompletion``."""
    if isinstance(value, dict):
        return value
    return {"text": value.text, "asset_count": value.asset_count, "finish_reason": value.finish_reason}


def _task(nodes) -> dict:
    task, issues = _compile(block(*nodes), expects=None, strict=False)
    assert issues == []
    return task


# Legacy event streams (a list of events), so a pre-request error can be checked as the single `error` event.
def _perceive_events(nodes, **opts):
    return list(perceive(*nodes, provider="perceptron", stream=True, **opts))


def _async_perceive_events(nodes, **opts):
    @async_perceive(provider="perceptron", stream=True, **opts)
    async def run():
        return block(*nodes)

    return _acollect(run())


def _client_stream_events(nodes, **opts):
    return list(Client(provider="perceptron").stream(_task(nodes), **opts))


def _async_client_stream_events(nodes, **opts):
    return _acollect(AsyncClient(provider="perceptron").stream(_task(nodes), **opts))


LEGACY_STREAMS = {
    "perceive_stream": _perceive_events,
    "async_perceive_stream": _async_perceive_events,
    "client_stream": _client_stream_events,
    "async_client_stream": _async_client_stream_events,
}


def _perceive(nodes, **opts):
    return _outcome(perceive(*nodes, provider="perceptron", **opts))


def _async_perceive(nodes, **opts):
    @async_perceive(provider="perceptron", **opts)
    async def run():
        return block(*nodes)

    return _outcome(asyncio.run(run()))


def _client_generate(nodes, **opts):
    return Client(provider="perceptron").generate(_task(nodes), **opts)


def _async_client_generate(nodes, **opts):
    return asyncio.run(AsyncClient(provider="perceptron").generate(_task(nodes), **opts))


def _messages(nodes) -> list[dict]:
    return [{"role": "user", "content": list(nodes)}]


def _create(nodes, **opts):
    return _outcome(Client().chat.completions.create(messages=_messages(nodes), **opts))


def _create_stream(nodes, **opts):
    stream = Client().chat.completions.create(messages=_messages(nodes), stream=True, **opts)
    return _outcome(stream.get_final_completion())


def _async_create(nodes, **opts):
    async def run():
        return await AsyncClient().chat.completions.create(messages=_messages(nodes), **opts)

    return _outcome(asyncio.run(run()))


def _async_create_stream(nodes, **opts):
    async def run():
        stream = await AsyncClient().chat.completions.create(messages=_messages(nodes), stream=True, **opts)
        return await stream.get_final_completion()

    return _outcome(asyncio.run(run()))


def _final_of(events_surface):
    """A legacy event-stream surface as one that returns its ``final`` result."""
    return lambda prompt, **opts: _final(events_surface(prompt, **opts))


# Surfaces that take a whole prompt (a list of DSL nodes) and return an outcome dict.
LEGACY_SEQUENCE = {
    "perceive": _perceive,
    **{name: _final_of(fn) for name, fn in LEGACY_STREAMS.items()},
    "async_perceive": _async_perceive,
    "client_generate": _client_generate,
    "async_client_generate": _async_client_generate,
}
MESSAGE_API = {
    "create": _create,
    "create_stream": _create_stream,
    "async_create": _async_create,
    "async_create_stream": _async_create_stream,
}


def _multilook(nodes, **opts):
    response = Client().chat.completions.multilook(context=_messages(nodes), prompts=[PROMPT], **opts)
    return {"text": response.results[0].completions[0].message.content}


def _async_multilook(nodes, **opts):
    async def run():
        return await AsyncClient().chat.completions.multilook(context=_messages(nodes), prompts=[PROMPT], **opts)

    response = asyncio.run(run())
    return {"text": response.results[0].completions[0].message.content}


MULTILOOK_SURFACES = {"multilook": _multilook, "async_multilook": _async_multilook}


# Surfaces that take one media node (the helpers add their own prompt).
def _question(media, **opts):
    return _outcome(question(media, PROMPT, provider="perceptron", **opts))


def _question_events(media, **opts):
    return list(question(media, PROMPT, provider="perceptron", stream=True, **opts))


def _caption(media, **opts):
    return _outcome(caption(media, provider="perceptron", **opts))


def _caption_events(media, **opts):
    return list(caption(media, provider="perceptron", stream=True, **opts))


HELPERS = {
    "question": _question,
    "question_stream": _final_of(_question_events),
    "caption": _caption,
    "caption_stream": _final_of(_caption_events),
}


def _with_prompt(fn):
    return lambda media, **opts: fn([media, text(PROMPT)], **opts)


LEGACY = {**{name: _with_prompt(fn) for name, fn in LEGACY_SEQUENCE.items()}, **HELPERS}
SURFACES = {**LEGACY, **{name: _with_prompt(fn) for name, fn in MESSAGE_API.items()}}
ALL_SURFACES = {**SURFACES, **{name: _with_prompt(fn) for name, fn in MULTILOOK_SURFACES.items()}}

# Pre-request errors raise everywhere except in legacy event streams, where they are the single `error` event.
STREAM_EVENT_SURFACES = {
    **{name: _with_prompt(fn) for name, fn in LEGACY_STREAMS.items()},
    "question_stream": _question_events,
    "caption_stream": _caption_events,
}
RAISING_SURFACES = {name: fn for name, fn in ALL_SURFACES.items() if name not in STREAM_EVENT_SURFACES}


def _messages_of(body: dict) -> list[dict]:
    """The request's messages (a multilook request's shared ``context``)."""
    return body["messages"] if "messages" in body else body["context"]


def _parts(body: dict) -> list[dict]:
    """Every content part of the request, in order."""
    return [
        part for message in _messages_of(body) if isinstance(message["content"], list) for part in message["content"]
    ]


def _media_parts(body: dict) -> list[dict]:
    return [part for part in _parts(body) if part["type"] in MEDIA_PART_TYPES]


def _hints(body: dict) -> list[str]:
    """The ``<hint>`` markup the request carries (system or user text)."""
    texts = [message["content"] for message in _messages_of(body) if isinstance(message["content"], str)]
    texts += [part["text"] for part in _parts(body) if part["type"] == "text"]
    return [value for value in texts if "<hint>" in value]


# ---------------------------------------------------------------------------
# Audio inputs (§13.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(("fmt", "kind"), SOURCES)
def test_audio_reaches_the_wire_through_every_surface(http, tmp_path, surface, fmt, kind):
    make, expected = _audio_source(fmt, kind, tmp_path)

    outcome = SURFACES[surface](make())

    assert outcome["text"] == "Hello"
    assert len(http.requests) == 1 and str(http.last.url) == CHAT_URL
    body = http.last_body
    assert body["model"] == "perceptron-mk1.5"
    assert _media_parts(body) == [expected]
    if expected["type"] == "input_audio":
        data = expected["input_audio"]["data"]
        assert not data.startswith("data:") and base64.b64decode(data, validate=True) == AUDIO[fmt]


@pytest.mark.parametrize("surface", MULTILOOK_SURFACES)
@pytest.mark.parametrize(("fmt", "kind"), [("mp3", "path"), ("flac", "bytes"), ("wav", "https"), ("wav", "file_id")])
def test_audio_in_multilook_context(http, tmp_path, surface, fmt, kind):
    make, expected = _audio_source(fmt, kind, tmp_path)

    assert MULTILOOK_SURFACES[surface]([make()])["text"] == "Hello"

    assert _media_parts(http.last_body) == [expected]


@pytest.mark.parametrize(
    ("data", "fmt"),
    [(WAV, "wav"), (_mp3(id3=False), "mp3"), (MP3, "mp3"), (FLAC, "flac")],
    ids=["wav", "mp3-frame-sync", "mp3-id3", "flac"],
)
def test_audio_format_comes_from_the_bytes_not_the_file_name(http, tmp_path, data, fmt):
    path = tmp_path / "recording.bin"  # the container is sniffed, whatever the extension says
    path.write_bytes(data)

    perceive(audio(path), text(PROMPT), provider="perceptron")

    assert _media_parts(http.last_body) == [_input_audio(data, fmt)]


@pytest.mark.parametrize(
    "data",
    [
        bytes([0xFF, 0xF1, 0x50, 0x80, 0x02, 0x1F, 0xFC]) + b"\x00" * 64,  # ADTS AAC: the frame sync, layer bits 00
        bytes([0xFF, 0xFD, 0x90, 0x04]) + b"\x00" * 64,  # MPEG-1 Layer II (.mp2)
    ],
    ids=["adts-aac", "mp2"],
)
def test_other_mpeg_audio_is_not_taken_for_mp3(http, tmp_path, data):
    # They share MP3's 11-bit frame sync; like OGG or M4A they fail client-side instead of being sent as format mp3.
    path = tmp_path / "voice.mp3"
    path.write_bytes(data)
    for source in (path, data):
        with pytest.raises(BadRequestError) as excinfo:
            perceive(audio(source), text(PROMPT), provider="perceptron")
        assert excinfo.value.code == "invalid_audio"
    assert http.requests == []


# CPython's urlsplit strips leading control and space characters since 3.10.12 / 3.11.4; on older patch releases such
# a string is a local path, for every media kind alike.
_leading_space_is_url = pytest.mark.skipif(
    urlsplit(" https://x").scheme != "https", reason="this Python's urlsplit keeps leading whitespace"
)
PASSTHROUGH_URLS = [
    pytest.param("http://example.com/media?sig=1", id="http"),
    pytest.param("https://example.com/media?sig=1", id="https"),
    pytest.param("HTTPS://example.com/media?sig=1", id="upper-case"),
    pytest.param("Http://example.com/media?sig=1", id="mixed-case"),
    pytest.param("https:example.com/media?sig=1", id="no-slashes"),
    pytest.param(" https://example.com/media?sig=1", id="leading-space", marks=_leading_space_is_url),
    pytest.param("\thttps://example.com/media?sig=1", id="leading-tab", marks=_leading_space_is_url),
]


@pytest.mark.parametrize("url", PASSTHROUGH_URLS)
@pytest.mark.parametrize(
    ("factory", "part_type"),
    [(image, "image_url"), (video, "video_url"), (audio, "audio_url")],
    ids=["image", "video", "audio"],
)
@pytest.mark.parametrize("surface", [_perceive, _create], ids=["perceive", "create"])
def test_http_urls_pass_through_for_every_media_kind(http, surface, factory, part_type, url):
    """Local files and bytes are encoded; whatever the DSL takes for an HTTP(S) URL passes through verbatim, for images
    (and video_frames frames) exactly as for video and audio."""
    surface([factory(url), video_frames([(url, 0), (url, 40)]), text(PROMPT)])

    frames = {
        "type": "video_frames",
        "video_frames": {"frames": [{"image_url": {"url": url}, "timestamp_ms": t} for t in (0, 40)]},
    }
    assert _media_parts(http.last_body) == [{"type": part_type, part_type: {"url": url}}, frames]


def test_dsl_media_entries_mark_urls_for_every_kind():
    """One check decides what is a URL for DSL input: image entries carry ``url`` like video and audio entries, so the
    lowering never re-guesses (base64 image entries are unchanged, without the key)."""
    url = "https:example.com/media"  # a URL to that check, though not to a prefix test

    task = _task([image(url), video(url), audio(url), image(PNG_BYTES)])

    assert [entry.get("url") for entry in task["content"]] == [True, True, True, None]
    assert "url" not in task["content"][3]


RAW_AUDIO_PARTS = [
    _input_audio(MP3, "mp3"),
    _audio_url(AUDIO_URL),
    {"type": "audio_file_id", "audio_file_id": {"file_id": FILE_ID}},
]


@pytest.mark.parametrize("surface", ["create", "create_stream", "async_create", "async_create_stream"])
def test_raw_audio_parts_in_messages_are_sent_verbatim_and_counted(http, surface):
    """Wire audio parts (and a mix of wire parts, DSL nodes and strings) keep their order and each count as an asset."""
    content = [RAW_AUDIO_PARTS[0], image(PNG_BYTES), "Then:", RAW_AUDIO_PARTS[1], audio(FLAC), RAW_AUDIO_PARTS[2]]

    outcome = MESSAGE_API[surface](content)

    sent = _user_content(http.last_body)
    assert sent[0] == RAW_AUDIO_PARTS[0] and sent[3] == RAW_AUDIO_PARTS[1] and sent[5] == RAW_AUDIO_PARTS[2]
    assert [part["type"] for part in sent] == [
        "input_audio",
        "image_url",
        "text",
        "audio_url",
        "input_audio",
        "audio_file_id",
    ]
    assert sent[4] == _input_audio(FLAC, "flac")
    assert outcome["asset_count"] == 5


RUN_TASK = {
    "client_generate": lambda task: Client(provider="perceptron").generate(task),
    "client_stream": lambda task: _final(Client(provider="perceptron").stream(task)),
    "async_client_generate": lambda task: asyncio.run(AsyncClient(provider="perceptron").generate(task)),
    "async_client_stream": lambda task: _final(_acollect(AsyncClient(provider="perceptron").stream(task))),
}


@pytest.mark.parametrize("surface", RUN_TASK)
def test_hand_written_audio_task_entries(http, surface):
    """``Client.generate/stream`` lower task entries of every audio form: base64 + format, URL, data URL, file id."""
    data_url = f"data:audio/flac;base64,{_b64(FLAC)}"
    task = {
        "content": [
            {"type": "audio", "role": "user", "content": _b64(MP3), "format": "mp3"},
            {"type": "audio", "role": "user", "content": AUDIO_URL, "url": True},
            {"type": "audio", "role": "user", "content": data_url},
            {"type": "audio", "role": "user", "file_id": FILE_ID},
            {"type": "text", "role": "user", "content": PROMPT},
        ]
    }
    result = RUN_TASK[surface](task)

    assert _user_content(http.last_body) == [
        _input_audio(MP3, "mp3"),
        _audio_url(AUDIO_URL),
        _audio_url(data_url),
        {"type": "audio_file_id", "audio_file_id": {"file_id": FILE_ID}},
        {"type": "text", "text": PROMPT},
    ]
    assert result["asset_count"] == 4


def test_audio_parts_stay_in_the_user_message_after_the_hint(http):
    perceive(audio(WAV), text(PROMPT), provider="perceptron", reasoning=True)

    messages = http.last_body["messages"]
    assert messages == [
        {"role": "system", "content": "<hint>THINK</hint>"},
        {"role": "user", "content": [_input_audio(WAV, "wav"), {"type": "text", "text": PROMPT}]},
    ]


# ---------------------------------------------------------------------------
# enable_audio_in_video (§13.4): explicit True and False are sent; unset sends no vision_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surface", LEGACY)
@pytest.mark.parametrize("value", [True, False, None], ids=["true", "false", "unset"])
def test_enable_audio_in_video_reaches_vision_config(http, surface, value):
    options = {} if value is None else {"enable_audio_in_video": value}

    assert LEGACY[surface](video(VIDEO_URL), **options)["text"] == "Hello"

    body = http.last_body
    if value is None:
        assert "vision_config" not in body
    else:
        assert body["vision_config"] == {"enable_audio_in_video": value}
        assert body["vision_config"]["enable_audio_in_video"] is value


@pytest.mark.parametrize("surface", [*MESSAGE_API, *MULTILOOK_SURFACES])
@pytest.mark.parametrize("value", [True, False, None], ids=["true", "false", "unset"])
def test_enable_audio_in_video_on_the_message_api(http, surface, value):
    options = {} if value is None else {"vision_config": {"enable_audio_in_video": value}}
    run = {**MESSAGE_API, **MULTILOOK_SURFACES}[surface]

    assert run([video(VIDEO_URL), text(PROMPT)], **options)["text"] == "Hello"

    body = http.last_body
    if value is None:
        assert "vision_config" not in body
    else:
        assert body["vision_config"] == {"enable_audio_in_video": value}


def _audio_in_video(surface: str, value) -> dict:
    """The soundtrack option as each surface takes it."""
    if surface in LEGACY:
        return {"enable_audio_in_video": value}
    return {"vision_config": {"enable_audio_in_video": value}}


NOT_BOOLS = ["false", "true", 0, 1]


@pytest.mark.parametrize("surface", RAISING_SURFACES)
@pytest.mark.parametrize("value", NOT_BOOLS)
def test_enable_audio_in_video_must_be_a_bool(http, surface, value):
    """``"false"`` must not be sent as true: every surface rejects a non-bool before any request."""
    with pytest.raises(BadRequestError) as excinfo:
        RAISING_SURFACES[surface](video(VIDEO_URL), **_audio_in_video(surface, value))

    assert excinfo.value.code == INVALID_VISION_CONFIG
    assert excinfo.value.param == "vision_config.enable_audio_in_video"  # one check for every surface
    assert http.requests == []


@pytest.mark.parametrize("surface", STREAM_EVENT_SURFACES)
@pytest.mark.parametrize("value", NOT_BOOLS)
def test_enable_audio_in_video_must_be_a_bool_in_legacy_streams(http, surface, value):
    events = STREAM_EVENT_SURFACES[surface](video(VIDEO_URL), enable_audio_in_video=value)

    assert [(event["type"], event["code"], event["param"]) for event in events] == [
        ("error", INVALID_VISION_CONFIG, "vision_config.enable_audio_in_video")
    ]
    assert http.requests == []


def test_enable_audio_in_video_with_an_audio_track_and_reasoning_effort(http):
    """The soundtrack flag, a separate audio part and a tier travel together without affecting one another."""
    perceive(
        video(VIDEO_URL),
        audio(AUDIO_URL),
        text(PROMPT),
        provider="perceptron",
        enable_audio_in_video=False,
        reasoning_effort="low",
    )

    body = http.last_body
    assert body["vision_config"] == {"enable_audio_in_video": False}
    assert body["reasoning_effort"] == "low"
    assert [part["type"] for part in _parts(body)] == ["video_url", "audio_url", "text"]


# ---------------------------------------------------------------------------
# reasoning_effort (§13.4, O2): every tier, normalized, top-level only, nothing derived from it
# ---------------------------------------------------------------------------

TIERS = [
    ("none", "none"),
    ("minimal", "minimal"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
    ("HIGH", "high"),
    (" Minimal ", "minimal"),
    ("None", "none"),  # the string, not the object
]


@pytest.mark.parametrize("surface", ALL_SURFACES)
@pytest.mark.parametrize(("given", "sent"), TIERS)
def test_every_reasoning_effort_tier_is_sent_top_level(http, surface, given, sent):
    assert ALL_SURFACES[surface](audio(AUDIO_URL), reasoning_effort=given)["text"] == "Hello"

    body = http.last_body
    assert body["reasoning_effort"] == sent
    assert "reasoning" not in body  # provider perceptron signals reasoning with the THINK hint only
    assert not any("THINK" in hint for hint in _hints(body))  # a tier never adds the THINK hint


@pytest.mark.parametrize("surface", ALL_SURFACES)
def test_reasoning_effort_is_absent_unless_given(http, surface):
    ALL_SURFACES[surface](audio(AUDIO_URL))

    assert "reasoning_effort" not in http.last_body


INVALID_TIERS = ["extreme", "", "max", "hi gh", 3]


@pytest.mark.parametrize("surface", RAISING_SURFACES)
@pytest.mark.parametrize("value", INVALID_TIERS)
def test_invalid_reasoning_effort_raises_before_any_request(http, surface, value):
    with pytest.raises(BadRequestError) as excinfo:
        RAISING_SURFACES[surface](audio(AUDIO_URL), reasoning_effort=value)

    assert excinfo.value.code == INVALID_REASONING_EFFORT
    assert http.requests == []


@pytest.mark.parametrize("surface", STREAM_EVENT_SURFACES)
@pytest.mark.parametrize("value", INVALID_TIERS)
def test_invalid_reasoning_effort_ends_legacy_streams_before_any_request(http, surface, value):
    """Legacy event streams report pre-request errors as their single terminal ``error`` event (§12.5)."""
    events = STREAM_EVENT_SURFACES[surface](audio(AUDIO_URL), reasoning_effort=value)

    assert [event["type"] for event in events] == ["error"]
    assert events[0]["code"] == INVALID_REASONING_EFFORT and events[0]["partial"] is None
    assert http.requests == []


@pytest.mark.parametrize("surface", LEGACY)
@pytest.mark.parametrize("tier", ["none", "high"])
def test_reasoning_true_sends_the_think_hint_and_the_tier(http, surface, tier):
    """#85: ``reasoning=True`` keeps the THINK hint (a system message on provider perceptron); the tier is sent as
    given. Neither is derived from the other and there is no conflict error."""
    LEGACY[surface](audio(AUDIO_URL), reasoning=True, reasoning_effort=tier)

    body = http.last_body
    assert body["reasoning_effort"] == tier
    assert "reasoning" not in body
    first = body["messages"][0]
    assert first["role"] == "system" and first["content"].startswith("<hint>")
    assert "THINK" in first["content"].split("</hint>")[0]
    assert [hint for hint in _hints(body) if "THINK" in hint] == [first["content"]]  # sent once


@pytest.mark.parametrize("surface", LEGACY)
def test_reasoning_true_alone_sends_no_tier(http, surface):
    LEGACY[surface](audio(AUDIO_URL), reasoning=True)

    body = http.last_body
    assert "reasoning_effort" not in body
    assert "THINK" in body["messages"][0]["content"]


@pytest.mark.parametrize("surface", LEGACY)
def test_reasoning_false_with_a_tier_sends_only_the_tier(http, surface):
    LEGACY[surface](audio(AUDIO_URL), reasoning=False, reasoning_effort="medium")

    body = http.last_body
    assert body["reasoning_effort"] == "medium"
    assert not any("THINK" in hint for hint in _hints(body))


def test_provider_names_are_case_insensitive_for_hints(http, monkeypatch):
    """The client resolves ``provider="Perceptron"`` case-insensitively, so ``perceive`` must pick the same registry
    entry: the hints stay a system message and ``reasoning=True`` is kept."""
    result = perceive(audio(AUDIO_URL), text(PROMPT), provider="Perceptron", reasoning=True, expects="box")

    first = http.last_body["messages"][0]
    assert (first["role"], first["content"]) == ("system", "<hint>BOX THINK</hint>")
    assert result.errors == []

    monkeypatch.setenv("PERCEPTRON_PROVIDER", "PERCEPTRON")
    question(audio(AUDIO_URL), PROMPT, reasoning=True)
    first = http.last_body["messages"][0]
    assert (first["role"], first["content"]) == ("system", "<hint>THINK</hint>")


def test_reasoning_effort_is_sent_on_fal_too(monkeypatch):
    """The tier is provider-independent: normalized and sent top-level on fal as well, with no THINK hint derived from
    it; an ``expects`` hint keeps fal's user-text encoding (O2)."""
    recorder = install(monkeypatch, lambda request: json_response(completion("Hello")))

    Client(provider="fal").generate(_task([text(PROMPT)]), model="isaac-0.1", expects="box", reasoning_effort="Medium")

    assert str(recorder.last.url) == "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"
    body = recorder.last_body
    assert body["reasoning_effort"] == "medium"
    assert "reasoning" not in body
    assert body["messages"] == [{"role": "user", "content": f"<hint>BOX</hint>{PROMPT}"}]


# ---------------------------------------------------------------------------
# Mixed-media ordering (§13.4): identical part order and asset_count on every surface
# ---------------------------------------------------------------------------

NODE_PARTS = {
    "image": (lambda: image(PNG_BYTES), "image_url"),
    "video": (lambda: video(VIDEO_URL), "video_url"),
    "audio": (lambda: audio(WAV), "input_audio"),
    "frames": (lambda: video_frames([(PNG_BYTES, 0), ("https://example.com/f1.jpg", 40)]), "video_frames"),
    "text": (lambda: text("Compare them."), "text"),
}
INTERLEAVINGS = [
    ("audio", "text", "image", "video", "frames"),
    ("text", "video", "audio", "text", "image"),
    ("frames", "audio", "audio", "text", "image", "text"),
    ("image", "audio", "video", "frames", "audio", "text"),
    ("audio", "image", "text", "audio", "video", "text", "frames"),
]
SEQUENCE_SURFACES = {**LEGACY_SEQUENCE, **MESSAGE_API}


def _nodes(order) -> list:
    return [NODE_PARTS[name][0]() for name in order]


def _user_content(body: dict) -> list[dict]:
    [message] = body["messages"]
    assert message["role"] == "user"
    return message["content"]


@pytest.mark.parametrize("surface", SEQUENCE_SURFACES)
@pytest.mark.parametrize("order", INTERLEAVINGS, ids="-".join)
def test_interleaved_media_keep_their_order_and_asset_count(http, surface, order):
    outcome = SEQUENCE_SURFACES[surface](_nodes(order))

    content = _user_content(http.last_body)
    assert [part["type"] for part in content] == [NODE_PARTS[name][1] for name in order]
    assert outcome["asset_count"] == sum(name != "text" for name in order)


def test_every_interleaving_lowers_identically_on_the_legacy_path_and_the_message_api(http):
    """All 120 orders of image/video/audio/video_frames/text: the legacy path and the message API send byte-identical
    user content, in the given order, and both report four assets."""
    for order in itertools.permutations(["image", "video", "audio", "frames", "text"]):
        results = [surface(_nodes(order)) for surface in (_perceive, _client_generate, _create)]

        bodies = [json.loads(request.content) for request in http.requests[-3:]]
        contents = [_user_content(body) for body in bodies]
        assert contents[0] == contents[1] == contents[2], order
        assert [part["type"] for part in contents[0]] == [NODE_PARTS[name][1] for name in order], order
        assert [result["asset_count"] for result in results] == [4, 4, 4], order
