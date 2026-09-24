"""DSL media forms: data URLs, uploaded-file ids, path errors, `video_frames`, and mixed-media ordering."""

from __future__ import annotations

import base64
import itertools
from dataclasses import dataclass

import pytest
from _http_mock import completion, install, json_response
from _image_fixtures import PNG_BYTES

from perceptron import Client, caption, detect, image, inspect_task, perceive, text
from perceptron import client as client_mod
from perceptron._lowering import count_assets
from perceptron.dsl.nodes import (
    MAX_VIDEO_FRAMES,
    VideoFrame,
    VideoFrames,
    agent,
    audio,
    point,
    tool_result,
    video,
    video_frames,
)
from perceptron.dsl.perceive import _compile
from perceptron.errors import (
    INVALID_DATA_URL,
    INVALID_FILE_ID,
    INVALID_VIDEO_FRAMES,
    UNSUPPORTED_PROVIDER_FEATURE,
    BadRequestError,
)
from perceptron.prompting import ModalityPrompt

FILE_ID = "file-abcdefghijklmnopqrstuv"
OTHER_FILE_ID = "file-ABCDEFGHIJKLMNOPQRSTUV"
FILES_URL = "https://api.perceptron.inc/v1/files/{}/content"
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24
MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
FLAC_BYTES = b"fLaC\x00\x00\x00\x22" + b"\x00" * 34
FACTORIES = {"image": image, "video": video, "audio": audio}
DATA_URLS = {
    "image": PNG_DATA_URL,
    "video": "data:video/mp4;base64,AAAAGGZ0eXBtcDQy",
    "audio": "data:audio/wav;base64,UklGRiQAAABXQVZF",
}


@dataclass
class File:
    """Stands in for the Files API's `File` (only `.id` matters)."""

    id: str
    filename: str = "upload.bin"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _compiled(*nodes):
    seq = nodes[0] if len(nodes) == 1 else sum(nodes[1:], nodes[0])
    task, issues = _compile(seq, expects=None, strict=False)
    assert issues == []
    return task


def _parts(*nodes, **lowering):
    task = _compiled(*nodes)
    messages = client_mod._task_to_openai_messages(task, **lowering)
    content = messages[0]["content"]
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


# ---------------------------------------------------------------------------
# data: URLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_data_url_passes_through(kind):
    url = DATA_URLS[kind]
    node = FACTORIES[kind](url)

    assert node.obj == url and node.file_id is None
    assert _compiled(node)["content"][0]["content"] == url
    assert _parts(node) == [{"type": f"{kind}_url", f"{kind}_url": {"url": url}}]  # audio too: not input_audio


@pytest.mark.parametrize(
    ("kind", "url"),
    [
        ("image", DATA_URLS["video"]),
        ("video", DATA_URLS["audio"]),
        ("audio", DATA_URLS["image"]),
        ("image", "data:image/png,not-base64"),
        ("audio", "data:;base64,AAAA"),
    ],
)
def test_data_url_family_must_match_the_node(kind, url):
    with pytest.raises(BadRequestError) as excinfo:
        FACTORIES[kind](url)
    assert excinfo.value.code == INVALID_DATA_URL


# ---------------------------------------------------------------------------
# Uploaded files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_file_id_keyword_and_file_objects(kind):
    for node in (FACTORIES[kind](file_id=FILE_ID), FACTORIES[kind](File(FILE_ID))):
        assert node.obj is None and node.file_id == FILE_ID
        assert _compiled(node)["content"] == [{"type": kind, "role": "user", "file_id": FILE_ID}]
        assert _parts(node) == [{"type": f"{kind}_file_id", f"{kind}_file_id": {"file_id": FILE_ID}}]


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
@pytest.mark.parametrize("bad", ["file-abc", "file-abcdefghijklmnopqrstu!", "abcdefghijklmnopqrstuvwxyz0"])
def test_invalid_file_ids_raise(kind, bad):
    for call in (lambda: FACTORIES[kind](file_id=bad), lambda: FACTORIES[kind](File(bad))):
        with pytest.raises(BadRequestError) as excinfo:
            call()
        assert excinfo.value.code == INVALID_FILE_ID


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_exactly_one_of_obj_and_file_id(kind):
    with pytest.raises(TypeError, match=rf"^{kind}\(\) takes exactly one of obj or file_id$"):
        FACTORIES[kind]("https://example.com/x", file_id=FILE_ID)
    with pytest.raises(TypeError, match=rf"^{kind}\(\) takes exactly one of obj or file_id$"):
        FACTORIES[kind]()


def test_file_id_like_strings_stay_paths_and_urls():
    # Only `file_id=` or a file object selects an upload; strings keep their path/URL meaning.
    assert image(FILE_ID).file_id is None
    assert image("https://example.com/file-abcdefghijklmnopqrstuv").file_id is None


def test_tool_result_images_accept_file_ids_and_data_urls():
    task = _compiled(
        text("Look it up."),
        agent(None, tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]),
        tool_result("c1", "found", image(file_id=FILE_ID), image(PNG_DATA_URL)),
    )
    tool = client_mod._task_to_openai_messages(task)[2]

    assert tool == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [
            {"type": "text", "text": "found"},
            {"type": "image_file_id", "image_file_id": {"file_id": FILE_ID}},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
        ],
    }


def test_file_ids_reach_the_request(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    perceive(video(File(FILE_ID)) + text("What happens?"), provider="perceptron")

    assert http.last_body["messages"][-1]["content"] == [
        {"type": "video_file_id", "video_file_id": {"file_id": FILE_ID}},
        {"type": "text", "text": "What happens?"},
    ]


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_file_ids_on_another_provider_raise_before_any_request(monkeypatch, kind):
    # Uploaded files live on the Perceptron API; provider fal, chosen or auto-selected (only FAL_KEY set), must not
    # receive them, just as file-id frames are refused.
    monkeypatch.delenv("PERCEPTRON_API_KEY")
    monkeypatch.setenv("FAL_KEY", "fal-key")
    http = install(monkeypatch, lambda request: json_response(completion()))
    node = FACTORIES[kind](file_id=FILE_ID)
    calls = [
        lambda: perceive(node, text("Describe.")),
        lambda: caption(node) if kind == "image" else perceive(node, text("x"), provider="fal"),
        lambda: Client(provider="fal").chat.completions.create(messages=[{"role": "user", "content": [node, "Hi"]}]),
    ]
    if kind == "image":
        calls.append(lambda: perceive(text("Look it up."), tool_result("c1", node), provider="fal"))
    for call in calls:
        with pytest.raises(BadRequestError) as excinfo:
            call()
        assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
        assert 'configure(provider="perceptron")' in str(excinfo.value)
    assert http.requests == []


# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_unreadable_paths_raise_invalid_media_path(tmp_path, kind):
    for path in (tmp_path / "missing.bin", tmp_path, str(tmp_path / "missing.bin")):
        with pytest.raises(BadRequestError) as excinfo:
            _compile(FACTORIES[kind](path), expects=None, strict=False)
        assert excinfo.value.code == "invalid_media_path"
        assert excinfo.value.details == {"origin": str(path)}
        assert kind in str(excinfo.value)


@pytest.mark.parametrize(("data", "fmt"), [(WAV_BYTES, "wav"), (MP3_BYTES, "mp3"), (FLAC_BYTES, "flac")])
def test_local_audio_files_are_encoded(tmp_path, data, fmt):
    path = tmp_path / f"clip.{fmt}"
    path.write_bytes(data)

    assert _parts(audio(path)) == [
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode("ascii"), "format": fmt}}
    ]


# ---------------------------------------------------------------------------
# video_frames
# ---------------------------------------------------------------------------


def test_video_frames_wire_part_through_inspect_task():
    @perceive()
    def clip():
        return video_frames(
            [
                (PNG_BYTES, 0),
                VideoFrame("https://cdn.example.com/f1.jpg", 500),
                (PNG_DATA_URL, 500),  # equal timestamps are allowed
                (image(PNG_BYTES), 1000),
            ]
        ) + text("What moves?")

    task, issues = inspect_task(clip)
    assert issues == []
    assert task["content"][0] == {
        "type": "video_frames",
        "role": "user",
        "frames": [
            {"url": PNG_DATA_URL, "timestamp_ms": 0},
            {"url": "https://cdn.example.com/f1.jpg", "timestamp_ms": 500},
            {"url": PNG_DATA_URL, "timestamp_ms": 500},
            {"url": PNG_DATA_URL, "timestamp_ms": 1000},
        ],
    }

    parts = client_mod._task_to_openai_messages(task)[0]["content"]
    assert parts[0] == {
        "type": "video_frames",
        "video_frames": {
            "frames": [
                {"image_url": {"url": PNG_DATA_URL}, "timestamp_ms": 0},
                {"image_url": {"url": "https://cdn.example.com/f1.jpg"}, "timestamp_ms": 500},
                {"image_url": {"url": PNG_DATA_URL}, "timestamp_ms": 500},
                {"image_url": {"url": PNG_DATA_URL}, "timestamp_ms": 1000},
            ]
        },
    }
    assert parts[1] == {"type": "text", "text": "What moves?"}


def test_video_frames_file_ids_become_files_content_urls(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))
    frames = video_frames([VideoFrame(image(file_id=FILE_ID), 0), (File(OTHER_FILE_ID), 40)])

    assert _compiled(frames)["content"][0]["frames"] == [
        {"file_id": FILE_ID, "timestamp_ms": 0},
        {"file_id": OTHER_FILE_ID, "timestamp_ms": 40},
    ]

    perceive(frames + text("Describe."), provider="perceptron")

    part = http.last_body["messages"][-1]["content"][0]
    assert part["video_frames"]["frames"] == [
        {"image_url": {"url": FILES_URL.format(FILE_ID)}, "timestamp_ms": 0},
        {"image_url": {"url": FILES_URL.format(OTHER_FILE_ID)}, "timestamp_ms": 40},
    ]


def test_video_frames_groups_keep_their_order_and_count_once_each():
    first = video_frames([("https://x/a0.jpg", 0), ("https://x/a1.jpg", 10), ("https://x/a2.jpg", 20)])
    second = video_frames([("https://x/b0.jpg", 5), ("https://x/b1.jpg", 5)])

    task = _compiled(first, text("vs"), second)
    messages = client_mod._task_to_openai_messages(task)

    parts = messages[0]["content"]
    assert [p["type"] for p in parts] == ["video_frames", "text", "video_frames"]
    assert [f["image_url"]["url"] for f in parts[0]["video_frames"]["frames"]] == [
        "https://x/a0.jpg",
        "https://x/a1.jpg",
        "https://x/a2.jpg",
    ]
    assert [f["image_url"]["url"] for f in parts[2]["video_frames"]["frames"]] == [
        "https://x/b0.jpg",
        "https://x/b1.jpg",
    ]
    assert count_assets(messages) == 2


def test_video_frames_in_create_content_lists(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))
    frames = video_frames([(PNG_BYTES, 0), (image(file_id=FILE_ID), 33)])

    result = Client().chat.completions.create(messages=[{"role": "user", "content": [frames, "What happens?"]}])

    assert http.last_body["messages"][0]["content"] == [
        {
            "type": "video_frames",
            "video_frames": {
                "frames": [
                    {"image_url": {"url": PNG_DATA_URL}, "timestamp_ms": 0},
                    {"image_url": {"url": FILES_URL.format(FILE_ID)}, "timestamp_ms": 33},
                ]
            },
        },
        {"type": "text", "text": "What happens?"},
    ]
    assert result.asset_count == 1


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ([("https://x/0.jpg", 0)], "2-256 frames; got 1"),
        ([("https://x/f.jpg", i) for i in range(MAX_VIDEO_FRAMES + 1)], "2-256 frames; got 257"),
        ([("https://x/0.jpg", 0), ("https://x/1.jpg", 1.0)], "frame 1 needs an integer timestamp_ms"),
        ([("https://x/0.jpg", True), ("https://x/1.jpg", 2)], "frame 0 needs an integer timestamp_ms"),
        ([("https://x/0.jpg", -1), ("https://x/1.jpg", 2)], "frame 0 needs an integer timestamp_ms"),
        ([("https://x/0.jpg", "0"), ("https://x/1.jpg", 2)], "frame 0 needs an integer timestamp_ms"),
        ([("https://x/0.jpg", 500), ("https://x/1.jpg", 499)], "must not decrease; frame 1 is at 499 ms after 500 ms"),
        ([("https://x/0.jpg", 0), "https://x/1.jpg"], "frame 1 must be a VideoFrame"),
        ([("https://x/0.jpg", 0, "extra"), ("https://x/1.jpg", 1)], "frame 0 must be a VideoFrame"),
        ((f for f in [("https://x/0.jpg", 0), ("https://x/1.jpg", 1)]), "takes a list"),
        (
            [(None, 0), ("https://x/1.jpg", 1)],
            r"frame 0 image must be an image \(anything image\(\) accepts\); got NoneType",
        ),
        ([("https://x/0.jpg", 0), (video("https://x/v.mp4"), 1)], r"frame 1 image must be an image .*; got Video\.$"),
        ([(audio("https://x/a.wav"), 0), ("https://x/1.jpg", 1)], r"frame 0 image must be an image .*; got Audio\.$"),
        ([(text("a frame"), 0), ("https://x/1.jpg", 1)], r"frame 0 image must be an image .*; got Text\.$"),
        (
            [(video_frames([("https://x/0.jpg", 0), ("https://x/1.jpg", 1)]), 0), ("https://x/1.jpg", 1)],
            r"frame 0 image must be an image .*; got VideoFrames\.$",
        ),
    ],
)
def test_video_frames_validation(frames, message):
    with pytest.raises(BadRequestError, match=message) as excinfo:
        video_frames(frames)
    assert excinfo.value.code == INVALID_VIDEO_FRAMES


def test_video_frames_limits_and_direct_construction():
    assert len(video_frames([("https://x/f.jpg", 0)] * MAX_VIDEO_FRAMES).frames) == MAX_VIDEO_FRAMES
    # The dataclass validates and normalizes too.
    node = VideoFrames([("https://x/0.jpg", 0), ("https://x/1.jpg", 1)])
    assert node.frames == [VideoFrame("https://x/0.jpg", 0), VideoFrame("https://x/1.jpg", 1)]
    with pytest.raises(BadRequestError):
        VideoFrames([("https://x/0.jpg", 1), ("https://x/1.jpg", 0)])


def test_video_frame_images_are_checked_when_built():
    with pytest.raises(BadRequestError) as excinfo:
        video_frames([(DATA_URLS["video"], 0), ("https://x/1.jpg", 1)])
    assert excinfo.value.code == INVALID_DATA_URL
    with pytest.raises(BadRequestError) as excinfo:
        video_frames([(File("file-short"), 0), ("https://x/1.jpg", 1)])
    assert excinfo.value.code == INVALID_FILE_ID


def test_video_frames_are_prompted_as_video(monkeypatch):
    frames = video_frames([("https://x/0.jpg", 0), ("https://x/1.jpg", 40)])
    assert ModalityPrompt(image="image", video="video", audio="audio").get(frames) == "video"

    tasks = []
    monkeypatch.setattr(client_mod.Client, "generate", lambda self, task, **kwargs: tasks.append(task) or {"text": ""})
    for run in (
        lambda media: caption(media, provider="perceptron"),
        lambda media: detect(media, classes=["cup"], provider="perceptron"),
    ):
        tasks.clear()
        run(video("https://x/v.mp4"))
        run(frames)
        as_video, as_frames = ([e for e in task["content"] if e["type"] == "text"] for task in tasks)
        assert as_frames == as_video and as_frames  # the same instructions as for video(...)
        assert [e["type"] for e in tasks[1]["content"]].count("video_frames") == 1


def test_video_frame_paths_fail_like_images(tmp_path):
    frames = video_frames([(tmp_path / "missing.png", 0), ("https://x/1.jpg", 1)])
    with pytest.raises(BadRequestError) as excinfo:
        _compile(frames, expects=None, strict=False)
    assert excinfo.value.code == "invalid_media_path"


# ---------------------------------------------------------------------------
# Mixed-media ordering and asset numbering
# ---------------------------------------------------------------------------


def _media_nodes():
    return {
        "image": image("https://x/i.png"),
        "video": video("https://x/v.mp4"),
        "audio": audio("https://x/a.wav"),
        "frames": video_frames([("https://x/f0.jpg", 0), ("https://x/f1.jpg", 40)]),
        "text": text("Describe."),
    }


def test_mixed_media_keep_part_order_and_asset_numbering():
    part_types = {
        "image": "image_url",
        "video": "video_url",
        "audio": "audio_url",
        "frames": "video_frames",
        "text": "text",
    }
    for order in itertools.permutations(["image", "video", "audio", "frames", "text"]):
        nodes = _media_nodes()
        media = [name for name in order if name != "text"]
        # One tag per media node, anchored with asset=; its asset_idx is the node's position among the media.
        tags = [point(1, 2, asset=nodes[name]) for name in media]
        task = _compiled(*(nodes[name] for name in order), *tags)
        messages = client_mod._task_to_openai_messages(task)

        assert len(messages) == 1, order
        parts = messages[0]["content"]
        assert [p["type"] for p in parts[: len(order)]] == [part_types[name] for name in order], order
        assert [p["text"] for p in parts[len(order) :]] == [
            f'<point asset_idx="{i}"> (1,2) </point>' for i in range(len(media))
        ], order
        assert count_assets(messages) == len(media)


def test_encoded_mixed_media_keep_their_order(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(WAV_BYTES)
    nodes = [
        audio(wav),
        text("then"),
        image(PNG_BYTES),
        video_frames([(PNG_BYTES, 0), (PNG_BYTES, 100)]),
        video(DATA_URLS["video"]),
        image(file_id=FILE_ID),
    ]

    parts = _parts(*nodes)

    assert [p["type"] for p in parts] == [
        "input_audio",
        "text",
        "image_url",
        "video_frames",
        "video_url",
        "image_file_id",
    ]
