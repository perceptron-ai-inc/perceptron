import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from perceptron import PerceiveResult
from perceptron.cli import (
    OutputFormat,
    _bucket_for_expects,
    _coerce_result_dict,
    _describe_point,
    _looks_like_video,
    _make_media_node,
    _resolve_media,
    _stream_render,
    app,
)
from perceptron.dsl.nodes import Image as ImageNode
from perceptron.dsl.nodes import Video as VideoNode
from perceptron.pointing.types import BoundingBox, Clip, ClipTimestamp, Polygon, SinglePoint


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    monkeypatch.setenv("COLUMNS", "180")


runner = CliRunner()


class _StubResult(PerceiveResult):
    def __init__(self, text: str):
        super().__init__(
            text=text,
            points=None,
            boxes=None,
            polygons=None,
            clips=None,
            parsed=None,
            reasoning=None,
            usage=None,
            errors=[],
            raw={"text": text},
        )


def test_caption_command(monkeypatch, tmp_path):
    image_path = tmp_path / "img.bin"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr("perceptron.cli.caption_image", lambda *a, **k: _StubResult("hello"))

    result = runner.invoke(app, ["caption", str(image_path)])
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_caption_command_json_output(monkeypatch):
    monkeypatch.setattr("perceptron.cli.caption_image", lambda *a, **k: _StubResult("hello"))
    result = runner.invoke(app, ["caption", "https://example.com/img", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["text"] == "hello"


def test_caption_command_text_expectation(monkeypatch):
    monkeypatch.setattr("perceptron.cli.caption_image", lambda *a, **k: _StubResult("caption"))
    result = runner.invoke(app, ["caption", "https://example.com/img", "--expects", "text"])
    assert result.exit_code == 0
    assert "caption" in result.stdout


def test_caption_command_directory(monkeypatch, tmp_path):
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")
    (tmp_path / "notes.txt").write_text("ignore me")

    def _fake_caption(data, **kwargs):
        assert isinstance(data, ImageNode)
        if data.obj == b"image-one":
            return _StubResult("caption-one")
        if data.obj == b"image-two":
            return _StubResult("caption-two")
        raise AssertionError("unexpected payload")

    monkeypatch.setattr("perceptron.cli.caption_image", _fake_caption)

    result = runner.invoke(app, ["caption", str(tmp_path)])
    assert result.exit_code == 0
    output_path = tmp_path / "captions.json"
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data == {"one.png": "caption-one", "two.jpg": "caption-two"}
    assert "captions.json" in result.stdout


def test_caption_command_directory_stream_not_supported(tmp_path):
    (tmp_path / "one.png").write_bytes(b"image-one")

    result = runner.invoke(app, ["caption", str(tmp_path), "--stream"])
    assert result.exit_code != 0
    assert "Streaming output is not supported" in result.stdout


def test_ocr_command_directory(monkeypatch, tmp_path):
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")

    def _fake_ocr(data, *, prompt=None):
        assert isinstance(data, ImageNode)
        if data.obj == b"image-one":
            return _StubResult("ocr-one")
        if data.obj == b"image-two":
            return _StubResult("ocr-two")
        raise AssertionError("unexpected payload")

    monkeypatch.setattr("perceptron.cli.ocr_image", _fake_ocr)

    result = runner.invoke(app, ["ocr", str(tmp_path), "--prompt", "read everything"])
    assert result.exit_code == 0
    data = json.loads((tmp_path / "ocr.json").read_text())
    assert data == {"one.png": "ocr-one", "two.jpg": "ocr-two"}


def test_ocr_command(monkeypatch):
    monkeypatch.setattr("perceptron.cli.ocr_image", lambda *a, **k: _StubResult("ocr text"))
    result = runner.invoke(app, ["ocr", "https://example.com/img"])
    assert result.exit_code == 0
    assert "ocr text" in result.stdout


def test_detect_command(monkeypatch):
    res = _StubResult("detected")
    res.points = []
    monkeypatch.setattr("perceptron.cli.detect_image", lambda *a, **k: res)
    result = runner.invoke(app, ["detect", "/tmp/img.png", "--classes", "person,bike"])
    assert result.exit_code == 0
    assert "detected" in result.stdout


def test_detect_command_directory(monkeypatch, tmp_path):
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")

    def _fake_detect(data, *, classes=None):
        assert classes == ["person"]
        assert isinstance(data, ImageNode)
        res = _StubResult("detected-one" if data.obj == b"image-one" else "detected-two")
        res.boxes = [
            BoundingBox(
                top_left=SinglePoint(1, 2, mention="person"),
                bottom_right=SinglePoint(3, 4),
                mention="person",
            )
        ]
        return res

    monkeypatch.setattr("perceptron.cli.detect_image", _fake_detect)

    result = runner.invoke(app, ["detect", str(tmp_path), "--classes", "person"])
    assert result.exit_code == 0
    output_path = tmp_path / "detections.json"
    data = json.loads(output_path.read_text())
    assert set(data.keys()) == {"one.png", "two.jpg"}
    assert data["one.png"]["text"] == "detected-one"
    boxes = data["one.png"].get("boxes")
    assert boxes and boxes[0]["type"] == "box"
    assert boxes[0]["top_left"]["x"] == 1
    assert boxes[0]["top_left"]["mention"] == "person"


def test_detect_command_stream(monkeypatch):
    events = [
        {"type": "text.delta", "chunk": "hi"},
        {"type": "final", "result": {"text": "done", "errors": []}},
    ]

    def _fake_detect(image, *, classes=None, stream=False):
        assert stream is True
        return iter(events)

    captured = {}

    def _fake_stream_render(ev, **kwargs):
        captured["events"] = list(ev)
        captured["kwargs"] = kwargs

    monkeypatch.setattr("perceptron.cli.detect_image", _fake_detect)
    monkeypatch.setattr("perceptron.cli._stream_render", _fake_stream_render)

    result = runner.invoke(app, ["detect", "https://example.com/img", "--stream"])
    assert result.exit_code == 0
    assert captured["events"] == events
    assert captured["kwargs"]["show_points_table"] is True


def test_question_command(monkeypatch):
    monkeypatch.setattr("perceptron.cli.question_image", lambda *a, **k: _StubResult("cat"))
    result = runner.invoke(app, ["question", "https://example.com/img", "What is shown?"])
    assert result.exit_code == 0
    assert "cat" in result.stdout


def test_question_command_forwards_reasoning_effort(monkeypatch):
    captured: dict[str, dict] = {}

    def _fake_question(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _StubResult("cat")

    monkeypatch.setattr("perceptron.cli.question_image", _fake_question)
    result = runner.invoke(app, ["question", "https://example.com/img", "What is shown?", "--reasoning-effort", "Low"])
    assert result.exit_code == 0, result.stdout
    assert captured["kwargs"]["reasoning_effort"] == "low"
    assert "enable_audio_in_video" not in captured["kwargs"]


def test_question_command_rejects_unknown_reasoning_effort(monkeypatch):
    monkeypatch.setattr("perceptron.cli.question_image", lambda *a, **k: _StubResult("cat"))
    result = runner.invoke(
        app, ["question", "https://example.com/img", "What is shown?", "--reasoning-effort", "extreme"]
    )
    assert result.exit_code != 0


def test_caption_command_forwards_reasoning_effort(monkeypatch):
    captured: dict[str, dict] = {}

    def _fake_caption(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _StubResult("hello")

    monkeypatch.setattr("perceptron.cli.caption_image", _fake_caption)
    result = runner.invoke(app, ["caption", "https://example.com/img", "--reasoning-effort", "high"])
    assert result.exit_code == 0, result.stdout
    assert captured["kwargs"]["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("media", "expected"),
    [
        ("https://example.com/img.png", "box"),
        ("https://example.com/clip.mp4", "text"),
        ("https://example.com/call.wav", "text"),
    ],
)
def test_caption_command_default_expects_follows_media(monkeypatch, media, expected):
    captured: dict[str, dict] = {}

    def _fake_caption(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _StubResult("hello")

    monkeypatch.setattr("perceptron.cli.caption_image", _fake_caption)
    result = runner.invoke(app, ["caption", media])
    assert result.exit_code == 0, result.stdout
    assert captured["kwargs"]["expects"] == expected


def test_caption_command_explicit_expects_overrides_media_default(monkeypatch):
    captured: dict[str, dict] = {}

    def _fake_caption(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _StubResult("hello")

    monkeypatch.setattr("perceptron.cli.caption_image", _fake_caption)
    result = runner.invoke(app, ["caption", "https://example.com/clip.mp4", "--expects", "box"])
    assert result.exit_code == 0, result.stdout
    assert captured["kwargs"]["expects"] == "box"


def test_question_command_forwards_audio_in_video_with_reasoning_effort(monkeypatch):
    captured: dict[str, dict] = {}

    def _fake_question(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _StubResult("cat")

    monkeypatch.setattr("perceptron.cli.question_image", _fake_question)
    result = runner.invoke(
        app,
        [
            "question",
            "https://example.com/clip.mp4",
            "What is said?",
            "--audio-in-video",
            "--reasoning-effort",
            "medium",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["kwargs"] == {"expects": "text", "enable_audio_in_video": True, "reasoning_effort": "medium"}


def test_question_command_box_json(monkeypatch):
    res = _StubResult("box answer")
    res.boxes = [
        BoundingBox(
            top_left=SinglePoint(1, 2, mention="item"),
            bottom_right=SinglePoint(3, 4),
            mention="item",
        )
    ]
    monkeypatch.setattr("perceptron.cli.question_image", lambda *a, **k: res)
    result = runner.invoke(
        app,
        [
            "question",
            "https://example.com/img",
            "Where is the item?",
            "--expects",
            "box",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["text"] == "box answer"
    assert payload["boxes"][0]["type"] == "box"


def test_config_command():
    result = runner.invoke(app, ["config", "--provider", "perceptron", "--api-key", "abc"])
    assert result.exit_code == 0
    assert "PERCEPTRON_PROVIDER=perceptron" in result.stdout
    assert "PERCEPTRON_API_KEY=abc" in result.stdout


def test_config_command_exports_a_fal_key_as_fal_key():
    result = runner.invoke(app, ["config", "--provider", "fal", "--api-key", "abc"])
    assert result.exit_code == 0
    assert "PERCEPTRON_PROVIDER=fal" in result.stdout
    assert "FAL_KEY=abc" in result.stdout
    assert "PERCEPTRON_API_KEY" not in result.stdout  # fal never reads it


# ---------------------------------------------------------------------------
# Per-kind JSON output coverage
# ---------------------------------------------------------------------------


def test_caption_command_box_json_emits_boxes_key(monkeypatch):
    res = _StubResult("describe")
    res.boxes = [
        BoundingBox(
            top_left=SinglePoint(10, 20, mention="lamp"),
            bottom_right=SinglePoint(30, 40),
            mention="lamp",
        )
    ]
    monkeypatch.setattr("perceptron.cli.caption_image", lambda *a, **k: res)
    result = runner.invoke(
        app,
        ["caption", "https://example.com/img", "--expects", "box", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["boxes"][0]["type"] == "box"
    assert "points" not in payload
    assert "polygons" not in payload


def test_question_command_point_json_emits_points_key(monkeypatch):
    res = _StubResult("center")
    res.points = [SinglePoint(50, 60, mention="middle")]
    monkeypatch.setattr("perceptron.cli.question_image", lambda *a, **k: res)
    result = runner.invoke(
        app,
        [
            "question",
            "https://example.com/img",
            "Where is the center?",
            "--expects",
            "point",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["points"][0]["type"] == "point"
    assert "boxes" not in payload
    assert "polygons" not in payload


def test_question_command_clip_json_emits_clips_key(monkeypatch):
    res = _StubResult("scene")
    res.clips = [Clip(timestamp=ClipTimestamp(at=1.5), mention="intro")]
    monkeypatch.setattr("perceptron.cli.question_image", lambda *a, **k: res)
    result = runner.invoke(
        app,
        [
            "question",
            "https://example.com/clip.mp4",
            "When does the intro happen?",
            "--expects",
            "clip",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["clips"][0]["type"] == "clip"
    assert payload["clips"][0]["mention"] == "intro"
    assert "boxes" not in payload
    assert "points" not in payload


def test_question_command_polygon_json_emits_polygons_key(monkeypatch):
    res = _StubResult("region")
    res.polygons = [
        Polygon(
            hull=[SinglePoint(0, 0), SinglePoint(10, 0), SinglePoint(5, 10)],
            mention="hull",
        )
    ]
    monkeypatch.setattr("perceptron.cli.question_image", lambda *a, **k: res)
    result = runner.invoke(
        app,
        [
            "question",
            "https://example.com/img",
            "Outline the region.",
            "--expects",
            "polygon",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["polygons"][0]["type"] == "polygon"
    assert "boxes" not in payload
    assert "points" not in payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_bucket_for_expects_routes_each_kind():
    res = _StubResult("x")
    res.points = [SinglePoint(1, 1)]
    res.boxes = [BoundingBox(top_left=SinglePoint(0, 0), bottom_right=SinglePoint(2, 2))]
    res.polygons = [Polygon(hull=[SinglePoint(0, 0), SinglePoint(2, 0), SinglePoint(1, 2)])]

    assert _bucket_for_expects(res, "point") == ("points", res.points)
    assert _bucket_for_expects(res, "box") == ("boxes", res.boxes)
    assert _bucket_for_expects(res, "polygon") == ("polygons", res.polygons)


def test_bucket_for_expects_returns_none_for_unsupported():
    res = _StubResult("x")
    res.boxes = [BoundingBox(top_left=SinglePoint(0, 0), bottom_right=SinglePoint(1, 1))]

    # `text`/`think`/None aren't in the bucket map.
    assert _bucket_for_expects(res, "text") is None
    assert _bucket_for_expects(res, "think") is None
    assert _bucket_for_expects(res, None) is None


def test_bucket_for_expects_returns_none_when_bucket_empty():
    res = _StubResult("x")
    # boxes is None — nothing to surface.
    assert _bucket_for_expects(res, "box") is None


def test_coerce_result_dict_normalizes_all_buckets():
    box = BoundingBox(top_left=SinglePoint(1, 1), bottom_right=SinglePoint(2, 2))
    coerced = _coerce_result_dict({"text": "hi", "boxes": [box]})

    # All three bucket fields are present (None for missing ones).
    assert coerced["text"] == "hi"
    assert coerced["boxes"] == [box]
    assert coerced["points"] is None
    assert coerced["polygons"] is None
    assert coerced["errors"] == []


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_render_routes_box_final_event_into_boxes_bucket(monkeypatch):
    """Streaming `final` event with a `boxes` field should reach the JSON output under `boxes`."""

    box = BoundingBox(
        top_left=SinglePoint(1, 2, mention="cat"),
        bottom_right=SinglePoint(3, 4),
        mention="cat",
    )
    events = [
        {"type": "text.delta", "chunk": "found one"},
        {"type": "final", "result": {"text": "found one", "boxes": [box]}},
    ]

    captured: dict[str, object] = {}

    def _fake_print_json(*, data):
        captured["payload"] = data

    monkeypatch.setattr("perceptron.cli.console.print_json", _fake_print_json)

    _stream_render(
        iter(events),
        title="Detect",
        output_format=OutputFormat.JSON,
        show_raw=False,
        show_points_table=True,
        expects="box",
    )

    payload = captured["payload"]
    assert payload["text"] == "found one"
    assert payload["boxes"][0]["type"] == "box"
    assert "points" not in payload
    assert "polygons" not in payload


def test_stream_render_buffers_points_delta_into_correct_bucket(monkeypatch):
    """Buffered `points.delta` events should be surfaced under the bucket matching `expects`."""

    poly = Polygon(hull=[SinglePoint(0, 0), SinglePoint(10, 0), SinglePoint(5, 10)])
    events = [
        {"type": "points.delta", "points": [poly]},
        {"type": "text.delta", "chunk": "ok"},
        # No `final` event; render falls back to the buffered points.
    ]

    captured: dict[str, object] = {}

    def _fake_print_json(*, data):
        captured["payload"] = data

    monkeypatch.setattr("perceptron.cli.console.print_json", _fake_print_json)

    _stream_render(
        iter(events),
        title="Question",
        output_format=OutputFormat.JSON,
        show_raw=False,
        show_points_table=False,
        expects="polygon",
    )

    payload = captured["payload"]
    assert payload["polygons"][0]["type"] == "polygon"
    assert "points" not in payload
    assert "boxes" not in payload


def test_stream_render_accumulates_text_deltas(monkeypatch):
    """Multiple `text.delta` events should be concatenated and surfaced as `text`."""

    events = [
        {"type": "text.delta", "chunk": "hello "},
        {"type": "text.delta", "chunk": "world"},
        # No final event — render must fall back to buffered text.
    ]

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "perceptron.cli.console.print_json",
        lambda *, data: captured.update(payload=data),
    )

    _stream_render(
        iter(events),
        title="Caption",
        output_format=OutputFormat.JSON,
        show_raw=False,
        show_points_table=False,
        expects=None,
    )

    assert captured["payload"]["text"] == "hello world"


def test_stream_render_final_text_overrides_buffer(monkeypatch):
    """If the `final` event carries `text`, it should replace the streamed buffer."""

    events = [
        {"type": "text.delta", "chunk": "draft"},
        {"type": "final", "result": {"text": "authoritative"}},
    ]

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "perceptron.cli.console.print_json",
        lambda *, data: captured.update(payload=data),
    )

    _stream_render(
        iter(events),
        title="Caption",
        output_format=OutputFormat.JSON,
        show_raw=False,
        show_points_table=False,
        expects=None,
    )

    assert captured["payload"]["text"] == "authoritative"


# ---------------------------------------------------------------------------
# Media-aware CLI helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "media",
    [
        "clip.mp4",
        "/local/path/clip.mp4",
        "https://example.com/clip.mp4",
        "https://example.com/clip.mp4?token=abc",
        "https://example.com/clip.mp4#fragment",
        "CLIP.MP4",  # case-insensitive
        "clip.webm",
        "/local/path/clip.webm",
        "https://example.com/clip.webm",
        "https://example.com/clip.webm?token=abc",
        "CLIP.WEBM",
    ],
)
def test_looks_like_video_recognizes_video_extensions(media):
    assert _looks_like_video(media) is True


@pytest.mark.parametrize(
    "media",
    [
        "image.jpg",
        "/local/path/image.png",
        "https://example.com/image.webp",
        "no-extension",
    ],
)
def test_looks_like_video_rejects_non_video(media):
    assert _looks_like_video(media) is False


def test_make_media_node_wraps_video_for_mp4_url():
    node = _make_media_node("https://example.com/clip.mp4", "https://example.com/clip.mp4")
    assert isinstance(node, VideoNode)


def test_make_media_node_wraps_image_for_png():
    node = _make_media_node("https://example.com/img.png", "https://example.com/img.png")
    assert isinstance(node, ImageNode)


def test_question_command_passes_video_node_to_sdk(monkeypatch):
    captured: dict[str, object] = {}

    def _capture(media, prompt, **kwargs):
        captured["media"] = media
        captured["prompt"] = prompt
        return _StubResult("ok")

    monkeypatch.setattr("perceptron.cli.question_image", _capture)
    result = runner.invoke(app, ["question", "https://example.com/clip.mp4", "What happens?"])
    assert result.exit_code == 0
    assert isinstance(captured["media"], VideoNode)


def test_question_command_passes_image_node_to_sdk(monkeypatch):
    captured: dict[str, object] = {}

    def _capture(media, prompt, **kwargs):
        captured["media"] = media
        return _StubResult("ok")

    monkeypatch.setattr("perceptron.cli.question_image", _capture)
    result = runner.invoke(app, ["question", "https://example.com/img.png", "What is shown?"])
    assert result.exit_code == 0
    assert isinstance(captured["media"], ImageNode)


def test_describe_point_renders_clip_moment():
    kind, coords, mention = _describe_point(Clip(timestamp=ClipTimestamp(at=1.5), mention="intro"))
    assert kind == "clip"
    assert coords == "@1.50s"
    assert mention == "intro"


def test_describe_point_renders_clip_range():
    kind, coords, mention = _describe_point(Clip(timestamp=ClipTimestamp(at=2.0, until=4.5), mention="hook"))
    assert kind == "clip"
    assert coords == "2.00s → 4.50s"
    assert mention == "hook"


def test_resolve_media_rejects_directory(tmp_path):
    """_resolve_media raises ValueError when given a directory."""
    with pytest.raises(ValueError, match="Expected media file"):
        _resolve_media(str(tmp_path))


def test_question_command_rejects_directory(tmp_path):
    """The question CLI command bails when given a directory (BadParameter -> exit code 2)."""
    result = runner.invoke(app, ["question", str(tmp_path), "What is shown?"])
    assert result.exit_code == 2


def test_question_command_clip_text_renders_clips_table(monkeypatch):
    res = _StubResult("found it")
    res.clips = [Clip(timestamp=ClipTimestamp(at=3.0, until=5.0), mention="shot")]
    monkeypatch.setattr("perceptron.cli.question_image", lambda *a, **k: res)
    result = runner.invoke(
        app,
        [
            "question",
            "https://example.com/clip.mp4",
            "When does the shot happen?",
            "--expects",
            "clip",
        ],
    )
    assert result.exit_code == 0
    assert "Clips" in result.stdout
    assert "3.00s" in result.stdout
    assert "5.00s" in result.stdout
    assert "shot" in result.stdout


# ---------------------------------------------------------------------------
# Real parsed output through the mocked transport (no stubbed results)
# ---------------------------------------------------------------------------

ANSWER = (
    'Intro <clip mention="intro" t="1.5 seconds 3 seconds" /> then '
    '<collection mention="person" asset_idx="0"><point_box> (1,2) (3,4) </point_box></collection> '
    '<track mention="car"><point_box t="0.5 seconds"> (5,6) (7,8) </point_box>'
    '<point_box t="1 seconds"> (6,7) (8,9) </point_box></track>'
)
USAGE = {
    "prompt_tokens": 12,
    "completion_tokens": 5,
    "total_tokens": 17,
    "prompt_tokens_details": {"audio_tokens": 3, "cached_tokens": 0},
}
AUDIO_LIMIT = {
    "error": {
        "message": "Audio input exceeds the token limit.",
        "type": "invalid_request_error",
        "param": None,
        "code": "audio_token_limit_exceeded",
    }
}


@pytest.fixture
def api(monkeypatch):
    """Route requests through `httpx.MockTransport`; `api.answer`/`api.status` shape the next responses."""
    from _http_mock import chunk, completion, install, json_response, sse_response

    for key in ("FAL_KEY", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "perceptron")
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    state = SimpleNamespace(answer=ANSWER, status=200, stream_events=None)

    def handler(request):
        headers = {"x-trace-id": "trace-1"}
        if state.status != 200:
            return json_response(AUDIO_LIMIT, status=state.status, headers=headers)
        if json.loads(request.content).get("stream"):
            events = state.stream_events or [
                chunk({"role": "assistant", "content": state.answer[:40]}),
                chunk({"content": state.answer[40:]}),
                chunk({}, finish_reason="stop"),
                chunk(choices=False, usage=USAGE),
            ]
            return sse_response(events, headers=headers)
        return json_response(completion(state.answer, usage=USAGE), headers=headers)

    state.http = install(monkeypatch, handler)
    return state


def test_question_clip_json_serializes_clips_tracks_and_parsed(api):
    result = runner.invoke(
        app, ["question", "https://example.com/clip.mp4", "When?", "--expects", "clip", "--format", "json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["clips"] == [{"type": "clip", "at": 1.5, "until": 3.0, "mention": "intro"}]
    assert payload["tracks"][0]["type"] == "track"
    assert [p["t"] for p in payload["tracks"][0]["points"]] == [0.5, 1.0]
    assert [seg["kind"] for seg in payload["parsed"]] == ["text", "clip", "text"]
    assert payload["parsed"][1]["value"]["type"] == "clip"
    assert payload["finish_reason"] == "stop"
    assert payload["usage"] == USAGE
    assert payload["request_id"] == "trace-1"
    assert payload["errors"] == []


def test_question_clip_text_lists_clips_and_tracks_with_status(api):
    result = runner.invoke(app, ["question", "https://example.com/clip.mp4", "When?", "--expects", "clip"])
    assert result.exit_code == 0, result.stdout
    assert "Clips" in result.stdout
    assert "1.50s → 3.00s" in result.stdout
    assert "2 box waypoints, 0.50s → 1.00s" in result.stdout
    assert "finish_reason stop" in result.stdout
    assert "tokens in 12 (audio 3, cached 0)" in result.stdout


def test_detect_directory_writes_boxes_tracks_and_parsed_json(api, tmp_path):
    from _image_fixtures import PNG_BYTES

    (tmp_path / "one.png").write_bytes(PNG_BYTES)
    result = runner.invoke(app, ["detect", str(tmp_path), "--classes", "person,car"])
    assert result.exit_code == 0, result.stdout
    data = json.loads((tmp_path / "detections.json").read_text())["one.png"]
    # Collection children and track waypoints are in the flat bucket, with the context their markup gives them.
    assert [(b["mention"], b.get("asset_idx"), b.get("t")) for b in data["boxes"]] == [
        ("person", 0, None),
        ("car", None, 0.5),
        ("car", None, 1.0),
    ]
    assert data["tracks"][0]["mention"] == "car"
    assert [seg["kind"] for seg in data["parsed"] if seg["kind"] != "text"] == ["collection", "track"]
    assert data["parsed"][1]["value"]["type"] == "collection"
    assert data["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens_details"] == {"audio_tokens": 3, "cached_tokens": 0}


def test_detect_text_renders_detections_table_with_tracks_and_assets(api):
    result = runner.invoke(app, ["detect", "https://example.com/frame.png", "--classes", "person,car"])
    assert result.exit_code == 0, result.stdout
    assert "Detections" in result.stdout
    assert "(1,2) → (3,4) @asset 0" in result.stdout
    assert "(5,6) → (7,8) t=0.50s" in result.stdout
    assert "track" in result.stdout


def test_detect_accepts_video_and_forwards_options(api):
    result = runner.invoke(
        app,
        [
            "detect",
            "https://example.com/clip.mp4",
            "--audio-in-video",
            "--reasoning-effort",
            "LOW",
            "--model",
            "perceptron-mk1.5",
            "--provider",
            "perceptron",
        ],
    )
    assert result.exit_code == 0, result.stdout
    request = api.http.last
    assert str(request.url) == "https://api.perceptron.inc/v1/chat/completions"
    body = api.http.last_body
    assert body["model"] == "perceptron-mk1.5"
    assert body["reasoning_effort"] == "low"
    assert body["vision_config"] == {"enable_audio_in_video": True}
    assert body["messages"][-1]["content"][-1] == {
        "type": "video_url",
        "video_url": {"url": "https://example.com/clip.mp4"},
    }


def test_detect_rejects_audio(api):
    result = runner.invoke(app, ["detect", "https://example.com/call.wav"])
    assert result.exit_code == 2
    assert api.http.requests == []


def test_no_audio_in_video_sends_explicit_false(api):
    result = runner.invoke(app, ["caption", "https://example.com/clip.mp4", "--no-audio-in-video", "--expects", "text"])
    assert result.exit_code == 0, result.stdout
    assert api.http.last_body["vision_config"] == {"enable_audio_in_video": False}


def test_audio_in_video_unset_sends_nothing(api):
    result = runner.invoke(app, ["question", "https://example.com/clip.mp4", "What is said?"])
    assert result.exit_code == 0, result.stdout
    assert "vision_config" not in api.http.last_body
    assert "reasoning_effort" not in api.http.last_body


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high"])
def test_reasoning_effort_tiers_reach_the_body(api, effort):
    result = runner.invoke(app, ["ocr", "https://example.com/doc.png", "--reasoning-effort", effort.upper()])
    assert result.exit_code == 0, result.stdout
    assert api.http.last_body["reasoning_effort"] == effort


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("caption", []),
        ("ocr", []),
        ("detect", []),
        ("question", ["What is shown?"]),
    ],
)
def test_every_command_forwards_model_and_provider(monkeypatch, command, extra):
    captured: dict[str, dict] = {}

    def _fake(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _StubResult("ok")

    target = {"caption": "caption_image", "ocr": "ocr_image", "detect": "detect_image", "question": "question_image"}
    monkeypatch.setattr(f"perceptron.cli.{target[command]}", _fake)
    result = runner.invoke(
        app,
        [command, "https://example.com/img.png", *extra, "--model", "perceptron-mk1", "--provider", "perceptron"],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["kwargs"]["model"] == "perceptron-mk1"
    assert captured["kwargs"]["provider"] == "perceptron"


def test_directory_mode_forwards_generation_options(monkeypatch, tmp_path):
    (tmp_path / "one.png").write_bytes(b"image-one")
    captured: list[dict] = []

    def _fake_caption(data, **kwargs):
        captured.append(kwargs)
        return _StubResult("caption")

    monkeypatch.setattr("perceptron.cli.caption_image", _fake_caption)
    result = runner.invoke(app, ["caption", str(tmp_path), "--model", "perceptron-mk1.5", "--reasoning-effort", "low"])
    assert result.exit_code == 0, result.stdout
    assert captured == [{"style": "concise", "expects": "box", "model": "perceptron-mk1.5", "reasoning_effort": "low"}]


def test_sdk_error_exits_with_code_and_request_id(api):
    api.status = 400
    result = runner.invoke(app, ["question", "https://example.com/call.wav", "Transcribe it."])
    assert result.exit_code == 1
    assert "audio_token_limit_exceeded" in result.stdout
    assert "trace-1" in result.stdout


def test_sdk_error_json_output(api):
    api.status = 400
    result = runner.invoke(app, ["question", "https://example.com/call.wav", "Transcribe it.", "--format", "json"])
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "audio_token_limit_exceeded"
    assert error["error_type"] == "invalid_request_error"
    assert error["status"] == 400
    assert error["request_id"] == "trace-1"
    assert error["details"]["code"] == "audio_token_limit_exceeded"


def test_stream_error_event_reports_code_details_and_exits_nonzero(api):
    api.status = 400
    result = runner.invoke(
        app, ["question", "https://example.com/call.wav", "Transcribe it.", "--stream", "--format", "json"]
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    error = payload["errors"][0]
    assert error["code"] == "audio_token_limit_exceeded"
    assert error["request_id"] == "trace-1"
    assert error["details"]["type"] == "invalid_request_error"


def test_stream_truncated_mid_answer_keeps_partial_text(api):
    from _http_mock import chunk

    api.stream_events = [chunk({"role": "assistant", "content": "Partial ans"})]
    api.answer = None

    def _truncated(request):
        from _http_mock import sse_response

        return sse_response(api.stream_events, done=False)

    api.http.handler = _truncated
    result = runner.invoke(app, ["question", "https://example.com/img.png", "Hi", "--stream", "--format", "json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["text"] == "Partial ans"
    assert payload["errors"][0]["code"] == "stream_truncated"


def test_stream_points_delta_context_and_status(api):
    result = runner.invoke(app, ["detect", "https://example.com/frame.png", "--stream"])
    assert result.exit_code == 0, result.stdout
    assert "(1,2) → (3,4) @asset 0" in result.stdout
    assert "Finish stop" in result.stdout
    assert "Tokens in 12 (audio 3, cached 0)" in result.stdout


def test_stream_json_payload_holds_boxes_tracks_and_usage(api):
    result = runner.invoke(app, ["detect", "https://example.com/frame.png", "--stream", "--format", "json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [(b["mention"], b.get("asset_idx"), b.get("t")) for b in payload["boxes"]] == [
        ("person", 0, None),
        ("car", None, 0.5),
        ("car", None, 1.0),
    ]
    assert payload["tracks"][0]["mention"] == "car"
    assert payload["finish_reason"] == "stop"
    assert payload["usage"] == USAGE
    assert payload["request_id"] == "trace-1"
    assert payload["errors"] == []


def test_stream_render_handles_tool_call_deltas(monkeypatch):
    events = [
        {"type": "tool_call.delta", "index": 0, "id": "call_1", "name": "get_weather", "arguments": '{"city":'},
        {"type": "tool_call.delta", "index": 0, "id": None, "name": None, "arguments": ' "Paris"}'},
        {"type": "final", "result": {"text": None, "finish_reason": "tool_calls", "errors": []}},
    ]
    captured: dict[str, object] = {}
    monkeypatch.setattr("perceptron.cli.console.print_json", lambda *, data: captured.update(payload=data))

    _stream_render(
        iter(events),
        title="Question",
        output_format=OutputFormat.JSON,
        show_raw=False,
        show_points_table=False,
        expects=None,
    )

    payload = captured["payload"]
    assert payload["finish_reason"] == "tool_calls"
    assert payload["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    ]


def test_stream_render_summary_marks_track_waypoints(monkeypatch):
    waypoint = BoundingBox(top_left=SinglePoint(1, 2), bottom_right=SinglePoint(3, 4), mention="car", t=0.5)
    events = [
        {
            "type": "points.delta",
            "points": [waypoint],
            "context": {"mention": "car", "t": 0.5, "asset_idx": None, "container": "track"},
        },
    ]
    from rich.console import Console

    recorder = Console(record=True, width=160)
    monkeypatch.setattr("perceptron.cli.console", recorder)
    _stream_render(
        iter(events),
        title="Question",
        output_format=OutputFormat.TEXT,
        show_raw=False,
        show_points_table=False,
        expects="box",
    )
    assert "1. box: (1,2) → (3,4) t=0.50s (car) [in track]" in recorder.export_text()


def test_describe_point_covers_tracks_assets_and_times():
    from perceptron import bbox, pt, track

    assert _describe_point(pt(5, 6, t=1.25, asset_idx=2)) == ("point", "(5,6) t=1.25s @asset 2", "")
    moving = track([pt(1, 1, t=0.0), pt(2, 2, t=2.0)], mention="ball", asset_idx=1)
    assert _describe_point(moving) == ("track", "2 point waypoints, 0.00s → 2.00s @asset 1", "ball")
    kind, coords, _ = _describe_point(Clip(timestamp=ClipTimestamp(at=1.0), asset_idx=0))
    assert (kind, coords) == ("clip", "@1.00s @asset 0")
    assert _describe_point(bbox(1, 2, 3, 4, mention="x"))[1] == "(1,2) → (3,4)"


def test_config_command_states_the_provider_rule():
    result = runner.invoke(app, ["config", "--api-key", "abc", "--model", "perceptron-mk1.5"])
    assert result.exit_code == 0
    assert "PERCEPTRON_API_KEY=abc" in result.stdout
    assert "PERCEPTRON_MODEL=perceptron-mk1.5" in result.stdout
    assert "PERCEPTRON_PROVIDER" not in result.stdout.split("Nothing is saved")[0]  # nothing to export: the default
    assert "Nothing is saved" in result.stdout
    notes = " ".join(result.stdout.split())
    assert "use the Perceptron API" in notes
    assert "'fal' is selected only when FAL_KEY is set and PERCEPTRON_API_KEY is not" in notes


def test_config_command_placeholders_select_the_perceptron_api():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "export PERCEPTRON_API_KEY=<your-key>" in result.stdout
    assert "PERCEPTRON_PROVIDER=" not in result.stdout  # the Perceptron API is the default


def test_answer_text_is_printed_literally_not_as_rich_markup(api):
    api.answer = "Step [/INST] then [bold]x[/bold]"
    result = runner.invoke(app, ["question", "https://example.com/img.png", "Hi"])
    assert result.exit_code == 0, result.stdout
    assert "[/INST]" in result.stdout
    assert "[bold]x[/bold]" in result.stdout


@pytest.mark.parametrize("stream", [False, True], ids=["result", "stream"])
@pytest.mark.parametrize("mention", ["[/INST] rack", "shelf [red]", "[bold]x"])
def test_annotation_mentions_are_printed_literally_not_as_rich_markup(api, mention, stream):
    api.answer = f'Found <point_box mention="{mention}"> (1,2) (3,4) </point_box>'
    result = runner.invoke(app, ["detect", "https://example.com/frame.png", *(["--stream"] if stream else [])])
    assert result.exit_code == 0, result.stdout
    rows = [line for line in result.stdout.splitlines() if "(1,2) → (3,4)" in line]  # the detections table
    assert rows and all(mention in row for row in rows)


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_stream_error_text_is_literal_and_names_the_request_id(api, output_format):
    from _http_mock import chunk, sse_response

    error = {"error": {"message": "boom [/x]", "type": "server_error", "code": "internal"}}
    api.http.handler = lambda request: sse_response(
        [chunk({"role": "assistant", "content": "part"}), error], headers={"x-trace-id": "trace-sse"}
    )
    result = runner.invoke(
        app, ["question", "https://example.com/img.png", "Hi", "--stream", "--format", output_format]
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception  # a clean exit, not a crash
    if output_format == "json":
        assert json.loads(result.stdout)["errors"][0]["message"] == "boom [/x]"
    else:
        assert "boom [/x]" in result.stdout
        assert "request id: trace-sse" in result.stdout


def test_stream_http_error_text_mode_prints_code_and_request_id(api):
    api.status = 400
    result = runner.invoke(app, ["question", "https://example.com/call.wav", "Transcribe it.", "--stream"])
    assert result.exit_code == 1
    assert "audio_token_limit_exceeded" in result.stdout
    assert "request id: trace-1" in result.stdout


def test_directory_mode_prints_paths_and_issues_literally(api, tmp_path, monkeypatch):
    from _image_fixtures import PNG_BYTES

    batch = tmp_path / "batch [bold]"  # would print as "batch" if read as markup
    batch.mkdir()
    (batch / "one.png").write_bytes(PNG_BYTES)
    api.answer = '<point_box mention="a"> (1,2) [/y] </point_box>'
    # A relative path keeps the printed path short, so the panel never wraps it whatever the temp dir is.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["detect", "batch [bold]"])
    assert result.exit_code == 0, result.stdout
    assert "batch [bold]" in result.stdout
    assert "(1,2) [/y]" in result.stdout
    assert (batch / "detections.json").exists()


@pytest.mark.parametrize(
    ("data_url", "part_type"),
    [
        ("data:video/mp4;base64,AAAA", "video_url"),
        ("data:audio/wav;base64,AAAA", "audio_url"),
        ("data:image/png;base64,AAAA", "image_url"),
        # Longer than a file name may be: never looked up on disk (no OSError "File name too long").
        ("data:image/png;base64," + "A" * 4096, "image_url"),
    ],
    ids=["video", "audio", "image", "long-image"],
)
def test_data_url_media_is_routed_by_mime_type(api, data_url, part_type):
    result = runner.invoke(app, ["question", data_url, "What is this?"])
    assert result.exit_code == 0, result.stdout
    media_parts = [part for part in api.http.last_body["messages"][-1]["content"] if part["type"] != "text"]
    assert media_parts == [{"type": part_type, part_type: {"url": data_url}}]


@pytest.mark.parametrize(
    ("command", "extra"),
    [("question", ["What?"]), ("caption", []), ("detect", []), ("ocr", [])],
)
def test_invalid_data_url_is_reported_with_its_code(api, command, extra):
    args = [command, "data:text/plain;base64,AAAA", *extra]
    result = runner.invoke(app, [*args, "--format", "json"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["error"]["code"] == "invalid_data_url"
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "[invalid_data_url]" in result.stdout
    assert api.http.requests == []


def test_ocr_rejects_a_video_data_url_with_its_code(api):
    result = runner.invoke(app, ["ocr", "data:video/mp4;base64,AAAA"])
    assert result.exit_code == 1
    assert "[invalid_data_url]" in result.stdout
    assert api.http.requests == []


PERCEPTRON_API = ("https://api.perceptron.inc/v1/chat/completions", "Bearer sk-test", "perceptron-mk1.5")
FAL = ("https://fal.run/perceptron/isaac-01/openai/v1/chat/completions", "Key fal-key", "isaac-0.1")
BOTH_KEYS = {"PERCEPTRON_API_KEY": "sk-test", "FAL_KEY": "fal-key"}


def _set_env(monkeypatch, env):
    for key in ("PERCEPTRON_PROVIDER", "PERCEPTRON_API_KEY"):
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize(
    ("env", "flags", "expected"),
    [
        ({"PERCEPTRON_API_KEY": "sk-test"}, [], PERCEPTRON_API),
        (BOTH_KEYS, [], PERCEPTRON_API),
        ({"FAL_KEY": "fal-key"}, [], FAL),
        (BOTH_KEYS, ["--provider", "fal"], FAL),
        ({**BOTH_KEYS, "PERCEPTRON_PROVIDER": "fal"}, ["--provider", "perceptron"], PERCEPTRON_API),
    ],
    ids=["api-key-only", "both-keys", "fal-key-only", "provider-flag-fal", "provider-flag-beats-env"],
)
def test_provider_follows_the_rule_and_the_flag_wins(api, monkeypatch, env, flags, expected):
    _set_env(monkeypatch, env)
    result = runner.invoke(app, ["question", "https://example.com/img.png", "Hi", *flags])
    assert result.exit_code == 0, result.stdout
    request = api.http.last
    assert (str(request.url), request.headers["authorization"], api.http.last_body["model"]) == expected


@pytest.mark.parametrize(
    ("env", "provider"),
    [({"PERCEPTRON_API_KEY": "sk-test"}, "fal"), ({"FAL_KEY": "fal-key"}, "perceptron")],
    ids=["fal-without-fal-key", "perceptron-without-perceptron-key"],
)
def test_a_key_is_never_sent_to_the_other_provider(api, monkeypatch, env, provider):
    _set_env(monkeypatch, env)
    result = runner.invoke(app, ["question", "https://example.com/img.png", "Hi", "--provider", provider])
    assert result.exit_code == 1
    assert "credentials_missing" in result.stdout
    assert api.http.requests == []


def test_directory_mode_reads_only_the_image_formats_the_api_accepts(monkeypatch, tmp_path):
    (tmp_path / "one.png").write_bytes(b"image-one")
    (tmp_path / "anim.gif").write_bytes(b"GIF89a")
    (tmp_path / "scan.tiff").write_bytes(b"II*\x00")
    seen: list[bytes] = []

    def _fake_caption(data, **kwargs):
        seen.append(data.obj)
        return _StubResult("caption")

    monkeypatch.setattr("perceptron.cli.caption_image", _fake_caption)
    result = runner.invoke(app, ["caption", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert seen == [b"image-one"]


def test_directory_without_supported_images_names_the_formats(tmp_path):
    (tmp_path / "anim.gif").write_bytes(b"GIF89a")
    result = runner.invoke(app, ["detect", str(tmp_path)])
    assert result.exit_code == 1
    assert "No image files (.jpeg, .jpg, .png, .webp) found" in result.stdout
