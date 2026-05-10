import json

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
    result = runner.invoke(app, ["config", "--provider", "fal", "--api-key", "abc"])
    assert result.exit_code == 0
    assert "PERCEPTRON_PROVIDER=fal" in result.stdout
    assert "PERCEPTRON_API_KEY=abc" in result.stdout


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
