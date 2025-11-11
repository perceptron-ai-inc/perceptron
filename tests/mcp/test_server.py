from __future__ import annotations

import pytest

from perceptron import BadRequestError
from perceptron.dsl.perceive import PerceiveResult
from perceptron.pointing.types import bbox, pt

pytest.importorskip("mcp")
from perceptron_mcp import server


def _result(**overrides):
    defaults = {
        "text": "ok",
        "points": None,
        "parsed": None,
        "usage": None,
        "errors": [],
        "raw": {},
    }
    defaults.update(overrides)
    return PerceiveResult(**defaults)


def test_caption_image_routes_arguments_and_formats(monkeypatch):
    captured = {}

    def fake_caption(image, *, style, expects, model, provider):
        captured["args"] = (image, style, expects, model, provider)
        return _result(text="caption")

    monkeypatch.setattr(server, "caption", fake_caption)

    payload = server.caption_image("img.png", style="detailed", expects="text", model="isaac", provider="fal")

    assert payload["text"] == "caption"
    assert payload["points"] is None
    assert captured["args"] == ("img.png", "detailed", "text", "isaac", "fal")


def test_detect_objects_serializes_boxes(monkeypatch):
    def fake_detect(image, *, classes, max_outputs, model, provider):
        assert image == "frame.png"
        assert classes == ["forklift"]
        assert max_outputs == 3
        assert model is None and provider is None
        return _result(points=[bbox(1, 2, 3, 4, mention="forklift")], parsed=[{"ok": True}], errors=[{"code": "warn"}])

    monkeypatch.setattr(server, "detect", fake_detect)

    payload = server.detect_objects("frame.png", classes=["forklift"], max_outputs=3)

    assert payload["text"] == "ok"
    assert payload["parsed"] == [{"ok": True}]
    assert payload["errors"] == [{"code": "warn"}]
    assert payload["points"] == [
        {
            "type": "box",
            "top_left": {"type": "point", "x": 1, "y": 2, "mention": None, "t": None},
            "bottom_right": {"type": "point", "x": 3, "y": 4, "mention": None, "t": None},
            "mention": "forklift",
            "t": None,
        }
    ]


def test_ocr_extract_passes_prompt(monkeypatch):
    captured = {}

    def fake_ocr(image, *, prompt, model, provider):
        captured["args"] = (image, prompt, model, provider)
        return _result(text="doc")

    monkeypatch.setattr(server, "ocr", fake_ocr)

    payload = server.ocr_extract("scan.png", prompt="labels", model="m1")

    assert payload["text"] == "doc"
    assert captured["args"] == ("scan.png", "labels", "m1", None)


def test_ask_image_supports_grounding(monkeypatch):
    captured = {}

    def fake_question(image, question_text, *, expects, model, provider):
        captured["args"] = (image, question_text, expects, model, provider)
        return _result(text="answer", points=[pt(5, 6, mention="spot")])

    monkeypatch.setattr(server, "question", fake_question)

    payload = server.ask_image("scene.png", "Where is the logo?", expects="point")

    assert payload["text"] == "answer"
    assert payload["points"] == [{"type": "point", "x": 5, "y": 6, "mention": "spot", "t": None}]
    assert captured["args"] == ("scene.png", "Where is the logo?", "point", None, None)


def test_set_defaults_requires_updates():
    with pytest.raises(BadRequestError):
        server.set_defaults()


def test_set_defaults_invokes_configure(monkeypatch):
    captured = {}

    def fake_configure(**kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(server, "configure", fake_configure)

    payload = server.set_defaults(provider="perceptron", max_tokens=2048)

    assert captured["kwargs"] == {"provider": "perceptron", "max_tokens": 2048}
    assert payload == {"updated": ["max_tokens", "provider"]}


@pytest.mark.anyio("asyncio")
async def test_fastmcp_call_tool_invokes_caption(monkeypatch):
    def fake_caption(image, *, style, expects, model, provider):
        assert image == "photo.jpg"
        assert style == "concise"
        assert expects == "text"
        return _result(text="integration")

    monkeypatch.setattr(server, "caption", fake_caption)

    content_blocks, structured = await server.app.call_tool("caption_image", {"image": "photo.jpg"})

    assert structured["text"] == "integration"
    assert content_blocks and "integration" in content_blocks[0].text


@pytest.fixture
def anyio_backend():
    return "asyncio"
