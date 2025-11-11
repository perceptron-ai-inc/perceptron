from __future__ import annotations

import time

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

    # FastMCP wraps the return value in a 'result' key
    assert structured["result"]["text"] == "integration"
    assert content_blocks and "integration" in content_blocks[0].text


def test_caption_image_handles_multiple_images(monkeypatch):
    call_count = 0

    def fake_caption(image, *, style, expects, model, provider):
        nonlocal call_count
        call_count += 1
        return _result(text=f"caption for {image}")

    monkeypatch.setattr(server, "caption", fake_caption)

    payload = server.caption_image(["img1.png", "img2.png"], style="concise", expects="text")

    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["text"] == "caption for img1.png"
    assert payload[1]["text"] == "caption for img2.png"
    assert call_count == 2


def test_detect_objects_handles_multiple_images(monkeypatch):
    def fake_detect(image, *, classes, max_outputs, model, provider):
        return _result(text=f"detected in {image}", points=[bbox(1, 2, 3, 4, mention=image)])

    monkeypatch.setattr(server, "detect", fake_detect)

    payload = server.detect_objects(["img1.png", "img2.png"], classes=["person"])

    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["text"] == "detected in img1.png"
    assert payload[1]["text"] == "detected in img2.png"
    assert payload[0]["points"][0]["mention"] == "img1.png"
    assert payload[1]["points"][0]["mention"] == "img2.png"


def test_ocr_extract_handles_multiple_images(monkeypatch):
    def fake_ocr(image, *, prompt, model, provider):
        return _result(text=f"text from {image}")

    monkeypatch.setattr(server, "ocr", fake_ocr)

    payload = server.ocr_extract(["doc1.png", "doc2.png"])

    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["text"] == "text from doc1.png"
    assert payload[1]["text"] == "text from doc2.png"


def test_ask_image_handles_multiple_images(monkeypatch):
    def fake_question(image, question_text, *, expects, model, provider):
        return _result(text=f"answer for {image}", points=[pt(5, 6, mention=image)])

    monkeypatch.setattr(server, "question", fake_question)

    payload = server.ask_image(["img1.png", "img2.png"], "What do you see?")

    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["text"] == "answer for img1.png"
    assert payload[1]["text"] == "answer for img2.png"


def test_multi_image_handles_exceptions(monkeypatch):
    def fake_caption(image, *, style, expects, model, provider):
        if image == "bad.png":
            raise BadRequestError("Invalid image")
        return _result(text=f"ok: {image}")

    monkeypatch.setattr(server, "caption", fake_caption)

    payload = server.caption_image(["good.png", "bad.png", "also_good.png"])

    assert isinstance(payload, list)
    assert len(payload) == 3
    assert payload[0]["text"] == "ok: good.png"
    assert payload[1]["text"] is None
    assert payload[1]["errors"][0]["type"] == "BadRequestError"
    assert "Invalid image" in payload[1]["errors"][0]["message"]
    assert payload[2]["text"] == "ok: also_good.png"


def test_single_image_returns_dict_not_list(monkeypatch):
    """Verify single image returns dict, not list with one element."""

    def fake_caption(image, *, style, expects, model, provider):
        return _result(text=f"caption for {image}")

    monkeypatch.setattr(server, "caption", fake_caption)

    # Single string should return dict
    payload = server.caption_image("single.png")
    assert isinstance(payload, dict)
    assert payload["text"] == "caption for single.png"


def test_multi_image_maintains_order(monkeypatch):
    """Verify results maintain input order even with parallel execution."""

    def fake_detect(image, *, classes, max_outputs, model, provider):
        # Simulate varying execution times
        if "slow" in image:
            time.sleep(0.1)
        return _result(text=f"result for {image}")

    monkeypatch.setattr(server, "detect", fake_detect)

    # Mix of fast and slow images
    images = ["fast1.png", "slow1.png", "fast2.png", "slow2.png", "fast3.png"]
    payload = server.detect_objects(images)

    assert isinstance(payload, list)
    assert len(payload) == 5
    # Verify order is maintained
    assert payload[0]["text"] == "result for fast1.png"
    assert payload[1]["text"] == "result for slow1.png"
    assert payload[2]["text"] == "result for fast2.png"
    assert payload[3]["text"] == "result for slow2.png"
    assert payload[4]["text"] == "result for fast3.png"


def test_multi_image_with_all_tool_parameters(monkeypatch):
    """Verify all parameters are correctly forwarded in multi-image mode."""
    captured_calls = []

    def fake_detect(image, *, classes, max_outputs, model, provider):
        captured_calls.append(
            {
                "image": image,
                "classes": classes,
                "max_outputs": max_outputs,
                "model": model,
                "provider": provider,
            }
        )
        return _result(text=f"detected {image}")

    monkeypatch.setattr(server, "detect", fake_detect)

    result = server.detect_objects(
        ["img1.png", "img2.png"],
        classes=["car", "person"],
        max_outputs=10,
        model="test-model",
        provider="test-provider",
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert len(captured_calls) == 2
    # Verify all parameters were forwarded to each call
    for call in captured_calls:
        assert call["classes"] == ["car", "person"]
        assert call["max_outputs"] == 10
        assert call["model"] == "test-model"
        assert call["provider"] == "test-provider"


def test_multi_image_with_empty_list(monkeypatch):
    """Verify empty list returns empty list."""

    def fake_caption(image, *, style, expects, model, provider):
        return _result(text=f"caption for {image}")

    monkeypatch.setattr(server, "caption", fake_caption)

    payload = server.caption_image([])
    assert isinstance(payload, list)
    assert len(payload) == 0


@pytest.fixture
def anyio_backend():
    return "asyncio"
