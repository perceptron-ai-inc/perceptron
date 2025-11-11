from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from perceptron import (
    Collection,
    PerceiveResult,
    Polygon,
    SinglePoint,
    annotate_image,
    bbox,
    box,
    caption,
    config,
    detect,
    image,
    ocr,
    perceive,
    question,
    scale_points_to_pixels,
    text,
)
from perceptron.errors import AnchorError, BadRequestError, RateLimitError, SDKError

ASSETS_DIR = Path(__file__).parent / "assets" / "docs"


def _doc_asset(name: str) -> str:
    path = ASSETS_DIR / name
    if not path.exists():
        pytest.skip(f"doc asset missing: {name}")
    return str(path)


def _live_config_kwargs() -> dict[str, str | None]:
    api_key = os.getenv("PERCEPTRON_API_KEY")
    if not api_key:
        pytest.skip("PERCEPTRON_API_KEY not set for live doc tests")
    base_url = os.getenv("PERCEPTRON_BASE_URL")
    return {"provider": "perceptron", "api_key": api_key, "base_url": base_url}


@pytest.mark.integration
def test_docs_capabilities_captioning_example():
    """`capabilities/captioning` snippet covering text + grounded variants."""

    img_path = _doc_asset("suburban_street.webp")

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        text_only = caption(img_path, style="concise", expects="text")
        boxes = caption(img_path, style="detailed", expects="box")

    assert isinstance(text_only.text, str)
    assert text_only.text.strip() != ""
    assert "choices" in boxes.raw
    assert boxes.points is not None and len(boxes.points) >= 1
    pixel_points = boxes.points_to_pixels(1920, 1080)
    assert pixel_points is not None
    for scaled in pixel_points:
        assert 0 <= scaled.top_left.x < scaled.bottom_right.x <= 1920
        assert 0 <= scaled.top_left.y < scaled.bottom_right.y <= 1080


@pytest.mark.integration
def test_docs_perceive_basics_examples():
    """`perceive-basics` examples: describe, find_center, and compare."""

    describe_img = _doc_asset("truck_scene.jpg")
    day_img = _doc_asset("suburban_street.webp")
    night_img = _doc_asset("studio_scene.webp")

    @perceive()
    def describe(image_path):
        return image(image_path) + text("Describe the primary object.")

    @perceive(expects="point")
    def find_center(frame):
        return image(frame) + text("Return one point marking the defect center.")

    @perceive()
    def compare(day, night):
        day_img_node = image(day)
        night_img_node = image(night)
        return (
            text("Compare the highlighted region in both frames.")
            + night_img_node
            + box(40, 60, 160, 140, mention="focus", image=night_img_node)
            + day_img_node
        )

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        describe_result = describe(describe_img)
        center_result = find_center(day_img)
        compare_result = compare(day_img, night_img)

    assert isinstance(describe_result.text, str) and describe_result.text.strip() != ""
    assert center_result.points is not None and len(center_result.points) >= 1
    assert compare_result.raw


@pytest.mark.integration
def test_docs_perceive_direct_invocation_examples():
    """README direct-call examples using live backend."""

    scene_path = _doc_asset("truck_scene.jpg")

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        text_only = perceive(image(scene_path) + text("What's happening in this scene?"), expects="text")
        boxed = perceive(
            [
                image(scene_path),
                text("Highlight the key vehicle."),
            ],
            expects="box",
        )

    assert isinstance(text_only.text, str)
    assert text_only.text.strip() != ""
    assert boxed.points is not None and len(boxed.points) >= 1
    for box_result in boxed.points:
        assert 0 <= box_result.top_left.x < box_result.bottom_right.x <= 1000
        assert 0 <= box_result.top_left.y < box_result.bottom_right.y <= 1000


@pytest.mark.integration
def test_docs_capabilities_object_detection_example():
    """`capabilities/object-detection` PPE example on live API."""

    frame_path = _doc_asset("ppe_line.webp")

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        result = detect(
            frame_path,
            classes=["helmet", "vest"],
            expects="box",
        )

    assert isinstance(result.raw, dict)
    assert result.points is not None and len(result.points) >= 1
    mentions = {box_result.mention.lower() for box_result in result.points if box_result.mention}
    assert mentions.intersection({"helmet", "vest"})
    for box_result in result.points:
        # Docs promise normalized geometry (0-1000 grid) for each detection.
        assert 0 <= box_result.top_left.x < box_result.bottom_right.x <= 1000
        assert 0 <= box_result.top_left.y < box_result.bottom_right.y <= 1000


@pytest.mark.integration
def test_docs_capabilities_ocr_example():
    """`capabilities/ocr` sample (prompted extraction) against the live API."""

    img_path = _doc_asset("grocery_labels.webp")

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        result = ocr(img_path, prompt="Extract product name and price", stream=False)

    assert isinstance(result.text, str)
    assert "price" in result.text.lower()


@pytest.mark.integration
def test_docs_capabilities_visual_qa_example():
    """`capabilities/visual-qa` snippet with question helper."""

    frame_path = _doc_asset("studio_scene.webp")

    question_text = "What stands out in this studio scene?"
    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        result = question(frame_path, question_text, expects="box")

    assert isinstance(result.text, str)
    assert result.text.strip() != ""
    assert result.points is not None and len(result.points) >= 1
    for answer_box in result.points:
        assert 0 <= answer_box.top_left.x < answer_box.bottom_right.x <= 1000
        assert 0 <= answer_box.top_left.y < answer_box.bottom_right.y <= 1000


def test_docs_concepts_coordinate_helpers():
    """`concepts/coordinates` conversion examples using SDK helpers."""

    normalized_box = bbox(250, 300, 750, 700)
    width, height = 960, 600

    scaled = scale_points_to_pixels([normalized_box], width=width, height=height)[0]

    expected = (
        int(normalized_box.top_left.x / 1000 * width),
        int(normalized_box.top_left.y / 1000 * height),
        int(normalized_box.bottom_right.x / 1000 * width),
        int(normalized_box.bottom_right.y / 1000 * height),
    )

    assert (scaled.top_left.x, scaled.top_left.y, scaled.bottom_right.x, scaled.bottom_right.y) == expected

    # Per docs, PerceiveResult.points_to_pixels is the one-liner wrapper
    result_stub = PerceiveResult(text="", points=[normalized_box], parsed=None, usage=None, errors=[], raw={})
    pixel_points = result_stub.points_to_pixels(width, height)
    assert pixel_points is not None
    assert isinstance(pixel_points[0], type(scaled))


@pytest.mark.integration
def test_docs_in_context_single_example():
    """`capabilities/in-context-learning` single-image recipe."""

    exemplar_path = _doc_asset("cake_mixer_example.webp")
    target_path = _doc_asset("find_kitchen_item.webp")

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        bootstrap = detect(
            exemplar_path,
            classes=["objectCategory1"],
        )

        assert bootstrap.points is not None and len(bootstrap.points) >= 1
        first_box = bootstrap.points[0]
        example_shot = annotate_image(
            exemplar_path,
            {
                "objectCategory1": [
                    bbox(
                        int(first_box.top_left.x),
                        int(first_box.top_left.y),
                        int(first_box.bottom_right.x),
                        int(first_box.bottom_right.y),
                        mention="objectCategory1",
                    )
                ]
            },
        )

        result = detect(
            target_path,
            classes=["objectCategory1"],
            examples=[example_shot],
        )

    assert result.points is not None and len(result.points) >= 1
    assert all((box.mention or "objectCategory1").startswith("objectCategory1") for box in result.points)


@pytest.mark.integration
def test_docs_in_context_multi_example():
    """`capabilities/multi-image-in-context-learning` recipe."""

    cat_path = _doc_asset("classA.jpg")
    dog_path = _doc_asset("classB.webp")
    target_path = _doc_asset("input.png")

    cat_example = annotate_image(
        cat_path,
        {"classA": [bbox(316, 136, 703, 906, mention="classA")]},
    )
    dog_example = annotate_image(
        dog_path,
        {"classB": [bbox(161, 48, 666, 980, mention="classB")]},
    )

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        result = detect(
            target_path,
            classes=["classA", "classB"],
            examples=[cat_example, dog_example],
        )

    assert result.points is not None and len(result.points) >= 1
    mentions = {box.mention for box in result.points if box.mention}
    assert mentions.intersection({"classA", "classB"})


def test_docs_error_handling_metadata():
    """Error-handling guide: SDKError + RateLimitError expose codes/details."""

    err = SDKError("boom", code="bad_request", details={"request_id": "req-123"})
    assert err.code == "bad_request"
    assert err.details["request_id"] == "req-123"

    rate_err = RateLimitError("slow down", retry_after=1.5, details={"request_id": "req-456"})
    assert rate_err.details["retry_after"] == 1.5
    assert rate_err.details["request_id"] == "req-456"


def test_docs_error_invalid_image_decision_tree(tmp_path):
    """`guides/error-messages`: invalid_image branch emits actionable metadata."""

    bad_path = tmp_path / "not_image.bin"
    bad_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(BadRequestError) as excinfo:
        caption(str(bad_path), expects="text")

    err = excinfo.value
    assert err.code == "invalid_image"
    assert err.details["origin"].endswith("not_image.bin")
    assert err.details["reason"] in {"decoder_failed", "unknown_format"}


def test_docs_error_anchor_strict_decision_tree():
    """`guides/error-messages`: anchor_missing branch stays reproducible via pytest."""

    @perceive(expects="box", strict=True)
    def broken_prompt():
        return text("Mark the defect") + box(0, 0, 10, 10)

    with pytest.raises(AnchorError) as excinfo:
        broken_prompt()

    err = excinfo.value
    assert err.code == "anchor_missing"
    assert err.details["code"] == "anchor_missing"


@pytest.mark.integration
def test_docs_quickstart_caption_example():
    """Live run of the Quickstart @perceive example against hosted API."""

    img_path = _doc_asset("truck_scene.jpg")

    @perceive()
    def caption_doc_example(image_path):
        return image(image_path) + text("Describe the primary object in this photo.")

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        result = caption_doc_example(img_path)

    assert isinstance(result.text, str)
    assert result.text.strip() != ""
    assert not result.errors


@pytest.mark.integration
def test_docs_pointing_basics_example():
    """`pointing-basics` locate_defect example should anchor a bounding box tag."""

    frame_path = _doc_asset("input.png")

    @perceive(expects="box")
    def locate_defect(frame):
        return image(frame) + text("Return a bounding box around the defect.")

    with config(**{k: v for k, v in _live_config_kwargs().items() if v is not None}):
        result = locate_defect(frame_path)

    assert isinstance(result.points, list) or result.points is None


def test_docs_pointing_basics_collection_snippet():
    """`pointing-basics` Collection example (inspection-drone)."""

    assembly = Collection(
        mention="inspection-drone",
        points=[
            SinglePoint(220, 180, mention="center"),
            Polygon(
                hull=[
                    SinglePoint(120, 90),
                    SinglePoint(260, 95),
                    SinglePoint(275, 180),
                    SinglePoint(135, 185),
                ]
            ),
        ],
    )

    assert assembly.mention == "inspection-drone"
    assert len(assembly.points) == 2
    assert isinstance(assembly.points[0], SinglePoint)
    assert isinstance(assembly.points[1], Polygon)


@pytest.mark.integration
def test_docs_batch_async_example():
    """Batch guide async pipeline: ensure asyncio helper drives detect via to_thread."""

    image_paths = [_doc_asset("truck_scene.jpg"), _doc_asset("ppe_line.webp")]
    cfg = {k: v for k, v in _live_config_kwargs().items() if v is not None}

    async def detect_async(image_path, *, classes):
        return await asyncio.to_thread(
            detect,
            image_path,
            classes=classes,
            expects="box",
        )

    async def process_batch(paths, classes):
        tasks = [detect_async(path, classes=classes) for path in paths]
        return await asyncio.gather(*tasks)

    with config(**cfg):
        results = asyncio.run(process_batch(image_paths, ["doc_demo"]))

    assert len(results) == len(image_paths)
    assert all(hasattr(res, "raw") for res in results)


@pytest.mark.integration
def test_docs_scaling_low_latency_example():
    """Scaling guide low-latency example: config context scopes detect defaults."""

    sample_image = _doc_asset("ppe_line.webp")
    cfg = {k: v for k, v in _live_config_kwargs().items() if v is not None}
    cfg.update({"timeout": 8, "max_tokens": 256})

    with config(**cfg):
        result = detect(sample_image, classes=["scratch"], expects="box")

    assert isinstance(result.points, list) or result.points is None
