import os

import pytest
from PIL import Image

from cookbook.utils import cookbook_asset
from perceptron import annotate_image, bbox, detect, scale_points_to_pixels
from perceptron import config as cfg
from perceptron.pointing.types import BoundingBox

pytestmark = pytest.mark.integration

_API_KEY = os.getenv("PERCEPTRON_API_KEY")
_TARGET_IMAGE = cookbook_asset("in-context-learning", "multi", "cat_dog_input.png")

requires_perceptron_key = pytest.mark.skipif(
    not _API_KEY,
    reason="PERCEPTRON_API_KEY not set; export it to exercise the live Perceptron backend.",
)


def _build_examples():
    cat_example = annotate_image(
        str(cookbook_asset("in-context-learning", "multi", "classA.jpg")),
        [bbox(316, 136, 703, 906, mention="classA")],
    )
    dog_example = annotate_image(
        str(cookbook_asset("in-context-learning", "multi", "classB.webp")),
        [bbox(161, 48, 666, 980, mention="classB")],
    )
    return [cat_example, dog_example]


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return (box.top_left.x, box.top_left.y, box.bottom_right.x, box.bottom_right.y)


def _intersection_over_union(pred: BoundingBox, expected: BoundingBox) -> float:
    px1, py1, px2, py2 = _box_tuple(pred)
    ex1, ey1, ex2, ey2 = _box_tuple(expected)
    ix1 = max(px1, ex1)
    iy1 = max(py1, ey1)
    ix2 = min(px2, ex2)
    iy2 = min(py2, ey2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    pred_area = (px2 - px1) * (py2 - py1)
    exp_area = (ex2 - ex1) * (ey2 - ey1)
    union = pred_area + exp_area - inter
    if union <= 0:
        return 0.0
    return inter / union


@requires_perceptron_key
def test_detect_examples_pipeline_hits_backend():
    examples = _build_examples()
    with cfg(provider="perceptron", api_key=_API_KEY):
        result = detect(
            str(_TARGET_IMAGE),
            classes=["classA", "classB"],
            examples=examples,
            temperature=0.0,
            max_tokens=256,
        )

    assert isinstance(result.raw, dict)
    assert result.raw.get("choices"), "Expected backend to return choices"
    assert result.points, "Backend did not return any bounding boxes"

    with Image.open(_TARGET_IMAGE) as im:
        width, height = im.size

    expected_box = bbox(38, 91, 934, 962, mention="classB")
    expected_pixels = scale_points_to_pixels([expected_box], width=width, height=height)[0]
    pixel_predictions = result.points_to_pixels(width, height)
    assert pixel_predictions, "Point scaling produced no boxes"

    best_match = max(pixel_predictions, key=lambda pt: _intersection_over_union(pt, expected_pixels))
    iou = _intersection_over_union(best_match, expected_pixels)
    assert iou >= 0.4, f"Predicted box too far from ground truth (IoU={iou:.3f})"
