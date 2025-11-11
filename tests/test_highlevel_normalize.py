import pytest

from perceptron.errors import BadRequestError
from perceptron.highlevel import _normalize_examples
from perceptron.pointing.types import bbox

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 12


def test_normalize_examples_supports_annotations_list():
    samples = [
        {
            "image": PNG_BYTES,
            "annotations": [
                bbox(0, 0, 4, 4, mention="cat"),
                bbox(1, 1, 5, 5, mention="dog"),
            ],
            "prompt": '<collection mention="A"> <point(0,0)> </collection>',
        }
    ]

    normalized = _normalize_examples(samples, class_order=["dog", "cat"])
    assert normalized[0].image == PNG_BYTES
    # annotations list should be processed even when provided under the single key
    assert normalized[0].tags.count("<point_box") == 2


def test_normalize_examples_requires_annotations():
    samples = [
        {
            "image": PNG_BYTES,
            "prompt": "Describe",
        }
    ]

    with pytest.raises(BadRequestError):
        _normalize_examples(samples, class_order=None)


def test_normalize_examples_accepts_type_lists():
    samples = [
        {
            "image": PNG_BYTES,
            "boxes": [bbox(0, 0, 2, 2, mention="apple")],
            "points": [(3, 3)],
        }
    ]

    normalized = _normalize_examples(samples, class_order=None)
    assert "apple" in normalized[0].tags
    assert "<point>" in normalized[0].tags
