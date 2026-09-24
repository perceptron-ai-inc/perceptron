import pytest
from _image_fixtures import PNG_BYTES

from perceptron.errors import BadRequestError
from perceptron.highlevel import _normalize_examples
from perceptron.pointing.types import bbox, clip, track


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


def test_normalize_examples_prompt_keeps_asset_idx_tracks_and_times():
    prompt = (
        'Reference: <collection mention="cup" asset_idx="0" t="1.5"> <point_box> (1,2) (3,4) </point_box> </collection> '
        'and <track mention="ball" asset_idx="1"> <point_box t="0.5 seconds"> (5,6) (7,8) </point_box> </track>'
    )
    samples = [{"image": PNG_BYTES, "boxes": [bbox(0, 0, 2, 2)], "prompt": prompt}]
    normalized = _normalize_examples(samples, class_order=None)
    assert normalized[0].prompt == (
        'Reference: <collection mention="cup" asset_idx="0"> <point_box t="1.5 seconds"> (1,2) (3,4) </point_box> '
        '</collection> and <track mention="ball" asset_idx="1"> <point_box t="0.5 seconds"> (5,6) (7,8) </point_box> '
        "</track>"
    )


def test_normalize_examples_forwards_clips_and_tracks():
    samples = [
        {
            "image": PNG_BYTES,
            "clips": [clip(1.0, 2.0, mention="goal")],
            "tracks": [track([bbox(0, 0, 2, 2, t=0.0)], mention="ball", asset_idx=0)],
        }
    ]
    tags = _normalize_examples(samples, class_order=None)[0].tags
    assert tags == (
        '<clip mention="goal" t="1 seconds 2 seconds" /> '
        '<track mention="ball" asset_idx="0"> <point_box t="0.0 seconds"> (0,0) (2,2) </point_box> </track>'
    )
    listed = _normalize_examples([{"image": PNG_BYTES, "annotations": [clip(3.0)]}], class_order=None)
    assert listed[0].tags == '<clip t="3 seconds" />'
