import pytest

from perceptron.dsl.perceive import PerceiveResult
from perceptron.pointing.geometry import (
    scale_box_to_pixels,
    scale_point_to_pixels,
    scale_points_to_pixels,
)
from perceptron.pointing.types import SinglePoint, bbox, collection, poly


def test_scale_point_to_pixels_clamps_edges():
    pt = SinglePoint(1000, 500, mention="center")
    scaled = scale_point_to_pixels(pt, width=640, height=480)

    assert scaled.x == 639
    assert scaled.y == 240
    assert scaled.mention == "center"


def test_scale_box_to_pixels_orders_coordinates():
    box = bbox(800, 200, 200, 900, mention="mixed")
    scaled = scale_box_to_pixels(box, width=100, height=50)

    assert scaled.top_left.x == 20
    assert scaled.bottom_right.x == 80
    assert scaled.top_left.y == 10
    assert scaled.bottom_right.y == 45
    assert scaled.mention == "mixed"


def test_scale_points_to_pixels_handles_collections():
    coll = collection(
        [
            bbox(0, 0, 1000, 1000, mention="img"),
            poly([(0, 0), (500, 500), (1000, 0)], mention="tri"),
        ],
        mention="scene",
    )

    scaled = scale_points_to_pixels([coll], width=200, height=100)
    assert scaled and scaled[0].mention == "scene"

    outer_box = scaled[0].points[0]
    assert outer_box.bottom_right.x == 199
    assert outer_box.bottom_right.y == 99

    tri = scaled[0].points[1]
    assert tri.hull[1].x == 100
    assert tri.hull[1].y == 50


def test_scale_points_to_pixels_none_passthrough():
    assert scale_points_to_pixels(None, width=10, height=10) is None


def test_scale_points_to_pixels_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        scale_points_to_pixels([], width=0, height=10)


def test_perceive_result_points_to_pixels_returns_copy():
    raw_box = bbox(0, 0, 1000, 1000, mention="full")
    result = PerceiveResult(
        text=None,
        points=[raw_box],
        parsed=None,
        usage=None,
        errors=[],
        raw=None,
    )

    scaled = result.points_to_pixels(400, 200)
    assert scaled and scaled[0].bottom_right.x == 399
    # Original remains normalized
    assert result.points[0].bottom_right.x == 1000
