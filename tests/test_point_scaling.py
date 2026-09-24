import pytest

from perceptron.dsl.perceive import PerceiveResult
from perceptron.pointing.geometry import (
    scale_annotations_by_asset,
    scale_box_to_pixels,
    scale_point_to_pixels,
    scale_points_to_pixels,
)
from perceptron.pointing.types import SinglePoint, Track, bbox, clip, collection, poly, pt, track


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


def test_perceive_result_boxes_to_pixels_returns_copy():
    raw_box = bbox(0, 0, 1000, 1000, mention="full")
    result = PerceiveResult(
        text=None,
        points=None,
        boxes=[raw_box],
        polygons=None,
        clips=None,
        parsed=None,
        reasoning=None,
        usage=None,
        errors=[],
        raw=None,
    )

    scaled = result.boxes_to_pixels(400, 200)
    assert scaled and scaled[0].bottom_right.x == 399
    # Original remains normalized
    assert result.boxes[0].bottom_right.x == 1000


def test_scaling_keeps_asset_idx_t_and_mention():
    scaled = scale_points_to_pixels(
        [
            pt(500, 500, mention="p", t=1.5, asset_idx=0),
            bbox(0, 0, 1000, 1000, mention="b", t=0.0, asset_idx=0),
            poly([(0, 0), (500, 0), (500, 500)], t=2.0, asset_idx=1),
            collection([pt(1000, 1000)], mention="c", t=3.0, asset_idx=0),
        ],
        width=100,
        height=50,
    )
    assert scaled[0] == pt(50, 25, mention="p", t=1.5, asset_idx=0)
    assert scaled[1] == bbox(0, 0, 99, 49, mention="b", t=0.0, asset_idx=0)
    assert (scaled[2].t, scaled[2].asset_idx) == (2.0, 1)
    assert (scaled[3].mention, scaled[3].t, scaled[3].asset_idx) == ("c", 3.0, 0)


def test_scaling_tracks_per_waypoint_and_passing_clips_through():
    tr = track([bbox(0, 0, 500, 500, t=0.0), bbox(500, 500, 1000, 1000, t=0.5)], mention="ball", asset_idx=0)
    c = clip(1.0, 2.0, mention="goal")
    scaled_track, scaled_clip = scale_points_to_pixels([tr, c], width=200, height=100)
    assert isinstance(scaled_track, Track)
    # waypoints carry the track's asset_idx (DESIGN §12.10)
    assert scaled_track.points == [bbox(0, 0, 100, 50, t=0.0, asset_idx=0), bbox(100, 50, 199, 99, t=0.5, asset_idx=0)]
    assert (scaled_track.mention, scaled_track.asset_idx) == ("ball", 0)
    assert scaled_clip is c


def test_scale_annotations_by_asset_uses_each_annotations_asset():
    anns = [
        pt(500, 500, asset_idx=0),
        pt(500, 500, asset_idx=1),
        collection([pt(1000, 1000), pt(1000, 1000, asset_idx=0)], asset_idx=1),
        track([pt(500, 500, t=0.0)], asset_idx=1),
        clip(1.0),
    ]
    scaled = scale_annotations_by_asset(anns, {0: (100, 100), 1: (1000, 500)})
    assert scaled[0] == pt(50, 50, asset_idx=0)
    assert scaled[1] == pt(500, 250, asset_idx=1)
    assert scaled[2].points == [pt(999, 499, asset_idx=1), pt(99, 99, asset_idx=0)]
    assert scaled[3].points == [pt(500, 250, t=0.0, asset_idx=1)]
    assert scaled[4] is anns[4]
    # a list of sizes is indexed by asset
    assert scale_annotations_by_asset([pt(500, 500, asset_idx=1)], [(10, 10), (20, 20)]) == [pt(10, 10, asset_idx=1)]
    assert scale_annotations_by_asset(None, [(10, 10)]) is None


def test_scale_annotations_by_asset_missing_selector_means_the_last_asset():
    # DESIGN §12.9: without `n_assets`, a missing selector is the highest index in `sizes`.
    assert scale_annotations_by_asset([pt(500, 500)], {0: (10, 20)}) == [pt(5, 10)]
    assert scale_annotations_by_asset([pt(500, 500)], [(10, 20)]) == [pt(5, 10)]
    assert scale_annotations_by_asset([pt(500, 500)], {3: (10, 20)}) == [pt(5, 10)]
    assert scale_annotations_by_asset([pt(500, 500)], [(10, 10), (20, 20)]) == [pt(10, 10)]
    assert scale_annotations_by_asset([pt(500, 500, asset_idx=3)], {3: (10, 20)}) == [pt(5, 10, asset_idx=3)]
    # Resolution never writes the index into the result.
    assert scale_annotations_by_asset([pt(500, 500)], [(10, 10), (20, 20)])[0].asset_idx is None


def test_scale_annotations_by_asset_uses_n_assets_for_missing_selectors():
    sizes = [(10, 10), (20, 20), (40, 40)]
    assert scale_annotations_by_asset([pt(500, 500)], sizes, n_assets=2) == [pt(10, 10)]
    assert scale_annotations_by_asset([pt(500, 500)], sizes, n_assets=1) == [pt(5, 5)]
    # Containers pass their selector down; a child without one (and no container selector) is the last asset.
    scaled = scale_annotations_by_asset(
        [collection([pt(500, 500)], asset_idx=0), collection([pt(500, 500)])], sizes, n_assets=3
    )
    assert [c.points for c in scaled] == [[pt(5, 5, asset_idx=0)], [pt(20, 20)]]
    with pytest.raises(ValueError, match="No size given for asset_idx 3"):
        scale_annotations_by_asset([pt(500, 500)], sizes, n_assets=4)
    with pytest.raises(ValueError, match="no asset"):
        scale_annotations_by_asset([pt(500, 500)], sizes, n_assets=0)
    assert scale_annotations_by_asset([pt(500, 500, asset_idx=0)], sizes, n_assets=0) == [pt(5, 5, asset_idx=0)]


def test_scale_annotations_by_asset_errors():
    with pytest.raises(ValueError, match="no asset"):
        scale_annotations_by_asset([pt(500, 500)], {})
    with pytest.raises(ValueError, match="asset_idx 5"):
        scale_annotations_by_asset([pt(500, 500, asset_idx=5)], [(10, 10)])
    with pytest.raises(ValueError):
        scale_annotations_by_asset([pt(1, 1, asset_idx=0)], [(0, 10)])
