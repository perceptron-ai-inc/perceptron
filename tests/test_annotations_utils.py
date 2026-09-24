import numpy as np
import pytest

from perceptron.annotations import (
    annotate_image,
    canonicalize_text_collections,
    coerce_annotation,
    serialize_annotations,
)
from perceptron.errors import BadRequestError
from perceptron.pointing.parser import PointParser
from perceptron.pointing.types import Polygon, Track, bbox, clip, collection, pt, track


def test_coerce_annotation_accepts_dict_type_hints():
    spec = {"type": "box", "bbox": (1, 2, 3, 4), "mention": "car"}
    result = coerce_annotation(spec)
    assert result.top_left.x == 1
    assert result.bottom_right.y == 4
    assert result.mention == "car"

    poly_spec = {"type": "polygon", "coords": [(0, 0), (2, 0), (2, 2)]}
    poly = coerce_annotation(poly_spec)
    assert isinstance(poly, Polygon)
    assert [(p.x, p.y) for p in poly.hull] == [(0, 0), (2, 0), (2, 2)]


def test_coerce_annotation_sequence_point_and_bad_collection():
    point = coerce_annotation((9, 8))
    assert point.x == 9 and point.y == 8

    with pytest.raises(BadRequestError):
        coerce_annotation({"type": "collection", "points": None})


def test_canonicalize_text_collections_orders_children():
    raw = '<collection mention="grp"> <point> (5,5) </point> <point> (1,1) </point> </collection>'
    canonical = canonicalize_text_collections(raw)
    assert canonical is not None
    assert canonical.index("(1,1)") < canonical.index("(5,5)")


def test_serialize_annotations_honors_mention_order():
    collections = [
        collection([bbox(5, 5, 6, 6)], mention="second"),
        collection([bbox(1, 1, 2, 2)], mention="first"),
    ]
    serialized = serialize_annotations(
        boxes=None,
        polygons=None,
        points=None,
        collections=collections,
        mention_order={"first": 0, "second": 1},
    )
    assert serialized.index('mention="first"') < serialized.index('mention="second"')


def test_annotate_image_rejects_unknown_annotation():
    class Unknown:
        pass

    with pytest.raises(BadRequestError):
        annotate_image("img", [Unknown()])


def test_canonicalize_text_collections_uses_pointparser_roundtrip():
    coll = collection([bbox(10, 10, 20, 20, mention="alpha"), pt(30, 30)], mention="group")
    text = f"prefix {PointParser.serialize(coll)} suffix"
    canonical = canonicalize_text_collections(text)
    assert canonical.startswith("prefix ")
    # Points should stay serialized through PointParser (ensures parse/serialize path runs)
    assert "<collection" in canonical and "</collection>" in canonical


# ---- t, asset_idx, clips and tracks -----------------------------------------------


def test_coerce_annotation_reads_t_and_asset_idx():
    assert coerce_annotation({"type": "point", "x": 1, "y": 2, "t": "1.5 seconds", "asset_idx": 0}) == pt(
        1, 2, t=1.5, asset_idx=0
    )
    box = coerce_annotation({"bbox": (1, 2, 3, 4), "t": 2, "asset_idx": 1, "label": "car"})
    assert (box.t, box.asset_idx, box.mention) == (2.0, 1, "car")
    polygon = coerce_annotation({"coords": [(0, 0), (2, 0), (2, 2)], "t": 0.5, "asset_idx": 0})
    assert (polygon.t, polygon.asset_idx) == (0.5, 0)
    coll = coerce_annotation({"type": "collection", "points": [(1, 2)], "asset_idx": 0, "t": "3s"})
    assert (coll.asset_idx, coll.t) == (0, 3.0)


def test_coerce_annotation_label_and_child_keys_use_is_not_none():
    assert coerce_annotation({"type": "point", "x": 1, "y": 2, "label": "", "mention": "m"}).mention == ""
    coll = coerce_annotation({"type": "collection", "points": [], "children": [(1, 2)]})
    assert coll.points == []


@pytest.mark.parametrize(
    "spec", [{"type": "point", "x": 1, "y": 2, "asset_idx": -1}, {"type": "point", "x": 1, "y": 2, "t": "soon"}]
)
def test_coerce_annotation_rejects_bad_context(spec):
    with pytest.raises(BadRequestError):
        coerce_annotation(spec)


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "point", "x": 1, "y": 2, "t": -3},
        {"type": "point", "x": 1, "y": 2, "t": "-3"},
        {"type": "point", "x": 1, "y": 2, "t": float("nan")},
        {"type": "box", "bbox": (1, 2, 3, 4), "t": float("inf")},
        {"type": "clip", "at": -1},
        {"type": "clip", "at": 1, "until": float("nan")},
        {"type": "point", "x": 1, "y": 2, "t": True},
    ],
)
def test_coerce_annotation_rejects_times_the_server_rejects(spec):
    with pytest.raises(BadRequestError) as exc_info:
        coerce_annotation(spec)
    assert exc_info.value.code == "invalid_time"


def test_coerce_annotation_accepts_numpy_scalars():
    spec = {"type": "point", "x": 1, "y": 2, "t": np.float32(1.5), "asset_idx": np.int64(1)}
    point = coerce_annotation(spec)
    assert point == pt(1, 2, t=1.5, asset_idx=1)
    assert type(point.t) is float and type(point.asset_idx) is int


def test_coerce_annotation_clip_and_track_specs():
    assert coerce_annotation({"type": "clip", "start": 1, "end": "3.2 seconds", "mention": "shot", "asset_idx": 0}) == (
        clip(1.0, 3.2, mention="shot", asset_idx=0)
    )
    assert coerce_annotation({"type": "clip", "at": 0}) == clip(0.0)
    tr = coerce_annotation(
        {
            "type": "track",
            "points": [{"x": 1, "y": 2, "t": 0}, {"x": 3, "y": 4, "t": 0.5}],
            "mention": "ball",
            "asset_idx": 0,
        }
    )
    assert isinstance(tr, Track)
    assert tr == track([pt(1, 2, t=0.0), pt(3, 4, t=0.5)], mention="ball", asset_idx=0)
    assert coerce_annotation(tr) is tr
    with pytest.raises(BadRequestError):
        coerce_annotation({"type": "track", "points": [(1, 2), (1, 2, 3, 4)]})
    with pytest.raises(BadRequestError):
        coerce_annotation({"type": "clip", "mention": "no time"})


def test_coerce_annotation_unknown_type_hint_raises():
    with pytest.raises(BadRequestError):
        coerce_annotation({"type": "rectangle", "points": [(1, 2)]})


def test_canonical_sort_key_orders_by_time_then_asset_then_position():
    coll = collection(
        [pt(9, 5, t=1.0), pt(5, 1, t=2.0), pt(1, 1, t=1.0, asset_idx=1), pt(0, 0, t=1.0, asset_idx=0), clip(0.5)],
        mention="c",
    )
    canonical = canonicalize_text_collections(PointParser.serialize(coll))
    assert canonical == (
        '<collection mention="c"> <clip t="0.5 seconds" /> <point t="1.0 seconds" asset_idx="0"> (0,0) </point> '
        '<point t="1.0 seconds" asset_idx="1"> (1,1) </point> <point t="1.0 seconds"> (9,5) </point> '
        '<point t="2.0 seconds"> (5,1) </point> </collection>'
    )


@pytest.mark.parametrize(
    "markup",
    [
        '<collection mention="objects" asset_idx="0"> <point_box> (1,1) (2,2) </point_box> <point_box asset_idx="1"> (3,3) (4,4) </point_box> </collection>',
        '<collection mention="events"> <clip mention="a" t="1 seconds" /> <point t="2.0 seconds"> (5,5) </point> </collection>',
        '<collection mention="players"> <track asset_idx="0"> <point_box t="1.0 seconds"> (10,10) (20,20) </point_box> </track> </collection>',
        '<track mention="ball" asset_idx="0"> <point_box t="0.0 seconds"> (1,2) (3,4) </point_box> <point_box t="0.5 seconds"> (2,2) (4,4) </point_box> </track>',
    ],
)
def test_canonicalize_text_collections_keeps_selectors_clips_tracks_and_times(markup):
    text = f"before {markup} after"
    assert canonicalize_text_collections(text) == text


def test_canonicalize_text_collections_normalizes_times_and_track_waypoints():
    text = (
        'x <collection mention="c" t="1.5"> <point> (1,1) </point> </collection> '
        '<track mention="b"> <point t="2" mention="leaf"> (5,5) </point> <point t="1s" asset_idx="3"> (6,6) </point> </track>'
    )
    # Collection t is pushed down with explicit units; waypoints are time-ordered, lose their own mention, and the
    # (uniform) waypoint selector moves onto the track.
    assert canonicalize_text_collections(text) == (
        'x <collection mention="c"> <point t="1.5 seconds"> (1,1) </point> </collection> '
        '<track mention="b" asset_idx="3"> <point t="1.0 seconds"> (6,6) </point> <point t="2.0 seconds"> (5,5) </point> </track>'
    )


@pytest.mark.parametrize(
    "text",
    [
        "use <collection> to group things",
        '<collection mention="A"> <point(0,0)> </collection>',
        "<collection> <collection> <point> (1,2) </point> </collection> </collection>",
        '<track mention="x"> <point t="0 seconds"> (1,2) </point>',
        # a malformed child is skipped by the lenient parse, but the text keeps it as written
        '<collection mention="c"> <point> (5,5) </point> <point> (1,2) (3,4) </point> </collection>',
    ],
)
def test_canonicalize_text_collections_leaves_malformed_or_unclosed_markup_untouched(text):
    assert canonicalize_text_collections(text) == text


def test_serialize_annotations_writes_clips_and_tracks():
    serialized = serialize_annotations(
        [bbox(1, 2, 3, 4, t=1.5, asset_idx=0)],
        None,
        None,
        None,
        clips=[clip(1.0, 2.0, mention="goal", asset_idx=0)],
        tracks=[track([pt(5, 5, t=1.0), pt(1, 1, t=0.0)], mention="ball", asset_idx=0)],
    )
    assert serialized == (
        '<point_box t="1.5 seconds" asset_idx="0"> (1,2) (3,4) </point_box> '
        '<clip mention="goal" asset_idx="0" t="1 seconds 2 seconds" /> '
        '<track mention="ball" asset_idx="0"> <point t="0.0 seconds"> (1,1) </point> <point t="1.0 seconds"> (5,5) </point> </track>'
    )


def test_annotate_image_and_serialize_annotations_keep_authored_times():
    example = annotate_image(
        "frame.png",
        [
            {"type": "point", "x": 5, "y": 5, "t": 0.15},
            {"type": "track", "mention": "ball", "points": [{"x": 2, "y": 2, "t": 0.033}, {"x": 1, "y": 1, "t": 0.0}]},
        ],
    )
    serialized = serialize_annotations(None, None, example["points"], None, tracks=example["tracks"])
    assert serialized == (
        '<point t="0.15 seconds"> (5,5) </point> '
        '<track mention="ball"> <point t="0.0 seconds"> (1,1) </point> <point t="0.033 seconds"> (2,2) </point> </track>'
    )


def test_tracks_whose_waypoints_name_different_assets_are_rejected():
    conflicting = Track([pt(1, 1, t=0.0, asset_idx=0), pt(2, 2, t=1.0, asset_idx=1)], "x")
    for call in (
        lambda: serialize_annotations(None, None, None, None, tracks=[conflicting]),
        lambda: serialize_annotations(None, None, None, [collection([conflicting])]),
        lambda: annotate_image("img", [conflicting]),
    ):
        with pytest.raises(BadRequestError) as exc_info:
            call()
        assert exc_info.value.code == "invalid_track_asset"
    with pytest.raises(BadRequestError, match="differ"):
        coerce_annotation({"type": "track", "points": [{"x": 1, "y": 1, "t": 0, "asset_idx": 0}], "asset_idx": 1})


def test_empty_tracks_are_never_authored():
    # The server does not parse `<track></track>`, so an example must not contain one.
    for spec in ({"type": "track", "mention": "ball", "points": []}, Track([], "ball")):
        with pytest.raises(BadRequestError, match="at least one waypoint"):
            serialize_annotations(None, None, None, None, tracks=[spec])


def test_parse_text_is_still_importable_from_annotations():
    from perceptron import annotations
    from perceptron.pointing import parser

    assert annotations.parse_text is parser.parse_text


def test_annotate_image_returns_clip_and_track_buckets():
    example = annotate_image(
        "img",
        [
            clip(3.0, mention="late"),
            {"type": "clip", "at": 1.0, "mention": "early"},
            track([pt(1, 1, t=1.0), pt(2, 2, t=0.0)], mention="obj", asset_idx=0),
            {"type": "box", "bbox": (1, 2, 3, 4), "asset_idx": 0, "t": 2},
        ],
    )
    assert [c.mention for c in example["clips"]] == ["early", "late"]
    assert [p.t for p in example["tracks"][0].points] == [0.0, 1.0]
    assert example["boxes"][0].asset_idx == 0 and example["boxes"][0].t == 2.0
