from dataclasses import replace

import numpy as np
import pytest

from perceptron.errors import ParseError
from perceptron.pointing.parser import (
    PointParser,
    _flatten,
    _scan_leaves,
    collect_annotations,
    extract_clips,
    extract_points,
    extract_tracks,
    parse_annotations,
    parse_text,
    resolve_asset_idx,
    scan_leaves,
    strip_tags,
)
from perceptron.pointing.types import (
    BoundingBox,
    Clip,
    ClipTimestamp,
    Collection,
    Polygon,
    SinglePoint,
    Track,
    bbox,
    clip,
    collection,
    poly,
    pt,
    track,
)


def test_point_serialize_and_parse():
    pt = SinglePoint(10, 20, mention="target")
    s = PointParser.serialize(pt)
    assert "<point" in s and "</point>" in s
    segs = parse_text(f"before {s} after")
    kinds = [seg["kind"] for seg in segs]
    assert kinds == ["text", "point", "text"]
    parsed_pt = segs[1]["value"]
    assert parsed_pt == pt


def test_box_and_polygon_extract():
    box = BoundingBox(SinglePoint(1, 2), SinglePoint(3, 4))
    poly = Polygon([SinglePoint(0, 0), SinglePoint(2, 0), SinglePoint(2, 2)])
    s = PointParser.serialize(box) + " and " + PointParser.serialize(poly)
    boxes = extract_points(s, expected="box")
    polys = extract_points(s, expected="polygon")
    assert boxes == [box]
    assert polys == [poly]


def test_strip_tags():
    pt = SinglePoint(5, 6)
    s = f"text {PointParser.serialize(pt)} more"
    stripped = strip_tags(s)
    assert "<point" not in stripped and "</point>" not in stripped


def test_extract_box_with_gt_in_mention():
    """Quoted ``>`` characters inside an attribute should not terminate the tag early."""

    text = '<point_box mention="a > b"> (10,20) (30,40) </point_box>'
    boxes = extract_points(text, expected="box")
    assert len(boxes) == 1
    assert boxes[0].mention == "a > b"
    assert boxes[0].top_left.x == 10 and boxes[0].bottom_right.x == 30


def test_point_parser_escapes_and_parses_mentions():
    pt = SinglePoint(1, 2, mention='door "A" & B', t=1.5)
    tag = PointParser.serialize(pt)
    assert "door &quot;A&quot; &amp; B" in tag
    segments = PointParser.parse(f"start {tag} end")
    assert len(segments) == 1
    parsed_pt = segments[0]["value"]
    assert parsed_pt.mention == 'door "A" & B'
    assert parsed_pt.t == 1.5


def test_extract_points_from_collection_propagates_attrs():
    child_box = BoundingBox(SinglePoint(1, 2), SinglePoint(3, 4))
    child_point = SinglePoint(5, 6, mention="inner")
    collection = Collection(points=[child_box, child_point], mention="group", t=2.5)
    text = PointParser.serialize(collection)

    boxes = extract_points(text, expected="box")
    points = extract_points(text, expected="point")
    assert len(boxes) == 1 and len(points) == 1
    assert boxes[0].mention == "group"
    assert boxes[0].t == 2.5
    assert points[0].mention == "inner"  # child mention preserved
    assert points[0].t == 2.5  # timestamp propagated from collection


def test_extract_points_collection_order_and_filtering():
    child_point = SinglePoint(9, 9)
    child_box = BoundingBox(SinglePoint(2, 2), SinglePoint(8, 8))
    child_poly = Polygon(
        [SinglePoint(0, 0), SinglePoint(1, 0), SinglePoint(1, 1)],
        mention="triangle",
        t=1.1,
    )
    trailing_point = SinglePoint(0, 0, mention="solo", t=5.0)
    collection = Collection(points=[child_point, child_box, child_poly], mention="bundle", t=4.2)
    text = f"pre {PointParser.serialize(collection)} mid {PointParser.serialize(trailing_point)} post"

    all_items = extract_points(text)
    assert [type(item).__name__ for item in all_items] == [
        "SinglePoint",
        "BoundingBox",
        "Polygon",
        "SinglePoint",
    ]

    propagated_point, propagated_box, preserved_poly, final_point = all_items
    assert propagated_point.mention == "bundle"
    assert propagated_point.t == 4.2
    assert propagated_box.mention == "bundle"
    assert propagated_box.t == 4.2
    assert preserved_poly.mention == "triangle"
    assert preserved_poly.t == 1.1
    only_points = extract_points(text, expected="point")
    assert only_points == [propagated_point, final_point]

    only_boxes = extract_points(text, expected="box")
    assert only_boxes == [propagated_box]

    only_polys = extract_points(text, expected="polygon")
    assert only_polys == [preserved_poly]


def test_collection_constructor_helper():
    child = SinglePoint(10, 20, mention="child")
    coll = collection([child], mention="group", t=2.5)
    assert isinstance(coll, Collection)
    assert coll.mention == "group"
    assert coll.t == 2.5
    assert coll.points[0] is child


class TestParseErrorOnMalformedTags:
    """Tests for ParseError raised when model returns malformed tags."""

    def test_point_tag_with_empty_body_raises_parse_error(self):
        text = '<point mention="some label"> </point>'
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == "invalid_point_coords"
        assert "expected coordinates like (x,y)" in str(exc_info.value)
        assert exc_info.value.details["body"] == " "

    def test_point_tag_with_no_coords_raises_parse_error(self):
        text = "<point>no coordinates here</point>"
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == "invalid_point_coords"

    def test_point_tag_with_json_in_mention_raises_parse_error(self):
        # This is the actual malformed response from the model
        text = '<point mention="{\\"price\\": \\"1.27\\"}"> </point>'
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == "invalid_point_coords"

    def test_box_tag_with_only_one_coord_raises_parse_error(self):
        text = "<point_box>(100,200)</point_box>"
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == "invalid_box_coords"
        assert "expected 2 coordinates" in str(exc_info.value)

    def test_box_tag_with_empty_body_raises_parse_error(self):
        text = "<point_box> </point_box>"
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == "invalid_box_coords"

    def test_polygon_tag_with_only_two_coords_raises_parse_error(self):
        text = "<polygon>(0,0) (10,10)</polygon>"
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == "invalid_polygon_coords"
        assert "expected at least 3 coordinates" in str(exc_info.value)
        assert exc_info.value.details["points_found"] == 2

    def test_polygon_tag_with_empty_body_raises_parse_error(self):
        text = "<polygon></polygon>"
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == "invalid_polygon_coords"
        assert exc_info.value.details["points_found"] == 0

    def test_valid_point_tag_does_not_raise(self):
        text = '<point mention="target">(100,200)</point>'
        segs = parse_text(text)
        assert len(segs) == 1
        assert segs[0]["kind"] == "point"
        assert segs[0]["value"].x == 100
        assert segs[0]["value"].y == 200

    def test_valid_box_tag_does_not_raise(self):
        text = "<point_box>(10,20) (30,40)</point_box>"
        segs = parse_text(text)
        assert len(segs) == 1
        assert segs[0]["kind"] == "box"

    def test_valid_polygon_tag_does_not_raise(self):
        text = "<polygon>(0,0) (10,0) (10,10)</polygon>"
        segs = parse_text(text)
        assert len(segs) == 1
        assert segs[0]["kind"] == "polygon"


class TestParseTextClipSegments:
    """With expects='clip', parse_text scans only <clip /> tags."""

    def test_top_level_clip_emits_clip_segment(self):
        text = 'before <clip mention="intro" t=1.5/> after'
        segs = parse_text(text, expects="clip")
        kinds = [s["kind"] for s in segs]
        assert kinds == ["text", "clip", "text"]
        clip_seg = segs[1]
        assert clip_seg["value"] == Clip(timestamp=ClipTimestamp(at=1.5), mention="intro")
        assert clip_seg["span"]["start"] == len("before ")

    def test_clip_range_t_attribute(self):
        text = '<clip mention="action" t="10 20"/>'
        segs = parse_text(text, expects="clip")
        assert len(segs) == 1
        assert segs[0]["kind"] == "clip"
        assert segs[0]["value"] == Clip(timestamp=ClipTimestamp(at=10.0, until=20.0), mention="action")

    def test_clip_mode_ignores_point_tags(self):
        """Clip mode does not scan geometry tags — they remain in the surrounding text segment."""
        text = "<point>(1,2)</point> and <clip t=1.0/>"
        segs = parse_text(text, expects="clip")
        kinds = [s["kind"] for s in segs]
        assert kinds == ["text", "clip"]
        assert "<point>" in segs[0]["text"]

    def test_clip_inside_collection_is_found_in_clip_mode(self):
        """expects='clip' scans the whole text — nesting under <collection> doesn't hide the clip."""
        text = '<collection mention="bundle"><point>(1,2)</point><clip t=1.0/></collection>'
        segs = parse_text(text, expects="clip")
        clip_segs = [s for s in segs if s["kind"] == "clip"]
        assert len(clip_segs) == 1
        assert clip_segs[0]["value"].timestamp == ClipTimestamp(at=1.0)

    def test_clip_without_t_raises_parse_error(self):
        text = '<clip mention="bare"/>'
        with pytest.raises(ParseError) as exc_info:
            parse_text(text, expects="clip")
        assert exc_info.value.code == "invalid_clip_timestamp"

    def test_clip_with_unparseable_t_raises_parse_error(self):
        text = '<clip t="not-a-number"/>'
        with pytest.raises(ParseError) as exc_info:
            parse_text(text, expects="clip")
        assert exc_info.value.code == "invalid_clip_timestamp"

    def test_default_expects_does_not_parse_clips(self):
        """Default mode preserves the geometry-only contract — clip tags stay in text segments."""
        text = "before <clip t=1.5/> after"
        segs = parse_text(text)
        assert len(segs) == 1
        assert segs[0]["kind"] == "text"
        assert "<clip" in segs[0]["text"]


# ---------------------------------------------------------------------------
# asset_idx, time grammar, tracks, containers (Mk1.5 annotation contract)
# ---------------------------------------------------------------------------


def _values(segments):
    return [seg["value"] for seg in segments if seg["kind"] != "text"]


class TestAssetIdx:
    def test_parse_and_serialize_asset_idx_zero(self):
        box = parse_text('<point_box asset_idx="0"> (1,2) (3,4) </point_box>')[0]["value"]
        assert box.asset_idx == 0
        assert PointParser.serialize(box) == '<point_box asset_idx="0"> (1,2) (3,4) </point_box>'

    def test_asset_idx_is_keyword_only_and_positional_constructors_unchanged(self):
        assert SinglePoint(1, 2, "m", 1.5) == SinglePoint(x=1, y=2, mention="m", t=1.5, asset_idx=None)
        with pytest.raises(TypeError):
            SinglePoint(1, 2, "m", 1.5, 0)  # type: ignore[misc]

    def test_collection_child_zero_overrides_nonzero_collection(self):
        text = (
            '<collection mention="cups" asset_idx="2"> <point_box> (10,10) (20,20) </point_box> '
            '<point_box asset_idx="0"> (30,30) (40,40) </point_box> </collection>'
        )
        boxes = extract_points(text, expected="box")
        assert [b.asset_idx for b in boxes] == [2, 0]
        assert [b.mention for b in boxes] == ["cups", "cups"]
        # the parsed tree holds the collection's value on children without their own (DESIGN §12.10)
        coll = parse_text(text)[0]["value"]
        assert coll.asset_idx == 2 and [c.asset_idx for c in coll.points] == [2, 0]

    @pytest.mark.parametrize("raw", ["-1", "1.0", "x", " "])
    def test_invalid_asset_idx_raises(self, raw):
        with pytest.raises(ParseError) as exc_info:
            parse_text(f'<point asset_idx="{raw}"> (1,2) </point>')
        assert exc_info.value.code == "invalid_asset_idx"

    @pytest.mark.parametrize(("raw", "expected"), [("", None), (" 1 ", 1), ("+1", 1), ("01", 1)])
    def test_asset_idx_forms(self, raw, expected):
        assert parse_text(f'<point asset_idx="{raw}"> (1,2) </point>')[0]["value"].asset_idx == expected


class TestSpatialTime:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1.5 seconds", 1.5),
            ("1.5 SECONDS", 1.5),
            ("2.0", 2.0),
            ("2", 2.0),
            ("1.5s", 1.5),
            ("1 second", 1.0),
            ("1.5seconds", 1.5),
            (".5", 0.5),
            ("1e2", 100.0),
            ("0", 0.0),
        ],
    )
    def test_accepted_forms_without_mention(self, raw, expected):
        pt = parse_text(f'<point t="{raw}"> (1,2) </point>')[0]["value"]
        assert pt.t == expected
        assert pt.mention is None

    @pytest.mark.parametrize("raw", ["nan", "inf", "-1 seconds", "1500ms", "0:05", "frame 3", "1e999"])
    def test_rejected_forms(self, raw):
        with pytest.raises(ParseError) as exc_info:
            parse_text(f'<point t="{raw}"> (1,2) </point>')
        assert exc_info.value.code == "invalid_time"

    def test_serialized_with_explicit_seconds(self):
        assert PointParser.serialize(SinglePoint(1, 2, t=0)) == '<point t="0.0 seconds"> (1,2) </point>'
        assert PointParser.serialize(SinglePoint(1, 2, t=2)) == '<point t="2.0 seconds"> (1,2) </point>'
        assert PointParser.serialize(SinglePoint(1, 2, t=1.25)) == '<point t="1.2 seconds"> (1,2) </point>'
        # a string t is normalized, never emitted as a bare number
        assert PointParser.serialize(SinglePoint(1, 2, t="2")) == '<point t="2.0 seconds"> (1,2) </point>'
        with pytest.raises(ParseError):
            PointParser.serialize(SinglePoint(1, 2, t="soon"))

    @pytest.mark.parametrize(
        "obj",
        [
            pt(1, 2, t=float("nan")),
            pt(1, 2, t=float("inf")),
            pt(1, 2, t=-1),
            pt(1, 2, t=True),
            clip(float("inf")),
            clip(1.0, -2.0),
            collection([pt(1, 2)], t=-0.5),
        ],
    )
    def test_serializer_rejects_times_the_parser_would_reject(self, obj):
        with pytest.raises(ValueError, match="invalid_time"):
            PointParser.serialize(obj)

    @pytest.mark.parametrize("asset_idx", [-1, 1.5, True, "1"])
    def test_serializer_rejects_invalid_asset_idx(self, asset_idx):
        with pytest.raises(ValueError, match="invalid_asset_idx"):
            PointParser.serialize(pt(1, 2, asset_idx=asset_idx))
        with pytest.raises(ValueError, match="invalid_asset_idx"):
            PointParser.serialize(clip(1.0, asset_idx=asset_idx))

    def test_serializer_accepts_numpy_scalars(self):
        obj = pt(1, 2, t=np.float32(1.5), asset_idx=np.int64(2))
        assert PointParser.serialize(obj) == '<point t="1.5 seconds" asset_idx="2"> (1,2) </point>'

    def test_leaf_attr_order_is_mention_t_asset_idx(self):
        tag = PointParser.serialize(BoundingBox(SinglePoint(1, 2), SinglePoint(3, 4), "car", 1.5, asset_idx=0))
        assert tag == '<point_box mention="car" t="1.5 seconds" asset_idx="0"> (1,2) (3,4) </point_box>'
        parsed = parse_text('<point asset_idx="1" t="2.5 seconds" mention="x"> (1,2) </point>')[0]["value"]
        assert parsed == SinglePoint(1, 2, "x", 2.5, asset_idx=1)

    def test_collection_t_is_pushed_down_not_emitted(self):
        coll = Collection(
            [SinglePoint(1, 2), SinglePoint(3, 4, t=1.0), Clip(ClipTimestamp(at=5.0))], mention="c", t=2.0
        )
        assert PointParser.serialize(coll) == (
            '<collection mention="c"> <point t="2.0 seconds"> (1,2) </point> '
            '<point t="1.0 seconds"> (3,4) </point> <clip t="5 seconds" /> </collection>'
        )


class TestAttributes:
    def test_single_quoted_unquoted_and_unknown_attrs(self):
        text = '<point_box mention=car asset_idx=\'1\' index="image_0" label="L" type="focus"> (1,2) (3,4) </point_box>'
        box = parse_text(text)[0]["value"]
        assert box == BoundingBox(SinglePoint(1, 2), SinglePoint(3, 4), "car", asset_idx=1)
        assert PointParser.serialize(box) == '<point_box mention="car" asset_idx="1"> (1,2) (3,4) </point_box>'

    def test_single_quoted_value_with_space(self):
        box = parse_text("<point_box mention='red ball'> (1,2) (3,4) </point_box>")[0]["value"]
        assert box.mention == "red ball"

    def test_empty_mention_is_absent_and_inherits(self):
        text = '<collection mention="group"> <point mention=""> (1,2) </point> </collection>'
        assert extract_points(text)[0].mention == "group"
        assert PointParser.serialize(SinglePoint(1, 2, mention="")) == "<point> (1,2) </point>"

    def test_t_inside_quoted_mention_is_not_an_attribute(self):
        pt = parse_text('<point mention="at t=9" t="1.0 seconds"> (1,2) </point>')[0]["value"]
        assert pt.t == 1.0 and pt.mention == "at t=9"


class TestCoordinates:
    def test_point_with_two_coordinates_raises(self):
        with pytest.raises(ParseError) as exc_info:
            parse_text("<point> (1,2) (3,4) </point>")
        assert exc_info.value.code == "invalid_point_coords"

    def test_box_with_three_coordinates_raises(self):
        with pytest.raises(ParseError) as exc_info:
            parse_text("<point_box> (1,2) (3,4) (5,6) </point_box>")
        assert exc_info.value.code == "invalid_box_coords"
        assert exc_info.value.details["points_found"] == 3

    def test_bare_number_fallback(self):
        box = parse_text('<point_box mention="car" t="1.5s"> 10  20  30  40 </point_box>')[0]["value"]
        assert box == BoundingBox(SinglePoint(10, 20), SinglePoint(30, 40), "car", 1.5)
        assert parse_text("<point> 7 8 </point>")[0]["value"] == SinglePoint(7, 8)

    @pytest.mark.parametrize("body", ["10 20 30", "-5 3", "1.5 2", "a b"])
    def test_bare_number_fallback_rejects_odd_or_non_digit_bodies(self, body):
        with pytest.raises(ParseError):
            parse_text(f"<point> {body} </point>")

    def test_signed_or_decimal_tuples_are_not_coordinates(self):
        with pytest.raises(ParseError):
            parse_text("<point> (-5,3) </point>")


class TestTracks:
    TRACK = (
        '<track mention="red ball" asset_idx="0">\n'
        ' <point_box t="0.0 seconds"> (100,150) (180,230) </point_box>\n'
        ' <point_box t="0.5 seconds"> (120,150) (200,230) </point_box>\n'
        "</track>"
    )

    def test_top_level_track_segment(self):
        segs = parse_text(f"see {self.TRACK} done")
        assert [s["kind"] for s in segs] == ["text", "track", "text"]
        tr = segs[1]["value"]
        assert isinstance(tr, Track)
        assert tr.mention == "red ball" and tr.asset_idx == 0 and tr.complete
        assert [p.t for p in tr.points] == [0.0, 0.5]
        assert all(p.mention is None and p.asset_idx == 0 for p in tr.points)  # the track's selector is pushed down

    def test_waypoints_flatten_into_boxes_with_context(self):
        boxes = extract_points(self.TRACK, expected="box")
        assert [(b.mention, b.t, b.asset_idx) for b in boxes] == [("red ball", 0.0, 0), ("red ball", 0.5, 0)]

    def test_extract_tracks_top_level_and_in_collection(self):
        text = (
            '<collection mention="each player"> <track> <point_box t="0.0 seconds"> (1,2) (3,4) </point_box> '
            '</track> <track asset_idx="1"> <point t="1.0 seconds"> (5,6) </point> </track> </collection> '
            f"{self.TRACK}"
        )
        tracks = extract_tracks(text)
        assert [(t.mention, t.asset_idx) for t in tracks] == [
            ("each player", None),
            ("each player", 1),
            ("red ball", 0),
        ]
        assert [t.mention for t in extract_tracks(text, expected="box")] == ["each player", "red ball"]
        assert len(extract_tracks(text, expected="point")) == 1

    def test_uniform_waypoint_asset_is_hoisted(self):
        tr = parse_text(
            '<track mention="x"> <point_box t="0.0 seconds" asset_idx="1"> (1,2) (3,4) </point_box> </track>'
        )
        assert tr[0]["value"] == Track([BoundingBox(SinglePoint(1, 2), SinglePoint(3, 4), t=0.0)], "x", asset_idx=1)

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            (
                '<track> <point_box t="0 seconds"> (1,2) (3,4) </point_box> <point t="1 seconds"> (1,2) </point> </track>',
                "invalid_track_geometry",
            ),
            ('<track> <clip t="1 seconds" /> </track>', "invalid_track_child"),
            (
                '<track> <point t="0 seconds" asset_idx="0"> (1,2) </point> <point t="1 seconds" asset_idx="1"> (1,2) '
                "</point> </track>",
                "invalid_track_asset",
            ),
            (
                '<track asset_idx="2"> <point t="0 seconds" asset_idx="1"> (1,2) </point> </track>',
                "invalid_track_asset",
            ),
        ],
    )
    def test_invalid_tracks_raise(self, text, code):
        with pytest.raises(ParseError) as exc_info:
            parse_text(text)
        assert exc_info.value.code == code

    def test_waypoint_mention_is_kept_raw_but_track_mention_is_effective(self):
        text = '<track mention="x"> <point_box mention="y" t="0.0 seconds"> (1,2) (3,4) </point_box> </track>'
        assert parse_text(text)[0]["value"].points[0].mention == "y"
        assert extract_points(text)[0].mention == "x"

    @pytest.mark.parametrize("markup", ['<track mention="ball"></track>', '<collection mention="dogs"> </collection>'])
    def test_closed_container_without_children_is_text(self, markup):
        # Like the server ("No valid child points found"): a container needs a valid child to be an annotation.
        text = f"Found: {markup}"
        text_only = [{"kind": "text", "text": text, "span": {"start": 0, "end": len(text)}}]
        assert parse_text(text) == text_only
        result = collect_annotations(text)
        assert (result.parsed, result.tracks, result.errors) == (text_only, [], [])
        assert extract_tracks(text) == []

    def test_unclosed_empty_track_is_still_arriving(self):
        result = collect_annotations('<track mention="ball">')
        assert result.tracks == [Track([], "ball", complete=False)]
        assert [e["code"] for e in result.errors] == ["incomplete_annotation"]

    def test_track_factory_takes_any_iterable_and_needs_a_waypoint(self):
        waypoints = [pt(1, 1, t=0.0), pt(2, 2, t=1.0)]
        assert track(iter(waypoints), mention="ball") == track(waypoints, mention="ball")
        assert len(track(p for p in waypoints).points) == 2
        with pytest.raises(ValueError, match="at least one waypoint"):
            track([])
        with pytest.raises(ValueError, match="at least one waypoint"):  # the server does not parse an empty track
            PointParser.serialize(Track([], "ball"))

    def test_track_serialization(self):
        tr = track([bbox(1, 2, 3, 4, t=0), bbox(2, 2, 4, 4, t=0.5)], mention="ball", asset_idx=0)
        assert PointParser.serialize(tr) == (
            '<track mention="ball" asset_idx="0"> <point_box t="0.0 seconds"> (1,2) (3,4) </point_box> '
            '<point_box t="0.5 seconds"> (2,2) (4,4) </point_box> </track>'
        )


class TestCollections:
    def test_collection_with_clip_and_track_children(self):
        text = (
            '<collection mention="events" asset_idx="1"> <clip t="1 seconds 2 seconds" /> '
            '<track> <point t="0.0 seconds"> (1,1) </point> </track> <point> (5,5) </point> </collection>'
        )
        coll = parse_text(text)[0]["value"]
        assert [type(c).__name__ for c in coll.points] == ["Clip", "Track", "SinglePoint"]
        assert PointParser.serialize(coll) == (
            '<collection mention="events" asset_idx="1"> <clip t="1 seconds 2 seconds" /> '
            '<track> <point t="0.0 seconds"> (1,1) </point> </track> <point> (5,5) </point> </collection>'
        )

    def test_nested_collection_raises(self):
        with pytest.raises(ParseError) as exc_info:
            parse_text(
                '<collection mention="a"> <collection mention="b"> <point> (1,2) </point> </collection> </collection>'
            )
        assert exc_info.value.code == "invalid_collection_child"

    def test_user_constructed_nested_collection_serializes_and_flattens(self):
        inner = collection([pt(1, 2)], mention="inner", asset_idx=0)
        outer = collection([inner, pt(3, 4)], mention="outer", t=1.0, asset_idx=2)
        assert PointParser.serialize(outer) == (
            '<collection mention="outer" asset_idx="2"> <collection mention="inner" asset_idx="0"> '
            '<point t="1.0 seconds"> (1,2) </point> </collection> <point t="1.0 seconds"> (3,4) </point> </collection>'
        )
        items, _ = _flatten([outer])
        assert items == [pt(1, 2, mention="inner", t=1.0, asset_idx=0), pt(3, 4, mention="outer", t=1.0, asset_idx=2)]
        assert collect_annotations("").points == []

    def test_unclosed_container_stays_text_in_parse_text(self):
        text = '<collection mention="x"> <point> (1,2) </point>'
        segs = parse_text(text)
        assert [s["kind"] for s in segs] == ["text"]

    def test_stray_open_tag_in_prose_does_not_swallow_the_next_tag(self):
        segs = parse_text("use <point and then <point_box> (1,2) (3,4) </point_box>")
        assert [s["kind"] for s in segs] == ["text", "box"]

    def test_stray_close_tag_at_top_level_is_prose(self):
        segs = parse_text("a </collection> b <point> (1,2) </point>")
        assert [s["kind"] for s in segs] == ["text", "point"]

    def test_mismatched_close_inside_container_raises(self):
        with pytest.raises(ParseError) as exc_info:
            parse_text('<collection> <track> <point t="0 seconds"> (1,2) </point> </collection>')
        assert exc_info.value.code == "unclosed_tag"

    def test_clip_mode_surfaces_collection_clips_with_context(self):
        text = '<collection mention="bundle" asset_idx="0"><point>(1,2)</point><clip t="1 seconds"/></collection>'
        segs = parse_text(text, expects="clip")
        assert [s["kind"] for s in segs] == ["text", "clip", "text"]
        assert segs[1]["value"] == Clip(ClipTimestamp(at=1.0), "bundle", asset_idx=0)


class TestParseAnnotations:
    def test_lenient_keeps_text_and_reports_error(self):
        text = 'a <point> (1,2) (3,4) </point> b <point_box asset_idx="0"> (1,2) (3,4) </point_box>'
        parsed = parse_annotations(text)
        assert [s["kind"] for s in parsed.segments] == ["text", "box"]
        assert parsed.segments[0]["text"] == "a <point> (1,2) (3,4) </point> b "
        assert parsed.errors == [
            {
                "code": "invalid_point_coords",
                "message": "Malformed <point> tag: expected coordinates like (x,y) but got: ' (1,2) (3,4) '",
                "span": {"start": 2, "end": 30},
            }
        ]

    def test_strict_raises(self):
        with pytest.raises(ParseError) as exc_info:
            parse_annotations("<point> </point>", strict=True)
        assert exc_info.value.code == "invalid_point_coords"
        assert exc_info.value.details["span"] == {"start": 0, "end": 16}

    def test_default_expects_structures_every_kind(self):
        text = '<clip t="1 seconds" /> <point> (1,2) </point>'
        assert [s["kind"] for s in parse_annotations(text).segments] == ["clip", "text", "point"]
        assert [s["kind"] for s in parse_annotations(text, expects="box").segments] == ["text", "point"]
        assert [s["kind"] for s in parse_annotations(text, expects="clip").segments] == ["clip", "text"]
        assert [s["kind"] for s in parse_annotations(text, expects="text").segments] == ["text"]

    def test_truncated_track_is_kept_incomplete(self):
        text = (
            '<track mention="basketball" asset_idx="0">\n'
            ' <point_box t="0.5 seconds"> (410,520) (450,570) </point_box>\n'
            ' <point_box t="1.0 seconds"> (470,300) (510,350) </point_box>\n'
        )
        parsed = parse_annotations(text)
        tr = parsed.segments[0]["value"]
        assert tr.complete is False and len(tr.points) == 2 and tr.asset_idx == 0
        assert [e["code"] for e in parsed.errors] == ["incomplete_annotation"]
        assert PointParser.serialize(tr).endswith("</track>")  # serializing an object always closes it
        with pytest.raises(ParseError) as exc_info:
            parse_annotations(text, strict=True)
        assert exc_info.value.code == "incomplete_annotation"

    def test_nested_collection_is_text_with_error(self):
        text = "<collection> <collection> <point> (1,2) </point> </collection> </collection> <point> (3,4) </point>"
        parsed = parse_annotations(text)
        assert [s["kind"] for s in parsed.segments] == ["text", "point"]
        assert [e["code"] for e in parsed.errors] == ["invalid_collection_child"]


class TestCollectAnnotations:
    def test_effective_context_and_buckets(self):
        text = (
            '<collection mention="each player" asset_idx="1">\n <track>\n'
            ' <point_box t="0.0 seconds"> (354,163) (397,265) </point_box>\n'
            ' <point_box t="0.5 seconds"> (355,164) (399,267) </point_box>\n</track>\n'
            ' <track asset_idx="0">\n <point_box t="0.0 seconds"> (506,610) (573,840) </point_box>\n</track>\n'
            ' <clip t="2 seconds" />\n <point t="3 seconds"> (1,1) </point>\n</collection>'
        )
        result = collect_annotations(text)
        assert result.errors == []
        assert [(t.mention, t.asset_idx, len(t.points)) for t in result.tracks] == [
            ("each player", 1, 2),
            ("each player", 0, 1),
        ]
        assert [(b.mention, b.t, b.asset_idx) for b in result.boxes] == [
            ("each player", 0.0, 1),
            ("each player", 0.5, 1),
            ("each player", 0.0, 0),
        ]
        assert result.clips == [Clip(ClipTimestamp(at=2.0), "each player", asset_idx=1)]
        assert result.points == [SinglePoint(1, 1, "each player", 3.0, asset_idx=1)]
        assert [s["kind"] for s in result.parsed] == ["collection"]

    def test_flattening_never_mutates_the_parsed_tree(self):
        text = '<collection mention="c" t="2 seconds"> <point> (1,2) </point> </collection>'
        result = collect_annotations(text)
        assert result.points[0] == SinglePoint(1, 2, "c", 2.0)
        assert result.parsed[0]["value"].points[0] == SinglePoint(1, 2)

    def test_waypoint_mention_falls_back_to_collection(self):
        text = '<collection mention="c"> <track> <point mention="leaf" t="0 seconds"> (1,2) </point> </track> </collection>'
        assert collect_annotations(text).points[0].mention == "c"

    def test_errors_are_reported(self):
        result = collect_annotations('<point t="soon"> (1,2) </point> <point> (3,4) </point>')
        assert result.points == [SinglePoint(3, 4)]
        assert [e["code"] for e in result.errors] == ["invalid_time"]


class TestInvalidChildren:
    """Lenient parsing skips an invalid child (reported) and keeps its container, like the server parser."""

    def test_malformed_leaf_in_collection_is_skipped(self):
        text = (
            '<collection mention="cars"> <point_box> (1,2) (3,4) </point_box> <point_box> (1,2) (3,4) (5,6) '
            "</point_box> <point_box> (5,6) (7,8) </point_box> </collection>"
        )
        result = collect_annotations(text)
        assert result.boxes == [bbox(1, 2, 3, 4, mention="cars"), bbox(5, 6, 7, 8, mention="cars")]
        assert [e["code"] for e in result.errors] == ["invalid_box_coords"]
        assert result.errors[0]["span"]["start"] == text.index("<point_box> (1,2) (3,4) (5,6)")
        assert len(result.parsed[0]["value"].points) == 2
        for parse in (parse_text, lambda t: collect_annotations(t, strict=True)):
            with pytest.raises(ParseError) as exc_info:
                parse(text)
            assert exc_info.value.code == "invalid_box_coords"

    def test_misplaced_track_child_is_skipped(self):
        text = (
            '<track mention="x"> <point t="0 seconds"> (1,2) </point> <clip t="1 seconds" /> '
            '<point t="1 seconds"> (3,4) </point> </track>'
        )
        result = collect_annotations(text)
        assert result.tracks == [Track([pt(1, 2, t=0.0), pt(3, 4, t=1.0)], "x")]
        assert result.points == [pt(1, 2, mention="x", t=0.0), pt(3, 4, mention="x", t=1.0)]
        assert result.clips == []
        assert result.errors == [
            {
                "code": "invalid_track_child",
                "message": "<track> may only contain point, point_box or polygon waypoints, not <clip>",
                "span": {"start": text.index("<clip"), "end": text.index("<clip") + len('<clip t="1 seconds" />')},
            }
        ]
        assert [leaf["kind"] for leaf in scan_leaves(text)] == ["point", "point"]  # never streamed either

    def test_invalid_track_is_skipped_inside_a_collection(self):
        text = (
            '<collection mention="p"> <track> <point_box t="0 seconds"> (1,2) (3,4) </point_box> '
            '<point t="1 seconds"> (1,2) </point> </track> <point> (5,5) </point> </collection>'
        )
        result = collect_annotations(text)
        assert result.tracks == [] and result.boxes == []
        assert result.points == [pt(5, 5, mention="p")]
        assert [e["code"] for e in result.errors] == ["invalid_track_geometry"]

    def test_malformed_waypoint_is_skipped_before_the_geometry_check(self):
        text = '<track> <point_box t="0 seconds"> (1,2) </point_box> <point t="1 seconds"> (3,4) </point> </track>'
        result = collect_annotations(text)
        assert result.tracks == [Track([pt(3, 4, t=1.0)])]
        assert [e["code"] for e in result.errors] == ["invalid_box_coords"]

    @pytest.mark.parametrize(
        ("text", "codes"),
        [
            # no valid child left
            ('<track> <clip t="1 seconds" /> </track>', ["invalid_track_child"]),
            (
                "<collection> <point> </point> <point_box> (1,2) </point_box> </collection>",
                ["invalid_point_coords", "invalid_box_coords"],
            ),
            # structural problems invalidate the whole container
            (
                '<track> <point_box t="0 seconds"> (1,2) (3,4) </point_box> <point t="1 seconds"> (1,2) </point> </track>',
                ["invalid_track_geometry"],
            ),
            (
                '<track asset_idx="2"> <point t="0 seconds" asset_idx="1"> (1,2) </point> </track>',
                ["invalid_track_asset"],
            ),
            ('<collection asset_idx="-1"> <point> (1,2) </point> </collection>', ["invalid_asset_idx"]),
            ("<collection> <point> (1,2) <point> (3,4) </point> </collection>", ["unclosed_tag"]),
        ],
    )
    def test_container_without_valid_children_or_with_structural_errors_is_text(self, text, codes):
        parsed = parse_annotations(text)
        assert [s["kind"] for s in parsed.segments] == ["text"]
        assert [e["code"] for e in parsed.errors] == codes


class TestTruncation:
    """Truncated or streamed output keeps every completed child; a partial element at the tail stays pending."""

    E7 = (
        '<collection mention="each player">\n <track>\n'
        ' <point_box t="0.0 seconds"> (354,163) (397,265) </point_box>\n'
        ' <point_box t="0.5 seconds"> (355,164) (399,267) </point_box>\n</track>\n <track>\n'
        ' <point_box t="0.0 seconds"> (506,610) (573,840) </point_box>\n</track>\n</collection>'
    )
    E15 = (
        '<track mention="basketball" asset_idx="0">\n'
        ' <point_box t="0.5 seconds"> (410,520) (450,570) </point_box>\n'
        ' <point_box t="1.0 seconds"> (470,300) (510,350) </point_box>\n'
    )
    CLIPS = '<collection mention="goals">\n <clip t="1 seconds" />\n <clip t="2 seconds 3.5 seconds" />\n</collection>'

    @pytest.mark.parametrize("text", [E7, E15, CLIPS], ids=["E7", "E15", "clips"])
    def test_every_prefix_keeps_completed_children_in_order(self, text):
        full = collect_annotations(text)
        for end in range(len(text) + 1):
            prefix = text[:end]
            result = collect_annotations(prefix)
            n_boxes, n_clips = prefix.count("</point_box>"), prefix.count("/>")
            assert result.boxes == full.boxes[:n_boxes], prefix
            assert sum(len(tr.points) for tr in result.tracks) == n_boxes, prefix
            assert result.clips == full.clips[:n_clips], prefix
            assert [e["code"] for e in result.errors] in ([], ["incomplete_annotation"]), prefix

    def test_cut_inside_a_waypoint_keeps_the_track(self):
        text = self.E15 + ' <point_box t="1.5 seconds"> (480,'
        result = collect_annotations(text)
        assert result.tracks == [
            Track(
                [bbox(410, 520, 450, 570, t=0.5), bbox(470, 300, 510, 350, t=1.0)],
                "basketball",
                asset_idx=0,
                complete=False,
            )
        ]
        assert [e["code"] for e in result.errors] == ["incomplete_annotation"]
        with pytest.raises(ParseError) as exc_info:
            collect_annotations(text, strict=True)
        assert exc_info.value.code == "incomplete_annotation"

    def test_cut_inside_a_clip_keeps_the_collection(self):
        result = collect_annotations('<collection mention="goals"> <clip t="1 seconds" /> <clip t="2 seconds">')
        assert result.clips == [clip(1.0, mention="goals")]
        assert [e["code"] for e in result.errors] == ["incomplete_annotation"]


class TestScanLeaves:
    COLLECTION = (
        '<collection mention="each player" asset_idx="2">\n <track>\n'
        ' <point_box t="0.0 seconds"> (354,163) (397,265) </point_box>\n'
        ' <point_box t="0.5 seconds"> (355,164) (399,267) </point_box>\n</track>\n'
        ' <track mention="goalie">\n <point_box t="0.0 seconds"> (506,610) (573,840) </point_box>\n</track>\n'
        " <point> (1,1) </point>\n</collection>"
    )

    def test_leaves_inside_unclosed_containers_have_context(self):
        partial = self.COLLECTION[: self.COLLECTION.index("</track>")]
        leaves = scan_leaves(partial)
        assert [leaf["kind"] for leaf in leaves] == ["box", "box"]
        assert leaves[0]["context"] == {"mention": "each player", "t": 0.0, "asset_idx": 2, "container": "track"}
        assert leaves[0]["value"] == BoundingBox(
            SinglePoint(354, 163), SinglePoint(397, 265), "each player", 0.0, asset_idx=2
        )
        assert leaves[0]["container_start"] == self.COLLECTION.index("<track>")

    @pytest.mark.parametrize("step", [1, 7])
    def test_chunked_scan_emits_each_leaf_once_in_order(self, step):
        seen: dict[tuple[int, int], dict] = {}
        order: list[tuple[int, int]] = []
        for end in range(step, len(self.COLLECTION) + step, step):
            for leaf in scan_leaves(self.COLLECTION[:end]):
                span = (leaf["span"]["start"], leaf["span"]["end"])
                if span not in seen:
                    seen[span] = leaf
                    order.append(span)
        final = scan_leaves(self.COLLECTION)
        assert order == [(leaf["span"]["start"], leaf["span"]["end"]) for leaf in final]
        assert [seen[span]["context"] for span in order] == [leaf["context"] for leaf in final]
        assert [leaf["context"]["mention"] for leaf in final] == ["each player", "each player", "goalie", "each player"]
        assert final[-1]["context"]["container"] == "collection"

    @pytest.mark.parametrize("step", [1, 7])
    def test_resumed_scans_match_full_scans(self, step):
        # A stream rescans from the end of the last complete top-level element; that finds exactly the leaves a full
        # scan finds after it, and nothing before it can change.
        text = f'{self.COLLECTION} then {self.COLLECTION} and <point mention="x"> (9,9) </point>'
        start = 0
        for end in range(step, len(text) + step, step):
            prefix = text[:end]
            leaves, resume = _scan_leaves(prefix, None, start)
            assert leaves == [leaf for leaf in scan_leaves(prefix) if leaf["span"]["start"] >= start], prefix
            assert start <= resume <= len(prefix)
            start = resume
        assert start == len(text)

    def test_expects_filters_kind_and_partial_leaf_is_pending(self):
        text = '<point> (1,2) </point> <clip t="1 seconds" /> <point_box> (1,2) (3,'
        assert [leaf["kind"] for leaf in scan_leaves(text)] == ["point", "clip"]
        assert [leaf["kind"] for leaf in scan_leaves(text, expects="clip")] == ["clip"]
        assert scan_leaves(text, expects="box") == []
        assert scan_leaves(text, expects="text") == []
        top = scan_leaves(text)[0]
        assert top["context"]["container"] is None and top["container_start"] is None


class TestResolveAssetIdx:
    def test_explicit_zero_wins(self):
        assert resolve_asset_idx(SinglePoint(1, 2, asset_idx=0), 3) == 0

    def test_missing_selector_means_the_last_asset(self):
        # DESIGN §12.9: no `asset_idx` refers to the last asset; None only without assets.
        assert resolve_asset_idx(SinglePoint(1, 2), 1) == 0
        assert resolve_asset_idx(SinglePoint(1, 2), 2) == 1
        assert resolve_asset_idx(SinglePoint(1, 2), 5) == 4
        assert resolve_asset_idx(SinglePoint(1, 2), 0) is None
        assert resolve_asset_idx(SinglePoint(1, 2), None) is None
        assert resolve_asset_idx(SinglePoint(1, 2, asset_idx=4), None) == 4

    def test_containers_and_clips_resolve_their_own_selector(self):
        assert resolve_asset_idx(Track([pt(1, 2, t=0.0)], asset_idx=0), 3) == 0
        assert resolve_asset_idx(clip(1.0), 3) == 2
        box = collect_annotations(
            '<collection asset_idx="1"> <point_box> (1,2) (3,4) </point_box> </collection>'
        ).boxes[0]
        assert resolve_asset_idx(box, 3) == 1

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            resolve_asset_idx(SinglePoint(1, 2, asset_idx=2), 2)
        with pytest.raises(ValueError, match="out of range"):
            resolve_asset_idx(SinglePoint(1, 2, asset_idx=0), 0)

    def test_parsed_objects_keep_none(self):
        assert parse_text("<point> (1,2) </point>")[0]["value"].asset_idx is None


def test_strip_tags_removes_tracks_containers_and_paired_clips():
    text = (
        'a <track mention="x"> <point t="0 seconds"> (1,2) </point> </track> b '
        '<clip t="1 seconds"></clip> c <collection> <clip t="2 seconds" /> </collection> d'
    )
    assert strip_tags(text) == "a  b  c  d"


def test_strip_tags_removes_a_closed_container_with_its_layout():
    # The server renders children on their own lines; 0.3.5 removed a whole collection, layout included.
    text = (
        'Two cats: <collection mention="cat">\n <point> (1,2) </point>\n <point> (3,4) </point>\n</collection>. '
        'Ball: <track mention="ball">\n <point t="0 seconds"> (1,2) </point>\n <point t="1 seconds"> (3,4) </point>'
        "\n</track>. Done"
    )
    assert strip_tags(text) == "Two cats: . Ball: . Done"
    # Inside an unclosed container, a closed child container still goes whole.
    assert strip_tags('<collection> <track> <point t="0 seconds"> (1,2) </point>\n</track>\n tail') == " \n tail"


def test_strip_tags_keeps_stray_leaf_tags_and_unclosed_container_text():
    assert strip_tags("use <point> tags") == "use <point> tags"
    assert strip_tags('<track mention="x"> <point t="0 seconds"> (1,2) </point> tail') == "  tail"


class TestToDictAndRepr:
    def test_to_dict_schema(self):
        assert pt(120, 200, mention="cup", t=1.5, asset_idx=0).to_dict() == {
            "type": "point",
            "x": 120,
            "y": 200,
            "mention": "cup",
            "t": 1.5,
            "asset_idx": 0,
        }
        assert bbox(1, 2, 3, 4, mention="car", asset_idx=1).to_dict() == {
            "type": "box",
            "top_left": {"x": 1, "y": 2},
            "bottom_right": {"x": 3, "y": 4},
            "mention": "car",
            "asset_idx": 1,
        }
        assert poly([(0, 0), (1, 0), (1, 1)], asset_idx=0).to_dict() == {
            "type": "polygon",
            "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
            "asset_idx": 0,
        }
        assert clip(1.0, 3.2, mention="shot", asset_idx=0).to_dict() == {
            "type": "clip",
            "at": 1.0,
            "until": 3.2,
            "mention": "shot",
            "asset_idx": 0,
        }
        tr = Track([bbox(1, 2, 3, 4, t=0.0)], "red ball", asset_idx=0, complete=False)
        assert tr.to_dict() == {
            "type": "track",
            "mention": "red ball",
            "asset_idx": 0,
            "points": [
                {
                    "type": "box",
                    "top_left": {"x": 1, "y": 2},
                    "bottom_right": {"x": 3, "y": 4},
                    "t": 0.0,
                    "asset_idx": 0,
                }
            ],
            "complete": False,
        }
        coll = collection([pt(1, 2), clip(2.0)], mention="objects", asset_idx=0)
        assert coll.to_dict() == {
            "type": "collection",
            "mention": "objects",
            "asset_idx": 0,
            "points": [{"type": "point", "x": 1, "y": 2, "asset_idx": 0}, {"type": "clip", "at": 2.0, "asset_idx": 0}],
        }
        assert ClipTimestamp(at=1.0).to_dict() == {"at": 1.0}

    def test_reprs_show_asset_idx_and_completeness(self):
        assert repr(pt(1, 2, asset_idx=0)) == "SinglePoint(x=1, y=2, asset_idx=0)"
        assert "asset_idx=1" in repr(bbox(1, 2, 3, 4, asset_idx=1))
        assert "asset_idx=0" in repr(poly([(0, 0), (1, 0), (1, 1)], asset_idx=0))
        assert repr(clip(1.0, asset_idx=0)) == "Clip(timestamp=ClipTimestamp(at=1.0), asset_idx=0)"
        assert repr(Track([pt(1, 1, t=0.0)], "x", complete=False)) == "Track(points=1, mention='x', complete=False)"
        assert repr(Collection([], asset_idx=0, complete=False)) == (
            "Collection(points=0, mention=None, t=None, asset_idx=0, complete=False)"
        )

    def test_track_factory_validates_waypoints(self):
        with pytest.raises(ValueError):
            track([pt(1, 1, t=0.0), bbox(1, 1, 2, 2, t=1.0)])
        with pytest.raises(TypeError):
            track([clip(1.0)])  # type: ignore[list-item]

    def test_track_waypoints_must_name_one_asset(self):
        with pytest.raises(ValueError, match="differ"):
            track([pt(1, 1, t=0.0, asset_idx=0), pt(2, 2, t=1.0, asset_idx=1)])
        with pytest.raises(ValueError, match="differ"):
            track([pt(1, 1, t=0.0, asset_idx=0)], asset_idx=1)
        with pytest.raises(ValueError, match="differ"):  # the serializer refuses markup the server rejects
            PointParser.serialize(Track([pt(1, 1, t=0.0, asset_idx=0)], asset_idx=1))
        assert track([pt(1, 1, t=0.0, asset_idx=0), pt(2, 2, t=1.0)], asset_idx=0).asset_idx == 0


# ---------------------------------------------------------------------------
# Containers push asset_idx down to their children (DESIGN §12.10)
# ---------------------------------------------------------------------------

# The owner's example: plain children, an explicit 0 override, a clip and a track, under a collection's selector.
OWNER_EXAMPLE = (
    '<collection mention="cup" asset_idx="1"> <point_box> (10,10) (20,20) </point_box> '
    '<point_box asset_idx="0"> (30,30) (40,40) </point_box> <clip t="1 seconds 2 seconds" /> '
    '<track> <point_box t="0.0 seconds"> (1,1) (2,2) </point_box> </track> </collection>'
)


class TestContainerAssetIdxPushDown:
    @pytest.mark.parametrize(
        "parse",
        [
            lambda text: parse_text(text)[0]["value"],
            lambda text: parse_annotations(text).segments[0]["value"],
            lambda text: parse_annotations(text, strict=True).segments[0]["value"],
            lambda text: collect_annotations(text).parsed[0]["value"],
        ],
        ids=["parse_text", "parse_annotations", "strict", "collect_annotations"],
    )
    def test_every_parse_path_sets_it_in_the_tree(self, parse):
        coll = parse(OWNER_EXAMPLE)
        assert [child.asset_idx for child in coll.points] == [1, 0, 1, 1]
        assert [waypoint.asset_idx for waypoint in coll.points[3].points] == [1]
        # mention inheritance is unchanged: only flattened copies get it
        assert coll.points[0].mention is None and coll.points[3].mention is None

    def test_flattened_views_and_scanned_leaves_agree(self):
        result = collect_annotations(OWNER_EXAMPLE)
        assert [b.asset_idx for b in result.boxes] == [1, 0, 1]
        assert [(c.asset_idx, c.mention) for c in result.clips] == [(1, "cup")]
        assert [(t.asset_idx, [w.asset_idx for w in t.points]) for t in result.tracks] == [(1, [1])]
        assert [b.asset_idx for b in extract_points(OWNER_EXAMPLE)] == [1, 0, 1]
        assert [w.asset_idx for t in extract_tracks(OWNER_EXAMPLE) for w in t.points] == [1]
        assert [c.asset_idx for c in extract_clips(OWNER_EXAMPLE)] == [1]
        leaves = scan_leaves(OWNER_EXAMPLE)
        assert [leaf["context"]["asset_idx"] for leaf in leaves] == [1, 0, 1, 1]
        assert [leaf["value"].asset_idx for leaf in leaves] == [1, 0, 1, 1]

    @pytest.mark.parametrize(
        "text",
        [
            '<track> <point t="0 seconds" asset_idx="0"> (1,1) </point> <point t="1 seconds"> (2,2) </point> </track>',
            '<collection asset_idx="1"> <track> <point t="0 seconds" asset_idx="0"> (1,1) </point>'
            ' <point t="1 seconds"> (2,2) </point> </track> </collection>',
        ],
        ids=["top_level", "in_collection"],
    )
    def test_a_track_takes_its_waypoints_selector_in_every_view(self, text):
        # The track has no selector of its own, so it takes its waypoints' shared one (not the collection's).
        assert [p.asset_idx for p in collect_annotations(text).points] == [0, 0]
        assert [w.asset_idx for w in extract_tracks(text)[0].points] == [0, 0]
        for prefix in (text, text[: text.index("</track>")]):  # closed, and still streaming
            leaves = scan_leaves(prefix)
            assert [leaf["context"]["asset_idx"] for leaf in leaves] == [0, 0]
            assert [leaf["value"].asset_idx for leaf in leaves] == [0, 0]

    def test_to_dict_shows_the_children_values(self):
        data = parse_text(OWNER_EXAMPLE)[0]["value"].to_dict()
        assert [child.get("asset_idx") for child in data["points"]] == [1, 0, 1, 1]
        assert [waypoint["asset_idx"] for waypoint in data["points"][3]["points"]] == [1]

    def test_serialization_stays_canonical(self):
        assert PointParser.serialize(parse_text(OWNER_EXAMPLE)[0]["value"]) == OWNER_EXAMPLE
        # Selectors that repeat the container's are dropped; track waypoints never write one.
        redundant = (
            '<collection asset_idx="2"> <point asset_idx="2"> (1,2) </point> <track asset_idx="2"> '
            '<point t="0 seconds" asset_idx="2"> (3,4) </point> </track> </collection>'
        )
        assert PointParser.serialize(parse_text(redundant)[0]["value"]) == (
            '<collection asset_idx="2"> <point> (1,2) </point> <track> <point t="0.0 seconds"> (3,4) </point> '
            "</track> </collection>"
        )
        # A value that differs from the container's is written, at every level.
        nested = collection([track([pt(1, 1, t=0.0)], asset_idx=0), pt(2, 2, asset_idx=3), clip(1.0)], asset_idx=1)
        assert PointParser.serialize(nested) == (
            '<collection asset_idx="1"> <track asset_idx="0"> <point t="0.0 seconds"> (1,1) </point> </track> '
            '<point asset_idx="3"> (2,2) </point> <clip t="1 seconds" /> </collection>'
        )
        with pytest.raises(ValueError, match="invalid_asset_idx"):  # a repeated value is still validated
            PointParser.serialize(collection([pt(1, 2, asset_idx=True)], asset_idx=1))

    @pytest.mark.parametrize(
        "build",
        [lambda children: Collection(children, asset_idx=2), lambda children: collection(children, asset_idx=2)],
        ids=["Collection", "collection"],
    )
    def test_construction_pushes_copies_and_never_mutates_inputs(self, build):
        leaf, explicit, waypoint = pt(1, 2), pt(3, 4, asset_idx=0), pt(5, 6, t=0.0)
        tr = Track([waypoint], "ball")
        children = [leaf, explicit, tr]

        coll = build(children)

        assert [child.asset_idx for child in coll.points] == [2, 0, 2]
        assert coll.points[2].points[0].asset_idx == 2
        assert coll.points[1] is explicit  # an explicit value (0 included) wins
        assert coll.points[0] is not leaf and coll.points[2] is not tr
        assert children[0] is leaf and children[2] is tr
        assert leaf.asset_idx is None and tr.asset_idx is None
        assert tr.points[0] is waypoint and waypoint.asset_idx is None

    def test_nested_containers_propagate_the_nearest_value(self):
        inner = collection([pt(1, 1), track([pt(2, 2, t=0.0)])])
        outer = collection([inner, collection([pt(3, 3)], asset_idx=0)], asset_idx=4)
        assert [child.asset_idx for child in outer.points[0].points] == [4, 4]
        assert outer.points[0].points[1].points[0].asset_idx == 4
        assert outer.points[1].points[0].asset_idx == 0
        assert inner.points[0].asset_idx is None

    @pytest.mark.parametrize("build", [Track, track], ids=["Track", "track"])
    def test_tracks_push_their_selector_onto_waypoints(self, build):
        waypoints = [pt(1, 1, t=0.0), pt(2, 2, t=1.0, asset_idx=3)]
        tr = build(waypoints, asset_idx=3)
        assert [w.asset_idx for w in tr.points] == [3, 3]
        assert waypoints[0].asset_idx is None
        # Without its own selector a track takes its waypoints' shared one (as the parser does), so it keeps it
        # inside a collection with another.
        hoisted = build([pt(1, 1, t=0.0, asset_idx=0), pt(2, 2, t=1.0)])
        assert (hoisted.asset_idx, [w.asset_idx for w in hoisted.points]) == (0, [0, 0])
        coll = collection([hoisted], asset_idx=1)
        assert coll.points[0] is hoisted
        assert _values(parse_annotations(PointParser.serialize(coll), strict=True).segments) == [coll]

    def test_setting_a_container_selector_later_is_not_propagated(self):
        coll = collection([pt(1, 2)])
        coll.asset_idx = 4
        assert coll.points[0].asset_idx is None
        # The markup still says it: the child inherits it when parsed back.
        assert collect_annotations(PointParser.serialize(coll)).points[0].asset_idx == 4

    def test_replace_does_not_retarget_the_children(self):
        parsed = parse_text(
            '<collection asset_idx="3"> <point> (1,2) </point> <track> <point t="0 seconds"> (3,4) </point> </track>'
            " </collection>"
        )[0]["value"]
        moved = replace(parsed, asset_idx=5)  # the children keep the value pushed when the tree was built
        assert [child.asset_idx for child in moved.points] == [3, 3]
        assert PointParser.serialize(moved).startswith('<collection asset_idx="5"> <point asset_idx="3">')
        tr = track([pt(1, 1, t=0.0)], asset_idx=2)
        assert replace(tr, asset_idx=None).asset_idx == 2  # re-taken from the waypoints
        with pytest.raises(ValueError, match="differ"):
            PointParser.serialize(replace(tr, asset_idx=4))
        # Rebuilding with the descendants' selectors cleared re-targets the whole tree.
        rebuilt = track([replace(w, asset_idx=None) for w in tr.points], asset_idx=4)
        assert [w.asset_idx for w in rebuilt.points] == [4]

    @pytest.mark.parametrize("wrap", [iter, tuple], ids=["generator", "tuple"])
    def test_containers_keep_every_child_of_any_iterable(self, wrap):
        coll = Collection(wrap([pt(0, 0), pt(1, 1), pt(2, 2)]), asset_idx=1)
        assert coll.points == [pt(0, 0, asset_idx=1), pt(1, 1, asset_idx=1), pt(2, 2, asset_idx=1)]
        tr = Track(wrap([pt(0, 0, t=0.0, asset_idx=2), pt(1, 1, t=1.0)]), "ball")
        assert (tr.asset_idx, tr.points) == (2, [pt(0, 0, t=0.0, asset_idx=2), pt(1, 1, t=1.0, asset_idx=2)])
        assert Collection(wrap([pt(3, 3)])).points == [pt(3, 3)]


# ---------------------------------------------------------------------------
# Round trip: parse → serialize → parse loses nothing
# ---------------------------------------------------------------------------

# (input, expects, canonical serialization, tokens intentionally dropped)
ROUND_TRIP_CASES = [
    ("<point> (1,2) </point>", None, "<point> (1,2) </point>", ()),
    ('<point mention="cup"> (1,2) </point>', None, '<point mention="cup"> (1,2) </point>', ()),
    ('<point mention="cup" t="1.5"> (1,2) </point>', None, '<point mention="cup" t="1.5 seconds"> (1,2) </point>', ()),
    ('<point t="1.5"> (1,2) </point>', None, '<point t="1.5 seconds"> (1,2) </point>', ()),
    ('<point t="1.5s"> (1,2) </point>', None, '<point t="1.5 seconds"> (1,2) </point>', ()),
    ('<point asset_idx="0"> (1,2) </point>', None, '<point asset_idx="0"> (1,2) </point>', ()),
    (
        '<point_box mention="a &amp; &quot;b&quot;" asset_idx="1"> (1,2) (3,4) </point_box>',
        None,
        '<point_box mention="a &amp; &quot;b&quot;" asset_idx="1"> (1,2) (3,4) </point_box>',
        (),
    ),
    (
        '<point_box mention="a > b"> (1,2) (3,4) </point_box>',
        None,
        '<point_box mention="a &gt; b"> (1,2) (3,4) </point_box>',
        (),
    ),
    (
        '<polygon mention="p" t="0 seconds"> (0,0) (10,0) (10,10) </polygon>',
        None,
        '<polygon mention="p" t="0.0 seconds"> (0,0) (10,0) (10,10) </polygon>',
        (),
    ),
    ('<point mention=""> (1,2) </point>', None, "<point> (1,2) </point>", ("mention=",)),
    ("<POINT> ( 1 , 2 ) </POINT>", None, "<point> (1,2) </point>", ()),
    (
        '<point_box mention="car" t="1.5s"> 10  20  30  40 </point_box>',
        None,
        '<point_box mention="car" t="1.5 seconds"> (10,20) (30,40) </point_box>',
        (),
    ),
    (
        '<point_box mention=car asset_idx=\'1\' index="image_0" label="L"> (1,2) (3,4) </point_box>',
        None,
        '<point_box mention="car" asset_idx="1"> (1,2) (3,4) </point_box>',
        (),
    ),
    (
        '<collection mention="c" asset_idx="0"> <point> (1,2) </point> <point asset_idx="1"> (3,4) </point> </collection>',
        None,
        '<collection mention="c" asset_idx="0"> <point> (1,2) </point> <point asset_idx="1"> (3,4) </point> </collection>',
        (),
    ),
    (
        '<collection mention="cups" asset_idx="2"> <point_box> (10,10) (20,20) </point_box> <point_box asset_idx="0"> (30,30) (40,40) </point_box> </collection>',
        None,
        '<collection mention="cups" asset_idx="2"> <point_box> (10,10) (20,20) </point_box> <point_box asset_idx="0"> (30,30) (40,40) </point_box> </collection>',
        (),
    ),
    (
        '<collection mention="c"> <point> (1,2) </point> <clip t="1 seconds" /> </collection>',
        None,
        '<collection mention="c"> <point> (1,2) </point> <clip t="1 seconds" /> </collection>',
        (),
    ),
    (
        '<collection mention="c"> <track> <point t="1 seconds"> (1,2) </point> <point t="2 seconds"> (2,2) </point> </track> </collection>',
        None,
        '<collection mention="c"> <track> <point t="1.0 seconds"> (1,2) </point> <point t="2.0 seconds"> (2,2) </point> </track> </collection>',
        (),
    ),
    (
        '<collection mention="balls" asset_idx="3">\n <track>\n <point_box t="0.0 seconds"> (1,2) (3,4) </point_box>\n</track>\n <track asset_idx="0">\n <point_box t="0.0 seconds"> (5,6) (7,8) </point_box>\n</track>\n</collection>',
        None,
        '<collection mention="balls" asset_idx="3"> <track> <point_box t="0.0 seconds"> (1,2) (3,4) </point_box> </track> <track asset_idx="0"> <point_box t="0.0 seconds"> (5,6) (7,8) </point_box> </track> </collection>',
        (),
    ),
    (
        '<track mention="ball" asset_idx="0">\n <point_box t="0.0 seconds"> (1,2) (3,4) </point_box>\n <point_box t="0.5 seconds"> (2,2) (4,4) </point_box>\n</track>',
        None,
        '<track mention="ball" asset_idx="0"> <point_box t="0.0 seconds"> (1,2) (3,4) </point_box> <point_box t="0.5 seconds"> (2,2) (4,4) </point_box> </track>',
        (),
    ),
    (
        '<track mention="x"> <point_box t="0.0 seconds" asset_idx="1"> (1,2) (3,4) </point_box> </track>',
        None,
        '<track mention="x" asset_idx="1"> <point_box t="0.0 seconds"> (1,2) (3,4) </point_box> </track>',
        (),
    ),
    (
        '<track mention="x"> <point_box mention="y" t="0.0 seconds"> (1,2) (3,4) </point_box> </track>',
        None,
        '<track mention="x"> <point_box mention="y" t="0.0 seconds"> (1,2) (3,4) </point_box> </track>',
        (),
    ),
    ('<clip mention="goal" t="3.2 seconds" />', "clip", '<clip mention="goal" t="3.2 seconds" />', ()),
    (
        '<clip mention="goal" asset_idx="0" t="1.0 seconds 3.2 seconds" />',
        "clip",
        '<clip mention="goal" asset_idx="0" t="1 seconds 3.2 seconds" />',
        (),
    ),
    (
        '<clip mention="pour" t="2s 4.5s" asset_idx="1" />',
        None,
        '<clip mention="pour" asset_idx="1" t="2 seconds 4.5 seconds" />',
        (),
    ),
    ('<clip mention="x" t="1 seconds"></clip>', "clip", '<clip mention="x" t="1 seconds" />', ()),
    (OWNER_EXAMPLE, None, OWNER_EXAMPLE, ()),
    (
        '<collection mention="c" asset_idx="1"> <point asset_idx="1"> (1,2) </point> <track asset_idx="1"> <point t="0.0 seconds" asset_idx="1"> (3,4) </point> </track> </collection>',
        None,
        '<collection mention="c" asset_idx="1"> <point> (1,2) </point> <track> <point t="0.0 seconds"> (3,4) </point> </track> </collection>',
        (),
    ),
]
# A collection's legacy `t` is pushed down to its children: the tree changes but the flattened annotations don't.
PUSH_DOWN_CASES = [
    (
        '<collection t="2"> <point> (1,2) </point> </collection>',
        None,
        '<collection> <point t="2.0 seconds"> (1,2) </point> </collection>',
        (),
    ),
]
_TOKENS = (
    "asset_idx=",
    "t=",
    "seconds",
    "mention=",
    "<track",
    "</track>",
    "<clip",
    "<collection",
    "</collection>",
    "<point",
    "<polygon",
)


def _flat(text, expects):
    result = collect_annotations(text, expects=expects, strict=True)
    return result.points, result.boxes, result.polygons, result.clips, result.tracks


@pytest.mark.parametrize(("text", "expects", "canonical", "dropped"), ROUND_TRIP_CASES + PUSH_DOWN_CASES)
def test_round_trip_loses_nothing(text, expects, canonical, dropped):
    objs = _values(parse_annotations(text, expects=expects, strict=True).segments)
    serialized = " ".join(PointParser.serialize(obj) for obj in objs)
    assert serialized == canonical
    assert _flat(serialized, expects) == _flat(text, expects)
    if (text, expects, canonical, dropped) not in PUSH_DOWN_CASES:
        assert _values(parse_annotations(serialized, expects=expects, strict=True).segments) == objs
    # canonical output is a fixed point
    assert (
        " ".join(PointParser.serialize(o) for o in _values(parse_annotations(serialized, expects=expects).segments))
        == serialized
    )
    lost = [tok for tok in _TOKENS if tok in text and tok not in serialized and tok not in dropped]
    assert lost == []


def test_round_trip_of_objects_with_zero_values_and_escaped_mentions():
    objs = [
        pt(0, 0, mention="a \"q\" <b> & 'c'", t=0.0, asset_idx=0),
        bbox(0, 0, 1000, 1000, t=0.0, asset_idx=0),
        poly([(0, 0), (1, 0), (1, 1)], asset_idx=0),
        clip(0.0, asset_idx=0),
        track([pt(1, 1, t=0.0), pt(2, 2, t=0.5)], mention="x", asset_idx=0),
        collection([pt(1, 1, asset_idx=0), clip(1.5, 2.5)], mention="c", asset_idx=1),
    ]
    text = " ".join(PointParser.serialize(obj) for obj in objs)
    assert _values(parse_annotations(text, strict=True).segments) == objs
