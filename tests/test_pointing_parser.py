import pytest

from perceptron.errors import ParseError
from perceptron.pointing.parser import (
    PointParser,
    extract_points,
    parse_text,
    strip_tags,
)
from perceptron.pointing.types import (
    BoundingBox,
    Clip,
    ClipTimestamp,
    Collection,
    Polygon,
    SinglePoint,
    collection,
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
        assert segs[0]["value"] == Clip(
            timestamp=ClipTimestamp(at=10.0, until=20.0), mention="action"
        )

    def test_clip_mode_ignores_point_tags(self):
        """Clip mode does not scan geometry tags — they remain in the surrounding text segment."""
        text = '<point>(1,2)</point> and <clip t=1.0/>'
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
        text = 'before <clip t=1.5/> after'
        segs = parse_text(text)
        assert len(segs) == 1
        assert segs[0]["kind"] == "text"
        assert "<clip" in segs[0]["text"]
