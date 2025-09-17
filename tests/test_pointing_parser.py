from perceptron.pointing.types import SinglePoint, BoundingBox, Polygon, Collection, collection
from perceptron.pointing.parser import (
    PointParser,
    ReasoningStreamCleaner,
    extract_points,
    extract_reasoning,
    parse_text,
    strip_tags,
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


def test_point_parser_escapes_and_parses_mentions():
    pt = SinglePoint(1, 2, mention='door "A" & B', t=1.5)
    tag = PointParser.serialize(pt)
    assert 'door &quot;A&quot; &amp; B' in tag
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
    child_poly = Polygon([SinglePoint(0, 0), SinglePoint(1, 0), SinglePoint(1, 1)], mention="triangle", t=1.1)
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


def test_extract_reasoning_returns_clean_text_and_segments():
    text = "Prefix <think> first thought </think> middle <think>second</think> end"
    extraction = extract_reasoning(text)
    assert extraction.text == "Prefix middle end"
    assert extraction.reasoning == ["first thought", "second"]


def test_reasoning_stream_cleaner_handles_partial_chunks():
    cleaner = ReasoningStreamCleaner()
    chunk1, reasons1 = cleaner.consume("Hello <think> inter")
    assert chunk1 == "Hello "
    assert reasons1 == []

    chunk2, reasons2 = cleaner.consume("nal </think> world")
    assert chunk2 == " world"
    assert reasons2 == ["internal"]

    chunk3, reasons3 = cleaner.consume("!")
    assert chunk3 == "!"
    assert reasons3 == []
