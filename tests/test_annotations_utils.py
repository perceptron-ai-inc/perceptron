import pytest

from perceptron.annotations import (
    annotate_image,
    canonicalize_text_collections,
    coerce_annotation,
    serialize_annotations,
)
from perceptron.errors import BadRequestError
from perceptron.pointing.parser import PointParser
from perceptron.pointing.types import Polygon, bbox, collection, pt


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
