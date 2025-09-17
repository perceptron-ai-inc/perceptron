"""Utilities for working with annotation examples (points, boxes, polygons, collections)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .errors import BadRequestError
from .pointing.types import (
    BoundingBox,
    Collection,
    Polygon,
    SinglePoint,
    bbox as make_bbox,
    collection as make_collection,
    poly as make_polygon,
    pt as make_point,
)
from .pointing.parser import PointParser, parse_text

__all__ = [
    "annotate_image",
    "coerce_annotation",
    "serialize_annotations",
    "canonicalize_text_collections",
]


AnnotationSpec = Any


def _coerce_bbox(spec: AnnotationSpec) -> BoundingBox:
    if isinstance(spec, BoundingBox):
        return spec
    if isinstance(spec, Mapping):
        if "bbox" in spec:
            coords = spec["bbox"]
        elif {"x1", "y1", "x2", "y2"}.issubset(spec):
            coords = (spec["x1"], spec["y1"], spec["x2"], spec["y2"])
        else:
            raise BadRequestError("Example box dict must include bbox tuple or x1/y1/x2/y2 keys")
        mention = spec.get("label") or spec.get("mention")
        x1, y1, x2, y2 = coords
        return make_bbox(int(x1), int(y1), int(x2), int(y2), mention=mention)
    if isinstance(spec, Sequence) and len(spec) == 4:
        x1, y1, x2, y2 = spec
        return make_bbox(int(x1), int(y1), int(x2), int(y2))
    raise BadRequestError(f"Unsupported box spec: {spec!r}")


def _coerce_point(spec: AnnotationSpec) -> SinglePoint:
    if isinstance(spec, SinglePoint):
        return spec
    if isinstance(spec, Mapping):
        if {"x", "y"}.issubset(spec):
            return make_point(int(spec["x"]), int(spec["y"]), mention=spec.get("label") or spec.get("mention"))
        if "point" in spec:
            x, y = spec["point"]
            return make_point(int(x), int(y), mention=spec.get("label") or spec.get("mention"))
        raise BadRequestError("Example point dict must include point or x/y keys")
    if isinstance(spec, Sequence) and len(spec) == 2:
        x, y = spec
        return make_point(int(x), int(y))
    raise BadRequestError(f"Unsupported point spec: {spec!r}")


def _coerce_polygon(spec: AnnotationSpec) -> Polygon:
    if isinstance(spec, Polygon):
        return spec
    if isinstance(spec, Mapping):
        coords = spec.get("coords") or spec.get("polygon")
        if not coords:
            raise BadRequestError("Example polygon dict must include coords/polygon")
        mention = spec.get("label") or spec.get("mention")
    else:
        coords = spec
        mention = None
    if not isinstance(coords, Iterable):
        raise BadRequestError("Polygon coords must be iterable")
    points: list[tuple[int, int]] = []
    for item in coords:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise BadRequestError("Polygon coordinate must be (x, y)")
        x, y = item
        points.append((int(x), int(y)))
    return make_polygon(points, mention=mention)


def _build_collection(spec: AnnotationSpec) -> Collection:
    if isinstance(spec, Collection):
        return spec
    if isinstance(spec, Mapping):
        mention = spec.get("label") or spec.get("mention")
        t = spec.get("t")
        child_specs = (
            spec.get("points")
            or spec.get("children")
            or spec.get("items")
            or spec.get("collection")
        )
        if child_specs is None:
            raise BadRequestError("Collection spec must include points/children/items list")
        children = [coerce_annotation(child) for child in child_specs]
        return make_collection(children, mention=mention, t=t)
    if isinstance(spec, Sequence):
        children = [coerce_annotation(child) for child in spec]
        return make_collection(children)
    raise BadRequestError(f"Unsupported collection spec: {spec!r}")


def coerce_annotation(spec: AnnotationSpec) -> Any:
    if isinstance(spec, (SinglePoint, BoundingBox, Polygon, Collection)):
        return spec
    if isinstance(spec, Mapping):
        type_hint = spec.get("type") or spec.get("kind") or spec.get("point_type")
        if type_hint:
            kind = str(type_hint).lower()
            if kind in {"point", "pt"}:
                return _coerce_point(spec)
            if kind in {"box", "bbox", "point_box"}:
                return _coerce_bbox(spec)
            if kind == "polygon":
                return _coerce_polygon(spec)
            if kind == "collection":
                return _build_collection(spec)
        if {"x", "y"}.issubset(spec) or "point" in spec:
            return _coerce_point(spec)
        if spec.get("bbox") is not None or {"x1", "y1", "x2", "y2"}.issubset(spec):
            return _coerce_bbox(spec)
        if spec.get("coords") is not None or spec.get("polygon") is not None:
            return _coerce_polygon(spec)
        if any(key in spec for key in ("points", "children", "items", "collection")):
            return _build_collection(spec)
    if isinstance(spec, Sequence):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            return _coerce_point(spec)
        if len(spec) == 4 and all(isinstance(v, (int, float)) for v in spec):
            return _coerce_bbox(spec)
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in spec):
            return _coerce_polygon(spec)
    raise BadRequestError(f"Unsupported annotation spec: {spec!r}")


def _point_sort_key(obj: Any) -> tuple[int, int]:
    if isinstance(obj, BoundingBox):
        return (obj.top_left.y, obj.top_left.x)
    if isinstance(obj, SinglePoint):
        return (obj.y, obj.x)
    if isinstance(obj, Polygon):
        hull = obj.hull
        if not hull:
            return (0, 0)
        first = min(hull, key=lambda p: (p.y, p.x))
        return (first.y, first.x)
    if isinstance(obj, Collection):
        if not obj.points:
            return (0, 0)
        first = min(obj.points, key=_point_sort_key)
        return _point_sort_key(first)
    return (0, 0)


def _canonicalize_collection(coll: Collection) -> Collection:
    children: list[Any] = []
    for child in coll.points:
        if isinstance(child, Collection):
            children.append(_canonicalize_collection(child))
        else:
            children.append(child)
    children.sort(key=_point_sort_key)
    return make_collection(children, mention=coll.mention, t=coll.t)


def serialize_annotations(
    boxes: Sequence[Any] | None,
    polygons: Sequence[Any] | None,
    points: Sequence[Any] | None,
    collections: Sequence[Any] | None,
    mention_order: Mapping[str, int] | None = None,
) -> str:
    tags: list[str] = []
    if boxes:
        for b in boxes:
            tags.append(PointParser.serialize(_coerce_bbox(b)))
    if polygons:
        for poly in polygons:
            tags.append(PointParser.serialize(_coerce_polygon(poly)))
    if points:
        for pt in points:
            tags.append(PointParser.serialize(_coerce_point(pt)))
    if collections:
        canonical: list[Collection] = []
        for coll in collections:
            canonical.append(_canonicalize_collection(_build_collection(coll)))

        def coll_key(c: Collection) -> tuple[int, tuple[int, int]]:
            rank = 10 ** 6
            if mention_order and c.mention is not None and c.mention in mention_order:
                rank = mention_order[c.mention]
            return (rank, _point_sort_key(c))

        canonical.sort(key=coll_key)
        for coll in canonical:
            tags.append(PointParser.serialize(coll))
    return " ".join(tags)


def annotate_image(image_obj: Any, annotations: Any) -> dict[str, Any]:
    boxes: list[BoundingBox] = []
    polys: list[Polygon] = []
    points: list[SinglePoint] = []
    collections: list[Collection] = []

    if isinstance(annotations, Mapping):
        for label, child_specs in annotations.items():
            child_objs = [coerce_annotation(child) for child in child_specs]
            collections.append(make_collection(child_objs, mention=str(label)))
    else:
        for item in annotations:
            obj = coerce_annotation(item)
            if isinstance(obj, BoundingBox):
                boxes.append(obj)
            elif isinstance(obj, Polygon):
                polys.append(obj)
            elif isinstance(obj, SinglePoint):
                points.append(obj)
            elif isinstance(obj, Collection):
                collections.append(obj)
            else:
                raise BadRequestError(f"Unsupported annotation: {item!r}")

    example: dict[str, Any] = {"image": image_obj}
    if boxes:
        boxes.sort(key=_point_sort_key)
        example["boxes"] = boxes
    if polys:
        polys.sort(key=_point_sort_key)
        example["polygons"] = polys
    if points:
        points.sort(key=_point_sort_key)
        example["points"] = points
    if collections:
        canonical = [_canonicalize_collection(coll) for coll in collections]
        canonical.sort(key=lambda c: ((c.mention or ""), _point_sort_key(c)))
        example["collections"] = canonical
    return example


def canonicalize_text_collections(text: str | None) -> str | None:
    if not text or "<collection" not in text:
        return text
    segments = parse_text(text)
    parts: list[str] = []
    idx = 0
    for seg in segments:
        start = seg["span"]["start"]
        end = seg["span"]["end"]
        if start > idx:
            parts.append(text[idx:start])
        if seg["kind"] == "collection":
            canonical = _canonicalize_collection(seg["value"])
            parts.append(PointParser.serialize(canonical))
        else:
            parts.append(text[start:end])
        idx = end
    if idx < len(text):
        parts.append(text[idx:])
    return "".join(parts)
