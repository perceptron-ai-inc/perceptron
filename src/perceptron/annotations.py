"""Utilities for working with annotation examples (points, boxes, polygons, clips, tracks, collections)."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .errors import BadRequestError, ParseError
from .pointing.parser import (
    PointParser,
    _parse_time_value,
    parse_annotations,
    parse_text,  # noqa: F401 - importable from here, as in 0.3.5
)
from .pointing.types import (
    BoundingBox,
    Clip,
    Collection,
    Polygon,
    SinglePoint,
    Track,
    _track_asset_idx,
)
from .pointing.types import (
    bbox as make_bbox,
)
from .pointing.types import (
    clip as make_clip,
)
from .pointing.types import (
    collection as make_collection,
)
from .pointing.types import (
    poly as make_polygon,
)
from .pointing.types import (
    pt as make_point,
)
from .pointing.types import (
    track as make_track,
)

__all__ = [
    "annotate_image",
    "canonicalize_text_collections",
    "coerce_annotation",
    "serialize_annotations",
]


AnnotationSpec = Any

_CHILD_KEYS = ("points", "children", "items", "collection")


def _first_present(spec: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = spec.get(key)
        if value is not None:
            return value
    return None


def _mention(spec: Mapping[str, Any]) -> str | None:
    return _first_present(spec, "label", "mention")


def _seconds(value: Any) -> float | None:
    """A time as seconds (finite, >= 0): a number, or a string such as "1.5", "1.5 seconds" or "1.5s"."""

    if value is None:
        return None
    if isinstance(value, str):
        try:
            return _parse_time_value(value)
        except ParseError as err:
            raise BadRequestError(str(err), code=err.code, details=err.details) from err
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not (math.isfinite(value) and value >= 0):
        raise BadRequestError(
            f"Annotation time must be a finite, non-negative number of seconds, got {value!r}", code="invalid_time"
        )
    return float(value)


def _asset_idx(spec: Mapping[str, Any]) -> int | None:
    value = spec.get("asset_idx")
    if value is None:
        return None
    if isinstance(value, str) and value.strip().isdecimal():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise BadRequestError(f"asset_idx must be a non-negative integer, got {value!r}", code="invalid_asset_idx")
    return int(value)


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
        x1, y1, x2, y2 = coords
        return make_bbox(
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            mention=_mention(spec),
            t=_seconds(spec.get("t")),
            asset_idx=_asset_idx(spec),
        )
    if isinstance(spec, Sequence) and len(spec) == 4:
        x1, y1, x2, y2 = spec
        return make_bbox(int(x1), int(y1), int(x2), int(y2))
    raise BadRequestError(f"Unsupported box spec: {spec!r}")


def _coerce_point(spec: AnnotationSpec) -> SinglePoint:
    if isinstance(spec, SinglePoint):
        return spec
    if isinstance(spec, Mapping):
        if {"x", "y"}.issubset(spec):
            x, y = spec["x"], spec["y"]
        elif "point" in spec:
            x, y = spec["point"]
        else:
            raise BadRequestError("Example point dict must include point or x/y keys")
        return make_point(int(x), int(y), mention=_mention(spec), t=_seconds(spec.get("t")), asset_idx=_asset_idx(spec))
    if isinstance(spec, Sequence) and len(spec) == 2:
        x, y = spec
        return make_point(int(x), int(y))
    raise BadRequestError(f"Unsupported point spec: {spec!r}")


def _coerce_polygon(spec: AnnotationSpec) -> Polygon:
    if isinstance(spec, Polygon):
        return spec
    context: dict[str, Any] = {}
    if isinstance(spec, Mapping):
        coords = _first_present(spec, "coords", "polygon")
        if not coords:
            raise BadRequestError("Example polygon dict must include coords/polygon")
        context = {"mention": _mention(spec), "t": _seconds(spec.get("t")), "asset_idx": _asset_idx(spec)}
    else:
        coords = spec
    if not isinstance(coords, Iterable):
        raise BadRequestError("Polygon coords must be iterable")
    points: list[tuple[int, int]] = []
    for item in coords:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise BadRequestError("Polygon coordinate must be (x, y)")
        x, y = item
        points.append((int(x), int(y)))
    return make_polygon(points, **context)


def _coerce_clip(spec: AnnotationSpec) -> Clip:
    if isinstance(spec, Clip):
        return spec
    if not isinstance(spec, Mapping):
        raise BadRequestError(f"Unsupported clip spec: {spec!r}")
    at = _seconds(_first_present(spec, "at", "start"))
    if at is None:
        raise BadRequestError("Clip dict must include at/start (seconds)")
    until = _seconds(_first_present(spec, "until", "end"))
    return make_clip(at, until, mention=_mention(spec), asset_idx=_asset_idx(spec))


def _coerce_track(spec: AnnotationSpec) -> Track:
    if isinstance(spec, Track):
        return spec
    if not isinstance(spec, Mapping):
        raise BadRequestError(f"Unsupported track spec: {spec!r}")
    child_specs = _first_present(spec, "points", "children")
    if child_specs is None:
        raise BadRequestError("Track spec must include a points/children list")
    waypoints = [coerce_annotation(child) for child in child_specs]
    try:
        return make_track(waypoints, mention=_mention(spec), asset_idx=_asset_idx(spec))
    except (TypeError, ValueError) as err:
        raise BadRequestError(f"Invalid track spec: {err}") from err


def _build_collection(spec: AnnotationSpec) -> Collection:
    if isinstance(spec, Collection):
        return spec
    if isinstance(spec, Mapping):
        child_specs = _first_present(spec, *_CHILD_KEYS)
        if child_specs is None:
            raise BadRequestError("Collection spec must include points/children/items list")
        children = [coerce_annotation(child) for child in child_specs]
        return make_collection(children, mention=_mention(spec), t=_seconds(spec.get("t")), asset_idx=_asset_idx(spec))
    if isinstance(spec, Sequence):
        children = [coerce_annotation(child) for child in spec]
        return make_collection(children)
    raise BadRequestError(f"Unsupported collection spec: {spec!r}")


_COERCE_BY_TYPE: dict[str, Callable[[AnnotationSpec], Any]] = {
    "point": _coerce_point,
    "pt": _coerce_point,
    "box": _coerce_bbox,
    "bbox": _coerce_bbox,
    "point_box": _coerce_bbox,
    "polygon": _coerce_polygon,
    "collection": _build_collection,
    "clip": _coerce_clip,
    "track": _coerce_track,
}


def coerce_annotation(spec: AnnotationSpec) -> Any:
    if isinstance(spec, (SinglePoint, BoundingBox, Polygon, Collection, Clip, Track)):
        return spec
    if isinstance(spec, Mapping):
        type_hint = spec.get("type") or spec.get("kind") or spec.get("point_type")
        if type_hint:
            coerce = _COERCE_BY_TYPE.get(str(type_hint).lower())
            if coerce is None:
                raise BadRequestError(f"Unsupported annotation type: {type_hint!r}")
            return coerce(spec)
        if {"x", "y"}.issubset(spec) or "point" in spec:
            return _coerce_point(spec)
        if spec.get("bbox") is not None or {"x1", "y1", "x2", "y2"}.issubset(spec):
            return _coerce_bbox(spec)
        if spec.get("coords") is not None or spec.get("polygon") is not None:
            return _coerce_polygon(spec)
        if any(key in spec for key in _CHILD_KEYS):
            return _build_collection(spec)
    if isinstance(spec, Sequence):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            return _coerce_point(spec)
        if len(spec) == 4 and all(isinstance(v, (int, float)) for v in spec):
            return _coerce_bbox(spec)
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in spec):
            return _coerce_polygon(spec)
    raise BadRequestError(f"Unsupported annotation spec: {spec!r}")


def _point_sort_key(obj: Any, asset_idx: int | None = None, t: Any = None) -> tuple[float, float, float, int, int]:
    """Canonical order: ``(start, end, asset_idx or +inf, min_y, min_x)``; containers sort by their first child."""

    own_asset = getattr(obj, "asset_idx", None)
    asset_idx = own_asset if own_asset is not None else asset_idx
    asset_rank = asset_idx if asset_idx is not None else math.inf
    if isinstance(obj, Clip):
        at = float(obj.timestamp.at)
        until = float(obj.timestamp.until) if obj.timestamp.until is not None else at
        return (at, until, asset_rank, 0, 0)
    if isinstance(obj, (Collection, Track)):
        if isinstance(obj, Collection) and obj.t is not None:
            t = obj.t
        if not obj.points:
            return (0.0, 0.0, asset_rank, 0, 0)
        return min(_point_sort_key(child, asset_idx, None if isinstance(obj, Track) else t) for child in obj.points)
    own_t = getattr(obj, "t", None)
    start = _seconds(own_t if own_t is not None else t) or 0.0
    if isinstance(obj, BoundingBox):
        return (start, start, asset_rank, obj.top_left.y, obj.top_left.x)
    if isinstance(obj, SinglePoint):
        return (start, start, asset_rank, obj.y, obj.x)
    if isinstance(obj, Polygon) and obj.hull:
        first = min(obj.hull, key=lambda p: (p.y, p.x))
        return (start, start, asset_rank, first.y, first.x)
    return (start, start, asset_rank, 0, 0)


def _canonicalize_track(track: Track) -> Track:
    # Waypoints in time order; the object's mention lives on the track only (and its asset selector is written there).
    if not track.points:
        raise BadRequestError("Invalid track: a track needs at least one waypoint")
    try:
        asset_idx = _track_asset_idx(track.points, track.asset_idx)
    except ValueError as err:
        raise BadRequestError(f"Invalid track: {err}", code="invalid_track_asset") from err
    waypoints = sorted((replace(p, mention=None) for p in track.points), key=_point_sort_key)
    return replace(track, points=waypoints, asset_idx=asset_idx)


def _canonicalize_collection(coll: Collection) -> Collection:
    children: list[Any] = []
    for child in coll.points:
        if isinstance(child, Collection):
            children.append(_canonicalize_collection(child))
        elif isinstance(child, Track):
            children.append(_canonicalize_track(child))
        else:
            children.append(child)
    children.sort(key=lambda child: _point_sort_key(child, coll.asset_idx, coll.t))
    return replace(coll, points=children)


def _ranked(mention_order: Mapping[str, int] | None) -> Callable[[Any], tuple[int, tuple]]:
    def key(obj: Any) -> tuple[int, tuple]:
        rank = 10**6
        if mention_order and obj.mention is not None and obj.mention in mention_order:
            rank = mention_order[obj.mention]
        return (rank, _point_sort_key(obj))

    return key


def serialize_annotations(  # noqa: PLR0913 - public signature
    boxes: Sequence[Any] | None,
    polygons: Sequence[Any] | None,
    points: Sequence[Any] | None,
    collections: Sequence[Any] | None,
    mention_order: Mapping[str, int] | None = None,
    *,
    clips: Sequence[Any] | None = None,
    tracks: Sequence[Any] | None = None,
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
        canonical = [_canonicalize_collection(_build_collection(coll)) for coll in collections]
        canonical.sort(key=_ranked(mention_order))
        tags.extend(PointParser.serialize(coll) for coll in canonical)
    if clips:
        tags.extend(PointParser.serialize(_coerce_clip(c)) for c in clips)
    if tracks:
        canonical_tracks = [_canonicalize_track(_coerce_track(tr)) for tr in tracks]
        canonical_tracks.sort(key=_ranked(mention_order))
        tags.extend(PointParser.serialize(tr) for tr in canonical_tracks)
    return " ".join(tags)


def annotate_image(image_obj: Any, annotations: Any) -> dict[str, Any]:
    buckets: dict[type, list[Any]] = {
        BoundingBox: [],
        Polygon: [],
        SinglePoint: [],
        Collection: [],
        Clip: [],
        Track: [],
    }

    if isinstance(annotations, Mapping):
        for label, child_specs in annotations.items():
            child_objs = [coerce_annotation(child) for child in child_specs]
            buckets[Collection].append(make_collection(child_objs, mention=str(label)))
    else:
        for item in annotations:
            obj = coerce_annotation(item)
            bucket = next((items for kind, items in buckets.items() if isinstance(obj, kind)), None)
            if bucket is None:
                raise BadRequestError(f"Unsupported annotation: {item!r}")
            bucket.append(obj)

    example: dict[str, Any] = {"image": image_obj}
    for key, kind in (("boxes", BoundingBox), ("polygons", Polygon), ("points", SinglePoint), ("clips", Clip)):
        if buckets[kind]:
            example[key] = sorted(buckets[kind], key=_point_sort_key)
    if buckets[Collection]:
        canonical = [_canonicalize_collection(coll) for coll in buckets[Collection]]
        canonical.sort(key=lambda c: ((c.mention or ""), _point_sort_key(c)))
        example["collections"] = canonical
    if buckets[Track]:
        example["tracks"] = sorted((_canonicalize_track(tr) for tr in buckets[Track]), key=_point_sort_key)
    return example


def canonicalize_text_collections(text: str | None) -> str | None:
    """Rewrite complete ``<collection>``/``<track>`` markup in ``text`` canonically; everything else is untouched.

    Malformed or unclosed markup is left exactly as written.
    """

    if not text:
        return text
    lowered = text.lower()
    if "<collection" not in lowered and "<track" not in lowered:
        return text
    parsed = parse_annotations(text)
    parts: list[str] = []
    idx = 0
    for seg in parsed.segments:
        value = seg.get("value")
        if not isinstance(value, (Collection, Track)) or not value.complete:
            continue
        start, end = seg["span"]["start"], seg["span"]["end"]
        if any(start <= err["span"]["start"] < end for err in parsed.errors):
            continue  # a malformed child was skipped: keep the markup as written rather than drop it
        canonical = _canonicalize_collection(value) if isinstance(value, Collection) else _canonicalize_track(value)
        parts.append(text[idx:start])
        parts.append(PointParser.serialize(canonical))
        idx = end
    parts.append(text[idx:])
    return "".join(parts)
