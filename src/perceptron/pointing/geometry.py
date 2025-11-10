"""Geometry helpers for converting normalized annotations into pixel space."""

from __future__ import annotations

from collections.abc import Sequence

from .types import BoundingBox, Collection, Polygon, SinglePoint

NORMALIZED_COORD_MAX = 1000.0

Annotation = SinglePoint | BoundingBox | Polygon | Collection


def _require_dimension(name: str, value: int | float) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError) as err:  # pragma: no cover - defensive
        raise TypeError(f"Image {name} must be an integer, got {value!r}") from err
    if dimension <= 0:
        raise ValueError(f"Image {name} must be a positive integer; got {dimension}")
    return dimension


def _normalize_dimensions(width: int | float, height: int | float) -> tuple[int, int]:
    return _require_dimension("width", width), _require_dimension("height", height)


def _scale_component(value: int | float, dimension: int, clamp: bool) -> int:
    scaled = round((float(value) / NORMALIZED_COORD_MAX) * dimension)
    if not clamp:
        return scaled
    upper = dimension - 1
    if scaled < 0:
        return 0
    if scaled > upper:
        return upper
    return scaled


def _scale_point(point: SinglePoint, dims: tuple[int, int], clamp: bool) -> SinglePoint:
    width, height = dims
    return SinglePoint(
        x=_scale_component(point.x, width, clamp),
        y=_scale_component(point.y, height, clamp),
        mention=point.mention,
        t=point.t,
    )


def _scale_box(box: BoundingBox, dims: tuple[int, int], clamp: bool) -> BoundingBox:
    scaled_top_left = _scale_point(box.top_left, dims, clamp)
    scaled_bottom_right = _scale_point(box.bottom_right, dims, clamp)
    left_x = min(scaled_top_left.x, scaled_bottom_right.x)
    right_x = max(scaled_top_left.x, scaled_bottom_right.x)
    top_y = min(scaled_top_left.y, scaled_bottom_right.y)
    bottom_y = max(scaled_top_left.y, scaled_bottom_right.y)
    return BoundingBox(
        top_left=SinglePoint(left_x, top_y),
        bottom_right=SinglePoint(right_x, bottom_y),
        mention=box.mention,
        t=box.t,
    )


def _scale_polygon(poly: Polygon, dims: tuple[int, int], clamp: bool) -> Polygon:
    scaled_hull = [_scale_point(point, dims, clamp) for point in poly.hull]
    return Polygon(hull=scaled_hull, mention=poly.mention, t=poly.t)


def _scale_collection(coll: Collection, dims: tuple[int, int], clamp: bool) -> Collection:
    scaled_children = [_scale_annotation(child, dims, clamp) for child in coll.points]
    return Collection(points=scaled_children, mention=coll.mention, t=coll.t)


def _scale_annotation(annotation: Annotation, dims: tuple[int, int], clamp: bool) -> Annotation:
    if isinstance(annotation, SinglePoint):
        return _scale_point(annotation, dims, clamp)
    if isinstance(annotation, BoundingBox):
        return _scale_box(annotation, dims, clamp)
    if isinstance(annotation, Polygon):
        return _scale_polygon(annotation, dims, clamp)
    if isinstance(annotation, Collection):
        return _scale_collection(annotation, dims, clamp)
    raise TypeError(f"Unsupported annotation type: {type(annotation)!r}")


def scale_point_to_pixels(point: SinglePoint, *, width: int, height: int, clamp: bool = True) -> SinglePoint:
    """Scale a single normalized point (0-1000 grid) into pixel coordinates."""

    dims = _normalize_dimensions(width, height)
    return _scale_point(point, dims, clamp)


def scale_box_to_pixels(box: BoundingBox, *, width: int, height: int, clamp: bool = True) -> BoundingBox:
    """Scale a normalized bounding box into pixel coordinates."""

    dims = _normalize_dimensions(width, height)
    return _scale_box(box, dims, clamp)


def scale_polygon_to_pixels(poly: Polygon, *, width: int, height: int, clamp: bool = True) -> Polygon:
    """Scale a normalized polygon into pixel coordinates."""

    dims = _normalize_dimensions(width, height)
    return _scale_polygon(poly, dims, clamp)


def scale_collection_to_pixels(coll: Collection, *, width: int, height: int, clamp: bool = True) -> Collection:
    """Scale a normalized collection (and its children) into pixel coordinates."""

    dims = _normalize_dimensions(width, height)
    return _scale_collection(coll, dims, clamp)


def scale_points_to_pixels(
    points: Sequence[Annotation] | None,
    *,
    width: int,
    height: int,
    clamp: bool = True,
) -> list[Annotation] | None:
    """Scale structured annotations from the normalized 0-1000 grid into pixel space.

    Args:
        points: Sequence of annotations (as returned by ``PerceiveResult.points``).
        width: Target image width in pixels.
        height: Target image height in pixels.
        clamp: When True (default), keep all coordinates within the image bounds.

    Returns:
        New annotations list with the same structure but expressed in pixel coordinates.
        Returns ``None`` when ``points`` is ``None``.
    """

    dims = _normalize_dimensions(width, height)
    if points is None:
        return None
    return [_scale_annotation(obj, dims, clamp) for obj in points]


__all__ = [
    "NORMALIZED_COORD_MAX",
    "scale_box_to_pixels",
    "scale_collection_to_pixels",
    "scale_point_to_pixels",
    "scale_points_to_pixels",
    "scale_polygon_to_pixels",
]
