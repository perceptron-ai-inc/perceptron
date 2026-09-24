"""Geometry helpers for converting normalized annotations into pixel space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .types import BoundingBox, Clip, Collection, Polygon, SinglePoint, Track

NORMALIZED_COORD_MAX = 1000.0

Annotation = SinglePoint | BoundingBox | Polygon | Collection | Clip | Track


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
    return replace(point, x=_scale_component(point.x, width, clamp), y=_scale_component(point.y, height, clamp))


def _scale_box(box: BoundingBox, dims: tuple[int, int], clamp: bool) -> BoundingBox:
    scaled_top_left = _scale_point(box.top_left, dims, clamp)
    scaled_bottom_right = _scale_point(box.bottom_right, dims, clamp)
    left_x = min(scaled_top_left.x, scaled_bottom_right.x)
    right_x = max(scaled_top_left.x, scaled_bottom_right.x)
    top_y = min(scaled_top_left.y, scaled_bottom_right.y)
    bottom_y = max(scaled_top_left.y, scaled_bottom_right.y)
    return replace(box, top_left=SinglePoint(left_x, top_y), bottom_right=SinglePoint(right_x, bottom_y))


def _scale_polygon(poly: Polygon, dims: tuple[int, int], clamp: bool) -> Polygon:
    return replace(poly, hull=[_scale_point(point, dims, clamp) for point in poly.hull])


def _scale_collection(coll: Collection, dims: tuple[int, int], clamp: bool) -> Collection:
    return replace(coll, points=[_scale_annotation(child, dims, clamp) for child in coll.points])


def _scale_annotation(annotation: Annotation, dims: tuple[int, int], clamp: bool) -> Annotation:
    if isinstance(annotation, SinglePoint):
        return _scale_point(annotation, dims, clamp)
    if isinstance(annotation, BoundingBox):
        return _scale_box(annotation, dims, clamp)
    if isinstance(annotation, Polygon):
        return _scale_polygon(annotation, dims, clamp)
    if isinstance(annotation, Collection):
        return _scale_collection(annotation, dims, clamp)
    if isinstance(annotation, Track):
        return replace(annotation, points=[_scale_annotation(p, dims, clamp) for p in annotation.points])
    if isinstance(annotation, Clip):  # temporal only: nothing to scale
        return annotation
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


def _scale_by_asset(
    annotation: Annotation,
    sizes: Mapping[int, tuple[int, int]],
    asset_idx: int | None,
    last_asset: int | None,
    clamp: bool,
) -> Annotation:
    own = getattr(annotation, "asset_idx", None)
    asset_idx = own if own is not None else asset_idx
    if isinstance(annotation, (Collection, Track)):
        children = [_scale_by_asset(child, sizes, asset_idx, last_asset, clamp) for child in annotation.points]
        return replace(annotation, points=children)
    if isinstance(annotation, Clip):
        return annotation
    if asset_idx is None:
        asset_idx = last_asset  # a missing selector means the last asset
        if asset_idx is None:
            raise ValueError(f"{annotation!r} has no asset_idx and there is no asset to resolve it to")
    if asset_idx not in sizes:
        raise ValueError(f"No size given for asset_idx {asset_idx}")
    return _scale_annotation(annotation, sizes[asset_idx], clamp)


def scale_annotations_by_asset(
    annotations: Sequence[Annotation] | None,
    sizes: Mapping[int, tuple[int, int]] | Sequence[tuple[int, int]],
    *,
    n_assets: int | None = None,
    clamp: bool = True,
) -> list[Annotation] | None:
    """Scale annotations into pixels using each annotation's own asset.

    Args:
        annotations: Annotations on the normalized 0-1000 grid (containers resolve ``asset_idx`` per child).
        sizes: ``(width, height)`` per asset, as a mapping ``{asset_idx: (w, h)}`` or a list indexed by asset.
        n_assets: The request's media asset count (e.g. ``result.asset_count``).
        clamp: When True (default), keep all coordinates within each asset's bounds.

    An annotation without an ``asset_idx`` refers to the last asset: ``n_assets - 1`` when ``n_assets`` is given,
    else the highest index in ``sizes``. A selector without a size raises ``ValueError``. Clips pass through
    unchanged. Returns ``None`` when ``annotations`` is ``None``.
    """

    items = sizes.items() if isinstance(sizes, Mapping) else enumerate(sizes)
    dims = {int(idx): _normalize_dimensions(*size) for idx, size in items}
    if annotations is None:
        return None
    if n_assets is None:  # the highest index with a size stands in for the last asset
        n_assets = max(dims, default=-1) + 1
    last_asset = n_assets - 1 if n_assets > 0 else None
    return [_scale_by_asset(obj, dims, None, last_asset, clamp) for obj in annotations]


__all__ = [
    "NORMALIZED_COORD_MAX",
    "scale_annotations_by_asset",
    "scale_box_to_pixels",
    "scale_collection_to_pixels",
    "scale_point_to_pixels",
    "scale_points_to_pixels",
    "scale_polygon_to_pixels",
]
