"""Annotation types: spatial leaves, clips, tracks and collections.

Coordinates are integers on the normalized 0-1000 grid. ``t`` is seconds from the start of a temporal asset.
``asset_idx`` is the 0-based position of the media asset (image, video or audio) the annotation refers to; ``None``
means neither the markup nor an enclosing container said. A container pushes its ``asset_idx`` down onto every
descendant that has none (collection → leaves, clips, tracks; track → waypoints): parsed and constructed trees alike
hold copies with the inherited value, and an explicit child value (including 0) wins. Setting a container's
``asset_idx`` after construction is not propagated, and ``dataclasses.replace(container, asset_idx=...)`` does not
re-target the children either: they keep the value pushed at construction (and a track re-takes its waypoints'
shared one). To move a tree to another asset, rebuild it with its descendants' ``asset_idx`` cleared (or set) as
needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any


def _context_repr(mention: str | None, t: Any = None, asset_idx: int | None = None) -> str:
    parts = ""
    if mention is not None:
        parts += f", mention={mention!r}"
    if t is not None:
        parts += f", t={t}"
    if asset_idx is not None:
        parts += f", asset_idx={asset_idx}"
    return parts


def _context_dict(data: dict[str, Any], mention: str | None, t: Any = None, asset_idx: int | None = None) -> dict:
    if mention is not None:
        data["mention"] = mention
    if t is not None:
        data["t"] = t
    if asset_idx is not None:
        data["asset_idx"] = asset_idx
    return data


def _point_fields(point: SinglePoint) -> dict[str, Any]:
    return _context_dict({"x": point.x, "y": point.y}, point.mention, point.t, point.asset_idx)


def _as_list(children: Any) -> list[Any]:
    """``children`` as a list: a tuple or a generator is copied (a generator could be walked only once)."""

    return children if isinstance(children, list) else list(children)


def _push_asset_idx(children: list[Any], asset_idx: int | None) -> list[Any]:
    """``children`` with ``asset_idx`` set on every annotation that has none, as ``dataclasses.replace`` copies (a
    copied container pushes it further down). The caller's objects and list are never mutated."""

    if asset_idx is None or not any(_lacks_asset_idx(child) for child in children):
        return children
    return [replace(child, asset_idx=asset_idx) if _lacks_asset_idx(child) else child for child in children]


def _lacks_asset_idx(child: Any) -> bool:
    return isinstance(child, _ANNOTATION_TYPES) and child.asset_idx is None


@dataclass(eq=True)
class SinglePoint:
    x: int
    y: int
    mention: str | None = None
    t: float | None = None
    asset_idx: int | None = field(default=None, kw_only=True)

    def __repr__(self) -> str:  # stable and concise
        return f"SinglePoint(x={self.x}, y={self.y}{_context_repr(self.mention, self.t, self.asset_idx)})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "point", **_point_fields(self)}


@dataclass(eq=True)
class BoundingBox:
    top_left: SinglePoint
    bottom_right: SinglePoint
    mention: str | None = None
    t: float | None = None
    asset_idx: int | None = field(default=None, kw_only=True)

    def __repr__(self) -> str:
        return (
            f"BoundingBox(top_left=({self.top_left.x},{self.top_left.y}), "
            f"bottom_right=({self.bottom_right.x},{self.bottom_right.y})"
            f"{_context_repr(self.mention, self.t, self.asset_idx)})"
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "type": "box",
            "top_left": _point_fields(self.top_left),
            "bottom_right": _point_fields(self.bottom_right),
        }
        return _context_dict(data, self.mention, self.t, self.asset_idx)


@dataclass(eq=True)
class Polygon:
    hull: list[SinglePoint]
    mention: str | None = None
    t: float | None = None
    asset_idx: int | None = field(default=None, kw_only=True)

    def __repr__(self) -> str:
        coords = ", ".join(f"({p.x},{p.y})" for p in self.hull)
        return f"Polygon(hull=[{coords}]{_context_repr(self.mention, self.t, self.asset_idx)})"

    def to_dict(self) -> dict[str, Any]:
        data = {"type": "polygon", "points": [_point_fields(p) for p in self.hull]}
        return _context_dict(data, self.mention, self.t, self.asset_idx)


@dataclass(eq=True)
class Collection:
    """A group of annotations; children inherit ``mention``, ``asset_idx`` and (legacy) ``t`` when unset.

    ``asset_idx`` is pushed down onto children (copies) at construction, so a later assignment or
    ``dataclasses.replace`` does not reach them; ``mention`` and ``t`` apply when flattened. ``complete`` is False
    when the markup was cut off before ``</collection>`` (e.g. a truncated response).
    """

    points: list[SinglePoint | BoundingBox | Polygon | Clip | Track | Collection]
    mention: str | None = None
    t: float | None = None
    asset_idx: int | None = field(default=None, kw_only=True)
    complete: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        self.points = _push_asset_idx(_as_list(self.points), self.asset_idx)

    def __repr__(self) -> str:
        base = f"Collection(points={len(self.points)}, mention={self.mention!r}, t={self.t}"
        if self.asset_idx is not None:
            base += f", asset_idx={self.asset_idx}"
        if not self.complete:
            base += ", complete=False"
        return base + ")"

    def to_dict(self) -> dict[str, Any]:
        data = _context_dict({"type": "collection"}, self.mention, self.t, self.asset_idx)
        data["points"] = [child.to_dict() for child in self.points]
        if not self.complete:
            data["complete"] = False
        return data


@dataclass(eq=True)
class ClipTimestamp:
    """Time anchor for a video clip. ``until is None`` ⇔ moment; otherwise a range [at, until]."""

    at: float
    until: float | None = None

    def __repr__(self) -> str:
        if self.until is None:
            return f"ClipTimestamp(at={self.at})"
        return f"ClipTimestamp(at={self.at}, until={self.until})"

    def to_dict(self) -> dict[str, Any]:
        return {"at": self.at} if self.until is None else {"at": self.at, "until": self.until}


@dataclass(eq=True)
class Clip:
    timestamp: ClipTimestamp
    mention: str | None = None
    asset_idx: int | None = field(default=None, kw_only=True)

    def __repr__(self) -> str:
        return f"Clip(timestamp={self.timestamp!r}{_context_repr(self.mention, asset_idx=self.asset_idx)})"

    def to_dict(self) -> dict[str, Any]:
        return _context_dict({"type": "clip", **self.timestamp.to_dict()}, self.mention, asset_idx=self.asset_idx)


@dataclass(eq=True)
class Track:
    """One object followed over time in one temporal asset.

    ``points`` are waypoints of a single kind (all points, all boxes or all polygons), each with its own ``t``.
    The object's ``mention`` and ``asset_idx`` live on the track: a track without an ``asset_idx`` takes the one its
    waypoints share, and waypoints without one get (copies with) the track's at construction, so a later assignment or
    ``dataclasses.replace`` does not reach them. ``complete`` is False when the markup was cut off before ``</track>``.
    """

    points: list[SinglePoint | BoundingBox | Polygon]
    mention: str | None = None
    asset_idx: int | None = field(default=None, kw_only=True)
    complete: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        self.points = _as_list(self.points)
        if self.asset_idx is None:  # hoist a uniform waypoint selector (conflicts are left for validation to report)
            shared = {p.asset_idx for p in self.points if getattr(p, "asset_idx", None) is not None}
            if len(shared) == 1:
                self.asset_idx = shared.pop()
        self.points = _push_asset_idx(self.points, self.asset_idx)

    def __repr__(self) -> str:
        base = f"Track(points={len(self.points)}{_context_repr(self.mention, asset_idx=self.asset_idx)}"
        return base + (")" if self.complete else ", complete=False)")

    def to_dict(self) -> dict[str, Any]:
        data = _context_dict({"type": "track"}, self.mention, asset_idx=self.asset_idx)
        data["points"] = [point.to_dict() for point in self.points]
        if not self.complete:
            data["complete"] = False
        return data


_ANNOTATION_TYPES = (SinglePoint, BoundingBox, Polygon, Clip, Track, Collection)


# Convenience constructors for annotations/examples
def pt(
    x: int, y: int, *, mention: str | None = None, t: float | None = None, asset_idx: int | None = None
) -> SinglePoint:
    return SinglePoint(x=x, y=y, mention=mention, t=t, asset_idx=asset_idx)


def bbox(  # noqa: PLR0913 - four coordinates plus keyword-only context
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    mention: str | None = None,
    t: float | None = None,
    asset_idx: int | None = None,
) -> BoundingBox:
    return BoundingBox(
        top_left=SinglePoint(x1, y1), bottom_right=SinglePoint(x2, y2), mention=mention, t=t, asset_idx=asset_idx
    )


def poly(
    coords: list[tuple[int, int]],
    *,
    mention: str | None = None,
    t: float | None = None,
    asset_idx: int | None = None,
) -> Polygon:
    return Polygon(hull=[SinglePoint(x, y) for x, y in coords], mention=mention, t=t, asset_idx=asset_idx)


def collection(
    points: Sequence[SinglePoint | BoundingBox | Polygon | Clip | Track | Collection],
    *,
    mention: str | None = None,
    t: float | None = None,
    asset_idx: int | None = None,
) -> Collection:
    return Collection(points=list(points), mention=mention, t=t, asset_idx=asset_idx)


def clip(at: float, until: float | None = None, *, mention: str | None = None, asset_idx: int | None = None) -> Clip:
    return Clip(timestamp=ClipTimestamp(at=at, until=until), mention=mention, asset_idx=asset_idx)


def _track_asset_idx(points: Sequence[Any], asset_idx: int | None) -> int | None:
    """The one asset a track follows: its own ``asset_idx`` or its waypoints' shared one (ValueError if they differ)."""

    assets = {p.asset_idx for p in points if p.asset_idx is not None}
    if asset_idx is not None:
        assets.add(asset_idx)
    if len(assets) > 1:
        raise ValueError(
            f"Track asset_idx values {sorted(assets)} differ; a track follows one object in one asset, so the track "
            "and its waypoints must name the same asset"
        )
    return next(iter(assets), None)


def track(
    points: Sequence[SinglePoint | BoundingBox | Polygon],
    *,
    mention: str | None = None,
    asset_idx: int | None = None,
) -> Track:
    """Build a track; waypoints must be points, boxes or polygons, all of one kind (give each a ``t``).

    There must be at least one waypoint (the server does not parse an empty ``<track>``). Waypoint ``asset_idx``
    values, when set, must match each other and ``asset_idx``; the track's value is pushed down onto (copies of) the
    waypoints that have none.
    """

    points = list(points)  # iterated more than once below
    if not points:
        raise ValueError("track() needs at least one waypoint")
    kinds = {type(p) for p in points}
    if not kinds <= {SinglePoint, BoundingBox, Polygon}:
        raise TypeError("track() waypoints must be SinglePoint, BoundingBox or Polygon objects")
    if len(kinds) > 1:
        raise ValueError("track() waypoints must all be the same kind (all points, all boxes or all polygons)")
    _track_asset_idx(points, asset_idx)
    return Track(points=points, mention=mention, asset_idx=asset_idx)
