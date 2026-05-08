from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(eq=True)
class SinglePoint:
    x: int
    y: int
    mention: str | None = None
    t: float | None = None

    def __repr__(self) -> str:  # stable and concise
        base = f"SinglePoint(x={self.x}, y={self.y}"
        if self.mention is not None:
            base += f", mention={self.mention!r}"
        if self.t is not None:
            base += f", t={self.t}"
        return base + ")"


@dataclass(eq=True)
class BoundingBox:
    top_left: SinglePoint
    bottom_right: SinglePoint
    mention: str | None = None
    t: float | None = None

    def __repr__(self) -> str:
        base = (
            f"BoundingBox(top_left=({self.top_left.x},{self.top_left.y}), "
            f"bottom_right=({self.bottom_right.x},{self.bottom_right.y})"
        )
        if self.mention is not None:
            base += f", mention={self.mention!r}"
        if self.t is not None:
            base += f", t={self.t}"
        return base + ")"


@dataclass(eq=True)
class Polygon:
    hull: list[SinglePoint]
    mention: str | None = None
    t: float | None = None

    def __repr__(self) -> str:
        coords = ", ".join(f"({p.x},{p.y})" for p in self.hull)
        base = f"Polygon(hull=[{coords}]"
        if self.mention is not None:
            base += f", mention={self.mention!r}"
        if self.t is not None:
            base += f", t={self.t}"
        return base + ")"


@dataclass(eq=True)
class Collection:
    points: list[SinglePoint | BoundingBox | Polygon | Collection]
    mention: str | None = None
    t: float | None = None

    def __repr__(self) -> str:
        return f"Collection(points={len(self.points)}, mention={self.mention!r}, t={self.t})"


@dataclass(eq=True)
class ClipTimestamp:
    """Time anchor for a video clip. ``until is None`` ⇔ moment; otherwise a range [at, until]."""

    at: float
    until: float | None = None

    def __repr__(self) -> str:
        if self.until is None:
            return f"ClipTimestamp(at={self.at})"
        return f"ClipTimestamp(at={self.at}, until={self.until})"


@dataclass(eq=True)
class Clip:
    timestamp: ClipTimestamp
    mention: str | None = None

    def __repr__(self) -> str:
        if self.mention is None:
            return f"Clip(timestamp={self.timestamp!r})"
        return f"Clip(timestamp={self.timestamp!r}, mention={self.mention!r})"


# Convenience constructors for annotations/examples
def pt(x: int, y: int, *, mention: str | None = None) -> SinglePoint:
    return SinglePoint(x=x, y=y, mention=mention)


def bbox(x1: int, y1: int, x2: int, y2: int, *, mention: str | None = None) -> BoundingBox:
    return BoundingBox(top_left=SinglePoint(x1, y1), bottom_right=SinglePoint(x2, y2), mention=mention)


def poly(coords: list[tuple[int, int]], *, mention: str | None = None) -> Polygon:
    return Polygon(hull=[SinglePoint(x, y) for x, y in coords], mention=mention)


def collection(
    points: Sequence[SinglePoint | BoundingBox | Polygon | Collection],
    *,
    mention: str | None = None,
    t: float | None = None,
) -> Collection:
    return Collection(points=list(points), mention=mention, t=t)


def clip(at: float, until: float | None = None, *, mention: str | None = None) -> Clip:
    return Clip(timestamp=ClipTimestamp(at=at, until=until), mention=mention)
