from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


class DSLNode:
    """Base class for DSL nodes used to compose prompts."""

    def __add__(self, other: "DSLNode | Sequence") -> "Sequence":
        if isinstance(other, Sequence):
            return Sequence([self, *other.nodes])
        return Sequence([self, other])


@dataclass
class Text(DSLNode):
    """User text content (role=user)."""

    content: str


@dataclass
class System(DSLNode):
    """System instruction text."""

    content: str


@dataclass
class Agent(DSLNode):
    """Assistant text content (few-shot / ICL examples)."""

    content: str


@dataclass
class Image(DSLNode):
    """Image content. Accepts path/bytes/PIL.Image/np.ndarray for encoding to base64."""

    obj: Any


@dataclass
class PointTag(DSLNode):
    x: int
    y: int
    image: Image | None = None
    mention: str | None = None
    t: float | None = None


@dataclass
class BoxTag(DSLNode):
    x1: int
    y1: int
    x2: int
    y2: int
    image: Image | None = None
    mention: str | None = None
    t: float | None = None


@dataclass
class PolygonTag(DSLNode):
    coords: List[tuple[int, int]]
    image: Image | None = None
    mention: str | None = None
    t: float | None = None


@dataclass
class Sequence(DSLNode):
    """A flat sequence of nodes; supports `+` composition."""

    nodes: List[DSLNode]

    def __add__(self, other: DSLNode | "Sequence") -> "Sequence":
        if isinstance(other, Sequence):
            return Sequence([*self.nodes, *other.nodes])
        return Sequence([*self.nodes, other])


def block(*nodes: Iterable[DSLNode | Sequence]) -> Sequence:
    """Concatenate nodes (or sequences) into a single sequence."""
    flat: list[DSLNode] = []
    for n in nodes:
        if isinstance(n, Sequence):
            flat.extend(n.nodes)
        else:
            flat.append(n)
    return Sequence(flat)


# Factory helpers preserving original DSL surface
def text(content: str) -> Text:
    """Create a user text node."""
    return Text(content)


def system(content: str) -> System:
    """Create a system instruction node."""
    return System(content)


def agent(content: str) -> Agent:
    """Create an assistant (agent) node, useful for ICL."""
    return Agent(content)


def image(obj: Any) -> Image:
    """Create an image node from path/bytes/PIL.Image/np.ndarray."""
    return Image(obj)


def point(
    x: int, y: int, *, image: Image | None = None, mention: str | None = None, t: float | None = None
) -> PointTag:
    """Create a point tag anchored to an image (explicit in multi-image cases)."""
    return PointTag(x, y, image=image, mention=mention, t=t)


def box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    image: Image | None = None,
    mention: str | None = None,
    t: float | None = None,
) -> BoxTag:
    """Create a bounding box tag anchored to an image."""
    return BoxTag(x1, y1, x2, y2, image=image, mention=mention, t=t)


def polygon(
    coords: list[tuple[int, int]], *, image: Image | None = None, mention: str | None = None, t: float | None = None
) -> PolygonTag:
    """Create a polygon tag anchored to an image; requires ≥3 vertices."""
    return PolygonTag(coords, image=image, mention=mention, t=t)
