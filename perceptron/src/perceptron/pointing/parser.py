"""Canonical pointing tag parser and helpers.

Supported tags
- <point [mention=...][t=FLOAT]> (x,y) </point>
- <point_box [mention=...][t=FLOAT]> (x1,y1) (x2,y2) </point_box>
- <polygon [mention=...][t=FLOAT]> (x1,y1) (x2,y2) (x3,y3) ... </polygon>
- <collection> ...child point/box/polygon tags... </collection>

Helpers
- parse_text(text) → ordered segments: text and structured tags with spans
- extract_points(text, expected) → filtered list of point/box/polygon
- strip_tags(text) → remove all canonical tags
"""

from __future__ import annotations

import re
from html import unescape, escape
from typing import Any, Literal
from dataclasses import dataclass, replace

from .types import SinglePoint, BoundingBox, Polygon, Collection


# Regex fragments
_WS = r"\s*"
_NUM = r"(?:\d+)"
_PT = rf"\({_WS}({_NUM}){_WS},{_WS}({_NUM}){_WS}\)"

_ATTR = r"(?:\s+[^>]+)?"  # we ignore unknown attrs; mention/t handled in parse
_FULL_TAG = re.compile(
    rf"<(?P<tag>point|point_box|polygon|collection){_ATTR}>(?P<body>[\s\S]*?)</(?P=tag)>",
    re.IGNORECASE,
)
_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _parse_attrs(tag_open: str) -> dict[str, str]:
    # naive attribute parsing: key="value" or key=value
    attrs: dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|([^\s>]+))", tag_open):
        key = m.group(1)
        val = m.group(2) or m.group(3) or ""
        attrs[key] = unescape(val)
    return attrs


def _parse_point_body(body: str) -> SinglePoint:
    m = re.search(_PT, body)
    if not m:
        raise ValueError("invalid point coords")
    x, y = int(m.group(1)), int(m.group(2))
    return SinglePoint(x, y)


def _parse_box_body(body: str) -> BoundingBox:
    pts = list(re.finditer(_PT, body))
    if len(pts) < 2:
        raise ValueError("invalid box coords")
    x1, y1 = int(pts[0].group(1)), int(pts[0].group(2))
    x2, y2 = int(pts[1].group(1)), int(pts[1].group(2))
    return BoundingBox(SinglePoint(x1, y1), SinglePoint(x2, y2))


def _parse_polygon_body(body: str) -> Polygon:
    pts = [SinglePoint(int(m.group(1)), int(m.group(2))) for m in re.finditer(_PT, body)]
    if len(pts) < 3:
        raise ValueError("invalid polygon coords")
    return Polygon(hull=pts)


def _parse_collection_body(body: str) -> Collection:
    # recursively parse child tags
    items: list[Any] = []
    idx = 0
    while True:
        m = _FULL_TAG.search(body, idx)
        if not m:
            break
        tag = m.group("tag").lower()
        inner_body = m.group("body") or ""
        tag_open = body[m.start() : m.start() + body[m.start() : m.end()].find(">") + 1]
        attrs = _parse_attrs(tag_open)
        if tag == "point":
            obj = _parse_point_body(inner_body)
        elif tag == "point_box":
            obj = _parse_box_body(inner_body)
        elif tag == "polygon":
            obj = _parse_polygon_body(inner_body)
        else:
            # nested collections not supported in MVP
            idx = m.end()
            continue
        if "mention" in attrs:
            obj.mention = attrs["mention"]
        if "t" in attrs:
            try:
                obj.t = float(attrs["t"])
            except ValueError:
                pass
        items.append(obj)
        idx = m.end()
    return Collection(points=items)


def _attr_string(mention: str | None, t: float | None) -> str:
    attrs = []
    if mention is not None:
        attrs.append(f'mention="{escape(mention, quote=True)}"')
    if t is not None:
        attrs.append(f"t={t}")
    return (" " + " ".join(attrs)) if attrs else ""


def PointParser_serialize(obj: Any) -> str:
    if isinstance(obj, SinglePoint):
        body = f"({obj.x},{obj.y})"
        attr = _attr_string(obj.mention, obj.t)
        return f"<point{attr}> {body} </point>"
    if isinstance(obj, BoundingBox):
        a, b = obj.top_left, obj.bottom_right
        body = f"({a.x},{a.y}) ({b.x},{b.y})"
        attr = _attr_string(obj.mention, obj.t)
        return f"<point_box{attr}> {body} </point_box>"
    if isinstance(obj, Polygon):
        body = " ".join(f"({p.x},{p.y})" for p in obj.hull)
        attr = _attr_string(obj.mention, obj.t)
        return f"<polygon{attr}> {body} </polygon>"
    if isinstance(obj, Collection):
        inner = " ".join(PointParser_serialize(p) for p in obj.points)
        attr = _attr_string(obj.mention, obj.t)
        return f"<collection{attr}> {inner} </collection>"
    raise TypeError(f"Unsupported type: {type(obj)}")


class PointParser:
    @staticmethod
    def serialize(obj: Any) -> str:
        return PointParser_serialize(obj)

    @staticmethod
    def parse(text: str) -> list[dict[str, Any]]:
        """Return structured tag segments parsed from text (excludes plain text)."""
        return [seg for seg in parse_text(text) if seg.get("kind") != "text"]


def parse_text(text: str) -> list[dict[str, Any]]:
    """Return ordered segments: text and tag segments with spans.

    Segment shapes:
      - {"kind": "text", "text": str, "span": {"start": int, "end": int}}
      - {"kind": "point"|"box"|"polygon"|"collection", "value": obj, "span": {...}}
    """
    segments: list[dict[str, Any]] = []
    idx = 0
    for m in _FULL_TAG.finditer(text):
        if m.start() > idx:
            segments.append({"kind": "text", "text": text[idx : m.start()], "span": {"start": idx, "end": m.start()}})
        tag = m.group("tag").lower()
        inner_body = m.group("body") or ""
        tag_open = text[m.start() : m.start() + text[m.start() : m.end()].find(">") + 1]
        attrs = _parse_attrs(tag_open)
        if tag == "point":
            obj = _parse_point_body(inner_body)
            kind = "point"
        elif tag == "point_box":
            obj = _parse_box_body(inner_body)
            kind = "box"
        elif tag == "polygon":
            obj = _parse_polygon_body(inner_body)
            kind = "polygon"
        else:  # collection
            obj = _parse_collection_body(inner_body)
            kind = "collection"
        if "mention" in attrs:
            obj.mention = attrs["mention"]
        if "t" in attrs:
            try:
                obj.t = float(attrs["t"])
            except ValueError:
                pass
        segments.append({"kind": kind, "value": obj, "span": {"start": m.start(), "end": m.end()}})
        idx = m.end()
    if idx < len(text):
        segments.append({"kind": "text", "text": text[idx:], "span": {"start": idx, "end": len(text)}})
    return segments


def _kind_of(obj: Any) -> str | None:
    if isinstance(obj, SinglePoint):
        return "point"
    if isinstance(obj, BoundingBox):
        return "box"
    if isinstance(obj, Polygon):
        return "polygon"
    return None


def _with_parent_attrs(obj: Any, mention: str | None, t: float | None) -> Any:
    # Propagate mention/t attributes from a collection to children when missing.
    original_mention = getattr(obj, "mention", None)
    original_t = getattr(obj, "t", None)
    new_mention = original_mention if original_mention is not None else mention
    new_t = original_t if original_t is not None else t
    if new_mention is original_mention and new_t is original_t:
        return obj
    kwargs = {}
    if new_mention is not original_mention:
        kwargs["mention"] = new_mention
    if new_t is not original_t:
        kwargs["t"] = new_t
    return replace(obj, **kwargs)


def _flatten_collection(
    collection: Collection,
    expected: Literal["point", "box", "polygon"] | None,
    inherited_mention: str | None = None,
    inherited_t: float | None = None,
) -> list[Any]:
    mention = collection.mention if collection.mention is not None else inherited_mention
    t = collection.t if collection.t is not None else inherited_t
    flattened: list[Any] = []
    for child in collection.points:
        if isinstance(child, Collection):
            flattened.extend(_flatten_collection(child, expected, mention, t))
            continue
        kind = _kind_of(child)
        if kind is None:
            continue
        if expected is None or kind == expected:
            flattened.append(_with_parent_attrs(child, mention, t))
    return flattened


def extract_points(text: str, expected: Literal["point", "box", "polygon"] | None = None) -> list[Any]:
    """Extract only the requested tag type (if provided) in order of appearance."""
    segs = parse_text(text)
    result: list[Any] = []
    for s in segs:
        kind = s["kind"]
        if kind in {"point", "box", "polygon"}:
            if expected is None or kind == expected:
                result.append(s["value"])
        elif kind == "collection":
            result.extend(_flatten_collection(s["value"], expected))
    return result


def strip_tags(text: str) -> str:
    """Remove all canonical tags and return plain text only."""
    return re.sub(_FULL_TAG, "", text)


@dataclass
class ReasoningExtraction:
    text: str | None
    reasoning: list[str]

    def as_tuple(self) -> tuple[list[str] | None, str | None]:
        return (self.reasoning or None, self.text)


class ReasoningStreamCleaner:
    """Incrementally strips <think>...</think> blocks while capturing reasoning."""

    def __init__(self) -> None:
        self._buffer: str = ""

    def consume(self, chunk: str) -> tuple[str, list[str]]:
        text = chunk or ""
        buf = self._buffer + text
        reasoning: list[str] = []
        sanitized_parts: list[str] = []

        while True:
            start = buf.find("<think>")
            if start == -1:
                sanitized_parts.append(buf)
                buf = ""
                break
            sanitized_parts.append(buf[:start])
            buf = buf[start + len("<think>") :]
            end = buf.find("</think>")
            if end == -1:
                self._buffer = "<think>" + buf
                return "".join(sanitized_parts), reasoning
            content = buf[:end].strip()
            if content:
                reasoning.append(content)
            buf = buf[end + len("</think>") :]

        self._buffer = buf
        sanitized = "".join(sanitized_parts)
        return sanitized, reasoning

    def reset(self) -> None:
        self._buffer = ""


def extract_reasoning(text: Any) -> ReasoningExtraction:
    """Return reasoning segments and text with <think> tags removed."""

    if not isinstance(text, str):
        return ReasoningExtraction(text=text, reasoning=[])

    segments = [seg.strip() for seg in _THINK_TAG.findall(text)]
    reasoning = [seg for seg in segments if seg]
    cleaned = _THINK_TAG.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return ReasoningExtraction(text=cleaned or None, reasoning=reasoning)
