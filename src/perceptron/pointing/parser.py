"""Canonical pointing tag parser and helpers.

Supported tags
- <point [mention=...][t=FLOAT]> (x,y) </point>
- <point_box [mention=...][t=FLOAT]> (x1,y1) (x2,y2) </point_box>
- <polygon [mention=...][t=FLOAT]> (x1,y1) (x2,y2) (x3,y3) ... </polygon>
- <collection> ...child point/box/polygon tags... </collection>
- <clip [mention=...] t=FLOAT[ FLOAT] />  (self-closing; moment or [at, until] range)

Helpers
- parse_text(text) → ordered segments: text and structured tags with spans
- extract_points(text, expected) → filtered list of point/box/polygon
- extract_clips(text) → list of clip annotations (with mention inheritance inside collections)
- strip_tags(text) → remove all canonical tags
"""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from html import escape, unescape
from typing import Any, Literal

from ..errors import ParseError
from .types import BoundingBox, Clip, ClipTimestamp, Collection, Polygon, SinglePoint

BOX_MIN_POINTS = 2
POLYGON_MIN_POINTS = 3

# Regex fragments
_WS = r"\s*"
_NUM = r"(?:\d+)"
_PT = rf"\({_WS}({_NUM}){_WS},{_WS}({_NUM}){_WS}\)"

# Tag attributes — sequence of either a complete quoted string (which may contain
# `>` characters) or a single non-quote/non-`>` character. The leading `\b` on
# the tag name keeps `<point` from matching `<point_box`.
_ATTRS = r'(?P<attrs>(?:"[^"]*"|[^>"])*)'
_FULL_TAG = re.compile(
    rf"<(?P<tag>point|point_box|polygon|collection)\b{_ATTRS}>(?P<body>[\s\S]*?)</(?P=tag)>",
    re.IGNORECASE,
)
# Self-closing <clip mention=... t=... /> — clips have no body, just attributes.
# Quoted values may contain `>` or `/>`, so attrs is a sequence of either a complete
# quoted string or a single non-quote/non-`>` character.
_CLIP_TAG = re.compile(
    r'<clip\b(?P<attrs>(?:"[^"]*"|[^>"])*?)\s*/>',
    re.IGNORECASE,
)
# Standalone <collection> regex used during clip extraction so clips inside a
# collection inherit the parent mention.
_COLLECTION_TAG = re.compile(
    rf"<collection\b{_ATTRS}>(?P<body>[\s\S]*?)</collection>",
    re.IGNORECASE,
)
# Accept t=1.5, t="1.5", t="1.5 seconds", t="1.5 2.0", t="1.5 seconds 2.0 seconds".
_T_VALUE = re.compile(r'\bt=(?:"([^"]*)"|(\S+))')


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
        raise ParseError(
            f"Malformed <point> tag: expected coordinates like (x,y) but got: {body!r}",
            code="invalid_point_coords",
            details={"body": body},
        )
    x, y = int(m.group(1)), int(m.group(2))
    return SinglePoint(x, y)


def _parse_box_body(body: str) -> BoundingBox:
    pts = list(re.finditer(_PT, body))
    if len(pts) < BOX_MIN_POINTS:
        raise ParseError(
            f"Malformed <point_box> tag: expected 2 coordinates like (x1,y1) (x2,y2) but got: {body!r}",
            code="invalid_box_coords",
            details={"body": body},
        )
    x1, y1 = int(pts[0].group(1)), int(pts[0].group(2))
    x2, y2 = int(pts[1].group(1)), int(pts[1].group(2))
    return BoundingBox(SinglePoint(x1, y1), SinglePoint(x2, y2))


def _parse_polygon_body(body: str) -> Polygon:
    pts = [SinglePoint(int(m.group(1)), int(m.group(2))) for m in re.finditer(_PT, body)]
    if len(pts) < POLYGON_MIN_POINTS:
        raise ParseError(
            f"Malformed <polygon> tag: expected at least 3 coordinates but got {len(pts)}: {body!r}",
            code="invalid_polygon_coords",
            details={"body": body, "points_found": len(pts)},
        )
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
        attrs = _parse_attrs(m.group("attrs") or "")
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
                with suppress(ValueError):
                    obj.t = float(attrs["t"])
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


def parse_text(text: str, *, expects: str | None = None) -> list[dict[str, Any]]:
    """Return ordered segments: text and tag segments with spans.

    The `expects` parameter selects which tag family to scan for, so callers
    don't pay for parsing tags they don't care about:
      - ``"clip"``: scan self-closing ``<clip />`` tags only
      - ``"point"`` | ``"box"`` | ``"polygon"`` | ``None``: scan
        point/point_box/polygon/collection tags (default — preserves the
        original geometry-only contract)
      - anything else: no tag parsing; the input is returned as a single
        text segment

    Segment shapes:
      - {"kind": "text", "text": str, "span": {"start": int, "end": int}}
      - {"kind": "point"|"box"|"polygon"|"collection", "value": obj, "span": {...}}
      - {"kind": "clip", "value": Clip, "span": {...}}
    """
    if expects == "clip":
        pattern, handler = _CLIP_TAG, _clip_match_to_segment
    elif expects is None or expects in {"point", "box", "polygon"}:
        pattern, handler = _FULL_TAG, _shape_match_to_segment
    else:
        return [{"kind": "text", "text": text, "span": {"start": 0, "end": len(text)}}] if text else []
    return _scan_segments(text, pattern, handler)


def _scan_segments(
    text: str,
    pattern: re.Pattern[str],
    handler: Callable[[re.Match[str]], tuple[str, Any]],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    idx = 0
    for m in pattern.finditer(text):
        if m.start() > idx:
            segments.append(
                {
                    "kind": "text",
                    "text": text[idx : m.start()],
                    "span": {"start": idx, "end": m.start()},
                }
            )
        kind, value = handler(m)
        segments.append({"kind": kind, "value": value, "span": {"start": m.start(), "end": m.end()}})
        idx = m.end()
    if idx < len(text):
        segments.append(
            {
                "kind": "text",
                "text": text[idx:],
                "span": {"start": idx, "end": len(text)},
            }
        )
    return segments


def _shape_match_to_segment(m: re.Match[str]) -> tuple[str, Any]:
    tag = m.group("tag").lower()
    inner_body = m.group("body") or ""
    attrs = _parse_attrs(m.group("attrs") or "")
    obj: SinglePoint | BoundingBox | Polygon | Collection
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
            with suppress(ValueError):
                obj.t = float(attrs["t"])
    return kind, obj


def _clip_match_to_segment(m: re.Match[str]) -> tuple[str, Any]:
    return "clip", _parse_clip_body(m.group("attrs") or "")


def _parse_clip_body(attrs: str) -> Clip:
    """Parse a self-closing `<clip />` tag's attributes into a Clip. Raises ParseError if t is missing or unparseable."""
    ts = _parse_clip_t(attrs)
    if ts is None:
        raise ParseError(
            f"Malformed <clip /> tag: expected a numeric t= attribute (e.g., t=1.5 or t=\"1.5 2.0\") but got attrs: {attrs!r}",
            code="invalid_clip_timestamp",
            details={"attrs": attrs},
        )
    mention = _parse_attrs(attrs).get("mention")
    return Clip(timestamp=ts, mention=mention)


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
    text = re.sub(_FULL_TAG, "", text)
    return _CLIP_TAG.sub("", text)


# ---------------------------------------------------------------------------
# Clip parsing (self-closing <clip /> tags with mention + t attributes)
# ---------------------------------------------------------------------------


def _parse_clip_t(attrs: str) -> ClipTimestamp | None:
    """Parse the ``t`` attribute. One number → moment; two → range. Trailing units (e.g., "seconds") are ignored."""

    m = _T_VALUE.search(attrs)
    if not m:
        return None
    value = m.group(1) if m.group(1) is not None else m.group(2)
    nums: list[float] = []
    for token in value.split():
        try:
            nums.append(float(token))
        except ValueError:
            continue
    if len(nums) == 1:
        return ClipTimestamp(at=nums[0])
    if len(nums) >= BOX_MIN_POINTS:
        return ClipTimestamp(at=nums[0], until=nums[1])
    return None


def extract_clips(text: str) -> list[Clip]:
    """Extract ``<clip />`` annotations. Clips inside a ``<collection>`` inherit the parent ``mention`` when unset."""

    results: list[Clip] = []

    def _on_collection(match: re.Match[str]) -> str:
        parent_mention = _parse_attrs(match.group("attrs") or "").get("mention")
        for inner in _CLIP_TAG.finditer(match.group("body")):
            inner_attrs = _parse_attrs(inner.group("attrs"))
            ts = _parse_clip_t(inner.group("attrs"))
            if ts is None:
                continue
            results.append(Clip(timestamp=ts, mention=inner_attrs.get("mention") or parent_mention))
        return ""

    remaining = _COLLECTION_TAG.sub(_on_collection, text)

    for m in _CLIP_TAG.finditer(remaining):
        attrs = _parse_attrs(m.group("attrs"))
        ts = _parse_clip_t(m.group("attrs"))
        if ts is None:
            continue
        results.append(Clip(timestamp=ts, mention=attrs.get("mention")))

    return results
