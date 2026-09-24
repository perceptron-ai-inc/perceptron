"""Canonical pointing tag parser and helpers.

Supported tags
- <point [mention=..][t=..][asset_idx=N]> (x,y) </point>
- <point_box [mention=..][t=..][asset_idx=N]> (x1,y1) (x2,y2) </point_box>
- <polygon [mention=..][t=..][asset_idx=N]> (x1,y1) (x2,y2) (x3,y3) ... </polygon>
- <clip [mention=..][asset_idx=N] t="S seconds[ E seconds]" />  (moment or [at, until] range; <clip ...></clip> too)
- <track [mention=..][asset_idx=N]> ...point|point_box|polygon waypoints of one kind, each with t... </track>
- <collection [mention=..][asset_idx=N]> ...leaf, clip and track children (no nested collections)... </collection>

Attribute values may be double-quoted, single-quoted or unquoted; they are HTML-unescaped and an empty value means
absent. Spatial ``t`` accepts "1.5", "1.5 seconds", "1.5s" and "1 second"; a clip ``t`` holds one or two such times.
Coordinates are unsigned integers written as (x,y) tuples, or as bare numbers ("10 20 30 40"). Tag names are
case-insensitive. Children inherit ``mention``/``asset_idx`` (and a collection's legacy ``t``) from their container;
a container's ``asset_idx`` is also set on its children in the parsed tree (see ``pointing.types``), and serialization
writes a child's ``asset_idx`` only where it differs from its container's.

Helpers
- parse_text(text, expects=None) → ordered segments; raises ParseError on malformed markup
- parse_annotations(text, expects=None, strict=False) → AnnotationParse(segments, errors); lenient by default
- collect_annotations(text, expects=None, strict=False) → AnnotationCollection of flattened annotations
- extract_points / extract_clips / extract_tracks → flattened annotations with inherited mention/t/asset_idx
- scan_leaves(text, expects=None) → closed leaves with their context, tolerating unclosed containers (streaming)
- resolve_asset_idx(annotation, n_assets) → the asset an annotation refers to
- strip_tags(text) → remove all annotation markup
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from html import escape, unescape
from typing import Any, Literal, NamedTuple

from ..errors import ParseError
from .types import BoundingBox, Clip, ClipTimestamp, Collection, Polygon, SinglePoint, Track, _track_asset_idx

BOX_MIN_POINTS = 2
POLYGON_MIN_POINTS = 3

# Leaf tag → segment kind.
_LEAF_KINDS = {"point": "point", "point_box": "box", "polygon": "polygon"}
_GEOMETRY_EXPECTS = frozenset({"point", "box", "polygon"})

# One markup token: an open tag, a close tag or a self-closing clip. Attributes are a sequence of complete quoted
# strings (which may contain `>`) or single characters other than quotes, `>` and `<` (so a stray `<point` in prose
# cannot swallow the next real tag). `\b` keeps `<point` from matching `<point_box`.
_TAG_TOKEN = re.compile(
    r"<(?P<close>/)?(?P<tag>point_box|point|polygon|clip|collection|track)\b"
    r"(?P<attrs>(?:\"[^\"]*\"|'[^']*'|[^<>\"'])*)>",
    re.IGNORECASE,
)
# name="v", name='v' or name=v; bare quoted strings are matched (and skipped) so a `t=` inside a quoted value is
# never read as an attribute.
_ATTR = re.compile(r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))|"[^"]*"|'[^']*'""")
_COORD = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_ASSET_IDX = re.compile(r"\s*\+?(\d+)\s*")
_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_TIME = rf"({_NUMBER})\s*(?:seconds|second|s)?"
_SPATIAL_T = re.compile(rf"\s*{_TIME}\s*", re.IGNORECASE)
_CLIP_T = re.compile(rf"\s*{_TIME}(?:(?:\s+|\s*[-,]\s*){_TIME})?\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Attribute, time and coordinate grammar
# ---------------------------------------------------------------------------


def _parse_attrs(tag_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for m in _ATTR.finditer(tag_attrs):
        if m.group(1) is None:
            continue
        value = next(v for v in m.group(2, 3, 4) if v is not None)
        attrs[m.group(1)] = unescape(value)
    return attrs


def _text_attr(attrs: dict[str, str], key: str) -> str | None:
    return attrs.get(key) or None


def _asset_idx_attr(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    m = _ASSET_IDX.fullmatch(raw)
    if m is None:
        raise ParseError(
            f"Invalid asset_idx {raw!r}: expected a non-negative integer",
            code="invalid_asset_idx",
            details={"value": raw},
        )
    return int(m.group(1))


def _parse_time_value(raw: str) -> float:
    """Parse a spatial time ("1.5", "1.5 seconds", "1.5s", "1 second") into seconds."""

    m = _SPATIAL_T.fullmatch(raw)
    value = float(m.group(1)) if m else math.nan
    if not math.isfinite(value):
        raise ParseError(
            f'Invalid t {raw!r}: expected seconds such as t="1.5 seconds"',
            code="invalid_time",
            details={"value": raw},
        )
    return value


def _time_attr(raw: str | None) -> float | None:
    return None if raw is None or raw == "" else _parse_time_value(raw)


def _parse_clip_t(raw: str | None) -> ClipTimestamp | None:
    """Parse a clip ``t``: one time (moment) or two times (range); None when absent or malformed."""

    m = _CLIP_T.fullmatch(raw) if raw else None
    if m is None:
        return None
    at = float(m.group(1))
    until = float(m.group(2)) if m.group(2) is not None else None
    if not math.isfinite(at) or (until is not None and not math.isfinite(until)):
        return None
    return ClipTimestamp(at=at, until=until)


def _parse_coords(body: str) -> list[SinglePoint]:
    pts = [SinglePoint(int(x), int(y)) for x, y in _COORD.findall(body)]
    if pts:
        return pts
    # Fallback for the raw coord-decoder form: whitespace-separated digit runs, taken in pairs.
    tokens = body.split()
    if tokens and len(tokens) % 2 == 0 and all(tok.isascii() and tok.isdigit() for tok in tokens):
        return [SinglePoint(int(tokens[i]), int(tokens[i + 1])) for i in range(0, len(tokens), 2)]
    return []


def _parse_leaf(tag: str, attrs_text: str, body: str) -> SinglePoint | BoundingBox | Polygon:
    pts = _parse_coords(body)
    details = {"body": body, "points_found": len(pts)}
    obj: SinglePoint | BoundingBox | Polygon
    if tag == "point":
        if len(pts) != 1:
            raise ParseError(
                f"Malformed <point> tag: expected coordinates like (x,y) but got: {body!r}",
                code="invalid_point_coords",
                details=details,
            )
        obj = pts[0]
    elif tag == "point_box":
        if len(pts) != BOX_MIN_POINTS:
            raise ParseError(
                f"Malformed <point_box> tag: expected 2 coordinates like (x1,y1) (x2,y2) but got: {body!r}",
                code="invalid_box_coords",
                details=details,
            )
        obj = BoundingBox(pts[0], pts[1])
    else:
        if len(pts) < POLYGON_MIN_POINTS:
            raise ParseError(
                f"Malformed <polygon> tag: expected at least 3 coordinates but got {len(pts)}: {body!r}",
                code="invalid_polygon_coords",
                details=details,
            )
        obj = Polygon(hull=pts)
    attrs = _parse_attrs(attrs_text)
    obj.mention = _text_attr(attrs, "mention")
    obj.t = _time_attr(attrs.get("t"))
    obj.asset_idx = _asset_idx_attr(attrs.get("asset_idx"))
    return obj


def _parse_clip(attrs_text: str) -> Clip:
    attrs = _parse_attrs(attrs_text)
    ts = _parse_clip_t(attrs.get("t"))
    if ts is None:
        raise ParseError(
            f'Malformed <clip /> tag: expected t="S seconds" or t="S seconds E seconds" but got attrs: {attrs_text!r}',
            code="invalid_clip_timestamp",
            details={"attrs": attrs_text},
        )
    return Clip(timestamp=ts, mention=_text_attr(attrs, "mention"), asset_idx=_asset_idx_attr(attrs.get("asset_idx")))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _authored_seconds(value: Any) -> float:
    """A time to write into markup: finite seconds >= 0, else ValueError (a string is parsed as a spatial ``t``)."""

    if isinstance(value, str):
        return _parse_time_value(value)  # unsigned grammar; ParseError(invalid_time) when malformed
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not (math.isfinite(value) and value >= 0):
        raise ValueError(f"invalid_time: expected a finite, non-negative number of seconds, got {value!r}")
    return float(value)


def _authored_asset_idx(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"invalid_asset_idx: expected a non-negative integer, got {value!r}")
    return int(value)


def _format_t(t: Any) -> str:
    return f"{round(_authored_seconds(t), 1)} seconds"


def _format_clip_time(x: Any) -> str:
    return f"{_authored_seconds(x):.6f}".rstrip("0").rstrip(".") + " seconds"


def _attr_string(mention: str | None, t: Any = None, asset_idx: int | None = None, *, clip_t: str | None = None) -> str:
    attrs = []
    if mention:
        attrs.append(f'mention="{escape(mention, quote=True)}"')
    if t is not None:
        attrs.append(f't="{_format_t(t)}"')
    if asset_idx is not None:
        attrs.append(f'asset_idx="{_authored_asset_idx(asset_idx)}"')
    if clip_t is not None:
        attrs.append(f't="{clip_t}"')
    return (" " + " ".join(attrs)) if attrs else ""


def _child_asset_idx(value: Any, inherited: int | None) -> int | None:
    """The ``asset_idx`` to write for an element: omitted when it repeats the one its container gives it."""

    if value is None:
        return None
    value = _authored_asset_idx(value)
    return None if value == inherited else value


def _serialize(obj: Any, inherited_t: Any = None, inherited_asset: int | None = None) -> str:
    # A collection's legacy `t` is never written; it is pushed down to direct spatial children that lack one. An
    # `asset_idx` is written only where it differs from the container's (track waypoints never carry one).
    if isinstance(obj, (SinglePoint, BoundingBox, Polygon)):
        t = obj.t if obj.t is not None else inherited_t
        attr = _attr_string(obj.mention, t, _child_asset_idx(obj.asset_idx, inherited_asset))
        if isinstance(obj, SinglePoint):
            return f"<point{attr}> ({obj.x},{obj.y}) </point>"
        if isinstance(obj, BoundingBox):
            a, b = obj.top_left, obj.bottom_right
            return f"<point_box{attr}> ({a.x},{a.y}) ({b.x},{b.y}) </point_box>"
        body = " ".join(f"({p.x},{p.y})" for p in obj.hull)
        return f"<polygon{attr}> {body} </polygon>"
    if isinstance(obj, Clip):
        ts = obj.timestamp
        clip_t = _format_clip_time(ts.at)
        if ts.until is not None:
            clip_t += " " + _format_clip_time(ts.until)
        asset_idx = _child_asset_idx(obj.asset_idx, inherited_asset)
        return f"<clip{_attr_string(obj.mention, asset_idx=asset_idx, clip_t=clip_t)} />"
    if isinstance(obj, Track):
        if not obj.points:
            raise ValueError("A track needs at least one waypoint (the server does not parse an empty <track>)")
        # The one asset the track follows (the server rejects a track whose waypoints disagree).
        track_asset = _track_asset_idx(obj.points, obj.asset_idx)
        inner = " ".join(_serialize(p, inherited_asset=track_asset) for p in obj.points)
        asset_idx = _child_asset_idx(track_asset, inherited_asset)
        return f"<track{_attr_string(obj.mention, asset_idx=asset_idx)}> {inner} </track>"
    if isinstance(obj, Collection):
        t = obj.t if obj.t is not None else inherited_t
        asset_idx = _child_asset_idx(obj.asset_idx, inherited_asset)
        effective = obj.asset_idx if obj.asset_idx is not None else inherited_asset
        inner = " ".join(_serialize(p, t, effective) for p in obj.points)
        return f"<collection{_attr_string(obj.mention, asset_idx=asset_idx)}> {inner} </collection>"
    raise TypeError(f"Unsupported type: {type(obj)}")


def PointParser_serialize(obj: Any) -> str:
    """Serialize an annotation as canonical markup (double-quoted attrs, explicit seconds, input order kept)."""

    return _serialize(obj)


class PointParser:
    @staticmethod
    def serialize(obj: Any) -> str:
        return PointParser_serialize(obj)

    @staticmethod
    def parse(text: str) -> list[dict[str, Any]]:
        """Return structured tag segments parsed from text (excludes plain text)."""
        return [seg for seg in parse_text(text) if seg.get("kind") != "text"]


# ---------------------------------------------------------------------------
# Tokenizer and container stack
# ---------------------------------------------------------------------------


class _Token(NamedTuple):
    tag: str
    closing: bool
    self_closing: bool
    attrs: str
    start: int
    end: int


def _tokenize(text: str, start: int = 0) -> list[_Token]:
    tokens: list[_Token] = []
    for m in _TAG_TOKEN.finditer(text, start):
        attrs = m.group("attrs")
        closing = m.group("close") is not None
        tag = m.group("tag").lower()
        stripped = attrs.rstrip()
        self_closing = not closing and stripped.endswith("/")
        # Close tags carry no attributes, and only clips self-close; anything else stays plain text.
        if (closing and stripped) or (self_closing and tag != "clip"):
            continue
        tokens.append(_Token(tag, closing, self_closing, stripped[:-1] if self_closing else attrs, m.start(), m.end()))
    return tokens


def _pairs(tok: _Token, nxt: _Token | None) -> bool:
    """True when ``tok`` opens a leaf or clip that ``nxt`` closes."""

    return nxt is not None and not tok.closing and not tok.self_closing and nxt.closing and nxt.tag == tok.tag


@dataclass
class _Issue:
    error: ParseError
    start: int
    end: int

    def __post_init__(self) -> None:
        self.error.details.setdefault("span", {"start": self.start, "end": self.end})

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.error.code, "message": str(self.error), "span": {"start": self.start, "end": self.end}}


def _issue(code: str, message: str, start: int, end: int) -> _Issue:
    return _Issue(ParseError(message, code=code), start, end)


@dataclass
class _Node:
    """One element of the markup tree; containers are finished (``value`` built) when their close tag arrives."""

    tag: str
    start: int
    end: int
    attrs: str = ""
    value: Any = None  # parsed object; None when the element is invalid (a container skips invalid children)
    issues: list[_Issue] = field(default_factory=list)  # problems with this element itself
    children: list[_Node] = field(default_factory=list)
    closed: bool = True
    mention: str | None = None  # container attributes, read leniently
    t: float | None = None
    asset_idx: int | None = None

    @property
    def kind(self) -> str:
        return _LEAF_KINDS.get(self.tag, self.tag)

    def all_issues(self) -> list[_Issue]:
        issues = list(self.issues)
        for child in self.children:
            issues.extend(child.all_issues())
        return issues


def _element(text: str, tok: _Token, close: _Token | None) -> _Node:
    """Parse a leaf (open + close token) or a clip (self-closing, or open + ``</clip>``)."""

    node = _Node(tok.tag, tok.start, close.end if close is not None else tok.end)
    body = text[tok.end : close.start] if close is not None else ""
    try:
        if tok.tag == "clip":
            if body.strip():
                raise ParseError("Malformed <clip> tag: a clip has no body", code="invalid_clip_body")
            node.value = _parse_clip(tok.attrs)
        else:
            node.value = _parse_leaf(tok.tag, tok.attrs, body)
    except ParseError as err:
        node.issues.append(_Issue(err, node.start, node.end))
    return node


def _check_children(node: _Node) -> None:
    """Record problems that invalidate the whole container; a misplaced child is skipped like any invalid child."""

    if node.tag == "collection":
        for child in node.children:
            if child.tag == "collection":
                node.issues.append(
                    _issue(
                        "invalid_collection_child",
                        "Nested <collection> is not allowed; a collection holds point, point_box, polygon, clip and "
                        "track children",
                        child.start,
                        child.end,
                    )
                )
        return
    for child in node.children:
        if child.tag not in _LEAF_KINDS:
            child.value = None
            child.issues.append(
                _issue(
                    "invalid_track_child",
                    f"<track> may only contain point, point_box or polygon waypoints, not <{child.tag}>",
                    child.start,
                    child.end,
                )
            )
    waypoints = [child.value for child in node.children if child.value is not None]
    if len({type(v) for v in waypoints}) > 1:
        node.issues.append(
            _issue("invalid_track_geometry", "Track waypoints must all be the same kind", node.start, node.end)
        )
    try:
        _track_asset_idx(waypoints, node.asset_idx)
    except ValueError as err:
        node.issues.append(_issue("invalid_track_asset", str(err), node.start, node.end))


def _finish(node: _Node, end: int, *, closed: bool) -> None:
    node.end = end
    node.closed = closed
    attrs = _parse_attrs(node.attrs)
    node.mention = _text_attr(attrs, "mention")
    open_end = node.start + len(node.tag) + len(node.attrs) + 2
    try:
        node.asset_idx = _asset_idx_attr(attrs.get("asset_idx"))
    except ParseError as err:
        node.issues.append(_Issue(err, node.start, open_end))
    try:
        if node.tag == "collection":  # a track's own `t` has no meaning and is ignored
            node.t = _time_attr(attrs.get("t"))
    except ParseError as err:
        node.issues.append(_Issue(err, node.start, open_end))
    _check_children(node)
    # Like the server, skip invalid children (they stay reported); a container with none left (or a closed one that
    # never had any) is not an annotation. An unclosed empty container is still arriving and stays pending.
    values = [child.value for child in node.children if child.value is not None]
    if node.issues or (not values and (node.children or closed)):
        return
    # The containers push their `asset_idx` down onto the children (a track first takes its waypoints' shared one).
    if node.tag == "collection":
        node.value = Collection(values, node.mention, node.t, asset_idx=node.asset_idx, complete=closed)
    else:
        node.value = Track(values, node.mention, asset_idx=node.asset_idx, complete=closed)


def _build_tree(text: str, start: int = 0) -> list[_Node]:
    """Parse markup (from ``start``) into top-level element nodes. Unclosed containers at the end stay open
    (``closed=False``)."""

    tokens = _tokenize(text, start)
    roots: list[_Node] = []
    stack: list[_Node] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        i += 1
        siblings = stack[-1].children if stack else roots
        if tok.tag in _LEAF_KINDS or tok.tag == "clip":
            if tok.self_closing or _pairs(tok, nxt):
                siblings.append(_element(text, tok, None if tok.self_closing else nxt))
                i += 0 if tok.self_closing else 1
            elif stack and (tok.closing or nxt is not None):
                # A stray tag inside a container is an error (at top level it is prose). An open tag with nothing
                # after it is a leaf still arriving (streamed or truncated output) and stays pending instead.
                code, what = ("unexpected_close_tag", "Unexpected") if tok.closing else ("unclosed_tag", "Unclosed")
                stack[-1].issues.append(_issue(code, f"{what} <{tok.tag}> tag", tok.start, tok.end))
        elif not tok.closing:
            node = _Node(tok.tag, tok.start, len(text), attrs=tok.attrs, closed=False)
            siblings.append(node)
            stack.append(node)
        else:
            _close(stack, tok)
    while stack:  # truncated output: keep what arrived, never synthesize the close tag
        _finish(stack.pop(), len(text), closed=False)
    return roots


def _close(stack: list[_Node], tok: _Token) -> None:
    depth = next((d for d in range(len(stack) - 1, -1, -1) if stack[d].tag == tok.tag), None)
    if depth is None:
        if stack:
            stack[-1].issues.append(_issue("unexpected_close_tag", f"Unexpected </{tok.tag}> tag", tok.start, tok.end))
        return
    while len(stack) > depth + 1:
        inner = stack.pop()
        inner.issues.append(_issue("unclosed_tag", f"Unclosed <{inner.tag}> tag", inner.start, tok.start))
        _finish(inner, tok.start, closed=False)
    _finish(stack.pop(), tok.end, closed=True)


# ---------------------------------------------------------------------------
# Segment views
# ---------------------------------------------------------------------------


def _family(expects: str | None, *, default: str) -> str | None:
    if expects is None:
        return default
    if expects == "clip":
        return "clip"
    return "geometry" if expects in _GEOMETRY_EXPECTS else None


def _incomplete(node: _Node) -> _Issue:
    return _issue(
        "incomplete_annotation",
        f"<{node.tag}> is not closed (the output may be truncated); completed children were kept",
        node.start,
        node.end,
    )


def _view(roots: list[_Node], family: str, *, keep_incomplete: bool) -> tuple[list[tuple[_Node, Any]], list[_Issue]]:
    """Select the structured elements for a family: ``all``, ``geometry`` (legacy default) or ``clip``.

    Invalid elements are reported and left as text. With ``keep_incomplete`` an unclosed container is kept
    (``complete=False``) and reported; otherwise it stays text.
    """

    elements: list[tuple[_Node, Any]] = []
    issues: list[_Issue] = []
    for node in roots:
        if node.tag in _LEAF_KINDS or node.tag == "clip":
            if family != "all" and (family == "clip") != (node.tag == "clip"):
                continue  # out of family: stays text
            if node.value is not None:
                elements.append((node, node.value))
            issues.extend(node.issues)
        elif node.closed or keep_incomplete:
            if family == "clip":
                _clip_view(node, elements, issues)
                continue
            problems = node.all_issues()
            if not node.closed:
                problems.append(_incomplete(node))
            if node.value is not None:
                elements.append((node, node.value))
            issues.extend(problems)
    return elements, issues


def _clip_view(node: _Node, elements: list[tuple[_Node, Any]], issues: list[_Issue]) -> None:
    # In clip mode a collection is a transparent wrapper: its clips surface with inherited context, the rest stays
    # text, and only structural problems or malformed clips count.
    if node.tag != "collection":
        return
    problems = node.issues + [issue for child in node.children if child.tag == "clip" for issue in child.issues]
    if not node.closed:
        problems.append(_incomplete(node))
    issues.extend(problems)
    if node.issues:
        return
    for child in node.children:
        if child.tag == "clip" and child.value is not None:
            elements.append((child, _inherit(child.value, mention=node.mention, asset_idx=node.asset_idx)))


def _segments(text: str, elements: list[tuple[_Node, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    idx = 0
    for node, value in elements:
        if node.start > idx:
            segments.append({"kind": "text", "text": text[idx : node.start], "span": {"start": idx, "end": node.start}})
        segments.append({"kind": node.kind, "value": value, "span": {"start": node.start, "end": node.end}})
        idx = node.end
    if idx < len(text):
        segments.append({"kind": "text", "text": text[idx:], "span": {"start": idx, "end": len(text)}})
    return segments


def parse_text(text: str, *, expects: str | None = None) -> list[dict[str, Any]]:
    """Return ordered segments: text and tag segments with spans. Raises ParseError on malformed markup.

    The `expects` parameter selects which tag family becomes structured; other tags stay in text segments:
      - ``"clip"``: ``<clip />`` tags, including clips inside collections (which carry the collection's
        mention/asset_idx)
      - ``"point"`` | ``"box"`` | ``"polygon"`` | ``None``: point/point_box/polygon/track/collection tags
        (collections and tracks are parsed completely; a top-level clip stays text)
      - anything else: no tag parsing; the input is returned as a single text segment

    Unclosed containers at the end of the text (partial output) stay text; use ``parse_annotations`` to keep them.

    Segment shapes:
      - {"kind": "text", "text": str, "span": {"start": int, "end": int}}
      - {"kind": "point"|"box"|"polygon"|"track"|"collection"|"clip", "value": obj, "span": {...}}
    """
    family = _family(expects, default="geometry")
    if family is None:
        return [{"kind": "text", "text": text, "span": {"start": 0, "end": len(text)}}] if text else []
    elements, issues = _view(_build_tree(text), family, keep_incomplete=False)
    if issues:
        raise min(issues, key=lambda issue: issue.start).error
    return _segments(text, elements)


@dataclass
class AnnotationParse:
    """Lenient parse result: ordered segments plus ``{"code", "message", "span"}`` error entries."""

    segments: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def parse_annotations(text: str, *, expects: str | None = None, strict: bool = False) -> AnnotationParse:
    """Parse markup leniently: a malformed element stays a text segment and adds an error entry.

    As on the server, an invalid child of a collection or track is skipped (and reported) while its container is
    kept. Structural problems (nested collection, mixed track geometry, conflicting track ``asset_idx``, a bad
    container attribute, stray tags), or no valid child left, make the whole container text.

    ``expects=None`` structures every tag kind; ``"point"``/``"box"``/``"polygon"``/``"clip"`` select a family as
    in ``parse_text``. An unclosed container at the end (truncated output) is kept with ``complete=False`` and
    reported as ``incomplete_annotation``; a partial element after its last complete child stays pending.
    ``strict=True`` raises the first problem as ``ParseError`` instead.
    """
    family = _family(expects, default="all")
    if not text or family is None:
        return AnnotationParse(
            [{"kind": "text", "text": text, "span": {"start": 0, "end": len(text)}}] if text else [], []
        )
    elements, issues = _view(_build_tree(text), family, keep_incomplete=True)
    issues.sort(key=lambda issue: issue.start)
    if strict and issues:
        raise issues[0].error
    return AnnotationParse(_segments(text, elements), [issue.as_dict() for issue in issues])


# ---------------------------------------------------------------------------
# Flattening (effective context)
# ---------------------------------------------------------------------------


def _with(obj: Any, **fields: Any) -> Any:
    changes = {key: value for key, value in fields.items() if getattr(obj, key) != value}
    return replace(obj, **changes) if changes else obj


def _inherit(obj: Any, **context: Any) -> Any:
    """Copy ``obj`` filling each unset (``is None``) field from ``context``; never mutates ``obj``."""

    return _with(obj, **{k: v for k, v in context.items() if v is not None and getattr(obj, k) is None})


_NO_CONTEXT: dict[str, Any] = {"mention": None, "t": None, "asset_idx": None}


def _flatten(values: Iterable[Any]) -> tuple[list[Any], list[Track]]:
    items: list[Any] = []
    tracks: list[Track] = []
    for value in values:
        _flatten_into(value, _NO_CONTEXT, items, tracks)
    return items, tracks


def _flatten_into(obj: Any, ctx: dict[str, Any], items: list[Any], tracks: list[Track]) -> None:
    if isinstance(obj, Collection):
        inner = {key: getattr(obj, key) if getattr(obj, key) is not None else value for key, value in ctx.items()}
        for child in obj.points:
            _flatten_into(child, inner, items, tracks)
    elif isinstance(obj, Track):
        track = _inherit(obj, mention=ctx["mention"], asset_idx=ctx["asset_idx"])
        tracks.append(track)
        for leaf in track.points:  # a waypoint names the track's object; its own mention is ignored
            own_asset = leaf.asset_idx if leaf.asset_idx is not None else track.asset_idx
            items.append(_with(leaf, mention=track.mention, asset_idx=own_asset))
    elif isinstance(obj, Clip):
        items.append(_inherit(obj, mention=ctx["mention"], asset_idx=ctx["asset_idx"]))
    else:
        items.append(_inherit(obj, **ctx))


def _kind_of(obj: Any) -> str | None:
    if isinstance(obj, SinglePoint):
        return "point"
    if isinstance(obj, BoundingBox):
        return "box"
    if isinstance(obj, Polygon):
        return "polygon"
    return None


def _structured(segments: list[dict[str, Any]]) -> list[Any]:
    return [seg["value"] for seg in segments if seg["kind"] != "text"]


@dataclass
class AnnotationCollection:
    """Flattened annotations with effective context (own ?? track ?? collection).

    Track waypoints appear in ``points``/``boxes``/``polygons`` too; ``tracks`` and ``parsed`` keep the tree.
    """

    points: list[SinglePoint]
    boxes: list[BoundingBox]
    polygons: list[Polygon]
    clips: list[Clip]
    tracks: list[Track]
    parsed: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def collect_annotations(text: str, *, expects: str | None = None, strict: bool = False) -> AnnotationCollection:
    """Parse leniently (see ``parse_annotations``) and flatten every annotation into kind buckets."""

    parsed = parse_annotations(text, expects=expects, strict=strict)
    items, tracks = _flatten(_structured(parsed.segments))
    return AnnotationCollection(
        points=[item for item in items if isinstance(item, SinglePoint)],
        boxes=[item for item in items if isinstance(item, BoundingBox)],
        polygons=[item for item in items if isinstance(item, Polygon)],
        clips=[item for item in items if isinstance(item, Clip)],
        tracks=tracks,
        parsed=parsed.segments,
        errors=parsed.errors,
    )


def extract_points(text: str, expected: Literal["point", "box", "polygon"] | None = None) -> list[Any]:
    """Extract only the requested tag type (if provided) in order of appearance.

    Collection children and track waypoints are included with their inherited mention/t/asset_idx.
    """
    items, _ = _flatten(_structured(parse_text(text)))
    return [item for item in items if _kind_of(item) is not None and (expected is None or _kind_of(item) == expected)]


def extract_tracks(text: str, expected: Literal["point", "box", "polygon"] | None = None) -> list[Track]:
    """Extract tracks (top level and inside collections) in order, with inherited mention/asset_idx resolved."""

    _, tracks = _flatten(_structured(parse_text(text)))
    if expected is None:
        return tracks
    return [tr for tr in tracks if tr.points and _kind_of(tr.points[0]) == expected]


def extract_clips(text: str) -> list[Clip]:
    """Extract ``<clip />`` annotations. Clips inside a ``<collection>`` inherit its ``mention``/``asset_idx``.

    Malformed clips are skipped.
    """

    return collect_annotations(text, expects="clip").clips


def _closed_container_ends(nodes: list[_Node], ends: dict[int, int]) -> dict[int, int]:
    """``{start: end}`` of the outermost closed collections and tracks (looking inside unclosed ones)."""

    for node in nodes:
        if node.tag in ("collection", "track"):
            if node.closed:
                ends[node.start] = node.end
            else:
                _closed_container_ends(node.children, ends)
    return ends


def strip_tags(text: str) -> str:
    """Remove all annotation markup (elements, container tags) and return plain text only.

    A closed collection or track is removed whole, with the layout whitespace between its children; an unclosed or
    stray container tag is removed on its own.
    """

    tokens = _tokenize(text)
    container_ends = _closed_container_ends(_build_tree(text), {}) if tokens else {}
    parts: list[str] = []
    idx = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        i += 1
        end = tok.end
        if tok.start in container_ends:
            end = container_ends[tok.start]
            while i < len(tokens) and tokens[i].start < end:
                i += 1
        elif tok.tag in _LEAF_KINDS or tok.tag == "clip":
            if nxt is not None and _pairs(tok, nxt):
                end = nxt.end
                i += 1
            elif not tok.self_closing:
                continue  # a stray leaf tag is prose
        parts.append(text[idx : tok.start])
        idx = end
    parts.append(text[idx:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Streaming and asset resolution
# ---------------------------------------------------------------------------


def scan_leaves(text: str, *, expects: str | None = None) -> list[dict[str, Any]]:
    """Return every closed, well-formed leaf (point/box/polygon/clip) in order, for streaming.

    Unclosed containers are fine: a leaf inside one still resolves its context from the open tags. Leaves of a
    container with a structural problem so far (bad attribute, nested collection, mixed track geometry or assets,
    stray tags) are skipped, as ``collect_annotations`` drops that container. Each entry is
    ``{"kind", "value", "span", "context": {"mention", "t", "asset_idx", "container"}, "container_start"}`` where
    ``value`` carries the effective mention/t/asset_idx and ``container`` is ``"track"``, ``"collection"`` or None.
    ``expects`` keeps one kind only.
    """
    if expects is not None and expects not in _GEOMETRY_EXPECTS and expects != "clip":
        return []
    return _scan_leaves(text, expects)[0]


def _scan_leaves(text: str, expects: str | None, start: int = 0) -> tuple[list[dict[str, Any]], int]:
    """:func:`scan_leaves` of ``text`` from ``start`` (spans still index ``text``), and where a scan of the same text
    with more appended can start instead: the end of the last complete top-level element, since nothing before it
    changes as text arrives. A stream rescans only the element still arriving."""
    roots = _build_tree(text, start)
    found: list[dict[str, Any]] = []
    _scan_into(roots, (None, None, None, None, None), expects, found)
    resume = start
    for node in roots:
        if node.closed:  # only an unclosed container (the last root) is still arriving
            resume = node.end
    return found, resume


def _scan_into(nodes: list[_Node], ctx: tuple, expects: str | None, found: list[dict[str, Any]]) -> None:
    mention, t, asset_idx, container, container_start = ctx
    for node in nodes:
        if node.tag in ("collection", "track"):
            if node.issues:  # structurally invalid (the collector drops it whole): none of its leaves count
                continue
            # A track without its own selector takes its waypoints' shared one, as in the tree (its built value has it).
            own = node.value.asset_idx if node.tag == "track" and node.value is not None else node.asset_idx
            inner = (
                node.mention if node.mention is not None else mention,
                None if node.tag == "track" else (node.t if node.t is not None else t),
                own if own is not None else asset_idx,
                node.tag,
                node.start,
            )
            _scan_into(node.children, inner, expects, found)
            continue
        if node.value is None or (expects is not None and node.kind != expects):
            continue
        if container == "track":  # a waypoint names the track's object
            value = _with(_inherit(node.value, asset_idx=asset_idx), mention=mention)
        elif node.tag == "clip":
            value = _inherit(node.value, mention=mention, asset_idx=asset_idx)
        else:
            value = _inherit(node.value, mention=mention, t=t, asset_idx=asset_idx)
        found.append(
            {
                "kind": node.kind,
                "value": value,
                "span": {"start": node.start, "end": node.end},
                "context": {
                    "mention": value.mention,
                    "t": getattr(value, "t", None),
                    "asset_idx": value.asset_idx,
                    "container": container,
                },
                "container_start": container_start,
            }
        )


def resolve_asset_idx(annotation: Any, n_assets: int | None) -> int | None:
    """Return the media asset an annotation refers to, given the request's asset count.

    An explicit ``asset_idx`` wins (``ValueError`` when it is out of range). Without one, the annotation refers to the
    last asset (``n_assets - 1``, so 0 for a single asset); None only when there is no asset (``n_assets`` 0 or None).
    Pass flattened annotations (e.g. from ``collect_annotations``) so inherited selectors are already applied. The
    result is never written back: indices are request-relative.
    """
    own = getattr(annotation, "asset_idx", None)
    if own is not None:
        if n_assets is not None and not 0 <= own < n_assets:
            raise ValueError(f"asset_idx {own} is out of range for a request with {n_assets} media asset(s)")
        return own
    return n_assets - 1 if n_assets else None
