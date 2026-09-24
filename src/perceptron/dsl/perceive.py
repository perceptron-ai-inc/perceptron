"""DSL compiler and `@perceive` decorator.

Compiles typed nodes (text/image/video/audio/video_frames/point/box/polygon) into
a Task shape and optionally executes it via the Client. Performs compile-time
validation of anchoring (``asset_idx``) and of coordinates on the normalized
0-1000 grid, returning issues (non-strict) or raising (strict).

PerceiveResult
- text: final text (if executed)
- points: list of parsed pointing objects if `expects` set and present
- parsed: ordered segments mixing text and all tags with spans
- errors: semantic/validation issues from compilation, and parse issues in the answer
- raw: the provider response
- finish_reason, tool_calls, usage, id, model, request_id, message: completion metadata (see `PerceiveResult`)
"""

from __future__ import annotations

import base64
import inspect
import numbers
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover
    PILImage = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from .._lowering import MEDIA_PART_TYPES, entry_to_part
from .._providers import _provider_key, _resolve_provider, missing_api_key_message, provider_api_key
from ..chat import _COMPLETE_FINISH_REASONS, ChatCompletionMessage, ToolCall
from ..client import (
    _PROVIDER_CONFIG,
    AsyncClient,
    Client,
    ResponseFormat,
    _inject_expectation_hint,
    _reject_unexpected_kwargs,
)
from ..config import settings
from ..errors import (
    ANCHOR_AMBIGUOUS,
    ANCHOR_MISSING,
    ANCHOR_UNKNOWN,
    BOUNDS_OUT_OF_RANGE,
    CREDENTIALS_MISSING,
    INVALID_MEDIA_PATH,
    INVALID_PARAMETER,
    INVALID_POLYGON,
    REASONING_DISABLED_FOR_THINKING_MODEL,
    REASONING_NOT_SUPPORTED,
    REASONING_REQUIRED_FOR_MODEL,
    AnchorError,
    AuthError,
    BadRequestError,
    ExpectationError,
    SDKError,
)
from ..files import _is_mp3_frame
from ..pointing.geometry import NORMALIZED_COORD_MAX, scale_points_to_pixels
from ..pointing.parser import PointParser_serialize, resolve_asset_idx
from ..pointing.types import BoundingBox, Clip, Polygon, SinglePoint
from .nodes import (
    Agent,
    DSLNode,
    Sequence,
    System,
    Text,
    ToolResult,
    VideoFrame,
    _frame_image,
)
from .nodes import (
    Audio as AudioNode,
)
from .nodes import (
    BoxTag as BoxTagNode,
)
from .nodes import (
    Image as ImageNode,
)
from .nodes import (
    PointTag as PointTagNode,
)
from .nodes import (
    PolygonTag as PolygonTagNode,
)
from .nodes import (
    Video as VideoNode,
)
from .nodes import (
    VideoFrames as VideoFramesNode,
)

# Media nodes; each takes the next `asset_idx` (images in tool results do too).
_MEDIA_NODES = (ImageNode, VideoNode, AudioNode, VideoFramesNode)
_TAG_NODES = (PointTagNode, BoxTagNode, PolygonTagNode)
_POLYGON_MIN_VERTICES = 3

_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",  # JPEG
    b"GIF87a",
    b"GIF89a",
    b"BM",  # BMP
    b"II*\x00",  # TIFF (little endian)
    b"MM\x00*",  # TIFF (big endian)
)

_WEBP_SIGNATURE_LENGTH = 12

# Maps PIL's format names to the wire-protocol MIME subtype.
_PIL_FORMAT_TO_WIRE = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}


def _is_webp(data: bytes) -> bool:
    return len(data) >= _WEBP_SIGNATURE_LENGTH and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _looks_like_image(data: bytes) -> bool:
    return any(data.startswith(sig) for sig in _IMAGE_SIGNATURES) or _is_webp(data)


def _validate_image_bytes(data: bytes, *, origin: str) -> dict[str, Any]:
    """Ensure the payload is an actual bitmap; raise otherwise."""

    meta: dict[str, Any] = {}
    pil_exc: Exception | None = None
    if PILImage is not None:
        try:
            with PILImage.open(BytesIO(data)) as im:
                meta["width"], meta["height"] = im.size
                if im.format and im.format in _PIL_FORMAT_TO_WIRE:
                    meta["format"] = _PIL_FORMAT_TO_WIRE[im.format]
                return meta
        except Exception as exc:
            pil_exc = exc
    if not _looks_like_image(data):
        reason = "decoder_failed" if pil_exc is not None else "unknown_format"
        details = {"origin": origin, "reason": reason}
        raise BadRequestError(
            "Image payload is not a decodable bitmap", code="invalid_image", details=details
        ) from pil_exc
    return meta


def _encode_bytes(data: bytes) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    if PILImage is not None:
        try:
            with PILImage.open(BytesIO(data)) as im:
                meta["width"], meta["height"] = im.size
                if im.format and im.format in _PIL_FORMAT_TO_WIRE:
                    meta["format"] = _PIL_FORMAT_TO_WIRE[im.format]
        except Exception:
            pass
    b64 = base64.b64encode(data).decode("ascii")
    return b64, meta


def _is_passthrough_url(obj: Any) -> bool:
    """True for an http(s) or ``data:`` URL, which is sent as given rather than read and encoded."""
    return isinstance(obj, str) and (obj.startswith("data:") or urlparse(obj).scheme in {"http", "https"})


def _read_media_file(path: str | Path, kind: str) -> tuple[Path, bytes]:
    """Read a local media file; I/O failures become ``BadRequestError(code="invalid_media_path")``."""
    p = Path(path)
    try:
        return p, p.read_bytes()
    except (OSError, ValueError) as exc:  # ValueError: e.g. an embedded null byte
        reason = getattr(exc, "strerror", None) or str(exc)
        raise BadRequestError(
            f"Could not read {kind} file {str(p)[:200]!r}: {reason}",
            code=INVALID_MEDIA_PATH,
            details={"origin": str(p)},
        ) from exc


def _to_b64_image(obj: Any) -> tuple[str, dict]:
    """Return base64 string and metadata with width/height.

    Accepts: Path/str (path, http/https URL or data URL), bytes, file-like, PIL.Image.Image, numpy.ndarray
    (HxWxC, uint8). URLs are returned verbatim with ``meta["url"]=True``, like video and audio URLs.
    """
    meta: dict[str, Any] = {}

    if _is_passthrough_url(obj):
        return obj, {"url": True}

    if isinstance(obj, (str, Path)):
        p, data = _read_media_file(obj, "image")
        meta = _validate_image_bytes(data, origin=str(p))
        b64 = base64.b64encode(data).decode("ascii")
        return b64, meta

    if isinstance(obj, bytes):
        b64, meta = _encode_bytes(obj)
        return b64, meta

    if PILImage is not None and isinstance(obj, PILImage.Image):  # type: ignore[attr-defined]
        meta["width"], meta["height"] = obj.size
        buf = BytesIO()
        obj.save(buf, format="PNG")
        meta["format"] = "png"
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, meta

    if np is not None and isinstance(obj, np.ndarray):  # type: ignore[arg-type]
        h, w = obj.shape[:2]
        meta["width"], meta["height"] = int(w), int(h)
        if PILImage is None:
            raise RuntimeError("Pillow is required to encode numpy arrays to PNG")
        im = PILImage.fromarray(obj)
        buf = BytesIO()
        im.save(buf, format="PNG")
        meta["format"] = "png"
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, meta

    raise TypeError(f"Unsupported image object: {type(obj)}")


def _detect_video_format(data: bytes) -> str | None:
    """Wire-protocol video format from magic bytes (mp4/webm only)."""

    # mp4 / quicktime: bytes [4:8] == b"ftyp"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "mp4"
    # webm / matroska: EBML header
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    return None


def _to_b64_video(obj: Any) -> tuple[str, dict[str, Any]]:
    """Return (base64-or-URL, metadata).

    Accepts: Path/str (path, http/https URL or data URL) or bytes. URLs are
    returned verbatim with ``meta["url"]=True``; bytes are base64-encoded and
    the format is detected from magic bytes (mp4 / webm).
    """

    meta: dict[str, Any] = {}

    if _is_passthrough_url(obj):
        meta["url"] = True
        return obj, meta

    if isinstance(obj, (str, Path)):
        p, data = _read_media_file(obj, "video")
        fmt = _detect_video_format(data)
        if fmt is None:
            raise BadRequestError(
                "Video format could not be detected. The wire protocol supports mp4 and webm.",
                code="invalid_video",
                details={"origin": str(p)},
            )
        meta["format"] = fmt
        b64 = base64.b64encode(data).decode("ascii")
        return b64, meta

    if isinstance(obj, bytes):
        fmt = _detect_video_format(obj)
        if fmt is None:
            raise BadRequestError(
                "Video format could not be detected from bytes. The wire protocol supports mp4 and webm.",
                code="invalid_video",
            )
        meta["format"] = fmt
        b64 = base64.b64encode(obj).decode("ascii")
        return b64, meta

    raise TypeError(f"Unsupported video object: {type(obj)}")


_WAV_SIGNATURE_LENGTH = 12


def _is_wav(data: bytes) -> bool:
    return len(data) >= _WAV_SIGNATURE_LENGTH and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _is_mp3(data: bytes) -> bool:
    # ID3v2 tag, or an MPEG Layer III frame header as the server checks it (ADTS AAC and MPEG Layer I/II audio share
    # the frame sync but are not MP3)
    return data.startswith(b"ID3") or _is_mp3_frame(data)


def _detect_audio_format(data: bytes) -> str | None:
    """Wire-protocol audio format from magic bytes (wav/mp3/flac only)."""

    if _is_wav(data):
        return "wav"
    if data.startswith(b"fLaC"):
        return "flac"
    if _is_mp3(data):
        return "mp3"
    return None


def _to_b64_audio(obj: Any) -> tuple[str, dict[str, Any]]:
    """Return (base64-or-URL, metadata).

    Accepts: Path/str (path, http/https URL or data URL) or bytes. URLs are
    returned verbatim with ``meta["url"]=True``; bytes are base64-encoded and
    the format is detected from magic bytes (wav / mp3 / flac).
    """

    meta: dict[str, Any] = {}

    if _is_passthrough_url(obj):
        meta["url"] = True
        return obj, meta

    if isinstance(obj, (str, Path)):
        p, data = _read_media_file(obj, "audio")
        fmt = _detect_audio_format(data)
        if fmt is None:
            raise BadRequestError(
                "Audio format could not be detected. The wire protocol supports wav, mp3, and flac.",
                code="invalid_audio",
                details={"origin": str(p)},
            )
        meta["format"] = fmt
        b64 = base64.b64encode(data).decode("ascii")
        return b64, meta

    if isinstance(obj, bytes):
        fmt = _detect_audio_format(obj)
        if fmt is None:
            raise BadRequestError(
                "Audio format could not be detected from bytes. The wire protocol supports wav, mp3, and flac.",
                code="invalid_audio",
            )
        meta["format"] = fmt
        b64 = base64.b64encode(obj).decode("ascii")
        return b64, meta

    raise TypeError(f"Unsupported audio object: {type(obj)}")


def _media_entry(node: ImageNode | VideoNode | AudioNode, *, role: str = "user") -> dict[str, Any]:
    """The task entry for an image, video or audio node: an uploaded file's id, a URL passed through, or base64."""
    kind = "image" if isinstance(node, ImageNode) else "video" if isinstance(node, VideoNode) else "audio"
    if node.file_id is not None:
        return {"type": kind, "role": role, "file_id": node.file_id}
    if kind == "image":
        b64, meta = _to_b64_image(node.obj)
        return {
            "type": "image",
            "role": role,
            "content": b64,
            "format": meta.get("format"),
            "metadata": {
                "width": meta.get("width"),
                "height": meta.get("height"),
            },
            # This check decides what is a URL (not the lowering's prefix fallback), as for video and audio.
            **({"url": True} if meta.get("url") else {}),
        }
    payload, meta = (_to_b64_video if kind == "video" else _to_b64_audio)(node.obj)
    return {
        "type": kind,
        "role": role,
        "content": payload,
        "format": meta.get("format"),
        "url": bool(meta.get("url")),
    }


def _frame_entry(frame: VideoFrame) -> dict[str, Any]:
    """A ``video_frames`` frame: its image as a URL (data URL for local images), or an uploaded file's id."""
    node = _frame_image(frame.image)
    if node.file_id is not None:
        return {"file_id": node.file_id, "timestamp_ms": frame.timestamp_ms}
    part = entry_to_part(_media_entry(node))  # the same URL an image(...) of this frame would send
    return {"url": part["image_url"]["url"], "timestamp_ms": frame.timestamp_ms}


def _video_frames_entry(node: VideoFramesNode) -> dict[str, Any]:
    return {"type": "video_frames", "role": "user", "frames": [_frame_entry(frame) for frame in node.frames]}


def _tool_call_dict(call: Any) -> dict[str, Any]:
    if isinstance(call, Mapping):
        return dict(call)
    to_dict = getattr(call, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"agent() tool_calls must be ToolCall objects or tool call dicts; got {type(call).__name__}")


def _agent_entry(node: Agent) -> dict[str, Any]:
    """A plain assistant text entry (merged like other text), or a standalone ``assistant_turn`` for a replayed turn."""
    if node.tool_calls is None and node.reasoning_content is None and isinstance(node.content, str):
        return {"type": "text", "role": "assistant", "content": node.content}
    tool_calls = node.tool_calls
    if tool_calls is not None and not isinstance(tool_calls, (list, tuple)):
        raise TypeError(f"agent() tool_calls must be a list; got {type(tool_calls).__name__}")
    if node.content is None and not tool_calls and node.reasoning_content is None:
        # The API rejects an assistant message with none of these.
        raise TypeError("agent() needs content, tool_calls or reasoning_content")
    return {
        "type": "assistant_turn",
        "role": "assistant",
        "content": node.content,
        "tool_calls": [_tool_call_dict(call) for call in tool_calls] if tool_calls else None,
        "reasoning_content": node.reasoning_content,
    }


def _tool_result_entry(node: ToolResult) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in node.content:
        if isinstance(item, Text):
            items.append({"type": "text", "role": "tool", "content": item.content})
        elif isinstance(item, ImageNode):
            items.append(_media_entry(item, role="tool"))
        else:
            raise TypeError(f"Tool results may contain only text and images; got {type(item).__name__}")
    return {"type": "tool_result", "role": "tool", "tool_call_id": node.tool_call_id, "content": items}


@dataclass
class _AssetLedger:
    """Media assets numbered in wire order (the ``asset_idx`` space): ``uses`` maps ``id(node)`` to the index of each
    use of a media node, and ``count`` is the number of assets."""

    uses: dict[int, list[int]] = field(default_factory=dict)
    count: int = 0


def _media_items(item: Any) -> list[Any]:
    """The media assets ``item`` puts on the wire, in order: a media node, the images of a tool result, the media of a
    sequence, or a media part given as a wire dict (in chat message content)."""
    if isinstance(item, Sequence):
        return [media for node in item.nodes for media in _media_items(node)]
    if isinstance(item, ToolResult):
        return [node for node in item.content if isinstance(node, ImageNode)]
    if isinstance(item, _MEDIA_NODES) or (isinstance(item, dict) and item.get("type") in MEDIA_PART_TYPES):
        return [item]
    return []


def _asset_ledger(items: Iterable[Any]) -> _AssetLedger:
    """Number the media assets of ``items`` in wire order: image, video, audio and video_frames nodes, images inside
    tool results, and media part dicts share one counter."""
    ledger = _AssetLedger()
    for item in items:
        for media in _media_items(item):
            ledger.uses.setdefault(id(media), []).append(ledger.count)
            ledger.count += 1
    return ledger


def _anchor(
    node: PointTagNode | BoxTagNode | PolygonTagNode, seen: int, ledger: _AssetLedger
) -> tuple[int | None, dict[str, str] | None]:
    """The ``asset_idx`` to write for a tag that follows ``seen`` media assets, and an anchoring issue (or None).

    The index is written only when the prompt has more than one media asset, or when given raw with ``asset_idx=``
    (always written); a single-asset prompt keeps its markup unchanged. ``image=``/``asset=`` is looked up by identity:
    a node used more than once resolves to its latest use before the tag (else its first use after it).
    """
    n_assets = ledger.count
    if node.asset_idx is not None:
        issue = None
        if node.asset_idx >= n_assets:
            issue = {
                "code": ANCHOR_UNKNOWN,
                "message": f"asset_idx={node.asset_idx} is out of range: the prompt has {n_assets} media asset(s)",
            }
        return node.asset_idx, issue
    target = node.asset if node.asset is not None else node.image
    if target is None:
        if n_assets == 1:
            return None, None
        if n_assets == 0:
            message = "Tag has no media asset to refer to"
        else:
            message = "Tag missing image=/asset= in a multi-asset prompt"
        return None, {"code": ANCHOR_MISSING, "message": message}
    if not isinstance(target, _MEDIA_NODES):
        return None, {
            "code": ANCHOR_MISSING,
            "message": "image=/asset= must reference an image(), video(), audio() or video_frames() node",
        }
    uses = ledger.uses.get(id(target))
    if not uses:
        return None, {
            "code": ANCHOR_UNKNOWN,
            "message": "image=/asset= references a media node that is not part of this prompt",
        }
    before = [asset_idx for asset_idx in uses if asset_idx < seen]
    asset_idx = before[-1] if before else uses[0]
    issue = None
    if len(uses) > 1:
        issue = {
            "code": ANCHOR_AMBIGUOUS,
            "message": f"image=/asset= references a media node used {len(uses)} times in this prompt; "
            f"anchored to asset_idx {asset_idx}",
        }
    return (asset_idx if n_assets > 1 else None), issue


def _report(issues: list[dict], issue: dict[str, str], *, strict: bool, error: type[SDKError]) -> None:
    if strict:
        raise error(issue["message"], code=issue["code"], details=issue)
    issues.append(issue)


def _grid_coord(value: Any) -> int | None:
    """``value`` as a coordinate of the normalized 0-1000 grid, or None (bools, non-integers, out of range)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    if not isinstance(value, numbers.Integral) and not float(value).is_integer():
        return None
    coord = int(value)
    return coord if 0 <= coord <= NORMALIZED_COORD_MAX else None


def _grid_point(label: str, x: Any, y: Any, issues: list[dict], *, strict: bool) -> tuple[SinglePoint, bool]:
    """A vertex checked against the 0-1000 grid, and whether it is valid. Integral floats become ints; an invalid
    vertex is reported and written as given."""
    gx, gy = _grid_coord(x), _grid_coord(y)
    if gx is None or gy is None:
        issue = {
            "code": BOUNDS_OUT_OF_RANGE,
            "message": f"{label} ({x},{y}) outside the 0-1000 normalized grid (coordinates are integers 0-1000)",
        }
        _report(issues, issue, strict=strict, error=ExpectationError)
        return SinglePoint(x, y), False
    return SinglePoint(gx, gy), True


def _tag_markup(
    node: PointTagNode | BoxTagNode | PolygonTagNode, asset_idx: int | None, issues: list[dict], *, strict: bool
) -> str:
    """Serialize a tag; coordinates are validated on the normalized grid, independent of any image's pixel size."""
    context: dict[str, Any] = {"mention": node.mention, "t": node.t, "asset_idx": asset_idx}
    if isinstance(node, PointTagNode):
        p, _ = _grid_point("point", node.x, node.y, issues, strict=strict)
        return PointParser_serialize(SinglePoint(p.x, p.y, **context))
    if isinstance(node, BoxTagNode):
        a, a_ok = _grid_point("box corner", node.x1, node.y1, issues, strict=strict)
        b, b_ok = _grid_point("box corner", node.x2, node.y2, issues, strict=strict)
        if a_ok and b_ok and (a.x > b.x or a.y > b.y):
            issue = {
                "code": BOUNDS_OUT_OF_RANGE,
                "message": f"box ({a.x},{a.y}) ({b.x},{b.y}) needs x1 <= x2 and y1 <= y2 (top-left corner first)",
            }
            _report(issues, issue, strict=strict, error=ExpectationError)
        return PointParser_serialize(BoundingBox(a, b, **context))
    if len(node.coords) < _POLYGON_MIN_VERTICES:
        issue = {
            "code": INVALID_POLYGON,
            "message": f"polygon needs at least {_POLYGON_MIN_VERTICES} vertices; got {len(node.coords)}",
        }
        _report(issues, issue, strict=strict, error=ExpectationError)
    hull = [_grid_point("polygon vertex", x, y, issues, strict=strict)[0] for (x, y) in node.coords]
    return PointParser_serialize(Polygon(hull, **context))


def _compile(
    nodes: DSLNode | Sequence,
    *,
    expects: str | None,
    strict: bool,
    ledger: _AssetLedger | None = None,
    first_asset: int = 0,
) -> tuple[dict, list[dict]]:
    """Compile DSL nodes into a Task JSON and return (task, issues).

    Media assets are numbered first (see ``_asset_ledger``) so a tag can anchor to an asset that comes after it. For
    nodes that are one part of a larger request (DSL nodes in chat message content), ``ledger`` numbers the whole
    request's assets and ``first_asset`` is the index of the first asset here.
    """
    seq = nodes if isinstance(nodes, Sequence) else Sequence([nodes])
    if ledger is None:
        ledger = _asset_ledger(seq.nodes)
    seen = first_asset  # media assets before the current node
    content: list[dict[str, Any]] = []
    issues: list[dict] = []

    for node in seq.nodes:
        if isinstance(node, Text):
            content.append({"type": "text", "role": "user", "content": node.content})
        elif isinstance(node, System):
            content.append({"type": "text", "role": "system", "content": node.content})
        elif isinstance(node, Agent):
            content.append(_agent_entry(node))
        elif isinstance(node, ToolResult):
            content.append(_tool_result_entry(node))
        elif isinstance(node, (ImageNode, VideoNode, AudioNode)):
            content.append(_media_entry(node))
        elif isinstance(node, VideoFramesNode):
            content.append(_video_frames_entry(node))
        elif isinstance(node, _TAG_NODES):
            asset_idx, issue = _anchor(node, seen, ledger)
            if issue is not None:
                _report(issues, issue, strict=strict, error=AnchorError)
            tag = _tag_markup(node, asset_idx, issues, strict=strict)
            content.append({"type": "text", "role": "user", "content": tag})
        else:
            raise TypeError(f"Unknown node type: {type(node)}")
        seen += len(_media_items(node))

    task = {"content": content, "expects": expects}
    return task, issues


@dataclass
class PerceiveResult:
    """The result of running a prompt.

    ``points``/``boxes``/``polygons``/``clips`` hold the ``expects`` kind, flattened: collection children and track
    waypoints included, each with the ``mention``/``t``/``asset_idx`` its markup gives it; ``tracks`` (every track, for
    any structured ``expects``) and ``parsed`` keep the tree. Malformed markup is listed in ``errors`` (``strict=True``
    raises ``ParseError`` with ``request_id`` and the answer as ``partial``).
    ``usage`` is the server's usage object as sent (e.g. ``prompt_tokens_details.audio_tokens``), or None.
    ``finish_reason`` is ``stop``, ``tool_calls``, ``length``, ...; ``tool_calls`` are the calls the model asked you to
    run (execute them only when ``complete``); ``request_id`` is the ``x-trace-id`` response header; ``message`` is the
    assistant message (``as_agent()`` replays it in a follow-up prompt); ``asset_count`` is the number of media assets
    in the request (the ``asset_idx`` space; see ``resolve_asset_idx``).
    """

    text: str | None
    points: list[SinglePoint] | None
    boxes: list[BoundingBox] | None
    polygons: list[Polygon] | None
    clips: list[Clip] | None
    parsed: list[dict] | None
    reasoning: str | None
    usage: dict | None
    errors: list[dict]
    raw: Any
    finish_reason: str | None = field(default=None, kw_only=True)
    tool_calls: list[ToolCall] | None = field(default=None, kw_only=True)
    id: str | None = field(default=None, kw_only=True)
    model: str | None = field(default=None, kw_only=True)
    request_id: str | None = field(default=None, kw_only=True)
    tracks: list[Any] | None = field(default=None, kw_only=True)
    message: ChatCompletionMessage | None = field(default=None, kw_only=True)
    asset_count: int | None = field(default=None, kw_only=True)

    @property
    def complete(self) -> bool:
        """True for a finished answer: ``stop``/``tool_calls``/``interrupted``, with tool calls present exactly when
        ``finish_reason == "tool_calls"``. ``length`` (cut off) is incomplete."""
        finish_reason = self.finish_reason
        return finish_reason in _COMPLETE_FINISH_REASONS and (finish_reason == "tool_calls") == bool(self.tool_calls)

    def as_agent(self) -> Agent:
        """This answer as an ``agent(...)`` node (with its tool calls and reasoning), to replay it in a follow-up
        prompt followed by one ``tool_result(...)`` per call."""
        message = self.message
        if message is None:
            return Agent(self.text, tool_calls=self.tool_calls, reasoning_content=self.reasoning)
        return Agent(message.content, tool_calls=message.tool_calls, reasoning_content=message.reasoning_content)

    def resolve_asset_idx(self, annotation: Any) -> int | None:
        """The request asset an annotation from this result refers to: its own (or inherited) ``asset_idx``, else the
        last asset (``asset_count - 1``). ``ValueError`` when the ``asset_idx`` is out of range; None without assets.
        Bucket items (``points``, ``boxes``, ...) already carry their container's selector."""
        return resolve_asset_idx(annotation, self.asset_count)

    def points_to_pixels(self, width: int, height: int, *, clamp: bool = True) -> list[SinglePoint] | None:
        """Return a pixel-space copy of ``points`` given the image dimensions."""

        return scale_points_to_pixels(self.points, width=width, height=height, clamp=clamp)

    def boxes_to_pixels(self, width: int, height: int, *, clamp: bool = True) -> list[BoundingBox] | None:
        """Return a pixel-space copy of ``boxes`` given the image dimensions."""

        return scale_points_to_pixels(self.boxes, width=width, height=height, clamp=clamp)

    def polygons_to_pixels(self, width: int, height: int, *, clamp: bool = True) -> list[Polygon] | None:
        """Return a pixel-space copy of ``polygons`` given the image dimensions."""

        return scale_points_to_pixels(self.polygons, width=width, height=height, clamp=clamp)


def _client_options(**options: Any) -> dict[str, Any]:
    """The generation options ``perceive`` forwards to the client: only those set, and ``strict`` only when True.

    ``allow_multiple``/``max_outputs`` are not among them: they never changed the request.
    """
    if not options.get("strict"):
        options.pop("strict", None)
    return {name: value for name, value in options.items() if value is not None}


def _stream_only_options(stream: bool, stream_options: Any) -> Any:
    """``stream_options`` for the client; it only applies to streams, so passing it without ``stream=True`` raises."""
    if stream_options is not None and not stream:
        raise BadRequestError(
            "stream_options applies only to streaming requests; pass stream=True.",
            code=INVALID_PARAMETER,
            param="stream_options",
        )
    return stream_options


def _prepare_client_kwargs(
    *,
    provider_override: str | None,
    model_override: str | None,
    expects: str | None,
    reasoning: bool | None,
    options: dict[str, Any],
):
    env = settings()
    provider_name = _provider_key(provider_override or env.provider)
    reasoning_enabled = reasoning if reasoning is not None else _expects_reasoning(expects)
    client_kwargs: dict[str, Any] = {
        "expects": expects,
        "provider": provider_name,
    }
    if reasoning_enabled:
        client_kwargs["reasoning"] = True
    if model_override is not None:
        client_kwargs["model"] = model_override
    client_kwargs.update(options)
    return env, provider_name, client_kwargs


def _require_credentials(
    *,
    stream: bool,
    provider_name: str,
    env,
    issues: list[dict],
    task: dict,
) -> None:
    """Raise ``AuthError`` (code ``credentials_missing``) when the provider has no API key (fal never uses
    ``PERCEPTRON_API_KEY``); an unknown provider raises ``BadRequestError``, as the client would."""
    provider_cfg = _resolve_provider(provider_name)
    if provider_api_key(env, provider_cfg):
        return
    issue = {"code": CREDENTIALS_MISSING, "message": missing_api_key_message(provider_cfg)}
    raise AuthError(
        issue["message"],
        code=issue["code"],
        details={"task": task, "errors": [*issues, issue], "stream": stream, "provider": provider_name},
    )


def _with_issues(event: Any, issues: list[dict]) -> Any:
    """``event`` with the compile ``issues`` put first in a ``final`` result's ``errors``, as the non-stream result
    has them."""
    if not issues or not isinstance(event, dict) or event.get("type") != "final":
        return event
    result = event.get("result") or {}
    return {**event, "result": {**result, "errors": [*issues, *(result.get("errors") or [])]}}


def _stream_with_issues(events: Any, issues: list[dict], client: Client) -> Iterator[Any]:
    """The stream's events, with the compile ``issues`` added to the ``final`` result (see :func:`_with_issues`).
    Closes the stream (a generator), then ``client`` (the one it came from), when it ends or is closed."""
    try:
        for event in events:
            yield _with_issues(event, issues)
    finally:
        events.close()
        client.close()


def _perceive_result_from_response(resp: dict, issues: list[dict]) -> PerceiveResult:
    text = resp.get("text")
    reasoning = resp.get("reasoning")
    tool_calls = resp.get("tool_calls")
    return PerceiveResult(
        text=text,
        points=resp.get("points"),
        boxes=resp.get("boxes"),
        polygons=resp.get("polygons"),
        clips=resp.get("clips"),
        parsed=resp.get("parsed"),
        reasoning=reasoning,
        usage=resp.get("usage"),
        errors=[*issues, *(resp.get("errors") or [])],
        raw=resp.get("raw"),
        finish_reason=resp.get("finish_reason"),
        tool_calls=tool_calls,
        id=resp.get("id"),
        model=resp.get("model"),
        request_id=resp.get("request_id"),
        tracks=resp.get("tracks"),
        message=ChatCompletionMessage(
            role="assistant",
            content=text if isinstance(text, str) else None,
            reasoning_content=reasoning,
            tool_calls=tool_calls or None,
        ),
        asset_count=resp.get("asset_count"),
    )


def _compile_nodes_sync(
    fn: Callable[..., Any], *, expects: str | None, strict: bool, args: tuple[Any, ...], kwargs: dict[str, Any]
):
    nodes = fn(*args, **kwargs)
    return _compile(nodes, expects=expects, strict=strict)


async def _compile_nodes_async(
    fn: Callable[..., Any],
    *,
    expects: str | None,
    strict: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
    nodes = fn(*args, **kwargs)
    if inspect.isawaitable(nodes):
        nodes = await nodes
    return _compile(nodes, expects=expects, strict=strict)


def _prepare_execution_context(
    *,
    task: dict,
    issues: list[dict],
    stream: bool,
    provider_override: str | None,
    model_override: str | None,
    expects: str | None,
    reasoning: bool | None,
    options: dict[str, Any],
):
    env, provider_name, client_kwargs = _prepare_client_kwargs(
        provider_override=provider_override,
        model_override=model_override,
        expects=expects,
        reasoning=reasoning,
        options=options,
    )

    provider_cfg = _PROVIDER_CONFIG.get(provider_name, {})
    model_name = model_override or env.model or provider_cfg.get("default_model")

    requires_reasoning = _requires_reasoning(model_name, provider_cfg)

    # Warn when a thinking model is used with reasoning explicitly disabled.
    if reasoning is False and _is_thinking_model(model_name):
        issues.append(
            {
                "code": REASONING_DISABLED_FOR_THINKING_MODEL,
                "message": f"Model '{model_name}' is a thinking model; setting reasoning=False will have no effect.",
            }
        )

    # Drop reasoning flag (and warn) for models that don't support it (registry-driven).
    if client_kwargs.get("reasoning") and not _supports_reasoning(model_name, provider_cfg):
        client_kwargs.pop("reasoning", None)
        issues.append(
            {
                "code": REASONING_NOT_SUPPORTED,
                "message": f"Model '{model_name}' does not support reasoning; flag was ignored.",
            }
        )

    # Force reasoning when the model requires it (registry-driven).
    if requires_reasoning and not client_kwargs.get("reasoning"):
        client_kwargs["reasoning"] = True
        issues.append(
            {
                "code": REASONING_REQUIRED_FOR_MODEL,
                "message": f"Model '{model_name}' requires reasoning; flag was enabled automatically.",
            }
        )

    _require_credentials(
        stream=stream,
        provider_name=provider_name,
        env=env,
        issues=issues,
        task=task,
    )
    return client_kwargs


def _expects_structured(expects: str | None) -> bool:
    return expects in {"point", "box", "polygon"}


def _expects_reasoning(expects: str | None) -> bool:
    return isinstance(expects, str) and expects.lower() == "think"


def _is_thinking_model(model_name: str | None) -> bool:
    if not isinstance(model_name, str):
        return False
    return "thinking" in model_name.lower()


def _supports_reasoning(model_name: str | None, provider_cfg: dict | None = None) -> bool:
    """Check if model supports reasoning based on registry config.

    Returns True if the model's registry entry has reasoning=True,
    or if the model is not in the registry (permissive default).
    """
    if not isinstance(model_name, str):
        return False

    models_cfg = provider_cfg.get("models") if isinstance(provider_cfg, dict) else None
    if isinstance(models_cfg, dict):
        entry = models_cfg.get(model_name)
        if isinstance(entry, dict) and "reasoning" in entry:
            return bool(entry["reasoning"])

    # Model not in registry - default to True (permissive)
    return True


def _requires_reasoning(model_name: str | None, provider_cfg: dict | None = None) -> bool:
    if not isinstance(model_name, str):
        return False
    models_cfg = provider_cfg.get("models") if isinstance(provider_cfg, dict) else None
    if isinstance(models_cfg, dict):
        entry = models_cfg.get(model_name)
        if isinstance(entry, dict) and entry.get("only_reasoning") is True:
            return True
    return False


def _prepare_task_with_hints(
    task: dict,
    expects: str | None,
    client_kwargs: dict,
) -> dict:
    """Inject expectation hints into task based on provider/model config.

    Resolves provider and model from client_kwargs and settings, then injects
    appropriate hints for structured expectations and/or reasoning.
    """
    env_local = settings()
    provider_name = _provider_key(client_kwargs.get("provider") or env_local.provider)
    provider_cfg = {"name": provider_name, **(_PROVIDER_CONFIG.get(provider_name) or {})}
    model_name = client_kwargs.get("model") or env_local.model or provider_cfg.get("default_model")
    include_reasoning = bool(
        client_kwargs.get("reasoning")
        or (expects and expects.lower() == "think")
        or _requires_reasoning(model_name, provider_cfg)
    )
    return _inject_expectation_hint(
        task,
        expects,
        model_name=model_name,
        provider_cfg=provider_cfg,
        include_reasoning=include_reasoning,
    )


def _collect_nodes(value: Any, acc: list[DSLNode]) -> None:
    if isinstance(value, Sequence):
        for node in value.nodes:
            _collect_nodes(node, acc)
        return
    if isinstance(value, DSLNode):
        acc.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_nodes(item, acc)
        return
    raise TypeError(
        "perceive direct invocation expects DSL nodes (text/image/...) or sequences; "
        f"received unsupported type {type(value)!r}"
    )


def _normalize_direct_nodes(values: tuple[Any, ...]) -> DSLNode | Sequence:
    if not values:
        raise TypeError("perceive direct invocation requires at least one DSL node")
    flat: list[DSLNode] = []
    for value in values:
        _collect_nodes(value, flat)
    if not flat:
        raise TypeError("perceive direct invocation did not receive any DSL nodes")
    if len(flat) == 1:
        return flat[0]
    return Sequence(flat)


def _execute_sync_task(
    *,
    task: dict,
    issues: list[dict],
    parse_points: bool,
    stream: bool,
    provider_override: str | None,
    model_override: str | None,
    expects: str | None,
    reasoning: bool | None,
    options: dict[str, Any],
):
    client_kwargs = _prepare_execution_context(
        task=task,
        issues=issues,
        stream=stream,
        provider_override=provider_override,
        model_override=model_override,
        expects=expects,
        reasoning=reasoning,
        options=options,
    )

    task = _prepare_task_with_hints(task, expects, client_kwargs)

    # A client for this call only, closed when it returns (or, for a stream, when the stream ends or is closed).
    client = Client()
    if stream:
        return _stream_with_issues(
            client.stream(
                task,
                parse_points=parse_points,
                **client_kwargs,
            ),
            issues,
            client,
        )

    try:
        resp = client.generate(task, **client_kwargs)
    finally:
        client.close()
    return _perceive_result_from_response(resp, issues)


def perceive(
    *nodes_or_fn: Any,
    expects: str | None = None,
    reasoning: bool | None = None,
    enable_audio_in_video: bool | None = None,
    reasoning_effort: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    strict: bool = False,
    allow_multiple: bool = False,
    max_outputs: int | None = 1,
    stream: bool = False,
    response_format: ResponseFormat | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
    extra_body: dict[str, Any] | None = None,
    stream_options: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Decorator (or direct helper) for building Tasks from DSL nodes.

    When called without nodes it returns a decorator; when passed nodes directly
    it immediately compiles and executes them. Each call runs on a new
    :class:`~perceptron.Client`, closed when the call returns (with
    ``stream=True``, when the stream ends or is closed). Without an API key for
    the provider (provider ``fal`` never uses ``PERCEPTRON_API_KEY``) it raises
    ``AuthError`` (code ``credentials_missing``) before anything is sent; use
    ``inspect_task`` to compile without executing.

    Args:
        response_format: Optional constraint for output format. Use
            :func:`~perceptron.json_schema_format` or :func:`~perceptron.regex_format`
            to construct this parameter. Enables constrained decoding on supported models.
        tools: Function tools the model may call (see :func:`~perceptron.chat.function_tool`), with
            ``tool_choice`` (``"auto"``/``"none"``) and ``parallel_tool_calls``. The result's ``tool_calls`` lists the
            calls to run; replay them with ``result.as_agent()`` followed by one ``tool_result(...)`` per call.
        extra_body: Extra request fields, merged into the body last and not validated.
        stream_options: Streaming options (``stream=True`` only), sent as given, e.g. ``{"include_usage": False}``.
            Streams to provider ``perceptron`` request ``{"include_usage": True}`` when you do not pass it; the
            usage then arrives on the ``final`` event.
        allow_multiple, max_outputs: Accepted for compatibility; they do not change the request.

    Unknown keyword arguments raise ``TypeError`` (the retired ``focus`` and ``visual_reasoning`` say so).
    """

    _reject_unexpected_kwargs("perceive", kwargs)
    parse_points = _expects_structured(expects)
    options = _client_options(
        enable_audio_in_video=enable_audio_in_video,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        extra_body=extra_body,
        stream_options=_stream_only_options(stream, stream_options),
        strict=strict,
    )

    def wrapper(fn: Callable[..., Any]):
        def _inspect(*args: Any, **kwargs: Any):
            return _compile_nodes_sync(fn, expects=expects, strict=strict, args=args, kwargs=kwargs)

        def _call(*args: Any, **kwargs: Any):
            task, issues = _inspect(*args, **kwargs)
            return _execute_sync_task(
                task=task,
                issues=issues,
                parse_points=parse_points,
                stream=stream,
                provider_override=provider,
                model_override=model,
                expects=expects,
                reasoning=reasoning,
                options=options,
            )

        _call.__perceptron_inspector__ = _inspect  # type: ignore[attr-defined]

        return _call

    if not nodes_or_fn:
        return wrapper

    if len(nodes_or_fn) == 1 and callable(nodes_or_fn[0]):
        return wrapper(nodes_or_fn[0])

    nodes = _normalize_direct_nodes(nodes_or_fn)
    task, issues = _compile(nodes, expects=expects, strict=strict)
    return _execute_sync_task(
        task=task,
        issues=issues,
        parse_points=parse_points,
        stream=stream,
        provider_override=provider,
        model_override=model,
        expects=expects,
        reasoning=reasoning,
        options=options,
    )


def async_perceive(
    *,
    expects: str | None = None,
    reasoning: bool | None = None,
    enable_audio_in_video: bool | None = None,
    reasoning_effort: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    strict: bool = False,
    allow_multiple: bool = False,
    max_outputs: int | None = 1,
    stream: bool = False,
    response_format: ResponseFormat | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
    extra_body: dict[str, Any] | None = None,
    stream_options: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Async counterpart to ``perceive`` (same parameters): each call runs on a new :class:`AsyncClient`, closed when
    the call returns (with ``stream=True``, when the stream ends or is closed).

    Args:
        response_format: Optional constraint for output format. Use
            :func:`~perceptron.json_schema_format` or :func:`~perceptron.regex_format`
            to construct this parameter. Enables constrained decoding on supported models.
    """

    _reject_unexpected_kwargs("async_perceive", kwargs)
    parse_points = _expects_structured(expects)
    options = _client_options(
        enable_audio_in_video=enable_audio_in_video,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        extra_body=extra_body,
        stream_options=_stream_only_options(stream, stream_options),
        strict=strict,
    )

    def wrapper(fn: Callable[..., Any]):
        async def _inspect_async(*args: Any, **kwargs: Any):
            return await _compile_nodes_async(fn, expects=expects, strict=strict, args=args, kwargs=kwargs)

        if stream:

            def _call(*args: Any, **kwargs: Any):
                async def _generator():
                    task, issues = await _inspect_async(*args, **kwargs)
                    client_kwargs = _prepare_execution_context(
                        task=task,
                        issues=issues,
                        stream=True,
                        provider_override=provider,
                        model_override=model,
                        expects=expects,
                        reasoning=reasoning,
                        options=options,
                    )
                    task_with_hint = _prepare_task_with_hints(task, expects, client_kwargs)
                    client = AsyncClient()  # for this stream only, closed when it ends or is closed
                    events = client.stream(
                        task_with_hint,
                        parse_points=parse_points,
                        **client_kwargs,
                    )
                    try:
                        async for event in events:
                            yield _with_issues(event, issues)
                    finally:
                        await events.aclose()
                        await client.aclose()

                return _generator()

            _call.__perceptron_inspector__ = _inspect_async  # type: ignore[attr-defined]

            return _call

        async def _call(*args: Any, **kwargs: Any):
            task, issues = await _inspect_async(*args, **kwargs)
            client_kwargs = _prepare_execution_context(
                task=task,
                issues=issues,
                stream=False,
                provider_override=provider,
                model_override=model,
                expects=expects,
                reasoning=reasoning,
                options=options,
            )
            task = _prepare_task_with_hints(task, expects, client_kwargs)

            client = AsyncClient()  # for this call only
            try:
                resp = await client.generate(task, **client_kwargs)
            finally:
                await client.aclose()
            return _perceive_result_from_response(resp, issues)

        _call.__perceptron_inspector__ = _inspect_async  # type: ignore[attr-defined]

        return _call

    return wrapper


def inspect_task(callable_obj: Callable[..., Any], *args: Any, **kwargs: Any):
    """Return the compiled Task dict (and issues) for a `perceive`/`async_perceive` function without executing it."""

    inspector = getattr(callable_obj, "__perceptron_inspector__", None)
    if inspector is None:
        raise TypeError("inspect_task expects a function produced by perceive/async_perceive")
    result = inspector(*args, **kwargs)
    return result


__all__ = ["PerceiveResult", "async_perceive", "inspect_task", "perceive"]
