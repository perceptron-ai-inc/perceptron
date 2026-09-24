from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .._lowering import validate_data_url, validate_file_id
from ..errors import INVALID_PARAMETER, INVALID_VIDEO_FRAMES, BadRequestError, ParseError
from ..pointing.parser import _authored_seconds

MIN_VIDEO_FRAMES = 2
MAX_VIDEO_FRAMES = 256


class DSLNode:
    """Base class for DSL nodes used to compose prompts."""

    def __add__(self, other: DSLNode | Sequence) -> Sequence:
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
    """Assistant content: few-shot / ICL examples, or a replayed assistant turn.

    ``tool_calls`` (``ToolCall`` objects or wire dicts) and ``reasoning_content`` make it a standalone assistant
    message, sent as given; ``content`` may then be None.
    """

    content: str | None
    tool_calls: list[Any] | None = field(default=None, kw_only=True)
    reasoning_content: str | None = field(default=None, kw_only=True)


@dataclass
class Image(DSLNode):
    """Image content. Accepts path/bytes/PIL.Image/np.ndarray for encoding to base64, an http(s) or ``data:`` URL
    (passed through), or an uploaded file's ``file_id``."""

    obj: Any
    file_id: str | None = field(default=None, kw_only=True)


@dataclass
class Video(DSLNode):
    """Video content. Accepts path/str (URL or file)/bytes (format auto-detected from magic bytes), a ``data:`` URL,
    or an uploaded file's ``file_id``."""

    obj: Any
    file_id: str | None = field(default=None, kw_only=True)


@dataclass
class Audio(DSLNode):
    """Audio content. Accepts path/str (URL or file)/bytes (format auto-detected from magic bytes), a ``data:`` URL,
    or an uploaded file's ``file_id``."""

    obj: Any
    file_id: str | None = field(default=None, kw_only=True)


@dataclass
class VideoFrame:
    """One frame of a :func:`video_frames` video: an image (anything :func:`image` accepts, or an ``image(...)`` node)
    and its time in integer milliseconds from the start of the video."""

    image: Any
    timestamp_ms: int


@dataclass
class VideoFrames(DSLNode):
    """A video given as timestamped frames: one media asset (one ``asset_idx``), frames sent in order.

    ``frames`` holds :class:`VideoFrame` objects or ``(image, timestamp_ms)`` pairs; 2-256 frames with integer
    ``timestamp_ms >= 0`` that never decrease, else ``BadRequestError(code="invalid_video_frames")``.
    """

    frames: list[VideoFrame]

    def __post_init__(self) -> None:
        if not isinstance(self.frames, (list, tuple)):
            raise _frames_error(f"takes a list of VideoFrame or (image, timestamp_ms) pairs; got {_kind(self.frames)}")
        if not MIN_VIDEO_FRAMES <= len(self.frames) <= MAX_VIDEO_FRAMES:
            raise _frames_error(f"takes {MIN_VIDEO_FRAMES}-{MAX_VIDEO_FRAMES} frames; got {len(self.frames)}")
        frames: list[VideoFrame] = []
        previous = 0
        for i, item in enumerate(self.frames):
            if isinstance(item, VideoFrame):
                frame = item
            elif isinstance(item, tuple) and len(item) == 2:  # noqa: PLR2004 - an (image, timestamp_ms) pair
                frame = VideoFrame(*item)
            else:
                raise _frames_error(
                    f"frame {i} must be a VideoFrame or an (image, timestamp_ms) pair; got {_kind(item)}"
                )
            timestamp_ms = frame.timestamp_ms
            if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
                raise _frames_error(
                    f"frame {i} needs an integer timestamp_ms >= 0 (milliseconds); got {timestamp_ms!r}"
                )
            if timestamp_ms < previous:
                raise _frames_error(
                    f"timestamps must not decrease; frame {i} is at {timestamp_ms} ms after {previous} ms"
                )
            previous = timestamp_ms
            if frame.image is None or (isinstance(frame.image, DSLNode) and not isinstance(frame.image, Image)):
                raise _frames_error(
                    f"frame {i} image must be an image (anything image() accepts); got {_kind(frame.image)}"
                )
            _frame_image(frame.image)  # an invalid file id or data URL fails here, not at send time
            frames.append(frame)
        self.frames = frames


def _kind(value: Any) -> str:
    return type(value).__name__


def _frames_error(message: str) -> BadRequestError:
    return BadRequestError(f"video_frames() {message}.", code=INVALID_VIDEO_FRAMES)


# The media nodes: each is one asset (one `asset_idx`), and a tag can anchor to any of them.
Media = Image | Video | Audio | VideoFrames


@dataclass
class ToolResult(DSLNode):
    """The result of one tool call (a ``tool`` message answering ``tool_call_id``); content is text and images."""

    tool_call_id: str
    content: list[DSLNode]


@dataclass
class PointTag(DSLNode):
    x: int
    y: int
    image: Media | None = None
    mention: str | None = None
    t: float | None = None
    asset: Media | None = field(default=None, kw_only=True)
    asset_idx: int | None = field(default=None, kw_only=True)


@dataclass
class BoxTag(DSLNode):
    x1: int
    y1: int
    x2: int
    y2: int
    image: Media | None = None
    mention: str | None = None
    t: float | None = None
    asset: Media | None = field(default=None, kw_only=True)
    asset_idx: int | None = field(default=None, kw_only=True)


@dataclass
class PolygonTag(DSLNode):
    coords: list[tuple[int, int]]
    image: Media | None = None
    mention: str | None = None
    t: float | None = None
    asset: Media | None = field(default=None, kw_only=True)
    asset_idx: int | None = field(default=None, kw_only=True)


@dataclass
class Sequence(DSLNode):
    """A flat sequence of nodes; supports `+` composition."""

    nodes: list[DSLNode]

    def __add__(self, other: DSLNode | Sequence) -> Sequence:
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


def agent(
    content: str | None,
    *,
    tool_calls: list[Any] | None = None,
    reasoning_content: str | None = None,
) -> Agent:
    """Create an assistant (agent) node, useful for ICL.

    Pass ``tool_calls`` (and the turn's ``reasoning_content``) to replay an assistant turn that called tools, e.g.
    ``agent(None, tool_calls=result.tool_calls)``; follow it with one :func:`tool_result` per call.
    """
    return Agent(content, tool_calls=tool_calls, reasoning_content=reasoning_content)


def tool_result(tool_call_id: str, *content: Any) -> ToolResult:
    """Create a tool result node answering the call ``tool_call_id``.

    ``content`` items may be strings or :func:`text` nodes, :func:`image` nodes, or dicts/lists (sent as JSON text).
    Tool results cannot carry video or audio.
    """
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise TypeError("tool_result() needs the tool call's id as a non-empty string")
    nodes: list[DSLNode] = []
    for item in content:
        if isinstance(item, (Text, Image)):
            nodes.append(item)
        elif isinstance(item, str):
            nodes.append(Text(item))
        elif isinstance(item, (dict, list)):
            nodes.append(Text(json.dumps(item, ensure_ascii=False)))  # the model reads characters, not \u escapes
        else:
            raise TypeError(
                f"tool_result() content must be str, dict, list, text() or image(); got {type(item).__name__}"
            )
    return ToolResult(tool_call_id, nodes)


def _media_source(kind: str, obj: Any, file_id: str | None) -> tuple[Any, str | None]:
    """``(obj, file_id)`` for ``image()``/``video()``/``audio()``: exactly one is given. An uploaded-file object (a string
    ``.id``, e.g. the ``File`` that ``client.files.upload`` returns) becomes its id; ids and ``data:`` URLs are checked
    here, before any request."""
    if (obj is None) == (file_id is None):
        raise TypeError(f"{kind}() takes exactly one of obj or file_id")
    if obj is not None and not isinstance(obj, (str, bytes, os.PathLike)):
        ref = getattr(obj, "id", None)
        if isinstance(ref, str):
            obj, file_id = None, ref
    if file_id is not None:
        return None, validate_file_id(file_id)
    if isinstance(obj, str) and obj.startswith("data:"):
        validate_data_url(obj, kind)
    return obj, None


def image(obj: Any = None, *, file_id: str | None = None) -> Image:
    """Create an image node from a path, bytes, PIL.Image or np.ndarray (base64-encoded), an http(s) URL or a
    ``data:image/...;base64,`` URL (passed through), or an uploaded file: ``file_id="file-..."`` or the ``File``
    object itself."""
    obj, file_id = _media_source("image", obj, file_id)
    return Image(obj, file_id=file_id)


def video(obj: Any = None, *, file_id: str | None = None) -> Video:
    """Create a video node from path/str (URL or file)/bytes, a ``data:video/...`` URL or an uploaded file.

    Format is auto-detected from magic bytes (mp4 / webm) for bytes and path
    input; HTTP(S) and data URLs are passed through and the server infers from
    the response. Uploaded files: ``file_id="file-..."`` or the ``File`` object.
    """
    obj, file_id = _media_source("video", obj, file_id)
    return Video(obj, file_id=file_id)


def audio(obj: Any = None, *, file_id: str | None = None) -> Audio:
    """Create an audio node from path/str (URL or file)/bytes, a ``data:audio/...`` URL or an uploaded file.

    Format is auto-detected from magic bytes (wav / mp3 / flac) for bytes and
    path input; HTTP(S) and data URLs are passed through and the server infers
    from the response. Uploaded files: ``file_id="file-..."`` or the ``File`` object.
    """
    obj, file_id = _media_source("audio", obj, file_id)
    return Audio(obj, file_id=file_id)


def _frame_image(obj: Any) -> Image:
    """A :class:`VideoFrame` image as an ``image(...)`` node (validating file ids and data URLs)."""
    return obj if isinstance(obj, Image) else image(obj)


def video_frames(frames: list[VideoFrame | tuple[Any, int]]) -> VideoFrames:
    """Create a video from timestamped frames: :class:`VideoFrame` objects or ``(image, timestamp_ms)`` pairs.

    Send 2-256 frames with integer millisecond timestamps (``>= 0``) that never decrease; otherwise
    ``BadRequestError(code="invalid_video_frames")``. Frame images take anything :func:`image` accepts: local files,
    bytes, PIL images and arrays become data URLs, http(s) and ``data:`` URLs pass through, and uploaded files
    (``image(file_id=...)`` or a ``File``) are sent as their files-content URL (provider ``perceptron`` only).

    The frames are one video, so one media asset (one ``asset_idx``), and output ``t`` values are seconds on the same
    timeline (``timestamp_ms / 1000``).
    """
    return VideoFrames(frames)


def _check_anchor(factory: str, image: Any, asset: Any, asset_idx: Any) -> None:
    if image is not None and asset is not None:
        raise TypeError(f"{factory}() takes image= or asset=, not both")
    if asset_idx is None:
        return
    if image is not None or asset is not None:
        raise TypeError(f"{factory}() takes image=/asset= or asset_idx=, not both")
    if isinstance(asset_idx, bool) or not isinstance(asset_idx, int) or asset_idx < 0:
        raise BadRequestError(
            f"{factory}() asset_idx must be an integer >= 0; got {asset_idx!r}.",
            code=INVALID_PARAMETER,
            param="asset_idx",
        )


def _check_time(factory: str, t: Any) -> None:
    """``t`` is seconds from the start of a temporal asset: what the markup serializer accepts (a finite number >= 0)."""
    if t is None:
        return
    try:
        _authored_seconds(t)
    except (ValueError, ParseError) as exc:
        raise BadRequestError(
            f"{factory}() t must be a finite number of seconds >= 0; got {t!r}.", code=INVALID_PARAMETER, param="t"
        ) from exc


# Tags: coordinates are integers on the normalized 0-1000 grid. Anchor a tag to a media node of the prompt with
# `image=` or `asset=` (any media node), or give the 0-based `asset_idx` directly. The markup carries `asset_idx` only
# when the prompt has more than one media asset (or `asset_idx=` was given), so single-asset prompts are unchanged.
def point(
    x: int,
    y: int,
    *,
    image: Media | None = None,
    mention: str | None = None,
    t: float | None = None,
    asset: Media | None = None,
    asset_idx: int | None = None,
) -> PointTag:
    """Create a point tag anchored to a media asset (explicit in multi-asset prompts)."""
    _check_anchor("point", image, asset, asset_idx)
    _check_time("point", t)
    return PointTag(x, y, image=image, mention=mention, t=t, asset=asset, asset_idx=asset_idx)


def box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    image: Media | None = None,
    mention: str | None = None,
    t: float | None = None,
    asset: Media | None = None,
    asset_idx: int | None = None,
) -> BoxTag:
    """Create a bounding box tag (top-left, bottom-right) anchored to a media asset."""
    _check_anchor("box", image, asset, asset_idx)
    _check_time("box", t)
    return BoxTag(x1, y1, x2, y2, image=image, mention=mention, t=t, asset=asset, asset_idx=asset_idx)


def polygon(
    coords: list[tuple[int, int]],
    *,
    image: Media | None = None,
    mention: str | None = None,
    t: float | None = None,
    asset: Media | None = None,
    asset_idx: int | None = None,
) -> PolygonTag:
    """Create a polygon tag anchored to a media asset; requires ≥3 vertices."""
    _check_anchor("polygon", image, asset, asset_idx)
    _check_time("polygon", t)
    return PolygonTag(coords, image=image, mention=mention, t=t, asset=asset, asset_idx=asset_idx)
