"""Files API: ``client.files`` (provider ``perceptron`` only).

Upload an image, video or audio file once and reference it by id in later requests (an ``image_file_id`` /
``video_file_id`` / ``audio_file_id`` content part). Files belong to the organization, never expire, and count toward
its storage quota until deleted::

    uploaded = client.files.upload("clip.mp4")
    messages = [{"role": "user", "content": [{"type": "video_file_id", "video_file_id": {"file_id": uploaded.id}}]}]
    client.files.delete(uploaded.id)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from . import _transport
from ._providers import surface_provider_cfg
from .chat import _int_or_none, _str_or_none
from .errors import INVALID_PARAMETER, BadRequestError

__all__ = ["AsyncFiles", "File", "FileDeleted", "FileList", "Files"]

FILE_PURPOSES = ("vision", "user_data")
SORT_ORDERS = ("asc", "desc")
# Per file; the server rejects larger uploads (413).
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_LIST_LIMIT = 10_000

_FEATURE = "The Files API"
_GENERIC_MIME = "application/octet-stream"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class File:
    """An uploaded file. ``bytes`` is its size and ``created_at`` a Unix timestamp; the server reports no MIME type
    (``content()``'s response ``Content-Type`` carries it)."""

    id: str
    bytes: int | None = None
    created_at: int | None = None
    filename: str | None = None
    purpose: str | None = None
    object: str = "file"

    @classmethod
    def from_dict(cls, data: Any) -> File:
        data = data if isinstance(data, Mapping) else {}
        return cls(
            id=_str_or_none(data.get("id")) or "",
            bytes=_int_or_none(data.get("bytes")),
            created_at=_int_or_none(data.get("created_at")),
            filename=_str_or_none(data.get("filename")),
            purpose=_str_or_none(data.get("purpose")),
            object=_str_or_none(data.get("object")) or "file",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "id": self.id,
            "bytes": self.bytes,
            "created_at": self.created_at,
            "filename": self.filename,
            "purpose": self.purpose,
        }


@dataclass
class FileList:
    """One page of ``files.list()``; iterating it yields its files. Pass ``after=page.last_id`` for the next page
    while ``has_more`` (or use ``files.iter()``). ``first_id``/``last_id`` are ``""`` on an empty page."""

    data: list[File] = field(default_factory=list)
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
    object: str = "list"

    def __iter__(self) -> Iterator[File]:
        return iter(self.data)

    @classmethod
    def from_dict(cls, data: Any) -> FileList:
        data = data if isinstance(data, Mapping) else {}
        items = data.get("data")
        return cls(
            data=[File.from_dict(item) for item in items] if isinstance(items, list) else [],
            first_id=_str_or_none(data.get("first_id")),
            last_id=_str_or_none(data.get("last_id")),
            has_more=data.get("has_more") is True,
            object=_str_or_none(data.get("object")) or "list",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "data": [item.to_dict() for item in self.data],
            "first_id": self.first_id,
            "last_id": self.last_id,
            "has_more": self.has_more,
        }


@dataclass
class FileDeleted:
    """The result of ``files.delete()``. ``deleted`` is always True: deleting a missing file raises ``NotFoundError``."""

    id: str
    deleted: bool
    object: str = "file"

    @classmethod
    def from_dict(cls, data: Any) -> FileDeleted:
        data = data if isinstance(data, Mapping) else {}
        return cls(
            id=_str_or_none(data.get("id")) or "",
            deleted=data.get("deleted") is True,
            object=_str_or_none(data.get("object")) or "file",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"object": self.object, "id": self.id, "deleted": self.deleted}


# ---------------------------------------------------------------------------
# Content-type sniffing
# ---------------------------------------------------------------------------

# The server sniffs every upload itself (the Rust `infer` crate) and rejects a declared part type that disagrees, so
# a type is declared only where `infer` is certain to report the same one; anything else goes as
# application/octet-stream and the server decides.
_PREFIX_TYPES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),
)
_RIFF_TYPES = {b"WEBP": "image/webp", b"WAVE": "audio/wav"}
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_MATROSKA_HEADER = b"\x93\x42\x82\x88matroska"
_MATROSKA_DOCTYPE_OFFSET = 31
# ISO BMFF major brands `infer` reports as video/mp4 (`M4V `, `M4A `, `3gp*`, `heic`, ... are other types).
_MP4_BRANDS = frozenset(
    {b"avc1", b"dash", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"isom", b"mmp4", b"mp41", b"mp42", b"mp4v"}
    | {b"mp71", b"MSNV", b"NDAS", b"NDSC", b"NSDC", b"NDSH", b"NDSM", b"NDSP", b"NDSS", b"NDXC", b"NDXH", b"NDXM"}
    | {b"NDXP", b"NDXS", b"F4V ", b"F4P "}
)
_QUICKTIME_BRAND = b"qt  "
# Any compatible brand `avif`/`avis` makes `infer` report image/avif.
_AVIF_BRANDS = frozenset({b"avif", b"avis"})
# An `ftyp` box of 256 bytes starts like an ICO file; boxes of 64 KiB or more like other formats.
_ICO_SIZED_BOX = 0x100
_MAX_FTYP_BOX = 0x10000
_FTYP_HEADER_BYTES = 16
# The server's check for an untagged MP3, an MPEG Layer III frame header: (byte, mask, value, must match) for the frame
# sync, a version that is not reserved, layer III, a valid bitrate and a sample rate that is not reserved.
_MP3_FRAME_RULES = (
    (0, 0xFF, 0xFF, True),
    (1, 0xE0, 0xE0, True),
    (1, 0x18, 0x08, False),
    (1, 0x06, 0x02, True),
    (2, 0xF0, 0xF0, False),
    (2, 0x0C, 0x0C, False),
)
_MP3_HEADER_BYTES = 3


def _iso_bmff_type(data: bytes) -> str | None:
    box_size = int.from_bytes(data[:4], "big")
    if box_size == _ICO_SIZED_BOX or box_size >= _MAX_FTYP_BOX or len(data) < _FTYP_HEADER_BYTES:
        return None
    compatible = {data[i : i + 4] for i in range(_FTYP_HEADER_BYTES, box_size - 3, 4)}
    if compatible & _AVIF_BRANDS:
        return None
    major = data[8:12]
    if major in _MP4_BRANDS:
        return "video/mp4"
    return "video/quicktime" if major == _QUICKTIME_BRAND else None


def _is_mp3_frame(data: bytes) -> bool:
    return len(data) >= _MP3_HEADER_BYTES and all(
        (data[i] & mask == value) is match for i, mask, value, match in _MP3_FRAME_RULES
    )


def _sniff_mime_type(data: bytes) -> str | None:
    """The supported MIME type of ``data`` when the server is certain to agree, else None."""
    for prefix, mime in _PREFIX_TYPES:
        if data.startswith(prefix):
            return mime
    if data[:4] == b"RIFF":
        return _RIFF_TYPES.get(data[8:12])
    if data[4:8] == b"ftyp":
        return _iso_bmff_type(data)
    if data[:4] == _EBML_MAGIC:
        # WebM unless it is recognizably Matroska (which the API does not accept).
        offset = _MATROSKA_DOCTYPE_OFFSET
        is_matroska = data[4:16] == _MATROSKA_HEADER or data[offset : offset + 8] == b"matroska"
        return None if is_matroska else "video/webm"
    return "audio/mpeg" if _is_mp3_frame(data) else None


# ---------------------------------------------------------------------------
# Request helpers (shared by the sync and async resources)
# ---------------------------------------------------------------------------


def _check_purpose(purpose: Any) -> str:
    if purpose not in FILE_PURPOSES:
        raise BadRequestError(
            f"purpose must be one of {', '.join(FILE_PURPOSES)}; got {purpose!r}.",
            code=INVALID_PARAMETER,
            param="purpose",
        )
    return purpose


def _too_large() -> BadRequestError:
    return BadRequestError(
        f"The file is larger than the upload limit of {MAX_UPLOAD_BYTES} bytes (128 MiB).",
        code=INVALID_PARAMETER,
        param="file",
    )


def _read_upload(file: Any) -> tuple[bytes, str | None]:
    """The bytes to upload (at most ``MAX_UPLOAD_BYTES``) and the filename they came with, if any."""
    if isinstance(file, (bytes, bytearray, memoryview)):
        data, name = bytes(file), None
    elif isinstance(file, (str, os.PathLike)):
        path = Path(file)
        try:
            if path.stat().st_size > MAX_UPLOAD_BYTES:
                raise _too_large()
            data, name = path.read_bytes(), path.name
        except OSError as exc:
            raise BadRequestError(
                f"Cannot read {str(path)!r}: {exc.strerror or exc}.", code=INVALID_PARAMETER, param="file"
            ) from exc
    elif callable(getattr(file, "read", None)):
        data = file.read(MAX_UPLOAD_BYTES + 1)
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("file objects must be opened in binary mode ('rb')")
        raw_name = getattr(file, "name", None)
        name = Path(raw_name).name if isinstance(raw_name, str) else None
    else:
        raise TypeError(f"file must be a path, bytes or a binary file object; got {type(file).__name__}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise _too_large()
    return bytes(data), name


def _upload_fields(file: Any, *, purpose: Any, filename: str | None) -> dict[str, Any]:
    """The multipart ``files``/``data`` of an upload. The body has a known length (the server requires
    ``Content-Length``); without a filename the part has none and the server names the file by its id."""
    purpose = _check_purpose(purpose)
    if filename is not None and not isinstance(filename, str):
        raise TypeError(f"filename must be a str; got {type(filename).__name__}")
    data, default_name = _read_upload(file)
    name = filename if filename is not None else default_name
    content_type = _sniff_mime_type(data) or _GENERIC_MIME
    return {"files": {"file": (name, data, content_type)}, "data": {"purpose": purpose}}


def _list_params(purpose: Any, limit: Any, order: Any, after: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if purpose is not None:
        params["purpose"] = _check_purpose(purpose)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_LIMIT:
            raise BadRequestError(
                f"limit must be an integer between 1 and {MAX_LIST_LIMIT}; got {limit!r}.",
                code=INVALID_PARAMETER,
                param="limit",
            )
        params["limit"] = limit
    if order is not None:
        if order not in SORT_ORDERS:
            raise BadRequestError(
                f"order must be 'asc' or 'desc'; got {order!r}.", code=INVALID_PARAMETER, param="order"
            )
        params["order"] = order
    if after is not None:
        if not isinstance(after, str) or not after:
            raise BadRequestError(
                "after must be a file id (the previous page's last_id).", code=INVALID_PARAMETER, param="after"
            )
        params["after"] = after
    return params


def _next_cursor(page: FileList) -> str | None:
    if not page.has_more or not page.data:
        return None
    return page.last_id or page.data[-1].id


def _file_path(file_id: Any, suffix: str = "") -> str:
    return _transport.resource_path("files", file_id, "file_id", suffix)


def _download_target(path: str | os.PathLike[str]) -> Path:
    """``path`` as a ``Path``, checked before the request: a directory raises ``BadRequestError(param="path")``."""
    target = Path(path)
    if target.is_dir():
        raise BadRequestError(
            f"path {str(target)!r} is a directory; pass the path of the file to write.",
            code=INVALID_PARAMETER,
            param="path",
        )
    return target


@contextmanager
def _replace_when_done(target: Path) -> Iterator[BinaryIO]:
    """Write to ``<target>.part`` and move it over ``target`` only when the block completes. A local file error raises
    ``BadRequestError(param="path")``."""
    partial = target.with_name(f"{target.name}.part")
    try:
        with partial.open("wb") as out:
            yield out
        partial.replace(target)
    except OSError as exc:
        raise BadRequestError(
            f"Cannot write {str(target)!r}: {exc.strerror or exc}.", code=INVALID_PARAMETER, param="path"
        ) from exc
    finally:
        partial.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


class Files:
    """``client.files``: upload, list, retrieve, read, download and delete files."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _cfg(self) -> dict[str, Any]:
        return surface_provider_cfg(self._client, feature=_FEATURE)

    def upload(self, file: Any, *, purpose: str = "vision", filename: str | None = None) -> File:
        """Upload a file (``POST /files``): a path, bytes, or a binary file object, up to 128 MiB.

        Supported types (detected from the bytes): JPEG, PNG, WebP, MP4, QuickTime, WebM, WAV, MP3 and FLAC.
        ``purpose`` is ``"vision"`` or ``"user_data"``; ``filename`` defaults to the path's or file object's name.
        """
        cfg = self._cfg()
        fields = _upload_fields(file, purpose=purpose, filename=filename)
        payload, _ = _transport.request_json(self._client, "POST", "/files", provider_cfg=cfg, **fields)
        return File.from_dict(payload)

    def list(
        self,
        *,
        purpose: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
    ) -> FileList:
        """One page of the organization's files (``GET /files``), newest first unless ``order="asc"``.

        ``limit`` is 1-10000 (server default 10000); ``after`` is the previous page's ``last_id``.
        """
        cfg = self._cfg()
        params = _list_params(purpose, limit, order, after)
        payload, _ = _transport.request_json(self._client, "GET", "/files", params=params, provider_cfg=cfg)
        return FileList.from_dict(payload)

    def iter(
        self,
        *,
        purpose: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
    ) -> Iterator[File]:
        """Every file across pages: ``list()`` again with ``after=last_id`` while ``has_more`` (``limit`` is the
        page size). Arguments are checked now; pages are fetched as you iterate."""
        cfg = self._cfg()
        return self._iterate(cfg, _list_params(purpose, limit, order, after))

    def _iterate(self, cfg: dict[str, Any], params: dict[str, Any]) -> Iterator[File]:
        while True:
            payload, _ = _transport.request_json(self._client, "GET", "/files", params=params, provider_cfg=cfg)
            page = FileList.from_dict(payload)
            yield from page.data
            cursor = _next_cursor(page)
            if cursor is None:
                return
            params = {**params, "after": cursor}

    def retrieve(self, file_id: str) -> File:
        """A file's metadata (``GET /files/{file_id}``); ``NotFoundError`` when it does not exist."""
        cfg = self._cfg()
        payload, _ = _transport.request_json(self._client, "GET", _file_path(file_id), provider_cfg=cfg)
        return File.from_dict(payload)

    def content(self, file_id: str) -> bytes:
        """A file's bytes (``GET /files/{file_id}/content``). Use ``download()`` to stream a large file to disk."""
        cfg = self._cfg()
        return _transport.request(self._client, "GET", _file_path(file_id, "/content"), provider_cfg=cfg).content

    def download(self, file_id: str, path: str | os.PathLike[str]) -> Path:
        """Stream a file's bytes to ``path`` and return it as a ``Path``. The bytes go to ``<path>.part`` first and
        replace ``path`` only once complete, so a failed download leaves no truncated file behind."""
        cfg = self._cfg()
        target = _download_target(path)
        with (
            _transport.stream_request(self._client, "GET", _file_path(file_id, "/content"), provider_cfg=cfg) as resp,
            _replace_when_done(target) as out,
        ):
            for chunk in _transport.iter_response_bytes(resp):
                out.write(chunk)
        return target

    def delete(self, file_id: str) -> FileDeleted:
        """Delete a file (``DELETE /files/{file_id}``); requests that reference it fail from now on."""
        cfg = self._cfg()
        payload, _ = _transport.request_json(self._client, "DELETE", _file_path(file_id), provider_cfg=cfg)
        return FileDeleted.from_dict(payload)


class AsyncFiles:
    """``AsyncClient.files``: the async :class:`Files`. ``iter()`` returns an async iterator."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _cfg(self) -> dict[str, Any]:
        return surface_provider_cfg(self._client, feature=_FEATURE)

    async def upload(self, file: Any, *, purpose: str = "vision", filename: str | None = None) -> File:
        """Async :meth:`Files.upload`."""
        cfg = self._cfg()
        fields = _upload_fields(file, purpose=purpose, filename=filename)
        payload, _ = await _transport.arequest_json(self._client, "POST", "/files", provider_cfg=cfg, **fields)
        return File.from_dict(payload)

    async def list(
        self,
        *,
        purpose: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
    ) -> FileList:
        """Async :meth:`Files.list`."""
        cfg = self._cfg()
        params = _list_params(purpose, limit, order, after)
        payload, _ = await _transport.arequest_json(self._client, "GET", "/files", params=params, provider_cfg=cfg)
        return FileList.from_dict(payload)

    def iter(
        self,
        *,
        purpose: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
    ) -> AsyncIterator[File]:
        """Async :meth:`Files.iter`: ``async for file in client.files.iter(): ...``."""
        cfg = self._cfg()
        return self._iterate(cfg, _list_params(purpose, limit, order, after))

    async def _iterate(self, cfg: dict[str, Any], params: dict[str, Any]) -> AsyncIterator[File]:
        while True:
            payload, _ = await _transport.arequest_json(self._client, "GET", "/files", params=params, provider_cfg=cfg)
            page = FileList.from_dict(payload)
            for item in page.data:
                yield item
            cursor = _next_cursor(page)
            if cursor is None:
                return
            params = {**params, "after": cursor}

    async def retrieve(self, file_id: str) -> File:
        """Async :meth:`Files.retrieve`."""
        cfg = self._cfg()
        payload, _ = await _transport.arequest_json(self._client, "GET", _file_path(file_id), provider_cfg=cfg)
        return File.from_dict(payload)

    async def content(self, file_id: str) -> bytes:
        """Async :meth:`Files.content`."""
        cfg = self._cfg()
        resp = await _transport.arequest(self._client, "GET", _file_path(file_id, "/content"), provider_cfg=cfg)
        return resp.content

    async def download(self, file_id: str, path: str | os.PathLike[str]) -> Path:
        """Async :meth:`Files.download`."""
        cfg = self._cfg()
        target = _download_target(path)
        path_on_server = _file_path(file_id, "/content")
        async with _transport.astream_request(self._client, "GET", path_on_server, provider_cfg=cfg) as resp:
            with _replace_when_done(target) as out:
                async for chunk in _transport.aiter_response_bytes(resp):
                    out.write(chunk)
        return target

    async def delete(self, file_id: str) -> FileDeleted:
        """Async :meth:`Files.delete`."""
        cfg = self._cfg()
        payload, _ = await _transport.arequest_json(self._client, "DELETE", _file_path(file_id), provider_cfg=cfg)
        return FileDeleted.from_dict(payload)
