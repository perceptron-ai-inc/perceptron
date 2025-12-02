"""Media upload client for handling presigned URL uploads to S3.

This module provides functionality to upload large media files (images, videos)
using presigned URLs from the Perceptron API, avoiding base64 encoding overhead.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from .config import settings
from .errors import AuthError, BadRequestError, SDKError, ServerError, TransportError

# Supported content types matching the server-side validation
SUPPORTED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp", "video/mp4"]

# Default API endpoint for media uploads
DEFAULT_MEDIA_API_BASE = "https://api.perceptron.inc"


@dataclass
class UploadedMedia:
    """Result of a successful media upload."""

    object_key: UUID
    file_name: str
    download_url: str | None = None


@dataclass
class PresignedUploadUrl:
    """Presigned URL for uploading media."""

    upload_url: str
    object_key: UUID
    file_name: str


def _get_content_type(file_path: Path) -> str:
    """Determine content type from file extension."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        # Default based on extension
        ext = file_path.suffix.lower()
        ext_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
        }
        mime_type = ext_map.get(ext)
    if mime_type is None:
        raise BadRequestError(f"Cannot determine content type for file: {file_path}")
    return mime_type


def _get_file_info(obj: Any) -> tuple[bytes, str, str]:
    """Extract file bytes, content type, and file name from input.

    Args:
        obj: File path (str/Path) or bytes with content_type

    Returns:
        Tuple of (file_bytes, content_type, file_name)
    """
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            raise BadRequestError(f"File not found: {p}")
        with open(p, "rb") as f:
            data = f.read()
        content_type = _get_content_type(p)
        file_name = p.name
        return data, content_type, file_name

    if isinstance(obj, bytes):
        # For raw bytes, try to detect type from magic bytes
        content_type = _detect_content_type_from_bytes(obj)
        file_name = f"upload.{_extension_for_content_type(content_type)}"
        return obj, content_type, file_name

    if isinstance(obj, dict) and "bytes" in obj:
        # Allow passing {"bytes": b"...", "content_type": "video/mp4", "file_name": "video.mp4"}
        data = obj["bytes"]
        content_type = obj.get("content_type") or _detect_content_type_from_bytes(data)
        file_name = obj.get("file_name", f"upload.{_extension_for_content_type(content_type)}")
        return data, content_type, file_name

    raise TypeError(f"Unsupported media object type: {type(obj)}")


def _detect_content_type_from_bytes(data: bytes) -> str:
    """Detect content type from magic bytes."""
    # JPEG: starts with FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # PNG: starts with 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # WebP: RIFF....WEBP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    # MP4/MOV: has "ftyp" box within first 12 bytes (box size + "ftyp")
    # The ftyp box can start at offset 4 (after box size)
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "video/mp4"
    raise BadRequestError("Cannot detect content type from bytes. Please provide content_type explicitly.")


def _extension_for_content_type(content_type: str) -> str:
    """Get file extension for a content type."""
    ext_map = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "video/mp4": "mp4",
    }
    return ext_map.get(content_type, "bin")


def _normalize_base_url(url: str) -> str:
    """Normalize base URL by stripping trailing /v1 or /v1/ to avoid duplication."""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _get_api_base_url() -> str:
    """Get the base URL for the media API."""
    env = settings()
    # Use the configured base_url if it looks like the perceptron API
    if env.base_url and "perceptron" in env.base_url:
        return _normalize_base_url(env.base_url)
    return DEFAULT_MEDIA_API_BASE


def _get_api_key() -> str:
    """Get the API key for authentication."""
    env = settings()
    api_key = env.api_key or os.getenv("PERCEPTRON_API_KEY")
    if not api_key:
        raise AuthError("API key required for media upload. Set PERCEPTRON_API_KEY environment variable.")
    return api_key


def _map_upload_error(resp: httpx.Response) -> SDKError:
    """Map HTTP response to appropriate SDK error."""
    try:
        data = resp.json()
        message = data.get("message") or data.get("error_code") or str(data)
    except Exception:
        message = resp.text or f"HTTP {resp.status_code}"

    if resp.status_code == 400:
        return BadRequestError(message)
    if resp.status_code in (401, 403):
        return AuthError(message)
    if resp.status_code == 429:
        return BadRequestError(f"Rate limit exceeded: {message}")
    return ServerError(message)


class MediaClient:
    """Client for uploading media files via presigned URLs."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 300.0) -> None:
        """Initialize the media client.

        Args:
            api_key: API key for authentication. Defaults to PERCEPTRON_API_KEY env var.
            base_url: Base URL for the API. Defaults to https://api.perceptron.inc
            timeout: Request timeout in seconds. Default 300s for large uploads.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        api_key = self._api_key or _get_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _get_base_url(self) -> str:
        """Get base URL for API requests."""
        return self._base_url or _get_api_base_url()

    def generate_upload_urls(
        self,
        files: list[dict[str, Any]],
    ) -> tuple[list[PresignedUploadUrl], int]:
        """Generate presigned upload URLs for files.

        Args:
            files: List of dicts with file_name, content_type, content_length

        Returns:
            Tuple of (list of PresignedUploadUrl, expires_in_seconds)
        """
        base_url = self._get_base_url()
        url = f"{base_url}/v1/media/upload-urls"

        request_body = {"files": files}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, headers=self._get_headers(), json=request_body)
        except httpx.TimeoutException as e:
            raise TransportError("Request timed out while generating upload URLs") from e
        except httpx.HTTPError as e:
            raise TransportError(str(e)) from e

        if resp.status_code != 200:
            raise _map_upload_error(resp)

        data = resp.json()
        urls = [
            PresignedUploadUrl(
                upload_url=item["upload_url"],
                object_key=UUID(item["object_key"]),
                file_name=item["file_name"],
            )
            for item in data["urls"]
        ]
        return urls, data["expires_in_seconds"]

    def generate_download_urls(
        self,
        object_keys: list[UUID],
    ) -> tuple[list[dict[str, Any]], int]:
        """Generate presigned download URLs for uploaded files.

        Args:
            object_keys: List of object UUIDs to get download URLs for

        Returns:
            Tuple of (list of download URL dicts, expires_in_seconds)
        """
        base_url = self._get_base_url()
        url = f"{base_url}/v1/media/download-urls"

        request_body = {"object_keys": [str(key) for key in object_keys]}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, headers=self._get_headers(), json=request_body)
        except httpx.TimeoutException as e:
            raise TransportError("Request timed out while generating download URLs") from e
        except httpx.HTTPError as e:
            raise TransportError(str(e)) from e

        if resp.status_code != 200:
            raise _map_upload_error(resp)

        data = resp.json()
        return data["urls"], data["expires_in_seconds"]

    def upload_to_presigned_url(
        self,
        upload_url: str,
        data: bytes,
        content_type: str,
    ) -> None:
        """Upload data to a presigned S3 URL.

        Args:
            upload_url: The presigned upload URL
            data: File bytes to upload
            content_type: MIME type of the file
        """
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.put(upload_url, content=data, headers=headers)
        except httpx.TimeoutException as e:
            raise TransportError("Upload timed out") from e
        except httpx.HTTPError as e:
            raise TransportError(str(e)) from e

        if resp.status_code not in (200, 201, 204):
            raise ServerError(f"Upload failed with status {resp.status_code}: {resp.text}")

    def upload(self, obj: Any) -> UploadedMedia:
        """Upload a media file and return the uploaded media info.

        Args:
            obj: File path (str/Path) or bytes/dict with content

        Returns:
            UploadedMedia with object_key and file_name
        """
        # Get file info
        file_bytes, content_type, file_name = _get_file_info(obj)

        if content_type not in SUPPORTED_CONTENT_TYPES:
            raise BadRequestError(
                f"Unsupported content type '{content_type}'. Supported types: {', '.join(SUPPORTED_CONTENT_TYPES)}"
            )

        # Generate presigned upload URL
        files = [
            {
                "file_name": file_name,
                "content_type": content_type,
                "content_length": len(file_bytes),
            }
        ]
        urls, _ = self.generate_upload_urls(files)
        presigned = urls[0]

        # Upload the file
        self.upload_to_presigned_url(presigned.upload_url, file_bytes, content_type)

        return UploadedMedia(
            object_key=presigned.object_key,
            file_name=presigned.file_name,
        )

    def upload_and_get_url(self, obj: Any) -> UploadedMedia:
        """Upload a media file and return info with download URL.

        Args:
            obj: File path (str/Path) or bytes/dict with content

        Returns:
            UploadedMedia with object_key, file_name, and download_url
        """
        uploaded = self.upload(obj)

        # Get download URL
        download_urls, _ = self.generate_download_urls([uploaded.object_key])
        if download_urls:
            uploaded.download_url = download_urls[0]["download_url"]

        return uploaded


class AsyncMediaClient:
    """Async client for uploading media files via presigned URLs."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 300.0) -> None:
        """Initialize the async media client.

        Args:
            api_key: API key for authentication. Defaults to PERCEPTRON_API_KEY env var.
            base_url: Base URL for the API. Defaults to https://api.perceptron.inc
            timeout: Request timeout in seconds. Default 300s for large uploads.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        api_key = self._api_key or _get_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _get_base_url(self) -> str:
        """Get base URL for API requests."""
        return self._base_url or _get_api_base_url()

    async def generate_upload_urls(
        self,
        files: list[dict[str, Any]],
    ) -> tuple[list[PresignedUploadUrl], int]:
        """Generate presigned upload URLs for files.

        Args:
            files: List of dicts with file_name, content_type, content_length

        Returns:
            Tuple of (list of PresignedUploadUrl, expires_in_seconds)
        """
        base_url = self._get_base_url()
        url = f"{base_url}/v1/media/upload-urls"

        request_body = {"files": files}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=self._get_headers(), json=request_body)
        except httpx.TimeoutException as e:
            raise TransportError("Request timed out while generating upload URLs") from e
        except httpx.HTTPError as e:
            raise TransportError(str(e)) from e

        if resp.status_code != 200:
            raise _map_upload_error(resp)

        data = resp.json()
        urls = [
            PresignedUploadUrl(
                upload_url=item["upload_url"],
                object_key=UUID(item["object_key"]),
                file_name=item["file_name"],
            )
            for item in data["urls"]
        ]
        return urls, data["expires_in_seconds"]

    async def generate_download_urls(
        self,
        object_keys: list[UUID],
    ) -> tuple[list[dict[str, Any]], int]:
        """Generate presigned download URLs for uploaded files.

        Args:
            object_keys: List of object UUIDs to get download URLs for

        Returns:
            Tuple of (list of download URL dicts, expires_in_seconds)
        """
        base_url = self._get_base_url()
        url = f"{base_url}/v1/media/download-urls"

        request_body = {"object_keys": [str(key) for key in object_keys]}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=self._get_headers(), json=request_body)
        except httpx.TimeoutException as e:
            raise TransportError("Request timed out while generating download URLs") from e
        except httpx.HTTPError as e:
            raise TransportError(str(e)) from e

        if resp.status_code != 200:
            raise _map_upload_error(resp)

        data = resp.json()
        return data["urls"], data["expires_in_seconds"]

    async def upload_to_presigned_url(
        self,
        upload_url: str,
        data: bytes,
        content_type: str,
    ) -> None:
        """Upload data to a presigned S3 URL.

        Args:
            upload_url: The presigned upload URL
            data: File bytes to upload
            content_type: MIME type of the file
        """
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.put(upload_url, content=data, headers=headers)
        except httpx.TimeoutException as e:
            raise TransportError("Upload timed out") from e
        except httpx.HTTPError as e:
            raise TransportError(str(e)) from e

        if resp.status_code not in (200, 201, 204):
            raise ServerError(f"Upload failed with status {resp.status_code}: {resp.text}")

    async def upload(self, obj: Any) -> UploadedMedia:
        """Upload a media file and return the uploaded media info.

        Args:
            obj: File path (str/Path) or bytes/dict with content

        Returns:
            UploadedMedia with object_key and file_name
        """
        # Get file info
        file_bytes, content_type, file_name = _get_file_info(obj)

        if content_type not in SUPPORTED_CONTENT_TYPES:
            raise BadRequestError(
                f"Unsupported content type '{content_type}'. Supported types: {', '.join(SUPPORTED_CONTENT_TYPES)}"
            )

        # Generate presigned upload URL
        files = [
            {
                "file_name": file_name,
                "content_type": content_type,
                "content_length": len(file_bytes),
            }
        ]
        urls, _ = await self.generate_upload_urls(files)
        presigned = urls[0]

        # Upload the file
        await self.upload_to_presigned_url(presigned.upload_url, file_bytes, content_type)

        return UploadedMedia(
            object_key=presigned.object_key,
            file_name=presigned.file_name,
        )

    async def upload_and_get_url(self, obj: Any) -> UploadedMedia:
        """Upload a media file and return info with download URL.

        Args:
            obj: File path (str/Path) or bytes/dict with content

        Returns:
            UploadedMedia with object_key, file_name, and download_url
        """
        uploaded = await self.upload(obj)

        # Get download URL
        download_urls, _ = await self.generate_download_urls([uploaded.object_key])
        if download_urls:
            uploaded.download_url = download_urls[0]["download_url"]

        return uploaded


# Module-level convenience functions using default client
_default_client: MediaClient | None = None


def _get_default_client() -> MediaClient:
    """Get or create the default media client."""
    global _default_client
    if _default_client is None:
        _default_client = MediaClient()
    return _default_client


def upload_media(obj: Any) -> UploadedMedia:
    """Upload a media file using the default client.

    Args:
        obj: File path (str/Path) or bytes/dict with content

    Returns:
        UploadedMedia with object_key and file_name
    """
    return _get_default_client().upload(obj)


def upload_media_and_get_url(obj: Any) -> UploadedMedia:
    """Upload a media file and get download URL using the default client.

    Args:
        obj: File path (str/Path) or bytes/dict with content

    Returns:
        UploadedMedia with object_key, file_name, and download_url
    """
    return _get_default_client().upload_and_get_url(obj)
