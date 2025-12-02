"""Tests for the media upload client."""

import asyncio
from uuid import UUID

import httpx
import pytest

import perceptron.media as media_mod
from perceptron.errors import AuthError, BadRequestError, ServerError, TransportError
from perceptron.media import (
    SUPPORTED_CONTENT_TYPES,
    AsyncMediaClient,
    MediaClient,
    PresignedUploadUrl,
    UploadedMedia,
    _detect_content_type_from_bytes,
    _extension_for_content_type,
    _get_content_type,
    _get_file_info,
    _map_upload_error,
    _normalize_base_url,
    upload_media,
    upload_media_and_get_url,
)


class TestContentTypeDetection:
    """Tests for content type detection from bytes."""

    def test_detect_mp4_from_bytes(self):
        """Test MP4 detection from ftyp box."""
        # MP4 files have a box size (4 bytes) followed by "ftyp"
        mp4_bytes = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00"
        assert _detect_content_type_from_bytes(mp4_bytes) == "video/mp4"

    def test_detect_jpeg_from_bytes(self):
        """Test JPEG detection from magic bytes."""
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert _detect_content_type_from_bytes(jpeg_bytes) == "image/jpeg"

    def test_detect_png_from_bytes(self):
        """Test PNG detection from magic bytes."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        assert _detect_content_type_from_bytes(png_bytes) == "image/png"

    def test_detect_webp_from_bytes(self):
        """Test WebP detection from magic bytes."""
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBP"
        assert _detect_content_type_from_bytes(webp_bytes) == "image/webp"

    def test_detect_unknown_bytes_raises(self):
        """Test that unknown bytes raise an error."""
        unknown_bytes = b"unknown data format"
        with pytest.raises(BadRequestError) as exc_info:
            _detect_content_type_from_bytes(unknown_bytes)
        assert "Cannot detect content type" in str(exc_info.value)


class TestNormalizeBaseUrl:
    """Tests for base URL normalization to avoid /v1 duplication."""

    def test_url_without_v1_unchanged(self):
        """Base URL without /v1 should remain unchanged."""
        assert _normalize_base_url("https://api.perceptron.inc") == "https://api.perceptron.inc"

    def test_url_with_trailing_v1_stripped(self):
        """Base URL with trailing /v1 should have it stripped."""
        assert _normalize_base_url("https://api.perceptron.inc/v1") == "https://api.perceptron.inc"

    def test_url_with_trailing_v1_slash_stripped(self):
        """Base URL with trailing /v1/ should have it stripped."""
        assert _normalize_base_url("https://api.perceptron.inc/v1/") == "https://api.perceptron.inc"

    def test_url_with_trailing_slash_only(self):
        """Base URL with only trailing slash should have it stripped."""
        assert _normalize_base_url("https://api.perceptron.inc/") == "https://api.perceptron.inc"

    def test_staging_url_with_v1(self):
        """Staging URL with /v1 should have it stripped."""
        assert _normalize_base_url("https://staging-api.perceptron.build/v1") == "https://staging-api.perceptron.build"


class TestExtensionForContentType:
    """Tests for getting file extension from content type."""

    def test_jpeg_extension(self):
        assert _extension_for_content_type("image/jpeg") == "jpg"

    def test_png_extension(self):
        assert _extension_for_content_type("image/png") == "png"

    def test_webp_extension(self):
        assert _extension_for_content_type("image/webp") == "webp"

    def test_mp4_extension(self):
        assert _extension_for_content_type("video/mp4") == "mp4"

    def test_unknown_extension(self):
        assert _extension_for_content_type("application/octet-stream") == "bin"


class TestGetContentType:
    """Tests for getting content type from file path."""

    def test_jpeg_from_path(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.touch()
        assert _get_content_type(p) == "image/jpeg"

    def test_png_from_path(self, tmp_path):
        p = tmp_path / "test.png"
        p.touch()
        assert _get_content_type(p) == "image/png"

    def test_mp4_from_path(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.touch()
        assert _get_content_type(p) == "video/mp4"

    def test_unknown_extension_raises(self, tmp_path):
        """Test that unknown extensions raise BadRequestError."""
        p = tmp_path / "test.unknownext123"
        p.touch()
        with pytest.raises(BadRequestError) as exc_info:
            _get_content_type(p)
        assert "Cannot determine content type" in str(exc_info.value)


class TestGetFileInfo:
    """Tests for extracting file info from various inputs."""

    def test_file_path_string(self, tmp_path):
        p = tmp_path / "test.mp4"
        content = b"\x00\x00\x00\x1cftypisom\x00\x00\x00\x00"
        p.write_bytes(content)

        data, content_type, file_name = _get_file_info(str(p))
        assert data == content
        assert content_type == "video/mp4"
        assert file_name == "test.mp4"

    def test_file_path_object(self, tmp_path):
        p = tmp_path / "image.png"
        content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        p.write_bytes(content)

        data, content_type, file_name = _get_file_info(p)
        assert data == content
        assert content_type == "image/png"
        assert file_name == "image.png"

    def test_bytes_input(self):
        content = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        data, content_type, file_name = _get_file_info(content)
        assert data == content
        assert content_type == "image/jpeg"
        assert file_name == "upload.jpg"

    def test_dict_input(self):
        content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        input_dict = {
            "bytes": content,
            "content_type": "image/png",
            "file_name": "custom.png",
        }
        data, content_type, file_name = _get_file_info(input_dict)
        assert data == content
        assert content_type == "image/png"
        assert file_name == "custom.png"

    def test_dict_input_auto_detect_content_type(self):
        # MP4 files have a box size (4 bytes) followed by "ftyp"
        content = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00"
        input_dict = {"bytes": content}
        _data, content_type, file_name = _get_file_info(input_dict)
        assert content_type == "video/mp4"
        assert file_name == "upload.mp4"

    def test_nonexistent_file_raises(self, tmp_path):
        p = tmp_path / "nonexistent.mp4"
        with pytest.raises(BadRequestError) as exc_info:
            _get_file_info(str(p))
        assert "File not found" in str(exc_info.value)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError) as exc_info:
            _get_file_info(12345)
        assert "Unsupported media object type" in str(exc_info.value)


class TestSupportedContentTypes:
    """Test the supported content types constant."""

    def test_supported_types_include_images(self):
        assert "image/jpeg" in SUPPORTED_CONTENT_TYPES
        assert "image/png" in SUPPORTED_CONTENT_TYPES
        assert "image/webp" in SUPPORTED_CONTENT_TYPES

    def test_supported_types_include_video(self):
        assert "video/mp4" in SUPPORTED_CONTENT_TYPES


class TestMediaClientInit:
    """Tests for MediaClient initialization."""

    def test_default_init(self):
        client = MediaClient()
        assert client._api_key is None
        assert client._base_url is None
        assert client._timeout == 300.0

    def test_custom_init(self):
        client = MediaClient(api_key="test-key", base_url="https://custom.api", timeout=60.0)
        assert client._api_key == "test-key"
        assert client._base_url == "https://custom.api"
        assert client._timeout == 60.0


class TestMediaClientUpload:
    """Tests for MediaClient upload functionality."""

    def test_upload_requires_api_key(self, tmp_path):
        """Test that upload fails without API key."""
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00\x00\x00\x1cftypisom\x00\x00\x00\x00")

        client = MediaClient()
        with pytest.raises(AuthError) as exc_info:
            client.upload(str(p))
        assert "API key required" in str(exc_info.value)

    def test_upload_rejects_unsupported_content_type(self, tmp_path, monkeypatch):
        """Test that unsupported content types are rejected."""
        p = tmp_path / "test.gif"
        # GIF magic bytes
        p.write_bytes(b"GIF89a" + b"\x00" * 100)

        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")
        client = MediaClient()

        with pytest.raises(BadRequestError) as exc_info:
            client.upload(str(p))
        assert "Unsupported content type" in str(exc_info.value)


class TestUploadedMediaDataclass:
    """Tests for the UploadedMedia dataclass."""

    def test_uploaded_media_creation(self):
        media = UploadedMedia(
            object_key=UUID("12345678-1234-5678-1234-567812345678"),
            file_name="test.mp4",
            download_url="https://example.com/download",
        )
        assert str(media.object_key) == "12345678-1234-5678-1234-567812345678"
        assert media.file_name == "test.mp4"
        assert media.download_url == "https://example.com/download"

    def test_uploaded_media_optional_download_url(self):
        media = UploadedMedia(
            object_key=UUID("12345678-1234-5678-1234-567812345678"),
            file_name="test.mp4",
        )
        assert media.download_url is None


class TestPresignedUploadUrlDataclass:
    """Tests for the PresignedUploadUrl dataclass."""

    def test_presigned_url_creation(self):
        url = PresignedUploadUrl(
            upload_url="https://s3.example.com/upload?signature=abc",
            object_key=UUID("12345678-1234-5678-1234-567812345678"),
            file_name="test.mp4",
        )
        assert url.upload_url == "https://s3.example.com/upload?signature=abc"
        assert str(url.object_key) == "12345678-1234-5678-1234-567812345678"
        assert url.file_name == "test.mp4"


class TestMediaClientWithMockedHTTP:
    """Tests for MediaClient with mocked HTTP responses."""

    def test_generate_upload_urls_success(self, monkeypatch):
        """Test successful generation of upload URLs."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "urls": [
                        {
                            "upload_url": "https://s3.example.com/upload?sig=abc",
                            "object_key": "12345678-1234-5678-1234-567812345678",
                            "file_name": "test.mp4",
                        }
                    ],
                    "expires_in_seconds": 3600,
                }

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        urls, expires = client.generate_upload_urls(
            [{"file_name": "test.mp4", "content_type": "video/mp4", "content_length": 1000}]
        )

        assert len(urls) == 1
        assert urls[0].file_name == "test.mp4"
        assert expires == 3600

    def test_generate_download_urls_success(self, monkeypatch):
        """Test successful generation of download URLs."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "urls": [
                        {
                            "download_url": "https://s3.example.com/download?sig=xyz",
                            "object_key": "12345678-1234-5678-1234-567812345678",
                        }
                    ],
                    "expires_in_seconds": 3600,
                }

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        urls, expires = client.generate_download_urls([UUID("12345678-1234-5678-1234-567812345678")])

        assert len(urls) == 1
        assert urls[0]["download_url"] == "https://s3.example.com/download?sig=xyz"
        assert expires == 3600

    def test_upload_to_presigned_url_success(self, monkeypatch):
        """Test successful upload to presigned URL."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 200
            text = "OK"

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def put(self, url, content=None, headers=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        # Should not raise
        client.upload_to_presigned_url(
            "https://s3.example.com/upload?sig=abc",
            b"test data",
            "video/mp4",
        )

    def test_upload_to_presigned_url_failure(self, monkeypatch):
        """Test failed upload to presigned URL."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 500
            text = "Internal Server Error"

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def put(self, url, content=None, headers=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(ServerError):
            client.upload_to_presigned_url(
                "https://s3.example.com/upload?sig=abc",
                b"test data",
                "video/mp4",
            )

    def test_upload_full_flow(self, monkeypatch, tmp_path):
        """Test full upload flow with mocked HTTP."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        # Create test file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")

        call_count = {"post": 0, "put": 0}

        class MockResponse:
            def __init__(self, status_code, data=None):
                self.status_code = status_code
                self._data = data
                self.text = "OK"

            def json(self):
                return self._data

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                call_count["post"] += 1
                return MockResponse(
                    200,
                    {
                        "urls": [
                            {
                                "upload_url": "https://s3.example.com/upload",
                                "object_key": "12345678-1234-5678-1234-567812345678",
                                "file_name": "test.mp4",
                            }
                        ],
                        "expires_in_seconds": 3600,
                    },
                )

            def put(self, url, content=None, headers=None):
                call_count["put"] += 1
                return MockResponse(200)

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        result = client.upload(str(video_file))

        assert result.file_name == "test.mp4"
        assert str(result.object_key) == "12345678-1234-5678-1234-567812345678"
        assert call_count["post"] == 1
        assert call_count["put"] == 1

    def test_upload_and_get_url_full_flow(self, monkeypatch, tmp_path):
        """Test upload_and_get_url with mocked HTTP."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        # Create test file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")

        call_count = {"post": 0, "put": 0}

        class MockResponse:
            def __init__(self, status_code, data=None):
                self.status_code = status_code
                self._data = data
                self.text = "OK"

            def json(self):
                return self._data

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                call_count["post"] += 1
                if "upload-urls" in url:
                    return MockResponse(
                        200,
                        {
                            "urls": [
                                {
                                    "upload_url": "https://s3.example.com/upload",
                                    "object_key": "12345678-1234-5678-1234-567812345678",
                                    "file_name": "test.mp4",
                                }
                            ],
                            "expires_in_seconds": 3600,
                        },
                    )
                else:  # download-urls
                    return MockResponse(
                        200,
                        {
                            "urls": [
                                {
                                    "download_url": "https://s3.example.com/download",
                                    "object_key": "12345678-1234-5678-1234-567812345678",
                                }
                            ],
                            "expires_in_seconds": 3600,
                        },
                    )

            def put(self, url, content=None, headers=None):
                call_count["put"] += 1
                return MockResponse(200)

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        result = client.upload_and_get_url(str(video_file))

        assert result.file_name == "test.mp4"
        assert result.download_url == "https://s3.example.com/download"
        assert call_count["post"] == 2  # upload-urls + download-urls
        assert call_count["put"] == 1


class TestMapUploadError:
    """Tests for _map_upload_error function."""

    def test_map_400_error(self):
        class MockResponse:
            status_code = 400
            text = "Bad Request"

            def json(self):
                return {"message": "Invalid file"}

        err = _map_upload_error(MockResponse())
        assert isinstance(err, BadRequestError)
        assert "Invalid file" in str(err)

    def test_map_401_error(self):
        class MockResponse:
            status_code = 401
            text = "Unauthorized"

            def json(self):
                return {"message": "Invalid API key"}

        err = _map_upload_error(MockResponse())
        assert isinstance(err, AuthError)

    def test_map_429_error(self):
        class MockResponse:
            status_code = 429
            text = "Too Many Requests"

            def json(self):
                return {"message": "Rate limit exceeded"}

        err = _map_upload_error(MockResponse())
        assert isinstance(err, BadRequestError)
        assert "Rate limit" in str(err)

    def test_map_500_error(self):
        class MockResponse:
            status_code = 500
            text = "Internal Server Error"

            def json(self):
                return {"message": "Server error"}

        err = _map_upload_error(MockResponse())
        assert isinstance(err, ServerError)

    def test_map_error_json_parse_failure(self):
        class MockResponse:
            status_code = 500
            text = "Internal Server Error"

            def json(self):
                raise ValueError("Invalid JSON")

        err = _map_upload_error(MockResponse())
        assert isinstance(err, ServerError)
        assert "Internal Server Error" in str(err)


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_upload_media_uses_default_client(self, monkeypatch, tmp_path):
        """Test that upload_media uses the default client."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        # Create test file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")

        class MockResponse:
            def __init__(self, status_code, data=None):
                self.status_code = status_code
                self._data = data
                self.text = "OK"

            def json(self):
                return self._data

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                return MockResponse(
                    200,
                    {
                        "urls": [
                            {
                                "upload_url": "https://s3.example.com/upload",
                                "object_key": "12345678-1234-5678-1234-567812345678",
                                "file_name": "test.mp4",
                            }
                        ],
                        "expires_in_seconds": 3600,
                    },
                )

            def put(self, url, content=None, headers=None):
                return MockResponse(200)

        monkeypatch.setattr(httpx, "Client", MockClient)

        # Reset default client
        media_mod._default_client = None

        result = upload_media(str(video_file))
        assert result.file_name == "test.mp4"

    def test_upload_media_and_get_url(self, monkeypatch, tmp_path):
        """Test upload_media_and_get_url module function."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        # Create test file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")

        class MockResponse:
            def __init__(self, status_code, data=None):
                self.status_code = status_code
                self._data = data
                self.text = "OK"

            def json(self):
                return self._data

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                if "upload-urls" in url:
                    return MockResponse(
                        200,
                        {
                            "urls": [
                                {
                                    "upload_url": "https://s3.example.com/upload",
                                    "object_key": "12345678-1234-5678-1234-567812345678",
                                    "file_name": "test.mp4",
                                }
                            ],
                            "expires_in_seconds": 3600,
                        },
                    )
                else:  # download-urls
                    return MockResponse(
                        200,
                        {
                            "urls": [
                                {
                                    "download_url": "https://s3.example.com/download",
                                    "object_key": "12345678-1234-5678-1234-567812345678",
                                }
                            ],
                            "expires_in_seconds": 3600,
                        },
                    )

            def put(self, url, content=None, headers=None):
                return MockResponse(200)

        monkeypatch.setattr(httpx, "Client", MockClient)

        # Reset default client
        media_mod._default_client = None

        result = upload_media_and_get_url(str(video_file))
        assert result.file_name == "test.mp4"
        assert result.download_url == "https://s3.example.com/download"


class TestHTTPErrorHandling:
    """Tests for HTTP error and timeout handling."""

    def test_generate_upload_urls_timeout(self, monkeypatch):
        """Test timeout handling in generate_upload_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(TransportError) as exc_info:
            client.generate_upload_urls(
                [{"file_name": "test.mp4", "content_type": "video/mp4", "content_length": 1000}]
            )
        assert "timed out" in str(exc_info.value)

    def test_generate_upload_urls_http_error(self, monkeypatch):
        """Test HTTP error handling in generate_upload_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                raise httpx.HTTPError("Connection refused")

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(TransportError) as exc_info:
            client.generate_upload_urls(
                [{"file_name": "test.mp4", "content_type": "video/mp4", "content_length": 1000}]
            )
        assert "Connection refused" in str(exc_info.value)

    def test_generate_upload_urls_non_200_response(self, monkeypatch):
        """Test non-200 response handling in generate_upload_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 400
            text = "Bad Request"

            def json(self):
                return {"message": "Invalid request"}

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(BadRequestError):
            client.generate_upload_urls(
                [{"file_name": "test.mp4", "content_type": "video/mp4", "content_length": 1000}]
            )

    def test_generate_download_urls_timeout(self, monkeypatch):
        """Test timeout handling in generate_download_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(TransportError) as exc_info:
            client.generate_download_urls([UUID("12345678-1234-5678-1234-567812345678")])
        assert "timed out" in str(exc_info.value)

    def test_generate_download_urls_http_error(self, monkeypatch):
        """Test HTTP error handling in generate_download_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, headers=None, json=None):
                raise httpx.HTTPError("Connection refused")

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(TransportError) as exc_info:
            client.generate_download_urls([UUID("12345678-1234-5678-1234-567812345678")])
        assert "Connection refused" in str(exc_info.value)

    def test_upload_to_presigned_url_timeout(self, monkeypatch):
        """Test timeout handling in upload_to_presigned_url."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def put(self, url, content=None, headers=None):
                raise httpx.TimeoutException("Upload timed out")

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(TransportError) as exc_info:
            client.upload_to_presigned_url(
                "https://s3.example.com/upload",
                b"test data",
                "video/mp4",
            )
        assert "timed out" in str(exc_info.value)

    def test_upload_to_presigned_url_http_error(self, monkeypatch):
        """Test HTTP error handling in upload_to_presigned_url."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def put(self, url, content=None, headers=None):
                raise httpx.HTTPError("Connection refused")

        monkeypatch.setattr(httpx, "Client", MockClient)

        client = MediaClient()
        with pytest.raises(TransportError) as exc_info:
            client.upload_to_presigned_url(
                "https://s3.example.com/upload",
                b"test data",
                "video/mp4",
            )
        assert "Connection refused" in str(exc_info.value)


class TestAsyncMediaClient:
    """Tests for AsyncMediaClient."""

    def test_async_client_init(self):
        """Test AsyncMediaClient initialization."""
        client = AsyncMediaClient()
        assert client._api_key is None
        assert client._base_url is None
        assert client._timeout == 300.0

    def test_async_client_custom_init(self):
        """Test AsyncMediaClient with custom parameters."""
        client = AsyncMediaClient(api_key="test-key", base_url="https://custom.api", timeout=60.0)
        assert client._api_key == "test-key"
        assert client._base_url == "https://custom.api"
        assert client._timeout == 60.0

    def test_async_generate_upload_urls(self, monkeypatch):
        """Test async generate_upload_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "urls": [
                        {
                            "upload_url": "https://s3.example.com/upload",
                            "object_key": "12345678-1234-5678-1234-567812345678",
                            "file_name": "test.mp4",
                        }
                    ],
                    "expires_in_seconds": 3600,
                }

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            return await client.generate_upload_urls(
                [{"file_name": "test.mp4", "content_type": "video/mp4", "content_length": 1000}]
            )

        urls, expires = asyncio.run(run_test())

        assert len(urls) == 1
        assert urls[0].file_name == "test.mp4"
        assert expires == 3600

    def test_async_generate_download_urls(self, monkeypatch):
        """Test async generate_download_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "urls": [
                        {
                            "download_url": "https://s3.example.com/download",
                            "object_key": "12345678-1234-5678-1234-567812345678",
                        }
                    ],
                    "expires_in_seconds": 3600,
                }

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            return await client.generate_download_urls([UUID("12345678-1234-5678-1234-567812345678")])

        urls, expires = asyncio.run(run_test())

        assert len(urls) == 1
        assert urls[0]["download_url"] == "https://s3.example.com/download"
        assert expires == 3600

    def test_async_upload_to_presigned_url(self, monkeypatch):
        """Test async upload_to_presigned_url."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 200
            text = "OK"

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def put(self, url, content=None, headers=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.upload_to_presigned_url(
                "https://s3.example.com/upload",
                b"test data",
                "video/mp4",
            )

        asyncio.run(run_test())

    def test_async_upload_to_presigned_url_failure(self, monkeypatch):
        """Test async upload_to_presigned_url failure."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockResponse:
            status_code = 500
            text = "Internal Server Error"

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def put(self, url, content=None, headers=None):
                return MockResponse()

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.upload_to_presigned_url(
                "https://s3.example.com/upload",
                b"test data",
                "video/mp4",
            )

        with pytest.raises(ServerError):
            asyncio.run(run_test())

    def test_async_upload_full_flow(self, monkeypatch, tmp_path):
        """Test async upload full flow."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        # Create test file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")

        class MockResponse:
            def __init__(self, status_code, data=None):
                self.status_code = status_code
                self._data = data
                self.text = "OK"

            def json(self):
                return self._data

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                return MockResponse(
                    200,
                    {
                        "urls": [
                            {
                                "upload_url": "https://s3.example.com/upload",
                                "object_key": "12345678-1234-5678-1234-567812345678",
                                "file_name": "test.mp4",
                            }
                        ],
                        "expires_in_seconds": 3600,
                    },
                )

            async def put(self, url, content=None, headers=None):
                return MockResponse(200)

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            return await client.upload(str(video_file))

        result = asyncio.run(run_test())

        assert result.file_name == "test.mp4"
        assert str(result.object_key) == "12345678-1234-5678-1234-567812345678"

    def test_async_upload_and_get_url(self, monkeypatch, tmp_path):
        """Test async upload_and_get_url."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        # Create test file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")

        class MockResponse:
            def __init__(self, status_code, data=None):
                self.status_code = status_code
                self._data = data
                self.text = "OK"

            def json(self):
                return self._data

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                if "upload-urls" in url:
                    return MockResponse(
                        200,
                        {
                            "urls": [
                                {
                                    "upload_url": "https://s3.example.com/upload",
                                    "object_key": "12345678-1234-5678-1234-567812345678",
                                    "file_name": "test.mp4",
                                }
                            ],
                            "expires_in_seconds": 3600,
                        },
                    )
                else:  # download-urls
                    return MockResponse(
                        200,
                        {
                            "urls": [
                                {
                                    "download_url": "https://s3.example.com/download",
                                    "object_key": "12345678-1234-5678-1234-567812345678",
                                }
                            ],
                            "expires_in_seconds": 3600,
                        },
                    )

            async def put(self, url, content=None, headers=None):
                return MockResponse(200)

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            return await client.upload_and_get_url(str(video_file))

        result = asyncio.run(run_test())

        assert result.file_name == "test.mp4"
        assert result.download_url == "https://s3.example.com/download"

    def test_async_generate_upload_urls_timeout(self, monkeypatch):
        """Test async timeout handling in generate_upload_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.generate_upload_urls(
                [{"file_name": "test.mp4", "content_type": "video/mp4", "content_length": 1000}]
            )

        with pytest.raises(TransportError) as exc_info:
            asyncio.run(run_test())
        assert "timed out" in str(exc_info.value)

    def test_async_generate_upload_urls_http_error(self, monkeypatch):
        """Test async HTTP error handling in generate_upload_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                raise httpx.HTTPError("Connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.generate_upload_urls(
                [{"file_name": "test.mp4", "content_type": "video/mp4", "content_length": 1000}]
            )

        with pytest.raises(TransportError) as exc_info:
            asyncio.run(run_test())
        assert "Connection refused" in str(exc_info.value)

    def test_async_generate_download_urls_timeout(self, monkeypatch):
        """Test async timeout handling in generate_download_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.generate_download_urls([UUID("12345678-1234-5678-1234-567812345678")])

        with pytest.raises(TransportError) as exc_info:
            asyncio.run(run_test())
        assert "timed out" in str(exc_info.value)

    def test_async_generate_download_urls_http_error(self, monkeypatch):
        """Test async HTTP error handling in generate_download_urls."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                raise httpx.HTTPError("Connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.generate_download_urls([UUID("12345678-1234-5678-1234-567812345678")])

        with pytest.raises(TransportError) as exc_info:
            asyncio.run(run_test())
        assert "Connection refused" in str(exc_info.value)

    def test_async_upload_to_presigned_url_timeout(self, monkeypatch):
        """Test async timeout handling in upload_to_presigned_url."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def put(self, url, content=None, headers=None):
                raise httpx.TimeoutException("Upload timed out")

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.upload_to_presigned_url(
                "https://s3.example.com/upload",
                b"test data",
                "video/mp4",
            )

        with pytest.raises(TransportError) as exc_info:
            asyncio.run(run_test())
        assert "timed out" in str(exc_info.value)

    def test_async_upload_to_presigned_url_http_error(self, monkeypatch):
        """Test async HTTP error handling in upload_to_presigned_url."""
        monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def put(self, url, content=None, headers=None):
                raise httpx.HTTPError("Connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        async def run_test():
            client = AsyncMediaClient()
            await client.upload_to_presigned_url(
                "https://s3.example.com/upload",
                b"test data",
                "video/mp4",
            )

        with pytest.raises(TransportError) as exc_info:
            asyncio.run(run_test())
        assert "Connection refused" in str(exc_info.value)
