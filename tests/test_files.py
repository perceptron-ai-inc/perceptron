"""`client.files`: multipart upload, content-type sniffing, listing and pagination, retrieve/content/download/delete,
error mapping, the provider rule, and async parity."""

import asyncio
import io

import httpx
import pytest
from _http_mock import Body, FailingBody, install, json_response
from _image_fixtures import JPEG_BYTES, PNG_BYTES, WEBP_BYTES

from perceptron import AsyncClient, Client, settings
from perceptron import config as cfg
from perceptron import files as files_mod
from perceptron.errors import (
    INVALID_PARAMETER,
    UNSUPPORTED_PROVIDER_FEATURE,
    BadRequestError,
    NotFoundError,
    TransportError,
)
from perceptron.errors import TimeoutError as SDKTimeoutError
from perceptron.files import AsyncFiles, File, FileDeleted, FileList, Files

FILE_ID = "file-Q7m2Lx9aPz3Kc8Rt1VbN0s"
FILE = {
    "object": "file",
    "id": FILE_ID,
    "bytes": len(PNG_BYTES),
    "created_at": 1790121600,
    "filename": "cat.png",
    "purpose": "vision",
}
TRACE = {"x-trace-id": "trace-files"}
NOT_FOUND = {
    "error": {
        "message": "No such File object: file-missing",
        "type": "invalid_request_error",
        "param": "id",
        "code": None,
    }
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _file(index: int) -> dict:
    return {**FILE, "id": f"file-{index:022d}", "filename": f"f{index}.png"}


def _page(ids, *, has_more=False) -> dict:
    data = [_file(i) for i in ids]
    return {
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else "",
        "last_id": data[-1]["id"] if data else "",
        "has_more": has_more,
    }


def _multipart(request: httpx.Request) -> dict[str, dict]:
    """The request's multipart parts by field name: ``{"disposition", "content_type", "body"}``."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    parts = {}
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, body = chunk[2:-2].split(b"\r\n\r\n", 1)
        headers = dict(line.split(": ", 1) for line in head.decode().split("\r\n"))
        name = headers["Content-Disposition"].split('name="')[1].split('"')[0]
        parts[name] = {
            "disposition": headers["Content-Disposition"],
            "content_type": headers.get("Content-Type"),
            "body": body,
        }
    return parts


def _ftyp(major: bytes, compatible=(), *, box_size=None) -> bytes:
    """An ISO BMFF file start: an `ftyp` box (major brand, minor version, compatible brands) and a `moov` box."""
    payload = major + b"\x00\x00\x02\x00" + b"".join(compatible)
    size = 8 + len(payload) if box_size is None else box_size
    return size.to_bytes(4, "big") + b"ftyp" + payload + b"\x00\x00\x00\x08moov"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def test_upload_sends_a_multipart_body_with_the_sniffed_type(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(FILE, headers=TRACE))

    uploaded = Client().files.upload(PNG_BYTES, filename="cat.png")

    request = http.last
    assert request.method == "POST"
    assert str(request.url) == "https://api.perceptron.inc/v1/files"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    # A body of known length: the server answers 411 to chunked uploads.
    assert int(request.headers["content-length"]) == len(request.content)
    assert "transfer-encoding" not in request.headers
    parts = _multipart(request)
    assert parts["purpose"]["body"] == b"vision"
    assert parts["file"]["disposition"] == 'form-data; name="file"; filename="cat.png"'
    assert parts["file"]["content_type"] == "image/png"
    assert parts["file"]["body"] == PNG_BYTES
    assert uploaded == File(
        id=FILE_ID, bytes=len(PNG_BYTES), created_at=1790121600, filename="cat.png", purpose="vision"
    )


def test_upload_from_a_path_uses_its_name(monkeypatch, tmp_path):
    http = install(monkeypatch, lambda request: json_response(FILE))
    path = tmp_path / "photo.jpg"
    path.write_bytes(JPEG_BYTES)

    Client().files.upload(path, purpose="user_data")
    parts = _multipart(http.last)
    assert parts["purpose"]["body"] == b"user_data"
    assert parts["file"]["disposition"] == 'form-data; name="file"; filename="photo.jpg"'
    assert parts["file"]["content_type"] == "image/jpeg"
    assert parts["file"]["body"] == JPEG_BYTES

    Client().files.upload(str(path), filename="renamed.jpg")
    assert 'filename="renamed.jpg"' in _multipart(http.last)["file"]["disposition"]


def test_upload_from_binary_file_objects(monkeypatch, tmp_path):
    http = install(monkeypatch, lambda request: json_response(FILE))
    path = tmp_path / "still.webp"
    path.write_bytes(WEBP_BYTES)

    with path.open("rb") as handle:
        Client().files.upload(handle)
    parts = _multipart(http.last)
    assert parts["file"]["disposition"] == 'form-data; name="file"; filename="still.webp"'
    assert parts["file"]["content_type"] == "image/webp"
    assert parts["file"]["body"] == WEBP_BYTES

    # No name: the part has no filename, and the server names the file by its id.
    Client().files.upload(io.BytesIO(PNG_BYTES))
    assert _multipart(http.last)["file"]["disposition"] == 'form-data; name="file"'


def test_bytes_the_sdk_cannot_vouch_for_go_as_octet_stream(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(FILE))

    Client().files.upload(b"GIF89a\x01\x00\x01\x00", filename="anim.gif")

    assert _multipart(http.last)["file"]["content_type"] == "application/octet-stream"


def _riff(kind: bytes) -> bytes:
    return b"RIFF\x24\x00\x00\x00" + kind + b"fmt \x10\x00\x00\x00"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (PNG_BYTES, "image/png"),
        (JPEG_BYTES, "image/jpeg"),
        (WEBP_BYTES, "image/webp"),
        (_riff(b"WAVE"), "audio/wav"),
        (b"fLaC\x00\x00\x00\x22" + b"\x00" * 16, "audio/flac"),
        (b"ID3\x04\x00\x00\x00\x00\x00\x00", "audio/mpeg"),
        (b"\xff\xfb\x90\x64\x00", "audio/mpeg"),  # MPEG-1 layer III
        (b"\xff\xf3\x90\x64\x00", "audio/mpeg"),  # untagged MPEG-2 layer III
        (_ftyp(b"isom", [b"isom", b"iso2", b"avc1", b"mp41"]), "video/mp4"),
        (_ftyp(b"mp42", [b"mp42", b"isom"]), "video/mp4"),
        (_ftyp(b"qt  ", [b"qt  "]), "video/quicktime"),
        (b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\x82\x84webm\x42\x87\x81\x04", "video/webm"),
        # Declared only when the server's sniffer is certain to agree; everything else is application/octet-stream.
        (b"\x1a\x45\xdf\xa3\x93\x42\x82\x88matroska\x42\x87", None),  # Matroska
        (_ftyp(b"M4V ", [b"M4V "]), None),
        (_ftyp(b"M4A ", [b"M4A "]), None),
        (_ftyp(b"heic", [b"mif1", b"heic"]), None),
        (_ftyp(b"isom", [b"isom", b"avif"]), None),  # read as AVIF by the server
        (_ftyp(b"isom", [b"isom"], box_size=256), None),  # starts like an ICO file
        (b"\xff\xf1\x50\x80\x02\x1f\xfc", None),  # AAC (ADTS), not layer III
        (b"\xff\xfb\xf0\x64\x00", None),  # invalid bitrate
        (_riff(b"AVI "), None),
        (b"GIF89a\x01\x00", None),
        (b"%PDF-1.7", None),
        (b"OggS\x00\x02", None),
        (b"", None),
    ],
)
def test_content_type_sniffing(data, expected):
    assert files_mod._sniff_mime_type(data) == expected


def test_upload_validates_purpose_and_size_before_sending(monkeypatch, tmp_path):
    http = install(monkeypatch, lambda request: json_response(FILE))
    monkeypatch.setattr(files_mod, "MAX_UPLOAD_BYTES", 8)
    files = Client().files

    with pytest.raises(BadRequestError) as excinfo:
        files.upload(PNG_BYTES[:8], purpose="fine-tune")
    assert (excinfo.value.code, excinfo.value.param) == (INVALID_PARAMETER, "purpose")

    big = tmp_path / "big.png"
    big.write_bytes(PNG_BYTES[:9])
    for source in (PNG_BYTES[:9], big, io.BytesIO(PNG_BYTES[:9])):
        with pytest.raises(BadRequestError, match="upload limit") as excinfo:
            files.upload(source)
        assert (excinfo.value.code, excinfo.value.param) == (INVALID_PARAMETER, "file")
    assert not http.requests

    files.upload(PNG_BYTES[:8])  # exactly at the limit
    assert len(http.requests) == 1


def test_upload_rejects_text_files_and_other_objects(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(FILE))

    with pytest.raises(TypeError, match="binary mode"):
        Client().files.upload(io.StringIO("text"))
    with pytest.raises(TypeError, match="path, bytes or a binary file object"):
        Client().files.upload(123)
    assert not http.requests


def test_upload_rejects_unreadable_paths_and_bad_filenames(monkeypatch, tmp_path):
    http = install(monkeypatch, lambda request: json_response(FILE))
    files = Client().files

    for path in (tmp_path / "missing.png", tmp_path):
        with pytest.raises(BadRequestError, match="Cannot read") as excinfo:
            files.upload(path)
        assert (excinfo.value.code, excinfo.value.param) == (INVALID_PARAMETER, "file")
        assert isinstance(excinfo.value.__cause__, OSError)
    with pytest.raises(TypeError, match="filename must be a str"):
        files.upload(PNG_BYTES, filename=123)
    with pytest.raises(TypeError, match="filename must be a str"):
        files.upload(tmp_path / "missing.png", filename=123)  # checked before the file is read
    assert not http.requests


def test_upload_errors_are_mapped(monkeypatch):
    error = {"error": {"message": "Unsupported content type 'image/gif'.", "type": "invalid_request_error"}}
    install(monkeypatch, lambda request: json_response(error, status=400, headers=TRACE))

    with pytest.raises(BadRequestError, match="image/gif") as excinfo:
        Client().files.upload(b"GIF89a")
    assert excinfo.value.request_id == "trace-files"


# ---------------------------------------------------------------------------
# List and iterate
# ---------------------------------------------------------------------------


def test_list_sends_only_the_set_parameters(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(_page([1, 2], has_more=True)))
    files = Client().files

    page = files.list()
    assert http.last.method == "GET"
    assert str(http.last.url) == "https://api.perceptron.inc/v1/files"

    files.list(purpose="vision", limit=2, order="asc", after="file-cursor")
    assert dict(http.last.url.params) == {"purpose": "vision", "limit": "2", "order": "asc", "after": "file-cursor"}

    assert isinstance(page, FileList)
    assert [f.id for f in page] == [_file(1)["id"], _file(2)["id"]]
    assert (page.first_id, page.last_id, page.has_more) == (_file(1)["id"], _file(2)["id"], True)
    assert page.to_dict() == _page([1, 2], has_more=True)


def test_empty_page():
    page = FileList.from_dict(_page([]))
    assert (list(page), page.first_id, page.last_id, page.has_more) == ([], "", "", False)


@pytest.mark.parametrize(
    ("kwargs", "param"),
    [
        ({"limit": 0}, "limit"),
        ({"limit": 10_001}, "limit"),
        ({"limit": True}, "limit"),
        ({"limit": "5"}, "limit"),
        ({"order": "newest"}, "order"),
        ({"purpose": "assistants"}, "purpose"),
        ({"after": ""}, "after"),
    ],
)
def test_list_validates_before_sending(monkeypatch, kwargs, param):
    http = install(monkeypatch, lambda request: json_response(_page([])))

    with pytest.raises(BadRequestError) as excinfo:
        Client().files.list(**kwargs)
    assert (excinfo.value.code, excinfo.value.param) == (INVALID_PARAMETER, param)
    with pytest.raises(BadRequestError):
        Client().files.iter(**kwargs)  # checked when called, before iterating
    assert not http.requests


def test_iter_follows_last_id_while_has_more(monkeypatch):
    pages = {
        None: _page([1, 2], has_more=True),
        _file(2)["id"]: _page([3, 4], has_more=True),
        _file(4)["id"]: _page([5], has_more=False),
    }
    http = install(monkeypatch, lambda request: json_response(pages[request.url.params.get("after")]))

    ids = [f.id for f in Client().files.iter(purpose="vision", limit=2)]

    assert ids == [_file(i)["id"] for i in range(1, 6)]
    assert [dict(r.url.params) for r in http.requests] == [
        {"purpose": "vision", "limit": "2"},
        {"purpose": "vision", "limit": "2", "after": _file(2)["id"]},
        {"purpose": "vision", "limit": "2", "after": _file(4)["id"]},
    ]


def test_iter_is_lazy_and_stops_on_an_empty_page(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(_page([], has_more=True)))

    iterator = Client().files.iter()
    assert not http.requests
    assert list(iterator) == []
    assert len(http.requests) == 1


# ---------------------------------------------------------------------------
# Retrieve, content, download, delete
# ---------------------------------------------------------------------------


def test_retrieve_content_and_delete(monkeypatch):
    def handler(request):
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=PNG_BYTES, headers={"content-type": "image/png"})
        if request.method == "DELETE":
            return json_response({"object": "file", "id": FILE_ID, "deleted": True})
        return json_response(FILE)

    http = install(monkeypatch, handler)
    files = Client().files

    assert files.retrieve(FILE_ID) == File.from_dict(FILE)
    assert (http.last.method, str(http.last.url)) == ("GET", f"https://api.perceptron.inc/v1/files/{FILE_ID}")

    assert files.content(FILE_ID) == PNG_BYTES
    assert str(http.last.url) == f"https://api.perceptron.inc/v1/files/{FILE_ID}/content"

    deleted = files.delete(FILE_ID)
    assert deleted == FileDeleted(id=FILE_ID, deleted=True)
    assert (http.last.method, str(http.last.url)) == ("DELETE", f"https://api.perceptron.inc/v1/files/{FILE_ID}")
    assert deleted.to_dict() == {"object": "file", "id": FILE_ID, "deleted": True}


def test_file_ids_are_path_escaped_and_checked(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(FILE))

    Client().files.retrieve("file-a/../b")
    assert http.last.url.raw_path == b"/v1/files/file-a%2F..%2Fb"

    with pytest.raises(BadRequestError) as excinfo:
        Client().files.retrieve("")
    assert excinfo.value.param == "file_id"
    with pytest.raises(TypeError, match="file_id must be a str"):
        Client().files.delete(None)
    assert len(http.requests) == 1


def test_missing_file_is_a_not_found_error(monkeypatch):
    install(monkeypatch, lambda request: json_response(NOT_FOUND, status=404, headers=TRACE))

    with pytest.raises(NotFoundError) as excinfo:
        Client().files.retrieve("file-missing")

    err = excinfo.value
    assert isinstance(err, BadRequestError)
    assert (err.status_code, err.param, err.error_type) == (404, "id", "invalid_request_error")
    assert err.request_id == "trace-files"
    assert str(err) == "No such File object: file-missing"


def test_download_streams_to_disk(monkeypatch, tmp_path):
    body = Body(PNG_BYTES)
    http = install(monkeypatch, lambda request: httpx.Response(200, stream=body, headers={"content-type": "image/png"}))
    target = tmp_path / "cat.png"

    result = Client().files.download(FILE_ID, target)

    assert result == target
    assert target.read_bytes() == PNG_BYTES
    assert str(http.last.url) == f"https://api.perceptron.inc/v1/files/{FILE_ID}/content"
    assert body.closed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cat.png"]

    assert Client().files.download(FILE_ID, str(tmp_path / "again.png")) == tmp_path / "again.png"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [(httpx.ReadError("connection reset"), TransportError), (httpx.ReadTimeout("slow"), SDKTimeoutError)],
)
def test_download_failing_mid_body_leaves_no_partial_file(monkeypatch, tmp_path, exc, expected):
    body = FailingBody(PNG_BYTES[:10], exc)
    install(monkeypatch, lambda request: httpx.Response(200, stream=body))
    target = tmp_path / "cat.png"
    target.write_bytes(b"previous download")

    with pytest.raises(expected):
        Client().files.download(FILE_ID, target)

    assert target.read_bytes() == b"previous download"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cat.png"]
    assert body.closed


def test_download_http_error_writes_nothing(monkeypatch, tmp_path):
    install(monkeypatch, lambda request: httpx.Response(404, stream=Body(b'{"error": {"message": "No such File"}}')))

    with pytest.raises(NotFoundError, match="No such File"):
        Client().files.download("file-missing", tmp_path / "out.png")
    assert list(tmp_path.iterdir()) == []


def test_download_to_a_directory_raises_before_any_request(monkeypatch, tmp_path):
    http = install(monkeypatch, lambda request: httpx.Response(200, stream=Body(PNG_BYTES)))

    async def _async_download():
        await AsyncClient().files.download(FILE_ID, tmp_path)

    for call in (lambda: Client().files.download(FILE_ID, tmp_path), lambda: asyncio.run(_async_download())):
        with pytest.raises(BadRequestError) as excinfo:
            call()
        assert (excinfo.value.code, excinfo.value.param) == ("invalid_parameter", "path")
    assert http.requests == []
    assert list(tmp_path.iterdir()) == []


def test_download_local_write_errors_are_sdk_errors(monkeypatch, tmp_path):
    install(monkeypatch, lambda request: httpx.Response(200, stream=Body(PNG_BYTES)))

    with pytest.raises(BadRequestError) as excinfo:
        Client().files.download(FILE_ID, tmp_path / "missing-dir" / "cat.png")
    assert excinfo.value.param == "path"


def test_parsing_is_lenient():
    parsed = File.from_dict({"id": FILE_ID, "bytes": "12", "unknown": 1})
    assert parsed == File(id=FILE_ID)
    assert File.from_dict(None) == File(id="")
    assert FileDeleted.from_dict({"id": FILE_ID}) == FileDeleted(id=FILE_ID, deleted=False)
    assert FileList.from_dict({"data": "nope"}).data == []


# ---------------------------------------------------------------------------
# Provider rule
# ---------------------------------------------------------------------------


def test_api_key_only_env_uses_the_perceptron_api(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(FILE))
    assert settings().provider == "fal"  # the legacy auto-detect is unchanged

    Client().files.upload(PNG_BYTES)

    assert str(http.last.url) == "https://api.perceptron.inc/v1/files"
    assert http.last.headers["authorization"] == "Bearer sk-test"


def test_explicit_non_perceptron_provider_is_rejected(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(FILE))

    def _clients():
        yield Client(provider="fal")
        with cfg(provider="fal"):
            yield Client()
        monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
        yield Client()

    for client in _clients():
        with pytest.raises(BadRequestError) as excinfo:
            client.files.list()
        assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
        assert "PERCEPTRON_PROVIDER=perceptron" in str(excinfo.value)
    assert not http.requests


def test_resources_are_the_real_classes():
    assert isinstance(Client().files, Files)
    assert isinstance(AsyncClient().files, AsyncFiles)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_async_parity(monkeypatch, tmp_path):
    pages = {None: _page([1], has_more=True), _file(1)["id"]: _page([2])}

    def handler(request):
        path = request.url.path
        if request.method == "POST":
            return json_response(FILE, headers=TRACE)
        if request.method == "DELETE":
            return json_response({"object": "file", "id": FILE_ID, "deleted": True})
        if path == "/v1/files":
            return json_response(pages[request.url.params.get("after")])
        if path.endswith("/content"):
            return httpx.Response(200, stream=Body(PNG_BYTES))
        if path.endswith("/file-missing"):
            return json_response(NOT_FOUND, status=404)
        return json_response(FILE)

    http = install(monkeypatch, handler)
    files = AsyncClient().files

    async def _run():
        uploaded = await files.upload(PNG_BYTES, filename="cat.png")
        page = await files.list(limit=1)
        ids = [f.id async for f in files.iter(limit=1)]
        retrieved = await files.retrieve(FILE_ID)
        content = await files.content(FILE_ID)
        downloaded = await files.download(FILE_ID, tmp_path / "cat.png")
        deleted = await files.delete(FILE_ID)
        with pytest.raises(NotFoundError):
            await files.retrieve("file-missing")
        return uploaded, page, ids, retrieved, content, downloaded, deleted

    uploaded, page, ids, retrieved, content, downloaded, deleted = asyncio.run(_run())

    assert uploaded == retrieved == File.from_dict(FILE)
    assert _multipart(http.requests[0])["file"]["content_type"] == "image/png"
    assert [f.id for f in page] == [_file(1)["id"]]
    assert ids == [_file(1)["id"], _file(2)["id"]]
    assert content == PNG_BYTES
    assert downloaded.read_bytes() == PNG_BYTES
    assert deleted.deleted is True


def test_async_download_failure_and_provider_rule(monkeypatch, tmp_path):
    http = install(monkeypatch, lambda request: httpx.Response(200, stream=FailingBody(b"abc", httpx.ReadError("x"))))

    async def _download():
        await AsyncClient().files.download(FILE_ID, tmp_path / "out.png")

    with pytest.raises(TransportError):
        asyncio.run(_download())
    assert list(tmp_path.iterdir()) == []

    async def _upload_on_fal():
        await AsyncClient(provider="fal").files.upload(PNG_BYTES)

    with pytest.raises(BadRequestError) as excinfo:
        asyncio.run(_upload_on_fal())
    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    assert len(http.requests) == 1
