"""A configured base URL (`Client(base_url=...)`, `configure`/`config`, `PERCEPTRON_BASE_URL`) is used by the message
API, files, models and multilook, like `Client.generate`: the request and the API key go where the caller pointed them,
also when only `PERCEPTRON_API_KEY` is set (the legacy fal auto-detect)."""

import asyncio
import json
from contextlib import contextmanager

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response

from perceptron import AsyncClient, Client, image, settings, video_frames
from perceptron import config as cfg

BASE = "http://localhost:8080/v1"
DEFAULT = "https://api.perceptron.inc/v1"
FILE_ID = "file-Q7m2Lx9aPz3Kc8Rt1VbN0s"
FILE = {"object": "file", "id": FILE_ID, "bytes": 3, "created_at": 1790121600, "filename": "a.png", "purpose": "vision"}
MODEL = {"id": "perceptron-mk1.5", "object": "model", "created": 1790035200, "owned_by": "perceptron"}
MULTILOOK = {
    "id": "mlcmpl-1",
    "object": "chat.completion.multilook",
    "model": "perceptron-mk1.5",
    "results": [
        {
            "prompt_index": 0,
            "completions": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }
    ],
}
USER = {"role": "user", "content": "Hi"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _handler(request):
    path = request.url.path
    if path.endswith("/multilook"):
        return json_response(MULTILOOK)
    if path.endswith("/chat/completions"):
        if json.loads(request.content).get("stream"):
            return sse_response([chunk({"content": "hi"}), chunk({}, finish_reason="stop")])
        return json_response(completion())
    if path.endswith("/models"):
        return json_response({"object": "list", "data": [MODEL]})
    return json_response(FILE)


@contextmanager
def _configured(monkeypatch, source):
    """Configure ``BASE`` the ``source`` way; yields the ``Client``/``AsyncClient`` overrides."""
    if source == "client":
        yield {"base_url": BASE}
    elif source == "configure":
        with cfg(base_url=BASE):
            yield {}
    else:
        monkeypatch.setenv("PERCEPTRON_BASE_URL", BASE)
        yield {}


SOURCES = ["client", "configure", "env"]


def _stream(client):
    return client.chat.completions.create(messages=[USER], stream=True).get_final_completion()


SYNC_SURFACES = [
    pytest.param(lambda c: c.chat.completions.create(messages=[USER]), "/chat/completions", id="create"),
    pytest.param(_stream, "/chat/completions", id="create-stream"),
    pytest.param(lambda c: c.files.retrieve(FILE_ID), f"/files/{FILE_ID}", id="files"),
    pytest.param(lambda c: c.models.list(), "/models", id="models"),
    pytest.param(
        lambda c: c.chat.completions.multilook(context=[], prompts=["q"]), "/chat/completions/multilook", id="multilook"
    ),
]


async def _acreate(client):
    return await client.chat.completions.create(messages=[USER])


async def _astream(client):
    stream = await client.chat.completions.create(messages=[USER], stream=True)
    return await stream.get_final_completion()


async def _afiles(client):
    return await client.files.retrieve(FILE_ID)


async def _amodels(client):
    return await client.models.list()


async def _amultilook(client):
    return await client.chat.completions.multilook(context=[], prompts=["q"])


ASYNC_SURFACES = [
    pytest.param(_acreate, "/chat/completions", id="create"),
    pytest.param(_astream, "/chat/completions", id="create-stream"),
    pytest.param(_afiles, f"/files/{FILE_ID}", id="files"),
    pytest.param(_amodels, "/models", id="models"),
    pytest.param(_amultilook, "/chat/completions/multilook", id="multilook"),
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(("call", "path"), SYNC_SURFACES)
def test_sync_surfaces_use_the_configured_base_url(monkeypatch, source, call, path):
    http = install(monkeypatch, _handler)

    with _configured(monkeypatch, source) as overrides:
        assert settings().provider == "fal"  # only PERCEPTRON_API_KEY is set: the legacy auto-detect is in effect
        call(Client(**overrides))

    assert str(http.last.url).split("?")[0] == BASE + path
    assert http.last.headers["authorization"] == "Bearer sk-test"


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(("call", "path"), ASYNC_SURFACES)
def test_async_surfaces_use_the_configured_base_url(monkeypatch, source, call, path):
    http = install(monkeypatch, _handler)

    with _configured(monkeypatch, source) as overrides:
        asyncio.run(call(AsyncClient(**overrides)))

    assert str(http.last.url).split("?")[0] == BASE + path
    assert http.last.headers["authorization"] == "Bearer sk-test"


@pytest.mark.parametrize(("call", "path"), SYNC_SURFACES)
def test_without_a_base_url_the_perceptron_api_is_used(monkeypatch, call, path):
    http = install(monkeypatch, _handler)

    call(Client())

    assert str(http.last.url).split("?")[0] == DEFAULT + path


@pytest.mark.parametrize(("call", "path"), ASYNC_SURFACES)
def test_async_without_a_base_url_the_perceptron_api_is_used(monkeypatch, call, path):
    http = install(monkeypatch, _handler)

    asyncio.run(call(AsyncClient()))

    assert str(http.last.url).split("?")[0] == DEFAULT + path


def test_explicit_fal_uses_the_base_url_with_fals_path_and_auth(monkeypatch):
    http = install(monkeypatch, _handler)

    Client(provider="fal", base_url=BASE).chat.completions.create(messages=[USER])

    assert str(http.last.url) == BASE + "/perceptron/isaac-01/openai/v1/chat/completions"
    assert http.last.headers["authorization"] == "Key sk-test"
    assert json.loads(http.last.content)["model"] == "isaac-0.1"


def test_one_client_talks_to_one_host(monkeypatch):
    http = install(monkeypatch, _handler)
    client = Client(base_url=BASE)

    client.generate({"content": [{"type": "text", "role": "user", "content": "Hi"}]})  # legacy: fal, auto-detected
    client.chat.completions.create(messages=[USER])
    client.files.retrieve(FILE_ID)

    assert {request.url.host for request in http.requests} == {"localhost"}


def _frames():
    return video_frames([(image(file_id=FILE_ID), 0), (image(file_id=FILE_ID), 40)])


def _frames_entry():
    frames = [{"file_id": FILE_ID, "timestamp_ms": 0}, {"file_id": FILE_ID, "timestamp_ms": 40}]
    return {"type": "video_frames", "role": "user", "frames": frames}


@pytest.mark.parametrize("source", SOURCES)
def test_uploaded_frames_use_the_configured_base_url(monkeypatch, source):
    http = install(monkeypatch, _handler)
    content_url = f"{BASE}/files/{FILE_ID}/content"

    with _configured(monkeypatch, source) as overrides:
        client = Client(**overrides)
        client.chat.completions.create(messages=[{"role": "user", "content": [_frames(), "What happens?"]}])
        created = json.loads(http.last.content)["messages"][0]["content"][0]
        client.chat.completions.multilook(context=[{"role": "user", "content": [_frames()]}], prompts=["q"])
        multilook = json.loads(http.last.content)["context"][0]["content"][0]
        with cfg(provider="perceptron"):
            Client(**overrides).generate({"content": [_frames_entry()]})
        generated = json.loads(http.last.content)["messages"][0]["content"][0]

    for part in (created, multilook, generated):
        assert [frame["image_url"]["url"] for frame in part["video_frames"]["frames"]] == [content_url, content_url]
    assert {request.url.host for request in http.requests} == {"localhost"}
