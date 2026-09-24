"""One provider rule for every surface, and fal's key isolation (DESIGN §14.1).

The provider you choose wins; otherwise ``fal`` only when ``FAL_KEY`` is set and ``PERCEPTRON_API_KEY`` is not;
otherwise ``perceptron``. Provider ``fal`` never receives a key read from ``PERCEPTRON_API_KEY``, and files, models and
multilook exist only on ``perceptron``. Requests go through `httpx.MockTransport` (see `_http_mock`).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response
from _image_fixtures import PNG_BYTES
from typer.testing import CliRunner

from perceptron import (
    AsyncClient,
    Client,
    async_perceive,
    caption,
    config,
    detect,
    image,
    ocr,
    perceive,
    question,
    settings,
    text,
)
from perceptron.cli import app
from perceptron.errors import CREDENTIALS_MISSING, UNSUPPORTED_PROVIDER_FEATURE, AuthError, BadRequestError

PERCEPTRON_CHAT = "https://api.perceptron.inc/v1/chat/completions"
FAL_CHAT = "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"
PERCEPTRON = (PERCEPTRON_CHAT, "Bearer sk-test", "perceptron-mk1.5")
FAL = (FAL_CHAT, "Key fal-key", "isaac-0.1")
USER = {"role": "user", "content": "Hi"}
TASK = {"content": [{"type": "text", "role": "user", "content": "Hi"}]}
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


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("PERCEPTRON_API_KEY", "FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


def _handler(request):
    path = request.url.path
    if path.endswith("/multilook"):
        return json_response(MULTILOOK)
    if path.endswith("/chat/completions"):
        if json.loads(request.content).get("stream"):
            return sse_response([chunk({"role": "assistant", "content": "ok"}), chunk({}, finish_reason="stop")])
        return json_response(completion("ok"))
    if path.endswith("/models"):
        return json_response({"object": "list", "data": [MODEL]})
    if request.method == "GET":
        return json_response({"object": "list", "data": [FILE], "first_id": FILE_ID, "last_id": FILE_ID})
    return json_response(FILE)


@pytest.fixture
def http(monkeypatch):
    return install(monkeypatch, _handler)


def _sent(http) -> tuple[str, str, str]:
    request = http.last
    return str(request.url), request.headers["authorization"], json.loads(request.content)["model"]


def _img():
    return image(PNG_BYTES)


async def _acollect(events):
    return [event async for event in events]


def _async_perceive():
    @async_perceive()
    def ask():
        return _img() + text("What is shown?")

    return asyncio.run(ask())


async def _acreate(client):
    return await client.chat.completions.create(messages=[USER])


def _cli(*flags):
    result = CliRunner().invoke(app, ["question", "https://example.com/img.png", "What is shown?", *flags])
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def _cli_ok():
    result = _cli()
    assert result.exit_code == 0, result.stdout


# Surfaces that send one chat completion request.
CHAT_SURFACES = {
    "perceive": lambda: perceive(_img(), text("What is shown?")),
    "perceive-stream": lambda: list(perceive(_img(), text("What is shown?"), stream=True)),
    "async-perceive": _async_perceive,
    "question": lambda: question(_img(), "What is shown?"),
    "question-stream": lambda: list(question(_img(), "What is shown?", stream=True)),
    "detect": lambda: detect(_img(), classes=["cat"]),
    "caption": lambda: caption(_img()),
    "ocr": lambda: ocr(_img()),
    "generate": lambda: Client().generate(TASK),
    "stream": lambda: list(Client().stream(TASK)),
    "async-generate": lambda: asyncio.run(AsyncClient().generate(TASK)),
    "async-stream": lambda: asyncio.run(_acollect(AsyncClient().stream(TASK))),
    "create": lambda: Client().chat.completions.create(messages=[USER]),
    "create-stream": lambda: Client().chat.completions.create(messages=[USER], stream=True).get_final_completion(),
    "async-create": lambda: asyncio.run(_acreate(AsyncClient())),
    "cli": _cli_ok,
}

# The Perceptron-only endpoints: (call, path, model sent).
RESOURCE_SURFACES = {
    "files-upload": (lambda: Client().files.upload(PNG_BYTES), "/v1/files", None),
    "files-list": (lambda: Client().files.list(), "/v1/files", None),
    "async-files": (lambda: asyncio.run(AsyncClient().files.retrieve(FILE_ID)), f"/v1/files/{FILE_ID}", None),
    "models": (lambda: Client().models.list(), "/v1/models", None),
    "async-models": (lambda: asyncio.run(AsyncClient().models.list()), "/v1/models", None),
    "multilook": (
        lambda: Client().chat.completions.multilook(context=[], prompts=["q"]),
        "/v1/chat/completions/multilook",
        "perceptron-mk1.5",
    ),
    "async-multilook": (
        lambda: asyncio.run(AsyncClient().chat.completions.multilook(context=[], prompts=["q"])),
        "/v1/chat/completions/multilook",
        "perceptron-mk1.5",
    ),
}


# ---------------------------------------------------------------------------
# The default rule, on every surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surface", CHAT_SURFACES)
def test_perceptron_api_key_alone_selects_the_perceptron_api(http, monkeypatch, surface):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    assert settings().provider == "perceptron"

    CHAT_SURFACES[surface]()

    assert _sent(http) == PERCEPTRON


@pytest.mark.parametrize("surface", RESOURCE_SURFACES)
def test_perceptron_api_key_alone_reaches_files_models_and_multilook(http, monkeypatch, surface):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    call, path, model = RESOURCE_SURFACES[surface]

    call()

    request = http.last
    assert (request.url.host, request.url.path) == ("api.perceptron.inc", path)
    assert request.headers["authorization"] == "Bearer sk-test"
    if model is not None:
        assert json.loads(request.content)["model"] == model


@pytest.mark.parametrize("surface", CHAT_SURFACES)
def test_fal_key_alone_selects_fal(http, monkeypatch, surface):
    monkeypatch.setenv("FAL_KEY", "fal-key")
    assert settings().provider == "fal"

    CHAT_SURFACES[surface]()

    assert _sent(http) == FAL


@pytest.mark.parametrize("surface", RESOURCE_SURFACES)
def test_fal_key_alone_rejects_files_models_and_multilook(http, monkeypatch, surface):
    monkeypatch.setenv("FAL_KEY", "fal-key")

    with pytest.raises(BadRequestError) as excinfo:
        RESOURCE_SURFACES[surface][0]()

    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    message = str(excinfo.value)
    assert "provider 'fal'" in message
    assert 'configure(provider="perceptron")' in message and "PERCEPTRON_PROVIDER=perceptron" in message
    assert http.requests == []


@pytest.mark.parametrize("surface", [*CHAT_SURFACES, *RESOURCE_SURFACES])
def test_both_keys_select_the_perceptron_api(http, monkeypatch, surface):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    monkeypatch.setenv("FAL_KEY", "fal-key")

    call = CHAT_SURFACES.get(surface) or RESOURCE_SURFACES[surface][0]
    call()

    assert http.last.url.host == "api.perceptron.inc"
    assert http.last.headers["authorization"] == "Bearer sk-test"


# ---------------------------------------------------------------------------
# Provider fal never receives PERCEPTRON_API_KEY
# ---------------------------------------------------------------------------


def _in_config(**settings_kwargs):
    with config(**settings_kwargs):
        return perceive(_img(), text("Hi"))


def _env_provider(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
    return question(_img(), "Hi")


FAL_CHOSEN = {
    "perceive-provider": lambda monkeypatch: perceive(_img(), text("Hi"), provider="fal"),
    "helper-provider": lambda monkeypatch: detect(_img(), classes=["cat"], provider="fal"),
    "configure": lambda monkeypatch: _in_config(provider="fal"),
    "env": _env_provider,
    "client-generate": lambda monkeypatch: Client(provider="fal").generate(TASK),
    "generate-provider": lambda monkeypatch: Client().generate(TASK, provider="fal"),
    "async-generate": lambda monkeypatch: asyncio.run(AsyncClient(provider="fal").generate(TASK)),
    "create": lambda monkeypatch: Client(provider="fal").chat.completions.create(messages=[USER]),
    "create-stream": lambda monkeypatch: Client(provider="fal").chat.completions.create(messages=[USER], stream=True),
    "async-create": lambda monkeypatch: asyncio.run(_acreate(AsyncClient(provider="fal"))),
}


@pytest.mark.parametrize("surface", FAL_CHOSEN)
def test_chosen_fal_never_gets_the_perceptron_api_key(http, monkeypatch, surface):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-secret")

    with pytest.raises(AuthError) as excinfo:
        FAL_CHOSEN[surface](monkeypatch)

    assert excinfo.value.code == CREDENTIALS_MISSING
    message = str(excinfo.value)
    assert "No API key for provider 'fal'. Set FAL_KEY or configure(api_key=...)." in message
    assert "PERCEPTRON_API_KEY is only sent to the Perceptron API" in message
    assert "sk-secret" not in message
    assert http.requests == []


def test_chosen_fal_without_its_key_ends_legacy_streams_and_the_cli_without_a_request(http, monkeypatch):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-secret")

    events = list(Client(provider="fal").stream(TASK))
    assert [(event["type"], event["code"]) for event in events] == [("error", CREDENTIALS_MISSING)]
    result = _cli("--provider", "fal")
    assert result.exit_code == 1
    assert CREDENTIALS_MISSING in result.stdout
    assert http.requests == []


def test_chosen_fal_uses_a_key_set_in_code(http, monkeypatch):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")  # present, but never sent to fal

    with config(provider="fal", api_key="fal-key"):
        perceive(_img(), text("Hi"))
        assert _sent(http) == FAL
        Client().chat.completions.create(messages=[USER])
        assert _sent(http) == FAL
    Client(provider="fal", api_key="fal-key").generate(TASK)
    assert _sent(http) == FAL

    monkeypatch.setenv("FAL_KEY", "fal-env-key")  # a key set in code wins over FAL_KEY
    with config(provider="fal", api_key="fal-key"):
        question(_img(), "Hi")
    assert _sent(http) == FAL


def test_chosen_fal_uses_fal_key_even_when_both_keys_are_set(http, monkeypatch):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    monkeypatch.setenv("FAL_KEY", "fal-key")

    for call in (
        lambda: perceive(_img(), text("Hi"), provider="fal"),
        lambda: Client(provider="fal").chat.completions.create(messages=[USER]),
    ):
        call()
        assert _sent(http) == FAL


def test_a_key_set_in_code_goes_to_the_perceptron_api_by_default(http):
    with config(api_key="sk-test"):
        question(_img(), "Hi")
        assert _sent(http) == PERCEPTRON
    Client(api_key="sk-test").chat.completions.create(messages=[USER])
    assert _sent(http) == PERCEPTRON


def test_client_keeps_its_provider_and_key_pairing(http, monkeypatch):
    """A client resolves its provider when built; a later FAL_KEY-only env does not move it (or its key) to fal."""
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    client = Client()
    monkeypatch.delenv("PERCEPTRON_API_KEY")
    monkeypatch.setenv("FAL_KEY", "fal-key")

    client.chat.completions.create(messages=[USER])
    client.generate(TASK)

    assert [str(request.url) for request in http.requests] == [PERCEPTRON_CHAT, PERCEPTRON_CHAT]
    assert {request.headers["authorization"] for request in http.requests} == {"Bearer sk-test"}
