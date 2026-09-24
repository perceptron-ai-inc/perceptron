"""`client.models`: list/retrieve with and without `extended`, open enums, errors, the provider rule, async parity."""

import asyncio

import pytest
from _http_mock import completion, install, json_response

from perceptron import AsyncClient, Client, settings
from perceptron.errors import INVALID_PARAMETER, UNSUPPORTED_PROVIDER_FEATURE, BadRequestError, NotFoundError
from perceptron.models import AsyncModels, Model, ModelInfo, ModelReasoning, Models

MODEL = {"id": "perceptron-mk1.5", "object": "model", "created": 1790035200, "owned_by": "perceptron"}
EXTENDED = {
    **MODEL,
    "name": "Perceptron Mk1.5",
    "capabilities": ["response_format_json_schema", "regex", "tool_calling", "some_future_capability"],
    "modalities": ["image", "video", "audio"],
    "output_formats": ["text", "point", "box", "polygon", "clip"],
    "sampling_parameters": ["temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty"],
    "reasoning": {"supported": True, "always_enabled": False},
    "max_context_tokens": 36864,
    "max_output_tokens": 8192,
    "description": "Perceptron Mk1.5 with image, video, and audio input, and tool calling.",
    "credits_per_million_input_tokens": 150000,
    "credits_per_million_output_tokens": 1500000,
    "credits_per_million_cache_read_tokens": 37500,
    "early_access": False,
}
ISAAC = {"id": "isaac-0.1", "object": "model", "created": 1758067200, "owned_by": "perceptron"}
MODEL_NOT_FOUND = {
    "error": {
        "message": "The model 'nonexistent-model' does not exist",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found",
    }
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _handler(request):
    extended = request.url.params.get("extended") == "true"
    if request.url.path == "/v1/models":
        data = [EXTENDED, {**ISAAC, "name": "Isaac 0.1"}] if extended else [MODEL, ISAAC]
        return json_response({"object": "list", "data": data})
    if request.url.path.endswith("/nonexistent-model"):
        return json_response(MODEL_NOT_FOUND, status=404, headers={"x-trace-id": "trace-models"})
    return json_response(EXTENDED if extended else MODEL)


def test_list_without_extended_sends_no_query(monkeypatch):
    http = install(monkeypatch, _handler)

    models = Client().models.list()

    assert http.last.method == "GET"
    assert str(http.last.url) == "https://api.perceptron.inc/v1/models"
    assert models == [Model.from_dict(MODEL), Model.from_dict(ISAAC)]
    assert all(type(m) is Model for m in models)
    assert models[0] == Model(id="perceptron-mk1.5", created=1790035200, owned_by="perceptron")
    assert models[0].to_dict() == MODEL


def test_list_extended(monkeypatch):
    http = install(monkeypatch, _handler)

    models = Client().models.list(extended=True)

    assert str(http.last.url) == "https://api.perceptron.inc/v1/models?extended=true"
    info = models[0]
    assert isinstance(info, ModelInfo) and isinstance(info, Model)
    assert (info.id, info.name, info.max_context_tokens, info.max_output_tokens) == (
        "perceptron-mk1.5",
        "Perceptron Mk1.5",
        36864,
        8192,
    )
    assert info.reasoning == ModelReasoning(supported=True, always_enabled=False)
    assert info.modalities == ["image", "video", "audio"]
    assert info.credits_per_million_cache_read_tokens == 37500
    assert info.early_access is False
    # Open lists: unknown names are kept, not rejected.
    assert info.capabilities[-1] == "some_future_capability"
    assert info.supports("tool_calling") and info.supports("some_future_capability")
    assert not info.supports("multilook")
    assert info.raw == EXTENDED
    assert info.to_dict() == EXTENDED

    # Sparse extended entries parse leniently.
    isaac = models[1]
    assert (isaac.name, isaac.capabilities, isaac.reasoning, isaac.description) == ("Isaac 0.1", [], None, None)


def test_extended_parsing_keeps_unknown_keys_in_raw_and_omits_absent_description():
    data = {k: v for k, v in EXTENDED.items() if k != "description"}
    info = ModelInfo.from_dict({**data, "new_field": {"x": 1}, "modalities": ["image", 3, "hologram"]})

    assert info.description is None
    assert "description" not in info.to_dict()
    assert info.raw["new_field"] == {"x": 1}
    assert info.modalities == ["image", "hologram"]


def test_retrieve(monkeypatch):
    http = install(monkeypatch, _handler)
    models = Client().models

    assert models.retrieve("perceptron-mk1.5") == Model.from_dict(MODEL)
    assert str(http.last.url) == "https://api.perceptron.inc/v1/models/perceptron-mk1.5"

    info = models.retrieve("perceptron-mk1.5", extended=True)
    assert str(http.last.url) == "https://api.perceptron.inc/v1/models/perceptron-mk1.5?extended=true"
    assert isinstance(info, ModelInfo)
    assert info.supports("regex")

    models.retrieve("org/model")
    assert http.last.url.raw_path == b"/v1/models/org%2Fmodel"


def test_unknown_model_is_a_not_found_error(monkeypatch):
    install(monkeypatch, _handler)

    with pytest.raises(NotFoundError) as excinfo:
        Client().models.retrieve("nonexistent-model")

    err = excinfo.value
    assert (err.code, err.param, err.status_code, err.request_id) == ("model_not_found", "model", 404, "trace-models")


def test_model_id_is_checked(monkeypatch):
    http = install(monkeypatch, _handler)

    with pytest.raises(BadRequestError) as excinfo:
        Client().models.retrieve("")
    assert (excinfo.value.code, excinfo.value.param) == (INVALID_PARAMETER, "model_id")
    with pytest.raises(TypeError):
        Client().models.retrieve(None)
    assert not http.requests


def test_list_is_lenient_about_the_payload(monkeypatch):
    install(monkeypatch, lambda request: json_response({"object": "list"}))
    assert Client().models.list() == []


def test_api_key_only_env_uses_the_perceptron_api_and_fal_is_rejected(monkeypatch):
    http = install(monkeypatch, _handler)
    assert settings().provider == "fal"

    Client().models.list()
    assert str(http.last.url) == "https://api.perceptron.inc/v1/models"
    assert http.last.headers["authorization"] == "Bearer sk-test"

    with pytest.raises(BadRequestError) as excinfo:
        Client(provider="fal").models.retrieve("perceptron-mk1.5")
    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    assert len(http.requests) == 1


def test_models_are_never_fetched_implicitly(monkeypatch):
    def handler(request):
        assert "/models" not in request.url.path
        if request.url.path.endswith("/multilook"):
            return json_response({"results": [{"prompt_index": 0, "completions": []}]})
        return json_response(completion())

    http = install(monkeypatch, handler)
    client = Client()
    client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], model="some-new-model")
    client.chat.completions.multilook(context=[], prompts=["hi"])

    assert len(http.requests) == 2


def test_resources_are_the_real_classes():
    assert isinstance(Client().models, Models)
    assert isinstance(AsyncClient().models, AsyncModels)


def test_async_parity(monkeypatch):
    http = install(monkeypatch, _handler)
    models = AsyncClient().models

    async def _run():
        listed = await models.list()
        extended = await models.list(extended=True)
        retrieved = await models.retrieve("perceptron-mk1.5", extended=True)
        with pytest.raises(NotFoundError):
            await models.retrieve("nonexistent-model")
        return listed, extended, retrieved

    listed, extended, retrieved = asyncio.run(_run())

    assert [type(m) for m in listed] == [Model, Model]
    assert extended[0] == retrieved == ModelInfo.from_dict(EXTENDED)
    assert [str(r.url) for r in http.requests[:3]] == [
        "https://api.perceptron.inc/v1/models",
        "https://api.perceptron.inc/v1/models?extended=true",
        "https://api.perceptron.inc/v1/models/perceptron-mk1.5?extended=true",
    ]

    async def _on_fal():
        await AsyncClient(provider="fal").models.list()

    with pytest.raises(BadRequestError) as excinfo:
        asyncio.run(_on_fal())
    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
