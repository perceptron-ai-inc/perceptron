"""Model registry and provider resolution (`perceptron._providers`)."""

import pytest
from _http_mock import install, json_response

from perceptron import _providers, perceive, settings, text
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.errors import MODEL_RENAMED, UNSUPPORTED_PROVIDER_FEATURE, BadRequestError

_ENV = ("PERCEPTRON_API_KEY", "FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)


def _cfg(name):
    return _providers._resolve_provider(name)


def test_old_private_names_still_import_from_client():
    assert client_mod._PROVIDER_CONFIG is _providers._PROVIDER_CONFIG
    assert client_mod._select_model is _providers._select_model
    assert client_mod._resolve_provider is _providers._resolve_provider
    assert client_mod.REASONING_EFFORTS == ("none", "minimal", "low", "medium", "high")
    assert client_mod._normalize_reasoning_effort(" High ") == "high"


def test_perceptron_registry_defaults_to_mk15_and_lists_released_models():
    perceptron = _providers._PROVIDER_CONFIG["perceptron"]
    assert perceptron["default_model"] == "perceptron-mk1.5"
    assert perceptron["supported_models"] == [
        "isaac-0.3-fast",
        "perceptron-mk1",
        "perceptron-mk1.5",
    ]
    assert set(perceptron["models"]) == set(perceptron["supported_models"])
    assert _providers._PROVIDER_CONFIG["fal"]["default_model"] == "isaac-0.1"


@pytest.mark.parametrize("model", ["perceptron-mk1.5", "perceptron-mk1", "isaac-0.3-fast"])
def test_perceptron_accepts_released_models(model):
    assert _providers._select_model(_cfg("perceptron"), model) == model


def test_perceptron_passes_unknown_ids_through():
    assert _providers._select_model(_cfg("perceptron"), "perceptron-mk2-experimental") == "perceptron-mk2-experimental"


def test_default_model_when_none_requested():
    assert _providers._select_model(_cfg("perceptron"), None) == "perceptron-mk1.5"
    assert _providers._select_model(_cfg("fal"), None) == "isaac-0.1"


@pytest.mark.parametrize("provider", ["perceptron", "fal"])
def test_preview_id_is_renamed(provider):
    with pytest.raises(BadRequestError) as excinfo:
        _providers._select_model(_cfg(provider), "perceptron-mk1.5-preview")

    assert excinfo.value.code == MODEL_RENAMED
    assert "renamed to perceptron-mk1.5" in str(excinfo.value)


def test_fal_rejects_perceptron_models_with_an_actionable_message():
    with pytest.raises(BadRequestError) as excinfo:
        _providers._select_model(_cfg("fal"), "perceptron-mk1.5")

    message = str(excinfo.value)
    assert "not supported for provider='fal'" in message
    assert 'configure(provider="perceptron")' in message
    assert "PERCEPTRON_PROVIDER=perceptron" in message
    assert "when FAL_KEY is set and neither PERCEPTRON_API_KEY nor a key set in code is" in message  # why fal is in use


# Retired ids (isaac-0.2-*) are no longer Perceptron API models, so fal must not point callers there.
@pytest.mark.parametrize("model", ["some-other-model", "isaac-0.2-1b", "isaac-0.2-2b-preview"])
def test_fal_rejects_other_ids_without_the_perceptron_hint(model):
    with pytest.raises(BadRequestError) as excinfo:
        _providers._select_model(_cfg("fal"), model)

    assert "PERCEPTRON_PROVIDER" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("env", "provider"),
    [
        ({}, "perceptron"),
        ({"PERCEPTRON_API_KEY": "sk-test"}, "perceptron"),
        ({"FAL_KEY": "fal-key"}, "fal"),
        ({"PERCEPTRON_API_KEY": "sk-test", "FAL_KEY": "fal-key"}, "perceptron"),
        ({"PERCEPTRON_API_KEY": "sk-test", "PERCEPTRON_PROVIDER": "fal"}, "fal"),
        ({"FAL_KEY": "fal-key", "PERCEPTRON_PROVIDER": "perceptron"}, "perceptron"),
    ],
    ids=["no-keys", "api-key-only", "fal-key-only", "both-keys", "env-provider-fal", "env-provider-perceptron"],
)
def test_default_provider_rule(monkeypatch, env, provider):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert settings().provider == provider
    assert client_mod.Client()._settings.provider == provider
    assert _providers.surface_provider_cfg(client_mod.Client())["name"] == provider


def test_no_provider_resolves_by_the_rule(monkeypatch):
    assert _providers._resolve_provider(None)["name"] == "perceptron"
    monkeypatch.setenv("FAL_KEY", "fal-key")
    assert _providers._resolve_provider(None)["name"] == "fal"
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    assert _providers._resolve_provider(None)["name"] == "perceptron"


def test_a_key_set_in_code_counts_as_a_perceptron_key(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-key")
    assert settings().provider == "fal"  # FAL_KEY is the only key
    assert client_mod.Client(api_key="sk-test")._settings.provider == "perceptron"
    assert client_mod.AsyncClient(api_key="sk-test")._settings.provider == "perceptron"
    assert client_mod.Client(provider="fal", api_key="fal-key")._settings.provider == "fal"
    with cfg(api_key="sk-test"):
        assert settings().provider == "perceptron"
        assert client_mod.Client()._settings.provider == "perceptron"
        assert _providers._resolve_provider(None)["name"] == "perceptron"
    with cfg(provider="fal", api_key="fal-key"):
        assert settings().provider == "fal"
    with cfg(api_key=""):  # an empty key is no key
        assert settings().provider == "fal"


def test_configured_provider_wins_over_the_rule(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-key")
    with cfg(provider="perceptron"):
        assert settings().provider == "perceptron"
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    with cfg(provider="fal"):
        assert settings().provider == "fal"
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "perceptron")
    with cfg(provider="fal"):
        assert settings().provider == "fal"  # configure() wins over PERCEPTRON_PROVIDER


def test_legacy_perceptron_default_model_is_mk15(monkeypatch):
    http = install(monkeypatch, lambda request: json_response({"choices": [{"message": {"content": "ok"}}]}))
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")

    with cfg(provider="perceptron"):
        assert perceive(text("Hi")).text == "ok"

    assert [str(request.url) for request in http.requests] == ["https://api.perceptron.inc/v1/chat/completions"]
    assert http.last_body["model"] == "perceptron-mk1.5"


# ---------------------------------------------------------------------------
# One provider per client, on every surface
# ---------------------------------------------------------------------------


def test_client_override_wins():
    client = client_mod.Client(provider="fal")
    assert client._settings.provider == "fal"
    assert _providers.surface_provider_cfg(client)["name"] == "fal"


def test_configure_is_used_by_every_surface():
    with cfg(provider="fal"):
        client = client_mod.Client()
        assert _providers.surface_provider_cfg(client)["name"] == "fal"
    assert _providers.surface_provider_cfg(client_mod.Client())["name"] == "perceptron"


def test_client_keeps_the_provider_it_was_built_with(monkeypatch):
    with cfg(provider="fal"):
        built_inside = client_mod.Client()
    built_outside = client_mod.Client()

    assert _providers.surface_provider_cfg(built_inside)["name"] == "fal"
    with cfg(provider="fal"):
        assert _providers.surface_provider_cfg(built_outside)["name"] == "perceptron"
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
    monkeypatch.setenv("FAL_KEY", "fal-key")
    assert _providers.surface_provider_cfg(built_outside)["name"] == "perceptron"
    with cfg(provider="perceptron"):
        assert _providers.surface_provider_cfg(built_inside)["name"] == "fal"


def test_env_provider_is_used(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
    assert _providers.surface_provider_cfg(client_mod.Client())["name"] == "fal"


def test_feature_on_non_perceptron_provider_is_rejected(monkeypatch):
    with pytest.raises(BadRequestError) as excinfo:
        _providers.surface_provider_cfg(client_mod.Client(provider="fal"), feature="Files")

    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    assert 'configure(provider="perceptron")' in str(excinfo.value)
    assert "PERCEPTRON_PROVIDER=perceptron" in str(excinfo.value)
    assert _providers.surface_provider_cfg(client_mod.Client(), feature="Files")["name"] == "perceptron"

    monkeypatch.setenv("FAL_KEY", "fal-key")  # fal auto-selected
    with pytest.raises(BadRequestError) as excinfo:
        _providers.surface_provider_cfg(client_mod.Client(), feature="Files")
    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE


def test_a_configured_base_url_always_applies(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    assert _providers.surface_provider_cfg(client_mod.Client())["base_url"] == "https://api.perceptron.inc/v1"

    # The request and its key go where the caller pointed them.
    monkeypatch.setenv("PERCEPTRON_BASE_URL", "https://proxy.example/v1")
    resolved = _providers.surface_provider_cfg(client_mod.Client())
    assert (resolved["name"], resolved["base_url"]) == ("perceptron", "https://proxy.example/v1")

    with cfg(provider="perceptron"):
        assert _providers.surface_provider_cfg(client_mod.Client())["base_url"] == "https://proxy.example/v1"

    resolved = _providers.surface_provider_cfg(client_mod.Client(provider="fal"))
    assert (resolved["name"], resolved["base_url"]) == ("fal", "https://proxy.example/v1")

    client = client_mod.Client(base_url="http://localhost:8080/v1")
    assert _providers.surface_provider_cfg(client)["base_url"] == "http://localhost:8080/v1"


def test_surface_model_prefers_argument_then_settings_then_default():
    perceptron = _cfg("perceptron")
    with cfg(model="perceptron-mk1"):
        settings_obj = settings()
        assert _providers.surface_model(perceptron, None, settings_obj) == "perceptron-mk1"
        assert _providers.surface_model(perceptron, "isaac-0.3-fast", settings_obj) == "isaac-0.3-fast"
    assert _providers.surface_model(perceptron, None, settings()) == "perceptron-mk1.5"
