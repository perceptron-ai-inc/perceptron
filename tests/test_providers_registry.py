"""Model registry and provider resolution (`perceptron._providers`)."""

from types import SimpleNamespace

import pytest

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
        "isaac-0.1",
        "isaac-0.2-1b",
        "isaac-0.2-2b-preview",
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


def test_fal_rejects_other_ids_without_the_perceptron_hint():
    with pytest.raises(BadRequestError) as excinfo:
        _providers._select_model(_cfg("fal"), "some-other-model")

    assert "PERCEPTRON_PROVIDER" not in str(excinfo.value)


def test_legacy_fal_auto_detect_is_unchanged(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    assert settings().provider == "fal"


def test_legacy_perceptron_default_model_is_mk15(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            captured["body"] = json
            return _Resp()

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Session())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")

    with cfg(provider="perceptron"):
        perceive(text("Hi"))

    assert captured["body"]["model"] == "perceptron-mk1.5"


# ---------------------------------------------------------------------------
# Explicit provider for the new surfaces
# ---------------------------------------------------------------------------


def test_surface_provider_defaults_to_perceptron_despite_fal_auto_detect(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    monkeypatch.setenv("FAL_KEY", "fal-key")
    client = client_mod.Client()

    assert client._settings.provider == "fal"  # the legacy surfaces keep the auto-detect
    assert _providers.explicit_provider() is None
    resolved = _providers.surface_provider_cfg(client)
    assert resolved["name"] == "perceptron"
    assert resolved["base_url"] == "https://api.perceptron.inc/v1"


def test_client_override_is_explicit():
    client = client_mod.Client(provider="fal")
    assert client._provider_override == "fal"
    assert _providers.surface_provider_cfg(client)["name"] == "fal"


def test_configure_is_explicit():
    with cfg(provider="fal"):
        client = client_mod.Client()
        assert _providers.explicit_provider() == "fal"
        assert _providers.surface_provider_cfg(client)["name"] == "fal"
    assert _providers.explicit_provider() is None


def test_client_keeps_the_explicit_provider_it_was_built_with(monkeypatch):
    with cfg(provider="fal"):
        built_inside = client_mod.Client()
    built_outside = client_mod.Client()

    assert _providers.surface_provider_cfg(built_inside)["name"] == "fal"
    with cfg(provider="fal"):
        assert _providers.surface_provider_cfg(built_outside)["name"] == "perceptron"
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
    assert _providers.surface_provider_cfg(built_outside)["name"] == "perceptron"
    with cfg(provider="perceptron"):
        assert _providers.surface_provider_cfg(built_inside)["name"] == "fal"


def test_objects_without_a_recorded_provider_resolve_it_at_call_time(monkeypatch):
    client = SimpleNamespace(_settings=settings())
    assert _providers.surface_provider_cfg(client)["name"] == "perceptron"
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
    assert _providers.surface_provider_cfg(client)["name"] == "fal"


def test_env_provider_is_explicit(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
    assert _providers.surface_provider_cfg(client_mod.Client())["name"] == "fal"


def test_feature_on_non_perceptron_provider_is_rejected():
    with pytest.raises(BadRequestError) as excinfo:
        _providers.surface_provider_cfg(client_mod.Client(provider="fal"), feature="Files")

    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    assert 'configure(provider="perceptron")' in str(excinfo.value)
    assert _providers.surface_provider_cfg(client_mod.Client(), feature="Files")["name"] == "perceptron"


def test_base_url_applies_only_when_configured_for_this_provider(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")
    monkeypatch.setenv("PERCEPTRON_BASE_URL", "https://proxy.example/v1")

    # Auto-detected fal settings: the base URL was meant for fal, not for the Perceptron surfaces.
    assert _providers.surface_provider_cfg(client_mod.Client())["base_url"] == "https://api.perceptron.inc/v1"

    with cfg(provider="perceptron"):
        assert _providers.surface_provider_cfg(client_mod.Client())["base_url"] == "https://proxy.example/v1"

    client = client_mod.Client(provider="fal")
    assert _providers.surface_provider_cfg(client)["base_url"] == "https://proxy.example/v1"


def test_surface_model_prefers_argument_then_settings_then_default():
    perceptron = _cfg("perceptron")
    with cfg(model="perceptron-mk1"):
        settings_obj = settings()
        assert _providers.surface_model(perceptron, None, settings_obj) == "perceptron-mk1"
        assert _providers.surface_model(perceptron, "isaac-0.2-1b", settings_obj) == "isaac-0.2-1b"
    assert _providers.surface_model(perceptron, None, settings()) == "perceptron-mk1.5"
