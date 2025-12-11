"""Tests for constrained decoding response_format feature.

Tests cover:
- json_schema_format() helper
- regex_format() helper
- pydantic_format() helper
- _build_response_format() validation
- Integration with client's _prepare_invocation()
"""

import pytest

from perceptron import client as client_mod
from perceptron import (
    image,
    json_schema_format,
    perceive,
    pydantic_format,
    regex_format,
    text,
)
from perceptron.config import config as cfg

# ---------------------------------------------------------------------------
# json_schema_format() tests
# ---------------------------------------------------------------------------


def test_json_schema_format_basic():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = json_schema_format(schema)

    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "response"
    assert result["json_schema"]["schema"] == schema
    assert "strict" not in result["json_schema"]


def test_json_schema_format_with_name():
    schema = {"type": "object"}
    result = json_schema_format(schema, name="my_custom_schema")

    assert result["json_schema"]["name"] == "my_custom_schema"


def test_json_schema_format_with_strict_true():
    schema = {"type": "object"}
    result = json_schema_format(schema, strict=True)

    assert result["json_schema"]["strict"] is True


def test_json_schema_format_with_strict_false():
    schema = {"type": "object"}
    result = json_schema_format(schema, strict=False)

    assert result["json_schema"]["strict"] is False


def test_json_schema_format_complex_schema():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer", "minimum": 0},
                    },
                    "required": ["name", "count"],
                },
            },
            "total": {"type": "integer"},
        },
        "required": ["items", "total"],
    }
    result = json_schema_format(schema, name="inventory")

    assert result["type"] == "json_schema"
    assert result["json_schema"]["schema"]["properties"]["items"]["type"] == "array"


# ---------------------------------------------------------------------------
# regex_format() tests
# ---------------------------------------------------------------------------


def test_regex_format_basic():
    result = regex_format(r"(yes|no)")

    assert result["type"] == "regex"
    assert result["regex"] == r"(yes|no)"


def test_regex_format_complex_pattern():
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    result = regex_format(email_pattern)

    assert result["type"] == "regex"
    assert result["regex"] == email_pattern


def test_regex_format_empty_pattern():
    result = regex_format("")

    assert result["type"] == "regex"
    assert result["regex"] == ""


# ---------------------------------------------------------------------------
# pydantic_format() tests
# ---------------------------------------------------------------------------


def test_pydantic_format_basic():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    result = pydantic_format(Person)

    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "Person"
    schema = result["json_schema"]["schema"]
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]


def test_pydantic_format_with_custom_name():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str

    result = pydantic_format(Person, name="custom_person")

    assert result["json_schema"]["name"] == "custom_person"


def test_pydantic_format_with_strict():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str

    result = pydantic_format(Person, strict=True)

    assert result["json_schema"]["strict"] is True


def test_pydantic_format_complex_model():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel, Field
    from typing import Optional

    class Address(BaseModel):
        street: str
        city: str
        zip_code: Optional[str] = None

    class Person(BaseModel):
        name: str
        age: int = Field(ge=0, le=150)
        address: Optional[Address] = None

    result = pydantic_format(Person)

    assert result["type"] == "json_schema"
    schema = result["json_schema"]["schema"]
    assert "properties" in schema


def test_pydantic_format_with_literals():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel
    from typing import Literal

    class Mood(BaseModel):
        feeling: Literal["happy", "sad", "neutral"]

    result = pydantic_format(Mood)

    assert result["type"] == "json_schema"


def test_pydantic_format_raises_for_non_pydantic():
    class NotPydantic:
        name: str

    with pytest.raises(TypeError, match="Expected a Pydantic model"):
        pydantic_format(NotPydantic)


def test_pydantic_format_raises_for_dict():
    with pytest.raises(TypeError, match="Expected a Pydantic model"):
        pydantic_format({"type": "object"})


# ---------------------------------------------------------------------------
# _build_response_format() tests
# ---------------------------------------------------------------------------


def test_build_response_format_none():
    result = client_mod._build_response_format(None)
    assert result is None


def test_build_response_format_json_schema():
    fmt = {
        "type": "json_schema",
        "json_schema": {"name": "test", "schema": {"type": "object"}},
    }
    result = client_mod._build_response_format(fmt)

    # Returns tuple: (field_name, value)
    assert isinstance(result, tuple)
    field_name, value = result
    assert field_name == "response_format"
    assert value["type"] == "json_schema"
    assert value["json_schema"]["name"] == "test"


def test_build_response_format_regex():
    fmt = {"type": "regex", "regex": r"\d+"}
    result = client_mod._build_response_format(fmt)

    # Returns tuple: (field_name, value) where regex goes to separate "regex" field
    assert isinstance(result, tuple)
    field_name, value = result
    assert field_name == "regex"
    assert value == r"\d+"


def test_build_response_format_invalid_json_schema():
    fmt = {"type": "json_schema", "json_schema": "not a dict"}

    with pytest.raises(ValueError, match="json_schema response_format requires"):
        client_mod._build_response_format(fmt)


def test_build_response_format_invalid_regex():
    fmt = {"type": "regex", "regex": 123}

    with pytest.raises(ValueError, match="regex response_format requires"):
        client_mod._build_response_format(fmt)


def test_build_response_format_unknown_type_raises():
    fmt = {"type": "custom_type", "custom_field": "value"}

    with pytest.raises(ValueError, match="Unknown response_format type: 'custom_type'"):
        client_mod._build_response_format(fmt)


# ---------------------------------------------------------------------------
# Integration with _prepare_invocation() tests
# ---------------------------------------------------------------------------


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def test_response_format_in_payload(monkeypatch):
    """Test that response_format is included in the request body."""
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"name": "test"}'}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    @perceive(response_format=json_schema_format(schema))
    def make_request():
        return image(PNG_BYTES) + text("Extract data")

    with cfg(provider="fal", base_url="https://mock.api"):
        make_request()

    payload = captured["payload"]
    assert "response_format" in payload
    assert payload["response_format"]["type"] == "json_schema"


def test_response_format_regex_in_payload(monkeypatch):
    """Test that regex is passed as separate 'regex' field in the request body."""
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "yes"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    @perceive(response_format=regex_format(r"(yes|no)"))
    def make_request():
        return image(PNG_BYTES) + text("Yes or no?")

    with cfg(provider="fal", base_url="https://mock.api"):
        make_request()

    payload = captured["payload"]
    # regex goes to separate "regex" field, not inside response_format
    assert "regex" in payload
    assert payload["regex"] == r"(yes|no)"
    assert "response_format" not in payload


def test_no_response_format_when_none(monkeypatch):
    """Test that response_format is not in payload when not specified."""
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "test"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    @perceive()
    def make_request():
        return image(PNG_BYTES) + text("Just text")

    with cfg(provider="fal", base_url="https://mock.api"):
        make_request()

    payload = captured["payload"]
    assert "response_format" not in payload


def test_pydantic_format_in_payload(monkeypatch):
    """Test that pydantic_format works end-to-end."""
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Output(BaseModel):
        value: str

    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"value": "test"}'}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    @perceive(response_format=pydantic_format(Output))
    def make_request():
        return image(PNG_BYTES) + text("Extract")

    with cfg(provider="fal", base_url="https://mock.api"):
        result = make_request()

    payload = captured["payload"]
    assert "response_format" in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "Output"

    # Verify result can be parsed back
    output = Output.model_validate_json(result.text)
    assert output.value == "test"

