import json

import httpx
from _http_mock import Body

from perceptron import client as client_mod
from perceptron.errors import AuthError, BadRequestError, RateLimitError, ServerError


def _response(status_code, payload, *, text=None, headers=None):
    """A read ``httpx.Response`` whose body is ``text``, else ``payload`` as JSON."""
    content = text if text is not None else json.dumps(payload)
    return httpx.Response(status_code, content=content.encode(), headers=headers or {})


# ========== Status Code Coverage ==========


def test_map_http_error_401_nested_error():
    payload = {
        "error": {
            "message": "Incorrect API key provided",
            "code": "invalid_api_key",
            "type": "invalid_request_error",
        }
    }
    resp = _response(401, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, AuthError)
    assert "Incorrect API key provided" in str(err)
    assert err.code == "invalid_api_key"
    assert err.details == payload["error"]


def test_map_http_error_403_forbidden():
    payload = {"error": {"message": "Access denied", "code": "forbidden"}}
    resp = _response(403, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, AuthError)
    assert "Access denied" in str(err)
    assert err.code == "forbidden"


def test_map_http_error_403_with_fallback():
    resp = _response(403, {}, text="Forbidden")

    err = client_mod._map_http_error(resp)

    assert isinstance(err, AuthError)
    assert str(err) == "Forbidden"
    assert err.code == "auth_error"


def test_map_http_error_404_not_found():
    payload = {"error": {"message": "Resource not found", "code": "not_found"}}
    resp = _response(404, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert "Resource not found" in str(err)
    assert err.code == "not_found"


def test_map_http_error_404_with_default_message():
    resp = _response(404, {})

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    # Empty dict serializes to '{}' as fallback text
    assert str(err) == "{}"


def test_map_http_error_429_rate_limit_with_retry_after():
    payload = {"error": {"message": "Rate limit exceeded", "code": "rate_limited"}}
    resp = _response(429, payload, headers={"Retry-After": "60"})

    err = client_mod._map_http_error(resp)

    assert isinstance(err, RateLimitError)
    assert "Rate limit exceeded" in str(err)
    assert err.details.get("retry_after") == 60.0


def test_map_http_error_429_without_retry_after():
    payload = {"message": "Too many requests"}
    resp = _response(429, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, RateLimitError)
    assert "Too many requests" in str(err)


def test_map_http_error_429_invalid_retry_after_header():
    payload = {"error": {"message": "Rate limited"}}
    resp = _response(429, payload, headers={"Retry-After": "invalid"})

    err = client_mod._map_http_error(resp)

    assert isinstance(err, RateLimitError)
    assert err.details.get("retry_after") is None


def test_map_http_error_422_other_4xx():
    payload = {"error": {"message": "Validation failed", "code": "validation_error"}}
    resp = _response(422, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert "Validation failed" in str(err)
    assert err.code == "validation_error"


def test_map_http_error_400_bad_request():
    resp = _response(400, payload=None, text="malformed request")

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert str(err) == "malformed request"


def test_map_http_error_500_server_error():
    payload = {"error": {"message": "Internal server error", "code": "server_error"}}
    resp = _response(500, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, ServerError)
    assert "Internal server error" in str(err)
    assert err.code == "server_error"


def test_map_http_error_503_service_unavailable():
    resp = _response(503, {}, text="Service Unavailable")

    err = client_mod._map_http_error(resp)

    assert isinstance(err, ServerError)
    assert "Service Unavailable" in str(err)


def test_map_http_error_500_with_default_message():
    resp = _response(502, {})

    err = client_mod._map_http_error(resp)

    assert isinstance(err, ServerError)
    # Empty dict serializes to '{}' as fallback text
    assert str(err) == "{}"


# ========== Payload Structure Coverage ==========


def test_extract_error_list_payload():
    payload = [
        {"message": "First error", "code": "error_1"},
        {"message": "Second error", "code": "error_2"},
    ]
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert "First error" in str(err)
    assert err.code == "error_1"


def test_extract_error_list_with_string():
    payload = ["  ", "", "First non-empty error"]
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert str(err) == "First non-empty error"


def test_extract_error_flat_dict_no_nested_error():
    payload = {"message": "Direct message", "code": "direct_code"}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert "Direct message" in str(err)
    assert err.code == "direct_code"


def test_extract_error_detail_field():
    payload = {"detail": "Detailed error message", "code": "detail_code"}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert "Detailed error message" in str(err)
    assert err.code == "detail_code"


def test_extract_error_string_in_error_field():
    payload = {"error": "Simple error string"}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert "Simple error string" in str(err)


def test_extract_error_nested_error_string():
    payload = {"error": "Error occurred"}
    resp = _response(500, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, ServerError)
    assert "Error occurred" in str(err)


def test_extract_error_string_payload():
    resp = _response(400, "Plain error string")

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    # String data is extracted directly
    assert str(err) == "Plain error string"


def test_extract_error_empty_payload():
    resp = _response(400, {})

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    # Empty dict serializes to '{}' as fallback text
    assert str(err) == "{}"


def test_extract_error_null_payload():
    resp = _response(500, None)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, ServerError)
    # A JSON null body has no error fields, so the body text 'null' is the message
    assert str(err) == "null"


# ========== Code Extraction Variants ==========


def test_extract_code_from_type_field():
    payload = {"error": {"message": "Invalid request", "type": "invalid_request_error"}}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    assert err.code == "invalid_request_error"


def test_extract_code_prefers_code_over_type():
    payload = {"error": {"message": "Error", "code": "specific_code", "type": "general_type"}}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert err.code == "specific_code"


def test_extract_code_missing():
    payload = {"error": {"message": "Error without code"}}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert err.code is None


def test_extract_code_from_flat_structure():
    payload = {"message": "Error", "code": "flat_code"}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert err.code == "flat_code"


# ========== Details Extraction ==========


def test_details_includes_nested_error_dict():
    payload = {"error": {"message": "Error", "code": "test", "extra": "metadata"}}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert err.details == payload["error"]
    assert err.details.get("extra") == "metadata"


def test_details_includes_full_dict_when_no_nesting():
    payload = {"message": "Error", "code": "test", "request_id": "12345"}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert err.details == payload
    assert err.details.get("request_id") == "12345"


def test_details_none_when_list_payload():
    payload = [{"message": "Error"}]
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    # Details should be the first dict item
    assert err.details == {"message": "Error"}


# ========== Edge Cases ==========


def test_first_nonempty_skips_whitespace():
    payload = {"error": {"message": "  \n  ", "detail": "Actual message"}}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert str(err) == "Actual message"


def test_response_text_exception_handled():
    # An unread body: both `json()` and `text` raise `httpx.ResponseNotRead`.
    resp = httpx.Response(500, stream=Body(b"bad json"))
    err = client_mod._map_http_error(resp)

    assert isinstance(err, ServerError)
    assert "server error: 500" in str(err)


def test_message_precedence_message_over_detail():
    payload = {"error": {"message": "Primary", "detail": "Secondary"}}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert str(err) == "Primary"


def test_message_precedence_detail_over_error_string():
    payload = {"error": {"detail": "Detail message", "error": "Error string"}}
    resp = _response(400, payload)

    err = client_mod._map_http_error(resp)

    assert str(err) == "Detail message"


def test_empty_list_payload():
    resp = _response(400, [])

    err = client_mod._map_http_error(resp)

    assert isinstance(err, BadRequestError)
    # Empty list serializes to '[]' as fallback text
    assert str(err) == "[]"


def test_retry_after_zero_is_valid():
    payload = {"message": "Rate limited"}
    resp = _response(429, payload, headers={"Retry-After": "0"})

    err = client_mod._map_http_error(resp)

    assert isinstance(err, RateLimitError)
    assert err.details.get("retry_after") == 0.0
