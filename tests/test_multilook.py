"""`client.chat.completions.multilook()`: request bodies, validation, partial failures, usage, error mapping, the
timeout floor, per-prompt asset counts, the provider rule, and async parity."""

import asyncio
import json
import warnings
from types import SimpleNamespace

import httpx
import pytest
from _http_mock import completion, install, json_response
from _image_fixtures import PNG_BYTES

from perceptron import AsyncClient, Client, image, settings, text
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.chat import ChatCompletionMessage, FunctionCall, ToolCall, Usage
from perceptron.dsl.nodes import box, point, polygon
from perceptron.errors import (
    ANCHOR_MISSING,
    ANCHOR_UNKNOWN,
    BOUNDS_OUT_OF_RANGE,
    INVALID_PARAMETER,
    INVALID_POLYGON,
    INVALID_REASONING_EFFORT,
    INVALID_RESPONSE,
    INVALID_TEMPERATURE,
    MODEL_RENAMED,
    UNSUPPORTED_PARAMETER,
    UNSUPPORTED_PROVIDER_FEATURE,
    BadRequestError,
    QuotaExceededError,
    ServerError,
)
from perceptron.multilook import (
    MultilookCompletion,
    MultilookPromptError,
    MultilookResponse,
    MultilookResult,
    MultilookUsage,
)
from perceptron.pointing.types import pt

URL = "https://api.perceptron.inc/v1/chat/completions/multilook"
VIDEO_PART = {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
IMAGE_PART = {"type": "image_url", "image_url": {"url": "https://example.com/ref.jpg"}}
CONTEXT = [
    {"role": "system", "content": "You are a precise video analyst."},
    {"role": "user", "content": [VIDEO_PART]},
]
PARTIAL_FAILURE = {
    "id": "mlcmpl-3f9c0d2b7a4e4c1b9e8f6a5d4c3b2a10",
    "object": "chat.completion.multilook",
    "model": "perceptron-mk1.5",
    "results": [
        {
            "prompt_index": 0,
            "completions": [
                {"index": 0, "message": {"role": "assistant", "content": "Yes, at 00:04."}, "finish_reason": "stop"},
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Yes.", "reasoning_content": "A person walks in."},
                    "finish_reason": "interrupted",
                },
            ],
            "usage": {"completion_tokens": 7},
        },
        {
            "prompt_index": 1,
            "error": {
                "message": "Could not fetch image at https://example.com/ref.jpg (HTTP 404).",
                "type": "invalid_request_error",
            },
        },
    ],
    "usage": {
        "prompt_tokens": 17120,
        "completion_tokens": 7,
        "total_tokens": 17127,
        "prompt_tokens_details": {"cached_tokens": 8440, "audio_tokens": 12},
    },
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _ok(prompt_count: int, n: int = 1) -> dict:
    return {
        "id": "mlcmpl-1",
        "object": "chat.completion.multilook",
        "model": "perceptron-mk1.5",
        "results": [
            {
                "prompt_index": i,
                "completions": [
                    {
                        "index": j,
                        "message": {"role": "assistant", "content": f"answer {i}.{j}"},
                        "finish_reason": "stop",
                    }
                    for j in range(n)
                ],
                "usage": {"completion_tokens": n},
            }
            for i in range(prompt_count)
        ],
    }


@pytest.fixture
def http(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        if "prompts" not in body:
            return json_response(completion())
        return json_response(_ok(len(body["prompts"]), body.get("n", 1)), headers={"x-trace-id": "trace-ml"})

    return install(monkeypatch, handler)


def _multilook(**kwargs):
    kwargs.setdefault("context", CONTEXT)
    kwargs.setdefault("prompts", ["Did a person enter the frame?"])
    return Client().chat.completions.multilook(**kwargs)


def _record_timeouts(monkeypatch) -> list:
    """Wrap both session factories (after `install`) to record the timeout each session gets."""
    seen: list = []
    sync_factory = client_mod._http_client
    async_httpx = client_mod.httpx

    def _sync(timeout):
        seen.append(timeout)
        return sync_factory(timeout)

    def _async(timeout):
        seen.append(timeout)
        return async_httpx.AsyncClient(timeout=timeout)

    monkeypatch.setattr(client_mod, "_http_client", _sync)
    monkeypatch.setattr(
        client_mod,
        "httpx",
        SimpleNamespace(
            AsyncClient=_async, TimeoutException=async_httpx.TimeoutException, HTTPError=async_httpx.HTTPError
        ),
    )
    return seen


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


def test_request_body(http):
    prompts = ["Did a person enter the frame?", {"content": [IMAGE_PART, {"type": "text", "text": "Same frame?"}]}]

    response = _multilook(
        prompts=prompts,
        n=3,
        temperature=0.7,
        max_completion_tokens=256,
        reasoning_effort="LOW",
        vision_config={"annotation_format": "clip"},
    )

    assert http.last.method == "POST"
    assert str(http.last.url) == URL
    assert http.last.headers["authorization"] == "Bearer sk-test"
    body = http.last_body
    assert body == {
        "model": "perceptron-mk1.5",
        "context": CONTEXT,
        "prompts": prompts,
        "n": 3,
        "max_completion_tokens": 256,
        "temperature": 0.7,
        "reasoning_effort": "low",
        "vision_config": {"annotation_format": "clip"},
    }
    assert list(body)[:3] == ["model", "context", "prompts"]
    assert isinstance(response, MultilookResponse)
    assert response.request_id == "trace-ml"


def test_only_what_was_set_is_sent(http):
    _multilook(context=[])
    assert http.last_body == {"model": "perceptron-mk1.5", "context": [], "prompts": ["Did a person enter the frame?"]}

    _multilook(model="perceptron-mk1", top_p=0.9, top_k=20, frequency_penalty=0.0, presence_penalty=1.5, n=1)
    body = http.last_body
    assert (body["model"], body["top_p"], body["top_k"], body["frequency_penalty"], body["presence_penalty"]) == (
        "perceptron-mk1",
        0.9,
        20,
        0.0,
        1.5,
    )
    assert body["n"] == 1


def test_configure_generation_defaults_count_as_set(http):
    with cfg(model="perceptron-mk1", temperature=0.4, max_tokens=64, top_k=5):
        _multilook(n=2)  # the configured temperature satisfies n > 1

    body = http.last_body
    assert (body["model"], body["temperature"], body["max_completion_tokens"], body["top_k"], body["n"]) == (
        "perceptron-mk1",
        0.4,
        64,
        5,
        2,
    )


def test_enable_thinking_is_deprecated_but_sent_and_the_warning_points_at_the_caller(http):
    async def _async():
        await AsyncClient().chat.completions.multilook(
            context=[], prompts=["q"], vision_config={"enable_thinking": False}
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _multilook(vision_config={"enable_thinking": True})
        asyncio.run(_async())

    assert [w.category for w in caught] == [DeprecationWarning, DeprecationWarning]
    assert [w.filename for w in caught] == [__file__, __file__]
    sent = [json.loads(request.content)["vision_config"] for request in http.requests]
    assert sent == [{"enable_thinking": True}, {"enable_thinking": False}]


def test_max_tokens_alias_and_extra_body(http):
    _multilook(max_tokens=100, extra_body={"temperature": 0.1, "custom": {"x": 1}}, temperature=0.5)
    body = http.last_body
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert (body["temperature"], body["custom"]) == (0.1, {"x": 1})  # merged last, unvalidated

    with pytest.raises(TypeError, match="not both"):
        _multilook(max_tokens=1, max_completion_tokens=1)
    with pytest.raises(TypeError, match="extra_body"):
        _multilook(extra_body=[("a", 1)])


def test_prompt_forms_and_dsl_nodes_are_lowered(http):
    context = [
        ChatCompletionMessage(role="assistant", content="Earlier answer."),
        {"role": "user", "content": [text("Context text"), image("https://example.com/a.png")]},
    ]
    prompts = [
        "plain",
        [image(PNG_BYTES), "Is this the part?"],
        image("https://example.com/b.png") + text("Seen before?"),
        {"content": [text("structured")]},
        {"content": [IMAGE_PART]},
    ]

    _multilook(context=context, prompts=prompts)

    body = http.last_body
    assert body["context"] == [
        {"role": "assistant", "content": "Earlier answer."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Context text"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
        },
    ]
    sent = body["prompts"]
    assert sent[0] == "plain"
    assert [p["type"] for p in sent[1]["content"]] == ["image_url", "text"]
    assert sent[1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent[1]["content"][1] == {"type": "text", "text": "Is this the part?"}
    assert sent[2] == {
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
            {"type": "text", "text": "Seen before?"},
        ]
    }
    assert sent[3] == {"content": [{"type": "text", "text": "structured"}]}
    assert sent[4] == {"content": [IMAGE_PART]}


# ---------------------------------------------------------------------------
# Validation before any request
# ---------------------------------------------------------------------------

_TOOL_CALL = ToolCall(id="call_1", function=FunctionCall(name="lookup", arguments="{}"))


@pytest.mark.parametrize(
    ("kwargs", "code", "param"),
    [
        ({"prompts": []}, INVALID_PARAMETER, "prompts"),
        ({"prompts": ["q"] * 17}, INVALID_PARAMETER, "prompts"),
        ({"n": 0}, INVALID_PARAMETER, "n"),
        ({"n": 9, "temperature": 1.0}, INVALID_PARAMETER, "n"),
        ({"n": True}, INVALID_PARAMETER, "n"),
        ({"n": 2.0, "temperature": 1.0}, INVALID_PARAMETER, "n"),
        ({"prompts": ["q"] * 9, "n": 8, "temperature": 1.0}, INVALID_PARAMETER, "n"),
        ({"n": 2}, INVALID_PARAMETER, "temperature"),
        ({"n": 2, "temperature": 0}, INVALID_PARAMETER, "temperature"),
        # An invalid temperature gets the API's own code, whatever n is.
        ({"n": 2, "temperature": "0.5"}, INVALID_TEMPERATURE, "temperature"),
        ({"n": 2, "temperature": True}, INVALID_TEMPERATURE, "temperature"),
        ({"n": 2, "temperature": float("nan")}, INVALID_TEMPERATURE, "temperature"),
        ({"n": 2, "temperature": -1}, INVALID_TEMPERATURE, "temperature"),
        ({"temperature": float("inf")}, INVALID_TEMPERATURE, "temperature"),
        ({"temperature": -0.5}, INVALID_TEMPERATURE, "temperature"),
        ({"max_completion_tokens": 0}, INVALID_PARAMETER, "max_completion_tokens"),
        (
            {"context": [*CONTEXT, {"role": "tool", "tool_call_id": "c", "content": "x"}]},
            UNSUPPORTED_PARAMETER,
            "context[2]",
        ),
        (
            {"context": [{"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL.to_dict()]}]},
            UNSUPPORTED_PARAMETER,
            "context[0]",
        ),
        (
            {"context": [ChatCompletionMessage(role="assistant", content=None, tool_calls=[_TOOL_CALL])]},
            UNSUPPORTED_PARAMETER,
            "context[0]",
        ),
        ({"reasoning_effort": "extreme"}, INVALID_REASONING_EFFORT, None),
        ({"vision_config": {"internal_tools": ["focus"]}}, UNSUPPORTED_PARAMETER, "vision_config.internal_tools"),
        ({"model": "perceptron-mk1.5-preview"}, MODEL_RENAMED, None),
    ],
)
def test_invalid_values_raise_before_sending(http, kwargs, code, param):
    with pytest.raises(BadRequestError) as excinfo:
        _multilook(**kwargs)

    assert (excinfo.value.code, excinfo.value.param) == (code, param)
    assert not http.requests


@pytest.mark.parametrize("temperature", ["warm", -0.5])
def test_configured_temperature_is_checked_like_an_argument(http, temperature):
    with cfg(temperature=temperature), pytest.raises(BadRequestError) as excinfo:
        _multilook()
    assert (excinfo.value.code, excinfo.value.param) == (INVALID_TEMPERATURE, "temperature")
    assert not http.requests


def test_limits_are_inclusive(http):
    _multilook(prompts=["q"] * 16, n=4, temperature=0.5)  # 64 completions
    assert len(http.last_body["prompts"]) == 16
    _multilook(context=[{"role": "assistant", "content": "ok", "tool_calls": []}])  # no calls is not tool traffic


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prompts": "one prompt"}, "prompts must be a list"),
        ({"prompts": [3]}, r"prompts\[0\] must be a str"),
        ({"prompts": ["ok", {"role": "user", "content": [IMAGE_PART]}]}, r"prompts\[1\] takes only a 'content' key"),
        ({"prompts": [{"content": "text"}]}, r"prompts\[0\]\['content'\] must be a list"),
        ({"prompts": [["ok", 3.5]]}, r"prompts\[0\]\.content\[1\]"),
        ({"context": "hello"}, "context must be a list"),
        ({"context": [3]}, r"context\[0\] must be a dict"),
        ({"context": [{"role": "user", "content": ["ok", 3.5]}]}, r"context\[0\]\.content\[1\]"),
    ],
)
def test_malformed_arguments_raise_type_errors(http, kwargs, match):
    with pytest.raises(TypeError, match=match):
        _multilook(**kwargs)
    assert not http.requests


@pytest.mark.parametrize("name", ["stream", "tools", "tool_choice", "response_format", "regex", "stop", "seed"])
def test_unsupported_parameters_are_not_accepted(http, name):
    with pytest.raises(TypeError, match=name):
        _multilook(**{name: True})
    assert not http.requests


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def test_partial_failure_response(monkeypatch):
    install(monkeypatch, lambda request: json_response(PARTIAL_FAILURE, headers={"x-trace-id": "trace-ml"}))

    response = _multilook(prompts=["Did a person enter?", {"content": [IMAGE_PART]}], n=2, temperature=0.7)

    assert (response.id, response.object, response.model) == (
        PARTIAL_FAILURE["id"],
        "chat.completion.multilook",
        "perceptron-mk1.5",
    )
    assert response.request_id == "trace-ml"
    assert response.raw == PARTIAL_FAILURE
    first, second = response.results
    assert first.ok and not second.ok
    assert response.succeeded == [first]
    assert response.failed == [second]

    assert [c.text for c in first.completions] == ["Yes, at 00:04.", "Yes."]
    assert first.completions[1].reasoning == "A person walks in."
    assert isinstance(first.completions[0].message, ChatCompletionMessage)
    assert [c.complete for c in first.completions] == [True, True]  # `interrupted` is a finished answer
    assert first.usage == Usage(completion_tokens=7)
    assert first.error is None

    assert second.completions is None and second.usage is None
    assert second.error == MultilookPromptError(
        message="Could not fetch image at https://example.com/ref.jpg (HTTP 404).", type="invalid_request_error"
    )

    usage = response.usage
    assert isinstance(usage, MultilookUsage) and isinstance(usage, Usage)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (17120, 7, 17127)
    assert (usage.cached_tokens, usage.audio_tokens) == (8440, 12)
    assert response.to_dict() == PARTIAL_FAILURE


def test_parsing_is_lenient():
    response = MultilookResponse.from_dict({"results": [{"completions": [{"message": None}]}, "junk"]})

    assert response.usage is None
    first, junk = response.results
    assert (first.prompt_index, junk.prompt_index) == (0, 1)
    completion = first.completions[0]
    assert (completion.index, completion.text, completion.finish_reason, completion.complete) == (0, None, None, False)
    assert not junk.ok
    assert MultilookUsage.from_dict({"prompt_tokens": 1}).cached_tokens is None


@pytest.mark.parametrize(("finish_reason", "complete"), [("stop", True), ("interrupted", True), ("length", False)])
def test_completion_complete(finish_reason, complete):
    message = ChatCompletionMessage(role="assistant", content="x")
    assert MultilookCompletion(index=0, message=message, finish_reason=finish_reason).complete is complete


def test_result_ok_requires_completions_and_no_error():
    assert MultilookResult(prompt_index=0, completions=[]).ok
    assert not MultilookResult(prompt_index=0).ok
    assert not MultilookResult(prompt_index=0, completions=[], error=MultilookPromptError(message="x")).ok


@pytest.mark.parametrize(
    ("status", "error", "headers", "expected"),
    [
        (400, {"message": "Could not fetch image.", "type": "invalid_request_error"}, {}, BadRequestError),
        (502, {"message": "Generation failed.", "type": "server_error", "code": "internal_error"}, {}, ServerError),
        (429, {"message": "You exceeded your current quota.", "type": "insufficient_quota"}, {}, QuotaExceededError),
        (
            503,
            {"message": "Overloaded.", "type": "server_error", "code": "model_overloaded"},
            {"Retry-After": "30"},
            ServerError,
        ),
    ],
)
def test_all_failed_requests_map_like_other_http_errors(monkeypatch, status, error, headers, expected):
    headers = {"x-trace-id": "trace-ml", **headers}
    install(monkeypatch, lambda request: json_response({"error": error}, status=status, headers=headers))

    with pytest.raises(expected) as excinfo:
        _multilook()

    err = excinfo.value
    assert type(err) is expected
    assert (str(err), err.status_code, err.request_id) == (error["message"], status, "trace-ml")
    if status == 503:
        assert (err.code, err.retry_after) == ("model_overloaded", 30.0)


@pytest.mark.parametrize("payload", [{"id": "mlcmpl-1"}, {"results": []}, ["not", "a", "dict"]])
def test_a_response_without_results_is_a_server_error(monkeypatch, payload):
    install(monkeypatch, lambda request: json_response(payload))

    with pytest.raises(ServerError) as excinfo:
        _multilook()
    assert excinfo.value.code == INVALID_RESPONSE


# ---------------------------------------------------------------------------
# Timeout floor, asset counts, provider rule
# ---------------------------------------------------------------------------


def test_timeout_defaults_to_at_least_305_seconds(http, monkeypatch):
    seen = _record_timeouts(monkeypatch)

    _multilook()
    with cfg(timeout=400.0):
        _multilook()
    _multilook(timeout=10.0)
    Client(timeout=500.0).chat.completions.multilook(context=[], prompts=["q"])
    Client().chat.completions.create(messages=[{"role": "user", "content": "hi"}])  # chat keeps the plain default

    assert seen == [305.0, 400.0, 10.0, 500.0, settings().timeout]


def test_each_completion_counts_the_context_and_its_prompts_assets(http):
    context = [
        {"role": "system", "content": "Compare against the references."},
        {"role": "user", "content": [VIDEO_PART, image("https://example.com/ref.png"), "Reference set."]},
    ]
    prompts = [
        "Text only.",
        [image(PNG_BYTES), "Is this in the video?"],
        {"content": [IMAGE_PART, {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}]},
    ]

    response = _multilook(context=context, prompts=prompts, n=2, temperature=0.5)

    counts = [[c.asset_count for c in result.completions] for result in response.results]
    assert counts == [[2, 2], [3, 3], [4, 4]]


def test_prompt_tags_anchor_after_the_context_media(http):
    shelf = image("https://example.com/shelf.png")
    context = [
        {"role": "system", "content": "Compare against the shelf."},
        {"role": "user", "content": [VIDEO_PART, shelf, "The shelf."]},
    ]
    crop = image(PNG_BYTES)
    prompts = [
        ["Is this cup on the shelf?", box(1, 2, 3, 4, image=shelf, mention="cup")],
        [crop, "Same cup?", point(5, 6, image=crop), box(1, 2, 3, 4, asset=shelf)],
        {"content": ["Anything else?", point(7, 8, asset_idx=0)]},
    ]

    response = _multilook(context=context, prompts=prompts)

    texts = [
        [part["text"] for part in prompt["content"] if part["type"] == "text"] for prompt in http.last_body["prompts"]
    ]
    # asset_idx counts the context's media (the video, then the shelf) before each prompt's own.
    assert texts == [
        ["Is this cup on the shelf?", '<point_box mention="cup" asset_idx="1"> (1,2) (3,4) </point_box>'],
        ["Same cup?", '<point asset_idx="2"> (5,6) </point>', '<point_box asset_idx="1"> (1,2) (3,4) </point_box>'],
        ["Anything else?", '<point asset_idx="0"> (7,8) </point>'],
    ]
    assert [result.completions[0].asset_count for result in response.results] == [2, 3, 2]


def test_single_asset_prompts_count_the_context_media(http):
    context = [{"role": "user", "content": [IMAGE_PART, "A shelf."]}]

    _multilook(context=context, prompts=[["Is this the cup?", point(5, 6)]])

    assert http.last_body["prompts"][0]["content"][1] == {"type": "text", "text": "<point> (5,6) </point>"}


@pytest.mark.parametrize(
    ("prompt", "code"),
    [
        (lambda: [image(PNG_BYTES), point(1, 2)], ANCHOR_MISSING),  # the context's image makes two assets
        (lambda: ["Here?", point(1, 2, image=image("https://example.com/elsewhere.png"))], ANCHOR_UNKNOWN),
        (lambda: ["Here?", point(1, 2, asset_idx=1)], ANCHOR_UNKNOWN),
        (lambda: ["Here?", point(5000, 2)], BOUNDS_OUT_OF_RANGE),
        (lambda: ["Here?", polygon([(1, 1), (2, 2)])], INVALID_POLYGON),
    ],
)
def test_prompt_tag_issues_raise_naming_the_prompt_item(http, prompt, code):
    context = [{"role": "user", "content": [IMAGE_PART]}]

    with pytest.raises(BadRequestError) as excinfo:
        _multilook(context=context, prompts=["ok", prompt()])

    assert (excinfo.value.code, excinfo.value.param) == (code, "prompts[1].content[1]")
    assert not http.requests


def _replayed(cup):
    """A context replayed with the same image node in two turns, and a prompt tag anchored to it too."""
    context = [
        {"role": "user", "content": [cup, "The cup is here:", point(1, 2, image=cup)]},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": [cup, "Remember it."]},
    ]
    return {"context": context, "prompts": [["Same cup?", point(5, 6, image=cup)]]}


_AMBIGUOUS = "image=/asset= references a media node used 2 times in this prompt; anchored to asset_idx {}"
# Each tag names the latest use of the node before it: the context's first image, then its second.
_REUSED_NODE_WARNINGS = [
    (UserWarning, f"context[0].content[2]: {_AMBIGUOUS.format(0)}", __file__),
    (UserWarning, f"prompts[0].content[1]: {_AMBIGUOUS.format(1)}", __file__),
]


def _reused_node_warnings(caught) -> list:
    return [(w.category, str(w.message), w.filename) for w in caught if "used 2 times" in str(w.message)]


def test_a_reused_media_node_warns_and_the_request_is_sent(http):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _multilook(**_replayed(image("https://example.com/cup.png")))

    assert _reused_node_warnings(caught) == _REUSED_NODE_WARNINGS
    body = http.last_body
    assert _context_texts(body) == ["The cup is here:", '<point asset_idx="0"> (1,2) </point>']
    assert body["prompts"][0]["content"][1] == {"type": "text", "text": '<point asset_idx="1"> (5,6) </point>'}


def test_async_a_reused_media_node_warns_and_the_request_is_sent(http):
    async def _run():
        await AsyncClient().chat.completions.multilook(**_replayed(image("https://example.com/cup.png")))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(_run())

    assert _reused_node_warnings(caught) == _REUSED_NODE_WARNINGS
    assert http.last_body["prompts"][0]["content"][1]["text"] == '<point asset_idx="1"> (5,6) </point>'


def _context_texts(body: dict) -> list[str]:
    return [part["text"] for part in body["context"][0]["content"] if part["type"] == "text"]


def test_context_tags_anchor_in_the_widest_prompts_asset_space(http):
    # The context is shared: a prompt that adds an image makes two assets, so a context tag names its asset even
    # though the text-only prompt sees one (the same bytes go with every prompt).
    cup = image("https://example.com/cup.png")
    context = [{"role": "user", "content": [cup, "The cup is here:", point(1, 2, image=cup)]}]

    response = _multilook(context=context, prompts=[[image(PNG_BYTES), "Is it the same cup?"], "Describe it."])

    assert _context_texts(http.last_body) == ["The cup is here:", '<point asset_idx="0"> (1,2) </point>']
    assert [result.completions[0].asset_count for result in response.results] == [2, 1]


def test_untagged_context_tags_are_ambiguous_once_a_prompt_adds_media(http):
    context = [{"role": "user", "content": [IMAGE_PART, "The cup is here:", point(1, 2)]}]

    with pytest.raises(BadRequestError) as excinfo:
        _multilook(context=context, prompts=["Describe it.", {"content": [IMAGE_PART, "Same cup?"]}])

    assert (excinfo.value.code, excinfo.value.param) == (ANCHOR_MISSING, "context[0].content[2]")
    assert not http.requests


def test_context_tags_are_unchanged_when_prompts_add_no_media(http):
    cup = image("https://example.com/cup.png")
    context = [{"role": "user", "content": [cup, "The cup is here:", point(1, 2, image=cup), box(1, 2, 3, 4)]}]

    _multilook(context=context, prompts=["Describe it.", ["Is it full?"]])

    assert _context_texts(http.last_body) == [
        "The cup is here:",
        "<point> (1,2) </point>",
        "<point_box> (1,2) (3,4) </point_box>",
    ]


def test_completion_annotations_resolve_in_their_prompts_asset_space(monkeypatch):
    answer = (
        "<point_box> (1,2) (3,4) </point_box>"
        ' <collection mention="cup" asset_idx="0"> <point> (5,6) </point> </collection>'
        ' <point asset_idx="4"> (7,8) </point>'
    )
    payload = _ok(2)
    for result in payload["results"]:
        result["completions"][0]["message"]["content"] = answer
    install(monkeypatch, lambda request: json_response(payload))

    response = _multilook(context=CONTEXT, prompts=["Where?", [IMAGE_PART, "And here?"]])

    first, second = (result.completions[0] for result in response.results)
    assert (first.asset_count, second.asset_count) == (1, 2)
    annotations = first.annotations()
    assert annotations.points[0] == pt(5, 6, mention="cup", asset_idx=0)
    assert annotations == first.message.annotations()
    assert first.annotations(expects="box").boxes == annotations.boxes
    box_, collection_point, stray = (*annotations.boxes, *annotations.points)
    # A missing selector is the last asset of the context plus that prompt.
    assert [first.resolve_asset_idx(box_), second.resolve_asset_idx(box_)] == [0, 1]
    assert [first.resolve_asset_idx(collection_point), second.resolve_asset_idx(collection_point)] == [0, 0]
    with pytest.raises(ValueError, match="out of range"):
        second.resolve_asset_idx(stray)
    assert box_.asset_idx is None  # resolution never writes back


def test_api_key_only_env_uses_the_perceptron_api(http):
    assert settings().provider == "perceptron"

    _multilook()

    assert str(http.last.url) == URL


def test_fal_chosen_or_auto_selected_is_rejected(http, monkeypatch):
    with pytest.raises(BadRequestError) as excinfo:
        Client(provider="fal").chat.completions.multilook(context=[], prompts=["q"])
    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    assert "Multilook" in str(excinfo.value)

    monkeypatch.setenv("PERCEPTRON_PROVIDER", "fal")
    with pytest.raises(BadRequestError):
        _multilook()
    monkeypatch.delenv("PERCEPTRON_PROVIDER")
    monkeypatch.delenv("PERCEPTRON_API_KEY")
    monkeypatch.setenv("FAL_KEY", "fal-key")  # fal auto-selected
    with pytest.raises(BadRequestError) as excinfo:
        _multilook()
    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    assert not http.requests


def test_base_url_configured_for_perceptron_is_honored(http):
    with cfg(provider="perceptron", base_url="https://proxy.example/v1/"):
        _multilook()
    assert str(http.last.url) == "https://proxy.example/v1/chat/completions/multilook"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_async_parity(http, monkeypatch):
    seen = _record_timeouts(monkeypatch)

    async def _run():
        return await AsyncClient().chat.completions.multilook(
            context=CONTEXT, prompts=["a", [image(PNG_BYTES), "b"]], n=2, temperature=0.3, reasoning_effort="high"
        )

    response = asyncio.run(_run())

    assert str(http.last.url) == URL
    body = http.last_body
    assert (body["n"], body["temperature"], body["reasoning_effort"]) == (2, 0.3, "high")
    assert body["prompts"][0] == "a"
    assert response.request_id == "trace-ml"
    assert [len(r.completions) for r in response.results] == [2, 2]
    assert [r.completions[0].asset_count for r in response.results] == [1, 2]
    assert seen == [305.0]


def test_async_validation_and_errors(monkeypatch):
    error = {"error": {"message": "Model 'isaac-0.1' does not support multilook", "type": "invalid_request_error"}}
    http = install(monkeypatch, lambda request: httpx.Response(400, json=error))

    async def _invalid():
        await AsyncClient().chat.completions.multilook(context=[], prompts=["q"], n=2)

    with pytest.raises(BadRequestError) as excinfo:
        asyncio.run(_invalid())
    assert excinfo.value.param == "temperature"
    assert not http.requests

    async def _unsupported_model():
        await AsyncClient().chat.completions.multilook(context=[], prompts=["q"], model="isaac-0.1")

    with pytest.raises(BadRequestError, match="does not support multilook"):
        asyncio.run(_unsupported_model())
    assert len(http.requests) == 1
