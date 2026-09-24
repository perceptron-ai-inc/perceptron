"""Every README example runs offline.

Each ```python block is executed as written, and each ``perceptron ...`` line of the ```bash blocks is invoked through
the CLI, in a temporary directory holding the media files the snippet names. HTTP goes to a small fake of the
Perceptron API (chat completions, streaming, tool calls, multilook, files, models); a request to any other host (for
example fal.run) fails the test, which keeps the README's provider story honest.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import httpx
import pytest
from _http_mock import chunk, completion, install, json_response, sse_response
from _image_fixtures import JPEG_BYTES, PNG_BYTES, WEBP_BYTES
from typer.testing import CliRunner

from perceptron import config, settings
from perceptron.cli import app

README = Path(__file__).resolve().parents[1] / "README.md"
FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)

MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42" + b"\x00" * 16
WEBM_BYTES = b"\x1a\x45\xdf\xa3" + b"\x00" * 32
WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24
MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
FLAC_BYTES = b"fLaC\x00\x00\x00\x22" + b"\x00" * 34
MEDIA_BYTES = {
    ".jpg": JPEG_BYTES,
    ".jpeg": JPEG_BYTES,
    ".png": PNG_BYTES,
    ".webp": WEBP_BYTES,
    ".mp4": MP4_BYTES,
    ".webm": WEBM_BYTES,
    ".wav": WAV_BYTES,
    ".mp3": MP3_BYTES,
    ".flac": FLAC_BYTES,
}
MEDIA_NAME = re.compile(r"""["']([\w./-]+\.(?:jpe?g|png|webp|mp4|webm|wav|mp3|flac))["']""")

FILE_ID = "file-AbCdEfGhIjKlMnOpQrStUv"
ANSWER = (
    "Two forklifts and a person. "
    '<point_box mention="forklift"> (100,120) (300,400) </point_box> '
    '<point mention="person"> (520,610) </point> '
    '<clip mention="door opens" t="1.0 seconds 2.5 seconds" /> '
    '<track mention="ball"><point t="0.5 seconds"> (100,200) </point><point t="1.0 seconds"> (120,210) </point></track>'
)
MULTI_ASSET_ANSWER = (
    ' <collection mention="new product" asset_idx="1"><point_box> (400,300) (600,650) </point_box></collection>'
)
MEDIA_PART_TYPES = {
    "image_url",
    "video_url",
    "audio_url",
    "input_audio",
    "image_file_id",
    "video_file_id",
    "audio_file_id",
    "video_frames",
}
AUDIO_PART_TYPES = {"audio_url", "input_audio", "audio_file_id"}
MODEL = {"id": "perceptron-mk1.5", "object": "model", "created": 1790035200, "owned_by": "perceptron"}
MODEL_INFO = {
    **MODEL,
    "name": "Perceptron Mk1.5",
    "capabilities": ["tool_calling", "regex", "response_format_json_schema"],
    "modalities": ["text", "image", "video", "audio"],
    "reasoning": {"supported": True, "always_enabled": False},
    "max_context_tokens": 131072,
    "max_output_tokens": 16384,
}
UPLOADED = {
    "id": FILE_ID,
    "object": "file",
    "bytes": len(JPEG_BYTES),
    "created_at": 1790000000,
    "filename": "warehouse.jpg",
    "purpose": "vision",
}


def _blocks(language: str) -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    return [
        (text.count("\n", 0, match.start(2)) + 1, match.group(2))
        for match in FENCE.finditer(text)
        if match.group(1) == language
    ]


PYTHON_BLOCKS = _blocks("python")
CLI_LINES = [
    (line_no + offset, line)
    for line_no, block in _blocks("bash")
    for offset, line in enumerate(block.splitlines())
    if line.startswith("perceptron ")
]


# ---------------------------------------------------------------------------
# A fake of the Perceptron API
# ---------------------------------------------------------------------------


def _parts(messages: list[dict]) -> list[dict]:
    return [part for message in messages if isinstance(message.get("content"), list) for part in message["content"]]


def _arguments(tool: dict) -> str:
    """Arguments for a tool's declared properties: a sample value per JSON type."""
    samples = {"string": "Paris", "integer": 500, "number": 0.5, "boolean": True}
    properties = (tool["function"].get("parameters") or {}).get("properties") or {}
    return json.dumps({name: samples.get(spec.get("type"), "x") for name, spec in properties.items()})


class FakePerceptronAPI:
    """Answers like the Perceptron API and records what the examples exercised."""

    def __init__(self) -> None:
        self.unexpected: list[str] = []
        self.endpoints: list[str] = []
        self.streams = 0
        self.tool_results = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url, method = request.url, request.method
        if url.host != "api.perceptron.inc" or not request.headers.get("authorization", "").startswith("Bearer "):
            self.unexpected.append(f"{method} {url}")
            raise AssertionError(f"README example called {method} {url}; expected the Perceptron API")
        self.endpoints.append(f"{method} {url.path}")
        if url.path == "/v1/chat/completions":
            return self._chat(json.loads(request.content))
        if url.path == "/v1/chat/completions/multilook":
            return self._multilook(json.loads(request.content))
        if url.path.startswith("/v1/files"):
            return self._files(method, url.path)
        if url.path.startswith("/v1/models"):
            info = url.params.get("extended") == "true"
            model = MODEL_INFO if info else MODEL
            return json_response({"object": "list", "data": [model]} if url.path == "/v1/models" else model)
        self.unexpected.append(f"{method} {url}")
        raise AssertionError(f"README example called an unexpected endpoint: {method} {url}")

    @staticmethod
    def _files(method: str, path: str) -> httpx.Response:
        if path.endswith("/content"):
            return httpx.Response(200, content=JPEG_BYTES)
        if method == "DELETE":
            return json_response({"id": FILE_ID, "object": "file", "deleted": True})
        if path == "/v1/files" and method == "GET":
            page = {"object": "list", "data": [UPLOADED], "first_id": FILE_ID, "last_id": FILE_ID, "has_more": False}
            return json_response(page)
        return json_response(UPLOADED)

    def _chat(self, body: dict) -> httpx.Response:
        messages = body["messages"]
        parts = _parts(messages)
        tools = body.get("tools")
        tool_calls = None
        content = ANSWER
        self.streams += bool(body.get("stream"))
        self.tool_results += messages[-1].get("role") == "tool"
        if tools and messages[-1].get("role") != "tool":
            tool = tools[0]
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool["function"]["name"], "arguments": _arguments(tool)},
                }
            ]
            content = None
        elif sum(part.get("type") in MEDIA_PART_TYPES for part in parts) > 1:
            content += MULTI_ASSET_ANSWER
        reasoning = "Counting carefully." if body.get("reasoning_effort") not in (None, "none") else None
        details = {"cached_tokens": 0}
        hears_video = (body.get("vision_config") or {}).get("enable_audio_in_video") is True
        if hears_video or any(part.get("type") in AUDIO_PART_TYPES for part in parts):
            details["audio_tokens"] = 120
        usage = {"prompt_tokens": 812, "completion_tokens": 34, "total_tokens": 846, "prompt_tokens_details": details}
        finish_reason = "tool_calls" if tool_calls else "stop"
        headers = {"x-trace-id": "trace-readme"}
        if not body.get("stream"):
            payload = completion(content, finish_reason=finish_reason, tool_calls=tool_calls, reasoning=reasoning)
            payload["usage"] = usage
            return json_response(payload, headers=headers)
        events = [chunk({"role": "assistant", **({"reasoning_content": reasoning} if reasoning else {})})]
        if tool_calls:
            events.append(chunk({"tool_calls": [{"index": 0, **tool_calls[0]}]}))
        else:
            third = len(content) // 3
            events += [chunk({"content": piece}) for piece in (content[:third], content[third:])]
        events += [chunk({}, finish_reason=finish_reason), chunk(choices=False, usage=usage)]
        return sse_response(events, headers=headers)

    def _multilook(self, body: dict) -> httpx.Response:
        results = [
            {
                "prompt_index": index,
                "completions": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"Answer {index}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 5},
            }
            for index, _ in enumerate(body["prompts"])
        ]
        usage = {
            "prompt_tokens": 900,
            "completion_tokens": 10,
            "total_tokens": 910,
            "prompt_tokens_details": {"cached_tokens": 400},
        }
        payload = {"id": "ml-1", "object": "chat.completion.multilook", "model": body.get("model"), "results": results}
        return json_response({**payload, "usage": usage}, headers={"x-trace-id": "trace-readme"})


@pytest.fixture
def fake_api(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    # What the README's Configuration section exports.
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "perceptron")
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk_live_test")
    api = FakePerceptronAPI()
    install(monkeypatch, api)
    yield api
    assert api.unexpected == []


def _create_media(workdir: Path, code: str) -> None:
    """Write the media files (and directories and datasets) a snippet names into ``workdir``."""
    names = set(MEDIA_NAME.findall(code))
    for token in shlex.split(code, comments=True) if code.startswith("perceptron ") else []:
        if Path(token).suffix.lower() in MEDIA_BYTES:
            names.add(token)
        elif token.startswith("./"):  # an input directory
            for index in range(2):
                (workdir / token / f"frame_{index}.png").parent.mkdir(parents=True, exist_ok=True)
                (workdir / token / f"frame_{index}.png").write_bytes(PNG_BYTES)
    for name in names:
        path = workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(MEDIA_BYTES[path.suffix.lower()])
    if "detect_from_coco" in code:
        dataset = workdir / "datasets" / "custom"
        (dataset / "train" / "images").mkdir(parents=True)
        (dataset / "annotations").mkdir()
        images = [{"id": index, "file_name": f"img_{index}.png", "width": 32, "height": 32} for index in (1, 2, 3)]
        for meta in images:
            (dataset / "train" / "images" / meta["file_name"]).write_bytes(PNG_BYTES)
        coco = {
            "images": images,
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10]},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [4, 4, 8, 8]},
                {"id": 3, "image_id": 3, "category_id": 1, "bbox": [1, 1, 5, 5]},
            ],
            "categories": [{"id": 1, "name": "defect"}, {"id": 2, "name": "ok"}],
        }
        (dataset / "annotations" / "instances_train.json").write_text(json.dumps(coco), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_readme_has_examples():
    assert len(PYTHON_BLOCKS) >= 15
    assert len(CLI_LINES) >= 8


@pytest.mark.parametrize(("line_no", "code"), PYTHON_BLOCKS, ids=[f"README.md:{n}" for n, _ in PYTHON_BLOCKS])
def test_python_example_runs(fake_api, tmp_path, monkeypatch, line_no, code):
    _create_media(tmp_path, code)
    monkeypatch.chdir(tmp_path)
    # Leading newlines keep traceback line numbers equal to README line numbers.
    compiled = compile("\n" * (line_no - 1) + code, str(README), "exec")
    with config():  # undo the example's configure() calls
        exec(compiled, {"__name__": "__readme__"})
    # The examples really do what they show.
    if "function_tool" in code:
        assert fake_api.tool_results >= 1, "the tool-calling example never sent a tool result"
    if "stream=True" in code:
        assert fake_api.streams >= 1
    for endpoint, used_by in (("/files", "client.files"), ("/models", "client.models"), ("/multilook", ".multilook(")):
        if used_by in code:
            assert any(endpoint in call for call in fake_api.endpoints), endpoint


@pytest.mark.parametrize(("line_no", "line"), CLI_LINES, ids=[f"README.md:{n}" for n, _ in CLI_LINES])
def test_cli_example_runs(fake_api, tmp_path, monkeypatch, line_no, line):
    monkeypatch.setenv("COLUMNS", "160")
    _create_media(tmp_path, line)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, shlex.split(line, comments=True)[1:])
    assert result.exit_code == 0, f"README.md:{line_no}: {line}\n{result.stdout}"


# The README's provider claims, checked directly.


def test_api_key_only_env_sends_helpers_to_fal_and_new_surfaces_to_perceptron(monkeypatch):
    from perceptron import Client, image, question
    from perceptron.errors import BadRequestError

    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk_live_test")
    seen: list[str] = []

    def handler(request):
        seen.append(f"{request.url.host}{request.url.path}")
        return json_response(completion("ok"))

    install(monkeypatch, handler)
    assert settings().provider == "fal"
    question(image(PNG_BYTES), "What is shown?")
    with pytest.raises(BadRequestError, match='configure\\(provider="perceptron"\\)'):
        question(image(PNG_BYTES), "What is shown?", model="perceptron-mk1.5")
    Client().chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
    with config(provider="perceptron"):
        question(image(PNG_BYTES), "What is shown?")
    assert seen == [
        "fal.run/perceptron/isaac-01/openai/v1/chat/completions",
        "api.perceptron.inc/v1/chat/completions",
        "api.perceptron.inc/v1/chat/completions",
    ]


def test_perceptron_provider_defaults_to_mk15(fake_api, monkeypatch):
    from perceptron import image, question

    recorded: list[dict] = []
    install(monkeypatch, lambda request: recorded.append(json.loads(request.content)) or fake_api(request))
    question(image(PNG_BYTES), "What is shown?")
    assert recorded[-1]["model"] == "perceptron-mk1.5"
