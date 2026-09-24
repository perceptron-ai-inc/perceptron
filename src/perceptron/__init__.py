"""
perceptron - Python SDK for Perceptron's perceptive-language models.

Public surface:
- Helpers: caption, question, detect, ocr (+ ocr_markdown, ocr_html), detect_from_coco
- DSL: perceive / async_perceive (decorator or direct call), inspect_task, and the nodes text, system, agent,
  image, video, audio, video_frames, tool_result, point, box, polygon, block
- Message API: Client / AsyncClient with ``client.chat.completions.create`` (and ``.multilook``), ``client.files``,
  ``client.models``; chat types (ChatCompletion, ToolCall, Usage, ...) and function_tool
- Annotations: SinglePoint, BoundingBox, Polygon, Collection, Clip, Track and their constructors; parse_text,
  parse_annotations, collect_annotations, extract_points / extract_clips / extract_tracks, scan_leaves,
  resolve_asset_idx, strip_tags; pixel scaling
- Config: configure, config (context manager), settings
- Errors: SDKError and its subclasses

``perceptron.tensorstream`` (optional torch dependency) is imported lazily on first access.
"""

import importlib

__version__ = "0.4.0"

from .annotations import annotate_image
from .chat import (
    AsyncChatCompletionStream,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionStream,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    FunctionCall,
    FunctionCallDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    function_tool,
)
from .client import (
    AsyncClient,
    Client,
    JsonSchemaFormat,
    JsonSchemaSpec,
    RegexFormat,
    ResponseFormat,
    json_schema_format,
    pydantic_format,
    regex_format,
)
from .config import config, configure, settings
from .dsl.nodes import (
    ToolResult,
    VideoFrame,
    VideoFrames,
    agent,
    audio,
    block,
    box,
    image,
    point,
    polygon,
    system,
    text,
    tool_result,
    video,
    video_frames,
)
from .dsl.perceive import PerceiveResult, async_perceive, inspect_task, perceive
from .errors import (
    AnchorError,
    AuthError,
    BadRequestError,
    ExpectationError,
    IncompleteStreamError,
    NotFoundError,
    ParseError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    SDKError,
    ServerError,
    TimeoutError,
    TransportError,
)
from .files import AsyncFiles, File, FileDeleted, FileList, Files
from .highlevel import caption, detect, detect_from_coco, ocr, ocr_html, ocr_markdown, question
from .models import AsyncModels, Model, ModelInfo, ModelReasoning, Models
from .multilook import (
    MultilookCompletion,
    MultilookPromptError,
    MultilookResponse,
    MultilookResult,
    MultilookUsage,
)
from .pointing.geometry import scale_annotations_by_asset, scale_points_to_pixels
from .pointing.parser import (
    AnnotationCollection,
    AnnotationParse,
    PointParser,
    collect_annotations,
    extract_clips,
    extract_points,
    extract_tracks,
    parse_annotations,
    parse_text,
    resolve_asset_idx,
    scan_leaves,
    strip_tags,
)
from .pointing.types import (
    BoundingBox,
    Clip,
    ClipTimestamp,
    Collection,
    Polygon,
    SinglePoint,
    Track,
    bbox,
    clip,
    collection,
    poly,
    pt,
    track,
)


# Lazy-load selected subpackages to allow attribute-style access like
# `perceptron.tensorstream` without importing it eagerly (and without forcing
# optional dependencies like torch unless used). It is not in `__all__`, so
# `from perceptron import *` works without torch.
def __getattr__(name):
    if name == "tensorstream":
        module = importlib.import_module(f"{__name__}.tensorstream")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnchorError",
    "AnnotationCollection",
    "AnnotationParse",
    "AsyncChatCompletionStream",
    "AsyncClient",
    "AsyncFiles",
    "AsyncModels",
    "AuthError",
    "BadRequestError",
    "BoundingBox",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "ChatCompletionStream",
    "Choice",
    "ChoiceDelta",
    "ChunkChoice",
    "Client",
    "Clip",
    "ClipTimestamp",
    "Collection",
    "ExpectationError",
    "File",
    "FileDeleted",
    "FileList",
    "Files",
    "FunctionCall",
    "FunctionCallDelta",
    "IncompleteStreamError",
    "JsonSchemaFormat",
    "JsonSchemaSpec",
    "Model",
    "ModelInfo",
    "ModelReasoning",
    "Models",
    "MultilookCompletion",
    "MultilookPromptError",
    "MultilookResponse",
    "MultilookResult",
    "MultilookUsage",
    "NotFoundError",
    "ParseError",
    "PerceiveResult",
    "PermissionDeniedError",
    "PointParser",
    "Polygon",
    "QuotaExceededError",
    "RateLimitError",
    "RegexFormat",
    "ResponseFormat",
    "SDKError",
    "ServerError",
    "SinglePoint",
    "TimeoutError",
    "ToolCall",
    "ToolCallDelta",
    "ToolResult",
    "Track",
    "TransportError",
    "Usage",
    "VideoFrame",
    "VideoFrames",
    "__version__",
    "agent",
    "annotate_image",
    "async_perceive",
    "audio",
    "bbox",
    "block",
    "box",
    "caption",
    "clip",
    "collect_annotations",
    "collection",
    "config",
    "configure",
    "detect",
    "detect_from_coco",
    "extract_clips",
    "extract_points",
    "extract_tracks",
    "function_tool",
    "image",
    "inspect_task",
    "json_schema_format",
    "ocr",
    "ocr_html",
    "ocr_markdown",
    "parse_annotations",
    "parse_text",
    "perceive",
    "point",
    "poly",
    "polygon",
    "pt",
    "pydantic_format",
    "question",
    "regex_format",
    "resolve_asset_idx",
    "scale_annotations_by_asset",
    "scale_points_to_pixels",
    "scan_leaves",
    "settings",
    "strip_tags",
    "system",
    "text",
    "tool_result",
    "track",
    "video",
    "video_frames",
]
