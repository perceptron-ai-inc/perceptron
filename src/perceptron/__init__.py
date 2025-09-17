"""
perceptron – Python SDK (v0.1 scaffolding)

Public surface (subject to refinement):
- DSL: perceive (decorator), text, system, agent, image, point, box, polygon, block
- Pointing: PointParser, parse_text, extract_points, strip_tags
- Data constructors for annotations/examples: pt, bbox, poly
- Config: configure, config (context manager), settings

This initial scaffold focuses on the compile/runtime pieces that do not require
network access. Transport and streaming are added in later phases.
"""

__version__ = "0.1.1"

from .config import configure, config, settings
from .client import Client, AsyncClient
from .errors import (
    SDKError,
    TransportError,
    TimeoutError,
    AuthError,
    RateLimitError,
    ServerError,
    BadRequestError,
    ExpectationError,
    AnchorError,
)
from .pointing.types import (
    SinglePoint,
    BoundingBox,
    Polygon,
    Collection,
    pt,
    bbox,
    poly,
    collection,
)
from .pointing.parser import (
    PointParser,
    ReasoningExtraction,
    ReasoningStreamCleaner,
    extract_points,
    extract_reasoning,
    parse_text,
    strip_tags,
)
from .dsl.nodes import text, system, agent, image, point, box, polygon, block
from .dsl.perceive import perceive, async_perceive, inspect_task, PerceiveResult
from .annotations import annotate_image
from .highlevel import caption, ocr, detect, detect_from_coco, question

__all__ = [
    # Config
    "configure",
    "config",
    "settings",
    "Client",
    "AsyncClient",
    # Errors
    "SDKError",
    "TransportError",
    "TimeoutError",
    "AuthError",
    "RateLimitError",
    "ServerError",
    "BadRequestError",
    "ExpectationError",
    "AnchorError",
    # Pointing types & constructors
    "SinglePoint",
    "BoundingBox",
    "Polygon",
    "Collection",
    "pt",
    "bbox",
    "poly",
    "collection",
    # Parser & helpers
    "PointParser",
    "ReasoningExtraction",
    "ReasoningStreamCleaner",
    "parse_text",
    "extract_points",
    "extract_reasoning",
    "strip_tags",
    # DSL nodes & decorator
    "text",
    "system",
    "agent",
    "image",
    "point",
    "box",
    "polygon",
    "block",
    "perceive",
    "async_perceive",
    "inspect_task",
    "PerceiveResult",
    # High-level helpers
    "annotate_image",
    "caption",
    "ocr",
    "detect",
    "question",
    "detect_from_coco",
    "__version__",
]
