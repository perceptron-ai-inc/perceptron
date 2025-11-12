"""
perceptron - Python SDK (v0.1 scaffolding)

Public surface (subject to refinement):
- DSL: perceive (decorator), text, system, agent, image, point, box, polygon, block
- Pointing: PointParser, parse_text, extract_points, strip_tags
- Data constructors for annotations/examples: pt, bbox, poly
- Config: configure, config (context manager), settings

This initial scaffold focuses on the compile/runtime pieces that do not require
network access. Transport and streaming are added in later phases.
"""

import importlib

__version__ = "0.1.4"

from .annotations import annotate_image
from .client import AsyncClient, Client
from .config import config, configure, settings
from .dsl.nodes import agent, block, box, image, point, polygon, system, text
from .dsl.perceive import PerceiveResult, async_perceive, inspect_task, perceive
from .errors import (
    AnchorError,
    AuthError,
    BadRequestError,
    ExpectationError,
    RateLimitError,
    SDKError,
    ServerError,
    TimeoutError,
    TransportError,
)
from .highlevel import caption, detect, detect_from_coco, ocr, ocr_html, ocr_markdown, question
from .pointing.geometry import scale_points_to_pixels
from .pointing.parser import (
    PointParser,
    ReasoningExtraction,
    ReasoningStreamCleaner,
    extract_points,
    extract_reasoning,
    parse_text,
    strip_tags,
)
from .pointing.types import (
    BoundingBox,
    Collection,
    Polygon,
    SinglePoint,
    bbox,
    collection,
    poly,
    pt,
)


# Lazy-load selected subpackages to allow attribute-style access like
# `perceptron.tensorstream` without importing it eagerly (and without forcing
# optional dependencies like torch unless used).
def __getattr__(name):
    if name == "tensorstream":
        module = importlib.import_module(f"{__name__}.tensorstream")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnchorError",
    "AsyncClient",
    "AuthError",
    "BadRequestError",
    "BoundingBox",
    "Client",
    "Collection",
    "ExpectationError",
    "PerceiveResult",
    "PointParser",
    "Polygon",
    "RateLimitError",
    "ReasoningExtraction",
    "ReasoningStreamCleaner",
    "SDKError",
    "ServerError",
    "SinglePoint",
    "TimeoutError",
    "TransportError",
    "__version__",
    "agent",
    "annotate_image",
    "async_perceive",
    "bbox",
    "block",
    "box",
    "caption",
    "collection",
    "config",
    "configure",
    "detect",
    "detect_from_coco",
    "extract_points",
    "extract_reasoning",
    "image",
    "inspect_task",
    "ocr",
    "ocr_html",
    "ocr_markdown",
    "parse_text",
    "perceive",
    "point",
    "poly",
    "polygon",
    "pt",
    "question",
    "scale_points_to_pixels",
    "settings",
    "strip_tags",
    "system",
    "tensorstream",
    "text",
]
