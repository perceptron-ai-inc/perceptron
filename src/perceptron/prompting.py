"""Prompt profiles for high-level helpers.

This module centralizes the text templates used by ``perceptron.highlevel`` so
that prompts can vary by model/provider combination. The registry is simple but
extensible: register a profile once, attach aliases or prefixes for the models
that should use it, and call ``resolve_prompt_profile`` at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptionPromptTemplate:
    style_prompts: dict[str, str]
    system_instruction: str | None = None


@dataclass(frozen=True)
class QuestionPromptTemplate:
    open_instruction: str | None
    grounded_instruction: str | None


@dataclass(frozen=True)
class OcrPromptTemplate:
    system_instruction: str | None
    prompts: Mapping[str, str | None]
    default_mode: str = "plain"


@dataclass(frozen=True)
class DetectPromptTemplate:
    general_instruction: str
    category_instruction_template: str


@dataclass(frozen=True)
class HighLevelPromptProfile:
    key: str
    caption: CaptionPromptTemplate
    question: QuestionPromptTemplate
    ocr: OcrPromptTemplate
    detect: DetectPromptTemplate


class PromptProfileRegistry:
    """Lightweight registry for associating models with prompt profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, HighLevelPromptProfile] = {}
        self._aliases: dict[str, str] = {}
        self._prefix_aliases: dict[str, str] = {}
        self._default_key: str | None = None

    def register(
        self,
        key: str,
        profile: HighLevelPromptProfile,
        *,
        is_default: bool = False,
        aliases: Iterable[str] | None = None,
        prefixes: Iterable[str] | None = None,
    ) -> None:
        normalized = key.lower()
        self._profiles[normalized] = profile
        if is_default or self._default_key is None:
            self._default_key = normalized
        for alias in aliases or ():
            self.register_alias(alias, normalized)
        for prefix in prefixes or ():
            self.register_prefix(prefix, normalized)

    def register_alias(self, alias: str, key: str) -> None:
        self._aliases[alias.lower()] = key.lower()

    def register_prefix(self, prefix: str, key: str) -> None:
        self._prefix_aliases[prefix.lower()] = key.lower()

    def resolve(self, model: str | None) -> HighLevelPromptProfile:
        if not self._profiles:
            raise RuntimeError("No prompt profiles registered")
        candidate = model.lower() if isinstance(model, str) else None
        if candidate and candidate in self._profiles:
            return self._profiles[candidate]
        if candidate:
            alias = self._aliases.get(candidate)
            if alias and alias in self._profiles:
                return self._profiles[alias]
            for prefix, target in self._prefix_aliases.items():
                if candidate.startswith(prefix) and target in self._profiles:
                    return self._profiles[target]
        if self._default_key is None:
            # Should not happen because register() always sets it, but fall back defensively.
            self._default_key = next(iter(self._profiles))
        return self._profiles[self._default_key]


PROMPT_REGISTRY = PromptProfileRegistry()


def prompt_registry() -> PromptProfileRegistry:
    """Return the global prompt registry (primarily for testing/extensibility)."""

    return PROMPT_REGISTRY


def resolve_prompt_profile(model: str | None) -> HighLevelPromptProfile:
    """Resolve the prompt profile associated with ``model`` (falls back to default)."""

    return PROMPT_REGISTRY.resolve(model)


_ISAAC_PROFILE = HighLevelPromptProfile(
    key="isaac-default",
    caption=CaptionPromptTemplate(
        style_prompts={
            "concise": "Provide a concise, human-friendly caption for the upcoming image.",
            "detailed": "Provide a detailed caption describing key objects, relationships, and context in the upcoming image.",
        },
    ),
    question=QuestionPromptTemplate(
        open_instruction=None,
        grounded_instruction=None,
    ),
    ocr=OcrPromptTemplate(
        system_instruction=(
            "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."
        ),
        prompts={
            "plain": None,
            "markdown": "Transcribe every readable word in the image using Markdown formatting with headings, lists, tables, and other structural elements as appropriate.",
            "html": "Transcribe every readable word in the image using HTML markup.",
        },
        default_mode="plain",
    ),
    detect=DetectPromptTemplate(
        general_instruction="Your goal is to segment out the objects in the scene",
        category_instruction_template="Your goal is to segment out the following categories: {categories}",
    ),
)

_QWEN_PROFILE = HighLevelPromptProfile(
    key="qwen3-vl-235b-a22b-thinking",
    caption=CaptionPromptTemplate(
        system_instruction=None,
        style_prompts={
            "concise": "Describe the primary subjects, their actions, and visible context in one vivid sentence.",
            "detailed": (
                "Provide a multi-sentence caption that calls out subjects, relationships, scene intent, and any text embedded in the image."
            ),
        },
    ),
    question=QuestionPromptTemplate(
        open_instruction=None,
        grounded_instruction=(
            "You are Qwen3-VL performing grounded reasoning. Give the answer and reference the relevant regions using structured tags when available. Report bbox coordinates in JSON format."
        ),
    ),
    ocr=OcrPromptTemplate(
        system_instruction=None,
        prompts={
            "plain": "Read all the text in the image.",
            "markdown": "qwenvl markdown",
            "html": "qwenvl html",
        },
        default_mode="plain",
    ),
    detect=DetectPromptTemplate(
        general_instruction="Locate every object of interest and report bounding box coordinates in JSON format.",
        category_instruction_template=(
            'Locate every instance that belongs to the following categories: "{categories}". '
            "Report bbox coordinates in JSON format."
        ),
    ),
)


PROMPT_REGISTRY.register(
    _ISAAC_PROFILE.key,
    _ISAAC_PROFILE,
    is_default=True,
    aliases=("default", "isaac", "perceptron", "isaac-0.1"),
    prefixes=("isaac-",),
)

PROMPT_REGISTRY.register(
    _QWEN_PROFILE.key,
    _QWEN_PROFILE,
    aliases=("qwen", "qwen3", "qwen3-vl", "qwen3-vl-235b"),
    prefixes=("qwen3-",),
)


__all__ = [
    "CaptionPromptTemplate",
    "DetectPromptTemplate",
    "HighLevelPromptProfile",
    "OcrPromptTemplate",
    "PromptProfileRegistry",
    "QuestionPromptTemplate",
    "prompt_registry",
    "resolve_prompt_profile",
]
