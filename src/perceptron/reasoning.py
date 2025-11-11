"""
Reasoning extraction and parsing utilities.

Supported tags:
- <think> reasoning content here </think>

Data structures:
- Reasoning: dataclass holding reasoning content

Functions:
- extract_reasoning(text) → single Reasoning object with concatenated content
- strip_reasoning(text) → remove all <think> tags
- parse_text(text) → ordered segments: text and reasoning with spans
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(eq=True)
class Reasoning:
    """
    Represents reasoning/chain-of-thought content extracted from <think> tags.

    Attributes:
        content: The accumulated reasoning text from all <think> tags
    """
    content: str


# Regex pattern for <think> tags (case-insensitive, dotall for multiline)
_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_reasoning(text: str) -> Reasoning | None:
    """
    Extract all <think> tag content and return as single Reasoning object.

    Multiple <think> tags are concatenated with newlines.
    Returns None if no <think> tags found.

    Args:
        text: Text containing potential <think> tags

    Returns:
        Reasoning object with concatenated content, or None if no tags found

    Example:
        >>> text = "First <think>reason 1</think> then <think>reason 2</think>"
        >>> reasoning = extract_reasoning(text)
        >>> reasoning.content
        'reason 1\\nreason 2'
    """
    if not isinstance(text, str):
        return None

    # Find all <think> tag contents
    segments = [seg.strip() for seg in _THINK_TAG.findall(text)]
    # Filter out empty segments
    reasoning_parts = [seg for seg in segments if seg]

    if not reasoning_parts:
        return None

    # Concatenate all reasoning parts with newlines
    content = "\n".join(reasoning_parts)
    return Reasoning(content=content)


def strip_reasoning(text: str) -> str:
    """
    Remove all <think> tags and return clean text.

    Args:
        text: Text containing potential <think> tags

    Returns:
        Text with all <think> tags removed

    Example:
        >>> text = "Answer: <think>reasoning here</think> 42"
        >>> strip_reasoning(text)
        'Answer:  42'
    """
    if not isinstance(text, str):
        return text

    # Remove all <think> tags
    cleaned = _THINK_TAG.sub("", text)
    # Collapse multiple spaces
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


def parse_text(text: str) -> list[dict[str, Any]]:
    """
    Parse text into ordered segments of text and reasoning with position tracking.

    Returns list of segments alternating between text and reasoning, with spans
    indicating their position in the original text.

    Args:
        text: Text to parse

    Returns:
        List of segment dictionaries with keys:
        - kind: "text" or "reasoning"
        - text: text content (for kind="text")
        - value: Reasoning object (for kind="reasoning")
        - span: {"start": int, "end": int} position in original text

    Example:
        >>> text = "Start <think>reasoning</think> end"
        >>> segments = parse_text(text)
        >>> segments[0]
        {"kind": "text", "text": "Start ", "span": {"start": 0, "end": 6}}
        >>> segments[1]
        {"kind": "reasoning", "value": Reasoning(content="reasoning"),
             "span": {"start": 6, "end": 28}}
    """
    if not isinstance(text, str):
        return [{"kind": "text", "text": str(text), "span": {"start": 0, "end": len(str(text))}}]

    segments = []
    idx = 0

    for match in _THINK_TAG.finditer(text):
        # Add text segment before the tag (if any)
        if match.start() > idx:
            segments.append({
                "kind": "text",
                "text": text[idx:match.start()],
                "span": {"start": idx, "end": match.start()}
            })

        # Add reasoning segment
        content = match.group(1).strip()
        if content:  # Only add non-empty reasoning
            segments.append({
                "kind": "reasoning",
                "value": Reasoning(content=content),
                "span": {"start": match.start(), "end": match.end()}
            })

        idx = match.end()

    # Add remaining text after last tag (if any)
    if idx < len(text):
        segments.append({
            "kind": "text",
            "text": text[idx:],
            "span": {"start": idx, "end": len(text)}
        })

    return segments
