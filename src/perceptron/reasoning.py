"""
Reasoning extraction and parsing utilities.

Supported tags:
- <think> reasoning content here </think>
- Points inside reasoning: <point>, <point_box>, <polygon>, <collection>

Data structures:
- Reasoning: dataclass holding reasoning content and parsed segments

Functions:
- extract_reasoning(text) → single Reasoning object with concatenated content and parsed points
- strip_reasoning(text) → remove all <think> tags
- parse_text(text) → ordered segments: text and reasoning with spans
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(eq=True)
class Reasoning:
    """
    Represents reasoning/chain-of-thought content extracted from <think> tags.

    Attributes:
        content: The accumulated reasoning text from all <think> tags
        parsed_segments: Optional list of parsed segments (text, points, etc.) from the reasoning content
    """
    content: str
    parsed_segments: list[dict[str, Any]] | None = None


# Regex pattern for <think> tags (case-insensitive, dotall for multiline)
_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _parse_points_in_content(content: str) -> list[dict[str, Any]] | None:
    """
    Helper function to parse point/box/polygon tags inside reasoning content.

    Args:
        content: Text content to parse for pointing tags

    Returns:
        List of parsed segments if pointing tags found, None otherwise
    """
    try:
        # Import here to avoid circular dependency
        from .pointing.parser import parse_text as parse_pointing_text

        parsed_segments = parse_pointing_text(content)
        # Only include if there are actual point/box/polygon/collection segments
        if not any(seg.get("kind") in {"point", "box", "polygon", "collection"} for seg in parsed_segments):
            return None
        return parsed_segments
    except ImportError as e:
        logger.warning("Failed to import pointing parser: %s", e)
        return None
    except (ValueError, TypeError, KeyError) as e:
        logger.warning("Failed to parse pointing tags in reasoning content: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error parsing pointing tags: %s", e)
        return None


def extract_reasoning(text: str, parse_points: bool = True) -> Reasoning | None:
    """
    Extract all <think> tag content and return as single Reasoning object.

    Multiple <think> tags are concatenated with newlines.
    Returns None if no <think> tags found.

    Args:
        text: Text containing potential <think> tags
        parse_points: If True, parse point/box/polygon tags inside reasoning content

    Returns:
        Reasoning object with concatenated content and optionally parsed segments, or None if no tags found

    Example:
        >>> text = "First <think>reason 1</think> then <think>reason 2</think>"
        >>> reasoning = extract_reasoning(text)
        >>> reasoning.content
        'reason 1\\nreason 2'

        >>> text = "<think>The point is at <point>(10,20)</point></think>"
        >>> reasoning = extract_reasoning(text, parse_points=True)
        >>> reasoning.parsed_segments  # Contains parsed point data
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

    # Parse points inside reasoning content if requested
    parsed_segments = None
    if parse_points:
        parsed_segments = _parse_points_in_content(content)

    return Reasoning(content=content, parsed_segments=parsed_segments)


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
        return str(text) if text is not None else ""

    # Remove all <think> tags
    cleaned = _THINK_TAG.sub("", text)
    # Collapse multiple spaces
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


def parse_text(text: str, parse_points: bool = True) -> list[dict[str, Any]]:
    """
    Parse text into ordered segments of text and reasoning with position tracking.

    Returns list of segments alternating between text and reasoning, with spans
    indicating their position in the original text. Reasoning content can contain
    parsed point/box/polygon tags.

    Args:
        text: Text to parse
        parse_points: If True, parse point/box/polygon tags inside reasoning content

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

        >>> text = "Text <think>Point at <point>(10,20)</point></think> more"
        >>> segments = parse_text(text, parse_points=True)
        >>> segments[1]["value"].parsed_segments  # Contains parsed point data
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

        # Add reasoning segment with point parsing
        content = match.group(1).strip()
        if content:  # Only add non-empty reasoning
            # Parse points inside reasoning content if requested
            parsed_segments = None
            if parse_points:
                parsed_segments = _parse_points_in_content(content)

            segments.append({
                "kind": "reasoning",
                "value": Reasoning(content=content, parsed_segments=parsed_segments),
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
