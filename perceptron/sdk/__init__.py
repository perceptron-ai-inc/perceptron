"""Mock SDK surface.

This module provides lightweight placeholders for the upcoming SDK so that
imports like `from perceptron import annotate_image` work today without heavy
dependencies.
"""

from __future__ import annotations

from typing import Any, Iterable


def annotate_image(image: Any, *, labels: Iterable[str] | None = None) -> dict:
    """Mock image annotation helper.

    Parameters
    - image: any python object representing an image; this is a stub.
    - labels: optional iterable of labels to include in the response.

    Returns
    - dict with mock metadata and no-op annotations.
    """
    return {
        "type": "annotation_result",
        "input_type": type(image).__name__,
        "labels": list(labels) if labels is not None else [],
        "annotations": [],  # placeholder for future boxes/masks/keypoints
    }


__all__ = ["annotate_image"]

