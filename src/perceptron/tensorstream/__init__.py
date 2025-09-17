"""TensorStream subpackage exports.

Core-only convenience imports so users can do:

    from perceptron.tensorstream import TensorStream, VisionType

Note: Importing this subpackage requires the optional torch dependency.
Install with: pip install perceptron[torch]
"""

from .tensorstream import (
    Event,
    Stream,
    TensorStream,
    TextType,
    VisionType,
    create_stream,
    group_streams,
)

__all__ = [
    "Event",
    "Stream",
    "TensorStream",
    "TextType",
    "VisionType",
    "create_stream",
    "group_streams",
]
