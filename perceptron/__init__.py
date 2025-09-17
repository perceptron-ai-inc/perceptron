"""Top-level package for perceptron.

Public API examples:
    - from perceptron import annotate_image

For TensorStream utilities, import from the subpackage:
    - from perceptron.tensorstream import TensorStream  # requires perceptron[torch]
"""

from .sdk import annotate_image

__all__ = ["annotate_image"]

# Single-source the package version for the build backend
__version__ = "0.1.0"
