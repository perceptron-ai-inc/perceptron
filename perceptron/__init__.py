"""Top-level package for perceptron.

This exposes the primary public API so you can:

    from perceptron import TensorStream
"""

from .tensorstream.tensor_stream import TensorStream

__all__ = [
    "TensorStream",
]

# Single-source the package version for the build backend
__version__ = "0.1.0"
