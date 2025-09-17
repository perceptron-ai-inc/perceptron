"""Top-level package for perceptron.

Public API examples:
    - from perceptron import annotate_image
    - from perceptron import TensorStream  # requires extra: perceptron[torch]
"""

from .sdk import annotate_image

# Expose a lazy attribute for TensorStream so importing perceptron does not
# require heavy optional dependencies (e.g., torch) unless actually used.
_TensorStream = None
_tensorstream_import_error = None

def __getattr__(name: str):
    if name == "TensorStream":
        global _TensorStream, _tensorstream_import_error
        if _TensorStream is not None:
            return _TensorStream
        try:
            from .tensorstream.tensor_stream import TensorStream as _TS
            _TensorStream = _TS
            return _TensorStream
        except Exception as e:  # defer import errors until attribute access
            _tensorstream_import_error = e
            raise ImportError(
                "TensorStream requires the optional 'torch' extra. "
                "Install with: pip install perceptron[torch]"
            ) from e
    raise AttributeError(f"module 'perceptron' has no attribute {name!r}")

__all__ = [
    "annotate_image",
    "TensorStream",
]

# Single-source the package version for the build backend
__version__ = "0.1.0"
