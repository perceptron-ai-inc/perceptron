"""
ExecuTorch export script for Perceptron models.

This script demonstrates how to export a Perceptron model to ExecuTorch format.
Model loading is copied from main.py to ensure consistency.

Example usage:
    uv run python -m huggingface.executorch_export --output-path model.pte
    uv run python -m huggingface.executorch_export --model-path PerceptronAI/Isaac-0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from loguru import logger
from PIL import Image as PILImage
from torch.export import Dim, export
from torch.utils._pytree import FlattenFunc, FlattenWithKeysFunc, UnflattenFunc
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, PreTrainedModel

from perceptron.tensorstream import Event, TensorStream, create_stream


def tensorstream_flatten_with_keys(ts: TensorStream):
    tensors, context = tensorstream_flatten(ts)
    key_children = [
        (pytree.SequenceKey(idx), tensor)
        for idx, tensor in enumerate(tensors)
    ]
    return key_children, context  # type: ignore[return-value]


def tensorstream_flatten(ts: TensorStream):
    """Flatten a TensorStream so torch.export can treat it as a pytree."""
    tensors: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for stream_idx, stream in enumerate(ts.streams):
        if len(stream.events) == 0:
            continue

        current_dtype: torch.dtype | None = None
        current_tensors: list[torch.Tensor] = []
        current_meta: list[dict[str, Any]] = []

        def emit_group():
            if not current_meta or current_dtype is None:
                return
            concatenated = (
                torch.cat(current_tensors, dim=0)
                if len(current_tensors) > 1
                else current_tensors[0]
            )
            tensors.append(concatenated)
            metadata.append(
                {
                    "stream_index": stream_idx,
                    "priority": stream.priority,
                    "dtype": str(current_dtype),
                    "events": list(current_meta),
                }
            )

        for event in stream.events:
            tensor = event.data
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("TensorStream events must carry torch.Tensor data")
            dtype = tensor.dtype
            if current_dtype is None:
                current_dtype = dtype
            if dtype != current_dtype:
                emit_group()
                current_dtype = dtype
                current_tensors = []
                current_meta = []

            flat_tensor = tensor.reshape(-1).contiguous()
            current_tensors.append(flat_tensor)
            current_meta.append(
                {
                    "type": event.type,
                    "time": event.time,
                    "role": event.role,
                    "dims_virtual": event.dims_virtual,
                    "dims_real": event.dims_real,
                    "idx_range": event.idx_range,
                    "shape": tuple(tensor.shape),
                    "numel": flat_tensor.numel(),
                    "dtype": str(dtype),
                }
            )

        emit_group()
    device = getattr(ts, "_device", None)
    return tensors, (metadata, str(device) if device is not None else None)


def tensorstream_unflatten(
    tensors,
    context,
):
    metadata_list, device_str = context
    device = torch.device(device_str) if device_str is not None else None
    stream_events_map: dict[int, list[Event]] = {}
    stream_priorities: dict[int, list[Any]] = {}
    fallback_device = device if device is not None else torch.device("cpu")

    for tensor, meta in zip(tensors, metadata_list):
        stream_index = int(meta.get("stream_index", 0))
        event_meta = meta["events"]
        priority = meta.get("priority", [])
        stream_priorities.setdefault(stream_index, priority)
        events_list = stream_events_map.setdefault(stream_index, [])

        cursor = 0
        for event_info in event_meta:
            numel = int(event_info.get("numel", 0))
            shape = tuple(event_info.get("shape", ()))
            dtype_meta = event_info.get("dtype", "torch.float32")
            if isinstance(dtype_meta, str):
                dtype_name = dtype_meta.split(".")[-1]
                dtype = getattr(torch, dtype_name)
            else:
                dtype = dtype_meta

            if isinstance(tensor, torch.Tensor):
                if numel > 0:
                    chunk = tensor.narrow(0, cursor, numel)
                else:
                    chunk = tensor.new_zeros((0,))
                cursor += numel
                chunk = chunk.reshape(shape)
            else:
                chunk = torch.zeros(shape, dtype=dtype, device=fallback_device)

            events_list.append(
                Event(
                    data=chunk,
                    type=event_info["type"],
                    time=event_info["time"],
                    role=event_info["role"],
                    dims_virtual=event_info["dims_virtual"],
                    dims_real=event_info["dims_real"],
                    idx_range=event_info.get("idx_range"),
                )
            )

    streams: list[Any] = []
    for idx in sorted(stream_events_map.keys()):
        events = stream_events_map[idx]
        priority = stream_priorities.get(idx, [])
        streams.append(create_stream(events, priority=priority, schedule=False))

    return TensorStream(streams=streams, _device=device)


pytree.register_pytree_node(
    TensorStream,
    cast(FlattenFunc, tensorstream_flatten),
    cast(UnflattenFunc, tensorstream_unflatten),
    serialized_type_name="perceptron.tensorstream.TensorStream",
    flatten_with_keys_fn=cast(FlattenWithKeysFunc, tensorstream_flatten_with_keys),
)


def tensorstream_dynamic_shapes_spec(ts: TensorStream) -> list[dict[int, Any] | None]:
    tensors, (metadata, _) = tensorstream_flatten(ts)
    dim_specs: list[dict[int, Any] | None] = []
    for tensor, _meta in zip(tensors, metadata, strict=False):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("TensorStream events must carry torch.Tensor data")

        if tensor.dim() > 0:
            dim_specs.append({0: Dim.AUTO})
        else:
            dim_specs.append(None)

    return dim_specs


def load_model(hf_path: str = "PerceptronAI/Isaac-0.1"):
    """Load processor, config, and model from HuggingFace checkpoint."""

    # Load processor and config from the HF checkpoint
    logger.info(f"Loading processor and config from HF checkpoint: {hf_path}")
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)

    # Load model from the HF checkpoint using AutoModelForCausalLM
    logger.info(f"Loading AutoModelForCausalLM from HF checkpoint: {hf_path}")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        hf_path, trust_remote_code=True
    )

    model.eval()
    logger.info(f"Model loaded successfully from {hf_path}")

    return model, processor, config

def create_example_inputs(processor, config, device: torch.device):
    dummy_images = [
        PILImage.new("RGB", (224, 224), color=(128, 128, 128)),
        PILImage.new("RGB", (320, 384), color=(64, 96, 128)),
    ]

    messages = [
        {"role": "user", "content": f"{config.vision_token} Here is the first view."},
        {
            "role": "user",
            "content": f"{config.vision_token} Compare it to this second shot and explain the differences in detail.",
        },
        {
            "role": "user",
            "content": "Also summarize both descriptions with extra narrative context so the text span is long.",
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=text, images=dummy_images, return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)
    return tensor_stream

def export_to_executorch(
    model,
    processor,
    config,
    output_path: str = "model.pte",
    use_quantization: bool = False,
    backend: str | None = None,
):
    """Export model to ExecuTorch format following official API patterns."""
    logger.info("=" * 60)
    logger.info("ExecuTorch Export")
    logger.info("=" * 60)

    logger.info("Preparing model for export...")
    device = next(model.parameters()).device

    logger.info("Creating example inputs...")
    example_tensor_stream = create_example_inputs(processor, config, device)

    model.eval()
    logger.info("Exporting model with torch.export.export()...")
    logger.info(f"Example input shape: {example_tensor_stream.shape}")

    dynamic_shapes_spec = tensorstream_dynamic_shapes_spec(example_tensor_stream)

    try:
        logger.info(
            "Exporting with TensorStream dynamic shape constraints"
        )
        exported_program = export(
            model,
            (),
            kwargs={"tensor_stream": example_tensor_stream, "use_cache": False},
            dynamic_shapes={
                "tensor_stream": dynamic_shapes_spec,
                "use_cache": None,
            },
            strict=False,
        )
        logger.info("✓ Model exported successfully")

        partitioner: list[Any] = []
        if backend == "xnnpack":
            partitioner = [XnnpackPartitioner()]
            logger.info("Using XNNPACK backend for CPU acceleration")
        elif backend is not None:
            logger.warning(f"Unknown backend '{backend}', using portable ops")

        logger.info("Converting to ExecuTorch Edge dialect with lowering...")
        partitioner_arg = cast(Any, partitioner if partitioner else None)
        edge_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=partitioner_arg,
        )
        logger.info("✓ Converted to Edge dialect")

        if use_quantization:
            logger.info("Applying quantization (not yet implemented)...")
            logger.warning("Quantization not yet implemented")

        logger.info("Converting to ExecuTorch program...")
        executorch_program = edge_program.to_executorch()
        logger.info("✓ Converted to ExecuTorch program")

        logger.info(f"Saving ExecuTorch program to: {output_path}")
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)

        logger.info(f"✓ Model exported successfully to {output_path}")
        file_size = output_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Exported model size: {file_size:.2f} MB")

    except Exception as exc:  # noqa: BLE001  # keep broad logging for CLI UX
        logger.error(f"Export failed during tracing/conversion: {exc}")
        logger.info("This model may have dynamic control flow or unsupported operations")
        logger.info("Consider:")
        logger.info("  - Exporting with strict=False (already enabled)")
        logger.info("  - Simplifying the model")
        logger.info("  - Using torch.cond for dynamic control flow")
        raise
            
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Export Perceptron models to ExecuTorch format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export (portable ops only)
  uv run python -m huggingface.executorch_export
  
  # Export with XNNPACK backend for CPU acceleration
  uv run python -m huggingface.executorch_export --backend xnnpack
  
  # Export with quantization (experimental)
  uv run python -m huggingface.executorch_export --quantize

For more info, see: https://docs.pytorch.org/executorch/stable/using-executorch-export.html
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="HuggingFace model path or local directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="isaac_0.1.pte",
        help="Output path for the ExecuTorch model file (.pte)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="xnnpack",
        choices=["xnnpack"],
        help="Hardware backend to target (default: portable ops only)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization during export (experimental)",
    )
    
    args = parser.parse_args()
    
    logger.info("Starting ExecuTorch export process...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Backend: {args.backend or 'portable (no acceleration)'}")
    logger.info(f"Quantization: {'enabled' if args.quantize else 'disabled'}")
    
    # Load the model using the same logic as main.py
    model, processor, config = load_model(args.model_path)
    
    # Export to ExecuTorch
    export_to_executorch(
        model,
        processor,
        config,
        output_path=args.output_path,
        use_quantization=args.quantize,
        backend=args.backend,
    )
    
    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info(f"Model saved to: {args.output_path}")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Test the model: see Runtime API Reference")
    logger.info("  2. Deploy to device: copy .pte file to target platform")
    logger.info("  3. Integrate: use ExecuTorch runtime to load and run the model")
    logger.info("\nDocs: https://docs.pytorch.org/executorch/stable/")


if __name__ == "__main__":
    main()
