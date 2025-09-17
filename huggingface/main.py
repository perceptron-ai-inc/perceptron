"""
Demonstration of HuggingFace Genesis pipeline using modular implementation.

This script shows how to:
1. Create a multimodal Document with text and images
2. Process it through GenesisProcessor to create TensorStreams
3. Run the TensorStream through GenesisForCausalLM model
4. Generate text continuations

Example usage:
    python -m genesis.huggingface.main
"""

from __future__ import annotations

import torch
import io
from PIL import Image as PILImage
import base64
from transformers import AutoTokenizer, AutoConfig
from loguru import logger

from genesis.data.schema import Document, Text, Image, Role
from perceptron.tensorstream import VisionType
from perceptron.tensorstream.ops import tensor_stream_token_view, modality_mask
from huggingface.modular_genesis import GenesisProcessor, GenesisForConditionalGeneration


# Create a dummy document with multimodal content
DUMMY_DOCUMENT = Document(
    content=[
        Text(content="<hint>BOX</hint>\nHello! I have an image to show you.", role=Role.USER),
        Image(
            # Small red dot image (base64 encoded PNG)
            content="iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==",
            role=Role.USER,
        ),
        Text(content="What do you see in this image?", role=Role.USER),
    ]
)


def document_to_messages(
    document: Document, vision_token: str = "<image>"
) -> tuple[list[dict[str, str]], list[PILImage.Image]]:
    """
    Convert a Document to messages format compatible with chat templates.
    Each content turn creates its own message entry.

    Args:
        document: Document containing Text and/or Image content
        vision_token: Token to use for image placeholder

    Returns:
        Tuple of (messages, images) where messages is a list of dicts with 'role' and 'content'
    """
    messages = []
    images = []

    for item in document.content:
        if isinstance(item, Text):
            if item.content:
                messages.append(
                    {
                        "role": item.role.value if item.role else "user",
                        "content": item.content,
                    }
                )
        elif isinstance(item, Image):
            if item.content:
                # Decode base64 image
                img = PILImage.open(io.BytesIO(base64.b64decode(item.content)))
                images.append(img)
                messages.append(
                    {
                        "role": item.role.value if item.role else "user",
                        "content": vision_token,
                    }
                )

    return messages, images


def decode_tensor_stream(tensor_stream, tokenizer):
    """Decode a TensorStream to see its text content."""
    token_view = tensor_stream_token_view(tensor_stream)
    mod = modality_mask(tensor_stream)

    # Get text tokens (excluding vision tokens)
    text_tokens = token_view[(mod != VisionType.image.value)]
    decoded = tokenizer.decode(text_tokens[0] if len(text_tokens.shape) > 1 else text_tokens)

    return decoded


def main():
    """Main demonstration of the HuggingFace Genesis pipeline."""
    logger.info("=" * 60)
    logger.info("HuggingFace Genesis Modular Implementation Demo")
    logger.info("=" * 60)

    # Use the Perceptron checkpoint directly - it will be converted automatically
    hf_path = "/home/akshat/models/dpo_v6/step-126/"

    # Load processor and config from the HF checkpoint
    logger.info(f"Loading processor and config from HF checkpoint: {hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=False)
    genesis_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    processor = GenesisProcessor(tokenizer=tokenizer, config=genesis_config)

    # Load model from the HF checkpoint
    logger.info(f"Loading GenesisForConditionalGeneration model from HF checkpoint: {hf_path}")
    model = GenesisForConditionalGeneration.from_pretrained(hf_path)

    # Move to appropriate device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()

    logger.info(f"Model loaded on {device} with dtype {dtype}")

    # Process the dummy document using chat templates
    logger.info("\nProcessing dummy document:")
    logger.info(f"Document content: {DUMMY_DOCUMENT}")

    # Convert document to messages format
    messages, images = document_to_messages(DUMMY_DOCUMENT, vision_token=genesis_config.vision_token)
    logger.info(f"\nConverted to messages: {messages}")
    logger.info(f"Number of images: {len(images)}")

    # Apply chat template to get formatted text
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logger.info(f"\nFormatted text after chat template:\n{text}")

    # Process with GenesisProcessor
    device = next(model.parameters()).device
    inputs = processor(text=text, images=images, return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(device)
    input_ids = inputs["input_ids"].to(device)

    # Decode and display the processed content
    decoded_content = decode_tensor_stream(tensor_stream, processor.tokenizer)
    logger.info(f"\nProcessed content (decoded):\n{decoded_content}")

    # Generate text using the model
    logger.info("\nGenerating text using the model...")

    with torch.no_grad():
        # Generate with TensorStream (multimodal generation)
        logger.info("\nMultimodal generation with TensorStream and chat template:")
        generated_ids = model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        # Decode the generated text
        logger.info(f"\nInput shape: {tensor_stream.shape}")
        logger.info(f"Generated shape: {generated_ids.shape}")

        # Decode full output
        generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        logger.info(f"\nFull generated output:\n{generated_text}")

        # Also show just the new tokens (excluding the input)
        if generated_ids.shape[1] > input_ids.shape[1]:
            new_tokens = generated_ids[0, input_ids.shape[1] :]
            new_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            logger.info(f"\nNew tokens only: '{new_text}' ({len(new_tokens)} tokens)")

    # Example of text-only generation using chat template
    logger.info("\n" + "=" * 60)
    logger.info("Text-only generation example:")

    # Create a simple text-only conversation
    text_messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    text_prompt = processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
    logger.info(f"\nText prompt:\n{text_prompt}")

    # Process and generate
    text_inputs = processor(text=text_prompt, images=None, return_tensors="pt")
    text_tensor_stream = text_inputs["tensor_stream"].to(device)

    with torch.no_grad():
        text_generated_ids = model.generate(
            tensor_stream=text_tensor_stream,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        text_output = processor.tokenizer.decode(text_generated_ids[0], skip_special_tokens=False)
        logger.info(f"\nGenerated response:\n{text_output}")


if __name__ == "__main__":
    main()
