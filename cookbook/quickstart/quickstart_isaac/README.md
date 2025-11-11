
# Quickstart — Running Isaac 0.1

Ship your first perceptive application with the Isaac 0.1 model. Follow these steps to mirror the developer quickstart experience.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/quickstart/quickstart_isaac/quickstart_isaac.ipynb)

## 1. Create and export an API key
1. Visit https://platform.perceptron.inc/ and generate a key.
2. Store it securely (for example in `~/.zshrc`).
3. Export it before running code:

```bash
# macOS / Linux
export PERCEPTRON_API_KEY="your_api_key_here"
```

```powershell
# Windows
setx PERCEPTRON_API_KEY "your_api_key_here"
```

## 2. Install the SDK and helpers
```bash
pip install --upgrade perceptron pillow
```

- `perceptron` — official SDK for authentication, uploads, and model calls.
- `Pillow` — lightweight image loader used to draw detection boxes.

## 3. Make your first request
Save this as `example.py` or run it inside the notebook:

```python
from pathlib import Path

from perceptron import configure, image, perceive, text
from PIL import Image, ImageDraw
from cookbook.utils import cookbook_asset

# Assume PERCEPTRON_API_KEY is already exported; configure() reads it automatically.
configure(
    provider="perceptron",
    # model="isaac-0.1",  # Enable once the SDK supports the model argument.
)

IMAGE_PATH = cookbook_asset("quickstart", "isaac", "truck_scene.jpg")

@perceive(expects="box", allow_multiple=True)
def detect_boxes(frame_path):
    scene = image(frame_path)
    return scene + text("Find every shipping box in the truck. Return one bounding box per item.")

result = detect_boxes(str(IMAGE_PATH))

img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

def to_px(point):
    return point.x / 1000 * img.width, point.y / 1000 * img.height

for idx, box in enumerate(result.points or []):
    top_left = to_px(box.top_left)
    bottom_right = to_px(box.bottom_right)
    draw.rectangle([top_left, bottom_right], outline="lime", width=3)
    label = box.mention or getattr(box, "label", None) or f"box {idx + 1}"
    draw.text((top_left[0], max(top_left[1] - 12, 0)), label, fill="lime")

img.save("truck_scene_annotated.jpg")
print("Annotated image saved to truck_scene_annotated.jpg")
```

> Perceptron's geometry is normalized to a **1–1000 grid**. Convert back to pixel coordinates before rendering overlays.

## 4. Run the script
```bash
python example.py
```
A file named `truck_scene_annotated.jpg` appears with the detected boxes.
> Tip: Run the script from the repo root (for example, `python cookbook/quickstart/quickstart_isaac/example.py`) so `import cookbook.utils` can locate the shared assets.

## Explore more
- [Captioning](../recipes/capabilities/captioning/README.md)
- [OCR](../recipes/capabilities/ocr/README.md)
- [Detection](../recipes/capabilities/object-detection/README.md)
- [Visual Q&A](../recipes/capabilities/visual-qa/README.md)
- [In-context learning (single example)](../recipes/capabilities/in-context-learning/README.md)
- [Multi-image in-context learning](../recipes/capabilities/multi-image-in-context-learning/README.md)
- [Isaac frame-by-frame tutorial](../recipes/tutorials/isaac_0.1_frame_by_frame/README.md)
