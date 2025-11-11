# Detection — PPE line

Prompt for the objects you care about and Isaac returns grounded geometry in Perceptron's normalized 1–1000 grid. This example spots hard hats and safety vests on a production line.

```python
from perceptron import image, perceive, text
from cookbook.utils import cookbook_asset

SCENE_PATH = cookbook_asset("capabilities", "detection", "ppe_line.webp")


@perceive(expects="box", allow_multiple=True)
def detect_ppe(frame_path):
    scene = image(frame_path)
    return scene + text("Find every safety helmet and high-visibility vest. Return one bounding box per item.")


result = detect_ppe(str(SCENE_PATH))
print(result.text)
print(result.points)
```

> **Tip:** Reach for `detect()` when you just need a quick helper the SDK builds for you. Reach for `@perceive` when you want to embed custom prompts, streaming, or expectations inside your own pipelines—the decorator gives you full control over the multimodal steps while still returning Perceptron's normalized geometry.

Need finer-grained geometry? Swap `expects` to `"point"` or `"polygon"`, or enable `stream=True` to process live video feeds.
