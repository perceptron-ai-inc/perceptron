# Multi-image in-context learning

Adapt Isaac to brand-new visual categories by supplying multiple exemplars with normalized geometry. This recipe mirrors the cat vs. dog detection flow from the developer docs.

> Install requirements first: `pip install --upgrade perceptron pillow`

## Step 1 — Prepare the exemplars
Two shots live in `_shared/assets/in-context-learning/multi/`:

- `classA.jpg` — reference for `classA` (cat)
- `classB.webp` — reference for `classB` (dog)

Each includes a normalized bounding box in the 0–1000 grid.

```python
from perceptron import annotate_image, bbox
from cookbook.utils import cookbook_asset

CAT_IMAGE = cookbook_asset("in-context-learning", "multi", "classA.jpg")
DOG_IMAGE = cookbook_asset("in-context-learning", "multi", "classB.webp")
TARGET_IMAGE = cookbook_asset("in-context-learning", "multi", "cat_dog_input.png")

cat_example = annotate_image(
    CAT_IMAGE,
    {'classA': [bbox(316, 136, 703, 906, mention='classA')]},
)

dog_example = annotate_image(
    DOG_IMAGE,
    {'classB': [bbox(161, 48, 666, 980, mention='classB')]},
)
```

## Step 2 — Detect with multi-shot guidance
```python
from perceptron import detect

result = detect(
    TARGET_IMAGE,
    classes=['classA', 'classB'],
    examples=[cat_example, dog_example],
)

for box in result.points or []:
    print(box.mention, box.top_left, box.bottom_right)
```

> Bounding boxes use Perceptron’s **1–1000** coordinate system. Convert to pixel space before drawing overlays or computing metrics.

## Step 3 — Visualize the predictions
Use Pillow (as shown in the accompanying `.py`/`.ipynb`) to draw the returned boxes on `cat_dog_input.png` and save `cat_dog_input_annotated.png` for quick inspection. When running scripts, call `cookbook_asset("in-context-learning", "multi", "cat_dog_input.png")` so you always pick up the canonical test asset.
