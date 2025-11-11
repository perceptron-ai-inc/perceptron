# Find kitchen item — single-shot ICL

Prime Isaac with a single annotated reference (the cake mixer) and detect the same kitchen item in a new frame.

> Install requirements first: `pip install --upgrade perceptron pillow`

## Step 1 — Bootstrap the exemplar box
Run `detect` once on the cake-mixer exemplar to harvest a normalized bounding box.

```python
from perceptron import annotate_image, bbox, detect
from cookbook.utils import cookbook_asset

EXEMPLAR = cookbook_asset("in-context-learning", "single", "cake_mixer_example.webp")
TARGET = cookbook_asset("in-context-learning", "single", "find_kitchen_item.webp")

bootstrap = detect(EXEMPLAR, classes=['objectCategory1'])
first_box = bootstrap.points[0]
example_shot = annotate_image(
    EXEMPLAR,
    {
        'objectCategory1': [
            bbox(
                int(first_box.top_left.x),
                int(first_box.top_left.y),
                int(first_box.bottom_right.x),
                int(first_box.bottom_right.y),
                mention='objectCategory1',
            )
        ]
    },
)
```

## Step 2 — Detect with the single-shot guidance
```python
from perceptron import detect

result = detect(
    TARGET,
    classes=['objectCategory1'],
    examples=[example_shot],
)

for box in result.points or []:
    print(box)
```

> Bounding boxes use Perceptron’s **1–1000** grid. Convert to pixels before drawing overlays.

## Step 3 — Visualize
Use Pillow to draw the prediction on `find_kitchen_item.webp` and save a file named `<original>_annotated.png` (for example, `findTeaPotExample_annotated.png`).
