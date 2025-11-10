# Grounded visual Q&A

Ask natural-language questions about any scene and get grounded citations along with the answer.

```python
from perceptron import question

QUESTION = "What stands out in this studio scene?"
result = question("studio_scene.webp", QUESTION, expects="box")
print(result.text)
for box in result.points or []:
    print(box.mention, box.top_left, box.bottom_right)
```

CLI equivalent:

```bash
perceptron question studio_scene.webp "What stands out?" --expects box --format json
```

Use `expects="text"` for pure answers or keep `"box"` / `"point"` to cite evidence.
