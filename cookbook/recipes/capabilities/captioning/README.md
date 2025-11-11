# Captioning

Generate concise or detailed descriptions for any image with one function call, or request grounded snippets for downstream overlays.

```python
from perceptron import caption
from cookbook.utils import cookbook_asset

SUBURBAN_STREET = cookbook_asset("capabilities", "caption", "suburban_street.webp")

result = caption(SUBURBAN_STREET, style="concise", expects="text")
print(result.text)
```

Need boxes for each mentioned region? Switch to `expects="box"` and iterate through `result.points` to draw overlays. The CLI equivalent is:

```bash
perceptron caption cookbook/_shared/assets/capabilities/caption/suburban_street.webp --style detailed
```
