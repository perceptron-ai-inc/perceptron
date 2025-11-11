# OCR with prompts

Use the OCR helper when you need structured transcriptions driven by your own instructions. Custom prompts let you capture SKUs, prices, serial numbers, or any other textual snippet you need.

```python
from perceptron import ocr
from cookbook.utils import cookbook_asset

LABELS = cookbook_asset("capabilities", "ocr", "grocery_labels.webp")

result = ocr(LABELS, prompt="Extract product name and price")
print(result.text)
```

The CLI provides the same experience:

```bash
perceptron ocr cookbook/_shared/assets/capabilities/ocr/grocery_labels.webp --prompt "Extract component labels"
```

Pair OCR output with detection or Q&A helpers to validate the surrounding visuals.
