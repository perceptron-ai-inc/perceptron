# Perceptron Prompting Guide

Optimal prompts for each SDK primitive with SDK and curl examples.

---

## Caption

| Style | Prompt |
|-------|--------|
| `concise` | `Provide a concise, human-friendly caption for the upcoming image.` |
| `detailed` | `Provide a detailed caption describing key objects, relationships, and context in the upcoming image.` |

### SDK

```python
from perceptron import configure, caption

configure(provider="perceptron", model="isaac-0.2-2b-preview", api_key="<your-api-key>")

result = caption("image.jpg", style="concise")
print(result.text)
```

### curl

```bash
curl -X POST "https://api.perceptron.inc/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
  "model": "isaac-0.2-2b-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "<image-url>"}},
        {"type": "text", "text": "Provide a concise, human-friendly caption for the upcoming image."}
      ]
    }
  ]
}'
```

---

## OCR

**System instruction:**
```
You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image.
```

### SDK

```python
from perceptron import configure, ocr

configure(provider="perceptron", model="isaac-0.2-2b-preview", api_key="<your-api-key>")

result = ocr("document.png")
print(result.text)

# With custom prompt
result = ocr("document.png", prompt="Extract the table data as CSV")
print(result.text)
```

### curl

```bash
curl -X POST "https://api.perceptron.inc/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
  "model": "isaac-0.2-2b-preview",
  "messages": [
    {
      "role": "system",
      "content": [
        {"type": "text", "text": "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "<image-url>"}}
      ]
    }
  ]
}'
```

---

## Detect

| Mode | Prompt |
|------|--------|
| General | `Your goal is to segment out the objects in the scene` |
| With classes | `Your goal is to segment out the following categories: {categories}` |

### SDK

```python
from perceptron import configure, detect

configure(provider="perceptron", model="isaac-0.2-2b-preview", api_key="<your-api-key>")

result = detect("warehouse.jpg", classes=["forklift", "person", "pallet"])

for box in result.points or []:
    print(f"{box.mention}: ({box.top_left.x}, {box.top_left.y}) to ({box.bottom_right.x}, {box.bottom_right.y})")
```

### curl

```bash
curl -X POST "https://api.perceptron.inc/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
  "model": "isaac-0.2-2b-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "<image-url>"}},
        {"type": "text", "text": "Your goal is to segment out the following categories: forklift, person, pallet"}
      ]
    }
  ]
}'
```

---

## Question

Pass your question directly as user content. For grounded responses, set `expects="box"` or `expects="point"`.

### SDK

```python
from perceptron import configure, question

configure(provider="perceptron", model="isaac-0.2-2b-preview", api_key="<your-api-key>")

# Simple Q&A
result = question("factory.jpg", "How many workers are visible?")
print(result.text)

# Grounded Q&A (with bounding boxes)
result = question("factory.jpg", "Where is the safety equipment?", expects="box")
for box in result.points or []:
    print(f"{box.mention}: ({box.top_left.x}, {box.top_left.y})")
```

### curl

```bash
curl -X POST "https://api.perceptron.inc/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
  "model": "isaac-0.2-2b-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "<image-url>"}},
        {"type": "text", "text": "Where is the safety equipment?"}
      ]
    }
  ]
}'
```

---

## Advanced: `@perceive` Decorator

For full control over prompts, reasoning, and structured output.

### Reasoning

```python
from perceptron import configure, perceive, image, text

configure(provider="perceptron", api_key="<your-api-key>")

@perceive(model="isaac-0.2-2b-preview", max_tokens=4096, reasoning=True)
def count_objects(img_url: str, query: str):
    return image(img_url) + text(query)

result = count_objects(
    "https://example.com/traffic.jpg",
    "Count the number of cars, excluding buses. Return JSON."
)
print(result.text)
```

### curl equivalent

```bash
curl -X POST "https://api.perceptron.inc/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
  "model": "isaac-0.2-2b-preview",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "<hint>THINK</hint>"}]
    },
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "<image-url>"}},
        {"type": "text", "text": "Count the number of cars, excluding buses. Return JSON."}
      ]
    }
  ]
}'
```

### Structured Output (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Literal
from perceptron import configure, perceive, image, text, pydantic_format

configure(provider="perceptron", api_key="<your-api-key>")

class SceneAnalysis(BaseModel):
    scene_type: str = Field(description="outdoor, indoor, urban, nature")
    main_subjects: list[str]
    mood: Literal["calm", "energetic", "dramatic", "peaceful", "tense"]

@perceive(model="isaac-0.2-1b", response_format=pydantic_format(SceneAnalysis))
def analyze_scene(img_path: str):
    return image(img_path) + text("Analyze this scene. Output JSON.")

result = analyze_scene("photo.jpg")
analysis = SceneAnalysis.model_validate_json(result.text)
print(f"Scene: {analysis.scene_type}, Mood: {analysis.mood}")
```
