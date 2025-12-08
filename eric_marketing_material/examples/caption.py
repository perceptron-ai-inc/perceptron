## READY FOR CEDRIC REVIEW
from pathlib import Path

from perceptron import configure, caption

IMAGE_NAME = "up.jpg"
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "assets" / "template-images" / IMAGE_NAME
ANNOTATED_PATH = Path(__file__).resolve().parent / ("annotated_" + IMAGE_NAME)
BOX_STROKE_WIDTH = 4

configure(
    provider="perceptron",
    api_key="ak.YTdqwv6A6bDXnew46uOHig.jusGKLeBUMx2B0tcKiSVJ46_VbTI2giNNAgn7L4-Iz0",
)

result = caption(image_obj=IMAGE_PATH, model="isaac-0.1", expects="text", style="concise")
print(result.text)

# Output: A vibrant hot air balloon, filled with a multitude of colorful balloons, floats gracefully in a clear blue sky dotted with fluffy white clouds. The balloon, with its vivid hues of red, orange, yellow, green, blue, and purple, creates a striking contrast against the serene backdrop. At the base of the balloon, a small, quaint house is suspended, adding a whimsical touch to the scene. The overall image exudes a sense of joy and wonder, capturing the magic of a hot air balloon ride on a beautiful day.