## READY FOR CEDRIC REVIEW
from pathlib import Path
from PIL import Image, ImageDraw
from perceptron import configure, ocr, scale_points_to_pixels 

IMAGE_NAME = "radio-city.jpg"
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "assets" / "template-images" / IMAGE_NAME
ANNOTATED_PATH = Path(__file__).resolve().parent / ("annotated_" + IMAGE_NAME)
BOX_STROKE_WIDTH = 4

configure(
    provider="perceptron",
    api_key="ak.YTdqwv6A6bDXnew46uOHig.jusGKLeBUMx2B0tcKiSVJ46_VbTI2giNNAgn7L4-Iz0",
)

result = ocr(image_obj=IMAGE_PATH, model="isaac-0.1", prompt="Read any signs.")
print(result.text)