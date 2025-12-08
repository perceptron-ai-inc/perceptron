## READY FOR CEDRIC REVIEW
from pathlib import Path
from PIL import Image, ImageDraw
from perceptron import configure, detect
from perceptron.pointing.geometry import scale_box_to_pixels

IMAGE_NAME = "construction-site.jpg"
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "assets" / "template-images" / IMAGE_NAME
ANNOTATED_PATH = Path(__file__).resolve().parent / ("annotated_" + IMAGE_NAME)
BOX_COLOR = "#FF9640"
BOX_STROKE_WIDTH = 6

configure(
    provider="perceptron",
    api_key="ak.YTdqwv6A6bDXnew46uOHig.jusGKLeBUMx2B0tcKiSVJ46_VbTI2giNNAgn7L4-Iz0",
)

result = detect(image_obj=IMAGE_PATH, model="isaac-0.1", classes=["heavy equipment"])
width, height = Image.open(IMAGE_PATH).size

img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

for box in result.points or []:
    scaled = scale_box_to_pixels(box, width=img.width, height=img.height)
    top_left = scaled.top_left
    bottom_right = scaled.bottom_right
    tlx, tly = int(round(top_left.x)), int(round(top_left.y))
    brx, bry = int(round(bottom_right.x)), int(round(bottom_right.y))
    draw.rectangle([tlx, tly, brx, bry], outline=BOX_COLOR, width=BOX_STROKE_WIDTH)

img.save(ANNOTATED_PATH)
print("Annotated image saved to", ANNOTATED_PATH)
