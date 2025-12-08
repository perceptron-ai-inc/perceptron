##
from pathlib import Path

from urllib.request import urlretrieve
from IPython.display import Image as IPyImage
from IPython.display import display
from PIL import Image, ImageDraw

from perceptron import configure, image, perceive, text
from perceptron.pointing.geometry import scale_box_to_pixels

text_prompt = "Find every shipping box in the truck. Return one bounding box per item."
IMAGE_NAME = "truck_scene.jpg"
IMAGE_PATH = Path(IMAGE_NAME)
ANNOTATED_PATH = Path("annotated_"+IMAGE_NAME)

configure(
    provider="perceptron",
    api_key="ak.YTdqwv6A6bDXnew46uOHig.jusGKLeBUMx2B0tcKiSVJ46_VbTI2giNNAgn7L4-Iz0",
)

@perceive(model="isaac-0.1", expects="box", allow_multiple=True)
def detect(frame_path: str):
    scene = image(frame_path)
    return scene + text(text_prompt)

result = detect(str(IMAGE_PATH))

img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

for idx, box in enumerate(result.points or []):
    scaled = scale_box_to_pixels(box, width=img.width, height=img.height)
    top_left = scaled.top_left
    bottom_right = scaled.bottom_right
    tlx, tly = int(round(top_left.x)), int(round(top_left.y))
    brx, bry = int(round(bottom_right.x)), int(round(bottom_right.y))
    draw.rectangle([tlx, tly, brx, bry], outline="lime", width=3)
    label = box.mention or getattr(box, "label", None) or f"box {idx + 1}"
    draw.text((tlx, max(tly - 12, 0)), label, fill="lime")

img.save(ANNOTATED_PATH)
print("Annotated image saved to", ANNOTATED_PATH)