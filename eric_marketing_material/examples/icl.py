##
from pathlib import Path

from PIL import Image, ImageDraw

from perceptron import configure, image, perceive, text
from perceptron.pointing.geometry import scale_box_to_pixels

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "template-images"
APPLE_EXAMPLE = ASSET_DIR / "apple.jpg"
CARROT_EXAMPLE = ASSET_DIR / "carrot.jpg"
TARGET_IMAGE_PATH = ASSET_DIR / "apples-carrots.webp"
ANNOTATED_PATH = Path(__file__).resolve().parent / ("annotated_" + TARGET_IMAGE_PATH.name)
BOX_COLOR = "#FF9640"
BOX_STROKE_WIDTH = 3

POLICY_TEXT = (
    "Produce quality check: Everything in Category 1 passes inspection. Everything in Category 2 must be discarded."
)

configure(
    provider="perceptron",
    api_key="ak.Lmtso-fLsOADgZrCiT_ZfA.9CJKGjEOD-lQgvcqMSV6E-ZkiD5drBoRoNzAaFxG96A",
    #api_key ="ak.sZCTuphOIMgcbxghVto06w.0qhUCNsaIetygu1gjGZjw1Eeivl8XBNEaDGLHn-Jd5k"
    #api_key="ak.NcIRkLVUXWxRjX8sCwNE2Q.L4aIoCUI6pTVtmHyeE5sj8rzl7gkknRcgefnu_9v7iE",
    #api_key="ak.YTdqwv6A6bDXnew46uOHig.jusGKLeBUMx2B0tcKiSVJ46_VbTI2giNNAgn7L4-Iz0",
)


@perceive(model="isaac-0.1", expects="box", allow_multiple=True)
def evaluate_produce():
    apple_example = image(APPLE_EXAMPLE)
    target_scene = image(TARGET_IMAGE_PATH)
    return (
        apple_example
        + text("This is Category 1. Action to take: cut into slices.")
        + target_scene
        + text(
            "For every instance of Category 1, output a bounding box labeled with the action to take."
        )
    )


result = evaluate_produce()

img = Image.open(TARGET_IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

for box in result.points or []:
    scaled = scale_box_to_pixels(box, width=img.width, height=img.height)
    top_left = scaled.top_left
    bottom_right = scaled.bottom_right
    tlx, tly = int(round(top_left.x)), int(round(top_left.y))
    brx, bry = int(round(bottom_right.x)), int(round(bottom_right.y))
    draw.rectangle([tlx, tly, brx, bry], outline=BOX_COLOR, width=BOX_STROKE_WIDTH)

img.save(ANNOTATED_PATH)
print(result.text or "No text output.")
print("Annotated image saved to", ANNOTATED_PATH)

# Output: <collection mention="Cut into slices"> <point_box> (545,60) (680,250) </point_box> <point_box> (680,60) (820,240) </point_box> <point_box> (630,240) (760,450) </point_box> <point_box> (380,370) (510,530) </point_box> </collection>
# Annotated image saved to /Users/ericpence/perceptron_repo/perceptron/eric_marketing_material/examples/annotated_apples-carrots.webp
