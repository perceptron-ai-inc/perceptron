from perceptron import ocr


def transcribe_region(img):
    """OCR the image."""
    return ocr(img)


if __name__ == "__main__":
    res = transcribe_region("/path/to/document.png")
    print("text:", res.text)
    print("errors:", res.errors)
