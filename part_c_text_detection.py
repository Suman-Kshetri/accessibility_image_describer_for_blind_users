"""
Capstone Project: Accessibility Image Describer
Part C: Text Detection / OCR (part_c_text_detection.py)

Reads any text visible in the image (signs, labels, documents) using
EasyOCR - a deep-learning scene-text reader that detects text regions
first, then reads them. This is the right tool for text embedded in
photos (signs on posts, storefronts, labels against busy backgrounds);
Tesseract (the other common option) is built for flat scanned
documents and struggles badly on this kind of image.

Free and fully local, CPU-friendly. First run downloads the detection
+ recognition models (a few hundred MB total); cached after that.

Test this file on its own first:
    python part_c_text_detection.py path/to/image.jpg
"""

from PIL import Image
import easyocr
import sys

CONFIDENCE_THRESHOLD = 0.2

_reader = None


def get_reader():
    """Load the EasyOCR reader once and reuse it (loading is slow)."""
    global _reader
    if _reader is None:
        # quantize=False: default dynamic quantization uses PyTorch's
        # FBGEMM backend (needs AVX2/AVX512-VNNI); without it, it
        # crashes with "Illegal instruction" instead of raising.
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False, quantize=False)
    return _reader


def extract_text(image: Image.Image) -> str:
    """
    Run OCR on the image and return any detected text above the
    confidence threshold, joined into one string. Low-confidence
    detections (usually noise from tree branches, edges, etc.) are
    dropped rather than read aloud. Returns "" if nothing confident
    is found.
    """
    import numpy as np

    reader = get_reader()
    results = reader.readtext(np.array(image))

    words = [text for (_, text, conf) in results if conf >= CONFIDENCE_THRESHOLD]
    return " ".join(words)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python part_c_text_detection.py <image_path>")
        sys.exit(1)

    image = Image.open(sys.argv[1]).convert("RGB")
    text = extract_text(image)
    print("Detected text:", text if text else "(none found)")