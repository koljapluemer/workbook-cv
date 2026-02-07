"""OCR detection and overlay drawing."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.core.types import BoundingBox, OCRWord
from src.config import OCR_OVERLAY_COLOR


def detect_text(image: np.ndarray) -> list[OCRWord]:
    """Detect text in an image using pytesseract.

    Args:
        image: RGB image as numpy array.

    Returns:
        List of OCRWord objects with detected text.
    """
    try:
        import pytesseract
    except ImportError:
        raise RuntimeError(
            "pytesseract is required for OCR. "
            "Install it with: pip install pytesseract"
        )

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Run OCR
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

    words = []
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        if w <= 0 or h <= 0:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1.0

        bbox = BoundingBox(x=x, y=y, w=w, h=h)
        words.append(OCRWord(text=text, bbox=bbox, confidence=confidence))

    return words


def draw_ocr_overlay(
    image: np.ndarray,
    words: list[OCRWord],
    color: tuple[int, int, int] = OCR_OVERLAY_COLOR,
    thickness: int = 2,
) -> np.ndarray:
    """Draw OCR bounding boxes and text labels on an image.

    Args:
        image: RGB image as numpy array.
        words: List of detected OCR words.
        color: RGB color for the overlay.
        thickness: Line thickness for boxes.

    Returns:
        Image with overlay drawn.
    """
    overlay = image.copy()

    for word in words:
        x, y, w, h = word.bbox.x, word.bbox.y, word.bbox.w, word.bbox.h

        # Draw rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

        # Draw text label above the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        label_y = max(y - 5, 15)
        cv2.putText(
            overlay,
            word.text,
            (x, label_y),
            font,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay
