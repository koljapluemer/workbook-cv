"""Text coverage calculation."""

from __future__ import annotations

import numpy as np

from src.core.types import OCRWord


def calculate_text_coverage(
    feature_shape: tuple[int, int],
    ocr_words: list[OCRWord],
) -> float:
    """Calculate the percentage of image area covered by detected text.

    Args:
        feature_shape: (height, width) of the feature image.
        ocr_words: List of detected OCR words.

    Returns:
        Coverage percentage (0-100).
    """
    h, w = feature_shape[:2]

    if h <= 0 or w <= 0:
        return 0.0

    # Create binary mask
    mask = np.zeros((h, w), dtype=np.uint8)

    for word in ocr_words:
        x1 = max(0, word.bbox.x)
        y1 = max(0, word.bbox.y)
        x2 = min(w, word.bbox.x + word.bbox.w)
        y2 = min(h, word.bbox.y + word.bbox.h)

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    # Calculate coverage
    total_pixels = h * w
    covered_pixels = np.count_nonzero(mask)

    return (covered_pixels / total_pixels) * 100
