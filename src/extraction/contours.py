"""Contour detection for feature rectangles."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.types import BoundingBox


def detect_feature_rectangles(
    image: np.ndarray,
    sensitivity: int = 5,
) -> list[BoundingBox]:
    """Detect rectangular features in an image using contour detection.

    Args:
        image: RGB image as numpy array.
        sensitivity: Detection sensitivity (1-10). Higher = more sensitive.

    Returns:
        List of BoundingBox objects sorted top-to-bottom, then left-to-right.
    """
    # Clamp sensitivity
    sensitivity = max(1, min(10, sensitivity))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold with sensitivity-based block size
    block_size = 21 - sensitivity
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        2,
    )

    # Morphological dilation
    kernel_w = max(1, 20 - sensitivity)
    kernel_h = max(1, 8 - sensitivity // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    dilated = cv2.dilate(binary, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate area thresholds
    image_area = image.shape[0] * image.shape[1]
    min_area_ratio = 0.0005 + (10 - sensitivity) * 0.00015
    min_area = int(image_area * min_area_ratio)
    max_area = int(image_area * 0.95)

    # Extract and filter bounding boxes
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if min_area <= area <= max_area:
            boxes.append(BoundingBox(x=x, y=y, w=w, h=h))

    # Sort by Y position (top to bottom), then X (left to right)
    boxes.sort(key=lambda b: (b.y, b.x))

    return boxes
