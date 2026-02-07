"""Contour detection for feature rectangles."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.types import BoundingBox


def merge_nearby_boxes(
    boxes: list[BoundingBox],
    horizontal_gap: int = 20,
    vertical_gap: int = 20,
) -> list[BoundingBox]:
    """Merge boxes that are close to each other.

    Args:
        boxes: List of bounding boxes.
        horizontal_gap: Max horizontal gap (in pixels) to merge.
        vertical_gap: Max vertical gap (in pixels) to merge.

    Returns:
        List of merged bounding boxes.
    """
    if not boxes:
        return []

    # Keep merging until no more merges possible
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue

            current = box1
            for j, box2 in enumerate(boxes):
                if j <= i or j in used:
                    continue

                # Check if boxes should be merged
                if should_merge(current, box2, horizontal_gap, vertical_gap):
                    current = merge_two_boxes(current, box2)
                    used.add(j)
                    merged = True

            new_boxes.append(current)
            used.add(i)

        boxes = new_boxes

    return boxes


def should_merge(
    box1: BoundingBox,
    box2: BoundingBox,
    horizontal_gap: int,
    vertical_gap: int,
) -> bool:
    """Check if two boxes should be merged based on proximity."""
    # Expand boxes by the gap thresholds
    x1_min = box1.x - horizontal_gap
    x1_max = box1.x + box1.w + horizontal_gap
    y1_min = box1.y - vertical_gap
    y1_max = box1.y + box1.h + vertical_gap

    x2_min = box2.x
    x2_max = box2.x + box2.w
    y2_min = box2.y
    y2_max = box2.y + box2.h

    # Check for overlap with expanded box1
    h_overlap = x1_min <= x2_max and x2_min <= x1_max
    v_overlap = y1_min <= y2_max and y2_min <= y1_max

    return h_overlap and v_overlap


def merge_two_boxes(box1: BoundingBox, box2: BoundingBox) -> BoundingBox:
    """Merge two boxes into one encompassing box."""
    x = min(box1.x, box2.x)
    y = min(box1.y, box2.y)
    x2 = max(box1.x + box1.w, box2.x + box2.w)
    y2 = max(box1.y + box1.h, box2.y + box2.h)
    return BoundingBox(x=x, y=y, w=x2 - x, h=y2 - y)


def detect_feature_rectangles(
    image: np.ndarray,
    sensitivity: int = 8,
    merge_horizontal: int = 0,
    merge_vertical: int = 0,
) -> list[BoundingBox]:
    """Detect rectangular features in an image using contour detection.

    Args:
        image: RGB image as numpy array.
        sensitivity: Detection sensitivity (1-10). Higher = more sensitive.
        merge_horizontal: Horizontal gap threshold for merging boxes (pixels).
        merge_vertical: Vertical gap threshold for merging boxes (pixels).

    Returns:
        List of BoundingBox objects sorted top-to-bottom, then left-to-right.
    """
    # Clamp sensitivity
    sensitivity = max(1, min(10, sensitivity))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold with sensitivity-based block size
    block_size = 25 - sensitivity
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

    # Morphological dilation - smaller kernel to avoid merging adjacent features
    kernel_w = max(1, 12 - sensitivity)
    kernel_h = max(1, 5 - sensitivity // 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate area thresholds - lower minimum to catch smaller features
    image_area = image.shape[0] * image.shape[1]
    min_area_ratio = 0.0002 + (10 - sensitivity) * 0.0001
    min_area = int(image_area * min_area_ratio)
    max_area = int(image_area * 0.95)

    # Extract and filter bounding boxes
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if min_area <= area <= max_area:
            boxes.append(BoundingBox(x=x, y=y, w=w, h=h))

    # Merge nearby boxes if thresholds are set
    if merge_horizontal > 0 or merge_vertical > 0:
        boxes = merge_nearby_boxes(boxes, merge_horizontal, merge_vertical)

    # Sort by Y position (top to bottom), then X (left to right)
    boxes.sort(key=lambda b: (b.y, b.x))

    return boxes
