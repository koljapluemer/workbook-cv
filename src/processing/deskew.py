import cv2
import numpy as np


def deskew(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Detect text angle using Hough lines and rotate to horizontal.

    Returns:
        Tuple of (deskewed_image, metadata_dict with angle)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return image, {"angle": 0.0, "lines_detected": 0}

    # Calculate angles of all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (within 45 degrees)
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return image, {"angle": 0.0, "lines_detected": len(lines)}

    # Use median angle to be robust against outliers
    median_angle = float(np.median(angles))

    # Only correct if angle is significant but not too extreme
    if abs(median_angle) < 0.1:
        return image, {"angle": median_angle, "lines_detected": len(lines)}

    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Calculate new bounding box size
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h), borderValue=(255, 255, 255)
    )

    return rotated, {"angle": median_angle, "lines_detected": len(lines)}
