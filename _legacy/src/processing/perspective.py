import cv2
import numpy as np


def detect_page_contour(image: np.ndarray) -> np.ndarray | None:
    """
    Detect the page/document contour in an image.

    Returns:
        4-point contour array or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 75, 200)

    # Dilate to close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:  # Check top 5 largest
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If it has 4 points, it's likely the page
        if len(approx) == 4:
            # Check if area is significant (at least 20% of image)
            img_area = image.shape[0] * image.shape[1]
            contour_area = cv2.contourArea(approx)
            if contour_area > 0.2 * img_area:
                return approx.reshape(4, 2)

    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_correction(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Detect page and apply perspective warp to flatten it.

    Returns:
        Tuple of (corrected_image, metadata_dict)
    """
    contour = detect_page_contour(image)

    if contour is None:
        return image, {"page_detected": False}

    # Order the points
    pts = order_points(contour.astype(np.float32))
    tl, tr, br, bl = pts

    # Compute width of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    # Compute height of new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    # Destination points
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    # Compute perspective transform and apply
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped, {
        "page_detected": True,
        "corners": pts.tolist(),
        "output_size": [max_width, max_height],
    }
